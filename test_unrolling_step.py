"""
Evaluates how reconstruction quality and adversarial robustness change when the number of MoDL unrolling steps is varied. 
It tests steps from 0 to 16.
"""

import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
from numpy.lib import format as npfmt
from torch.utils.data import DataLoader, Dataset

from models.didn import DIDN
from options.test_options import TestOptions
from util.metrics import PSNR
from util.util import complex_conj, complex_matmul, fft2, ifft2


opt = TestOptions().parse()
opt.batchSize = 1

PAPER_DEFAULT_TRAIN_VALI_SIZE = 3032
PAPER_DEFAULT_TEST_SIZE = 64
PAPER_PGD_EPSILON = 0.02
PAPER_UNROLLING_STEPS = list(range(1, 17))
CG_MAX_ITER = int(os.environ.get('SMUG_CG_MAX_ITER', '30'))

if opt.train_valiSize == 336:
    opt.train_valiSize = PAPER_DEFAULT_TRAIN_VALI_SIZE
if opt.testSize != PAPER_DEFAULT_TEST_SIZE:
    opt.testSize = opt.testSize


device = torch.device(('cuda:' + str(opt.gpu_ids[0])) if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')


class WeightEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=16):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(5):
            layers.extend([
                nn.Conv2d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = hidden_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.output(self.fc(x))
        return x.view(x.shape[0], 1, 1, 1)


def load_model(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    netG = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=opt.n_res_blocks)
    weight_encoder = None

    if isinstance(checkpoint, dict) and 'netG' in checkpoint and 'weight_encoder' in checkpoint:
        netG.load_state_dict(checkpoint['netG'])
        hidden_channels = checkpoint['weight_encoder']['features.0.weight'].shape[0]
        weight_encoder = WeightEncoder(in_channels=2, hidden_channels=hidden_channels)
        weight_encoder.load_state_dict(checkpoint['weight_encoder'])
    elif isinstance(checkpoint, dict) and 'netG' in checkpoint:
        netG.load_state_dict(checkpoint['netG'])
    else:
        netG.load_state_dict(checkpoint)

    netG = netG.float().to(device).eval()
    if weight_encoder is not None:
        weight_encoder = weight_encoder.float().to(device).eval()
    return netG, weight_encoder


netG, weight_encoder = load_model(opt.netGpath)
if opt.smoothing == 'WeightedSMUG' and weight_encoder is None:
    raise ValueError('WeightedSMUG requires a combined checkpoint containing netG and weight_encoder.')

mse_loss = nn.MSELoss().to(device)
ssim_loss = pytorch_msssim.SSIM(data_range=2.0, channel=2).to(device)


def read_npz_shape(npz_path, key='k_r'):
    with zipfile.ZipFile(npz_path) as zf, zf.open(f'{key}.npy') as f:
        version = npfmt.read_magic(f)
        if version == (1, 0):
            shape, _, _ = npfmt.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = npfmt.read_array_header_2_0(f)
        else:
            shape, _, _ = npfmt._read_array_header(f, version)
    return tuple(shape)


def load_masks_by_shape(mask_root):
    masks = {}
    for filename in sorted(os.listdir(mask_root)):
        if not filename.endswith('.npy'):
            continue
        mask = np.load(os.path.join(mask_root, filename), 'r').astype(np.bool_)
        if mask.ndim == 2:
            masks[tuple(mask.shape)] = mask
    if not masks:
        raise FileNotFoundError(f'No 2D .npy masks found in {mask_root}')
    return masks


def adjoint_from_kspace(kspace, smap, mask):
    _, num_coils, _, _, _ = kspace.shape
    mask_coil = mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    k_under = kspace * mask_coil
    im_u = ifft2(k_under.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    return complex_matmul(im_u, complex_conj(smap)).sum(1)


class KspaceDataset(Dataset):
    def __init__(self, dataroot, mask_root, start, count):
        dataroot = Path(dataroot)
        self.paths = sorted(p for p in dataroot.glob('*.npz'))[start:start + count]
        self.masks_by_shape = load_masks_by_shape(mask_root)
        if len(self.paths) == 0:
            raise RuntimeError(f'No test files found from start={start}, count={count} in {dataroot}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        npz_path = self.paths[index]
        with np.load(npz_path, 'r') as data:
            s_r = data['s_r'].astype(np.float32) / 32767.0
            s_i = data['s_i'].astype(np.float32) / 32767.0
            k_r = data['k_r'].astype(np.float32) / 32767.0
            k_i = data['k_i'].astype(np.float32) / 32767.0

        _, nx, ny = s_r.shape
        top = nx // 2 - 160
        left = ny // 2 - 160
        mask_np = self.masks_by_shape.get((nx, ny))
        if mask_np is None:
            raise KeyError(f'Missing mask for shape {(nx, ny)} from {npz_path.name}')

        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, top:top + 320, left:left + 320], s_i[:, top:top + 320, left:left + 320]), axis=0)
        mask_crop = mask_np[top:top + 320, left:left + 320]

        mask = torch.tensor(np.repeat(mask_crop[np.newaxis], 2, axis=0), dtype=torch.float32)
        k_full = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        image = ifft2(k_full.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        image = image[:, :, top:top + 320, left:left + 320]
        smap = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)

        sos = torch.sum(complex_matmul(image, complex_conj(smap)), dim=0)
        image = image / torch.max(torch.abs(sos))
        kspace = fft2(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        direct = adjoint_from_kspace(kspace.unsqueeze(0), smap.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
        target = adjoint_from_kspace(kspace.unsqueeze(0), smap.unsqueeze(0), torch.ones_like(mask).unsqueeze(0)).squeeze(0)
        return kspace, direct, target, smap, mask


def apply_ata(image, smap, mask, lam):
    _, num_coils, _, _, _ = smap.shape
    image_coil = image.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    image_s = complex_matmul(image_coil, smap)
    k_full = fft2(image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    mask_coil = mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    im_u = ifft2((k_full * mask_coil).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    return complex_matmul(im_u, complex_conj(smap)).sum(1) + lam * image


def cg_differentiable(dn, tol, lam, smap, mask, aliased_image):
    lam = torch.as_tensor(lam, dtype=dn.dtype, device=dn.device)
    b0 = dn * lam + aliased_image
    xk = dn
    rk = b0 - apply_ata(xk, smap, mask, lam)
    pk = rk
    for _ in range(CG_MAX_ITER):
        if float(torch.norm(rk).detach().cpu()) <= tol:
            break
        rtr = torch.pow(torch.norm(rk), 2)
        apk = apply_ata(pk, smap, mask, lam)
        alpha = rtr / (torch.sum(complex_matmul(complex_conj(pk), apk)) + 1e-12)
        xk = xk + alpha * pk
        rk_next = rk - alpha * apk
        beta = torch.pow(torch.norm(rk_next), 2) / (rtr + 1e-12)
        pk = rk_next + beta * pk
        rk = rk_next
    return xk


def average_repeated_batch(tensor, batch_size, num_sample):
    return tensor.reshape(num_sample, batch_size, *tensor.shape[1:]).mean(dim=0)


def weighted_average_repeated_batch(values, weights, batch_size, num_sample):
    values = values.reshape(num_sample, batch_size, *values.shape[1:])
    weights = weights.reshape(num_sample, batch_size, 1, 1, 1)
    return (weights * values).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-8)


def kspace_mask_like(kspace, mask):
    _, num_coils, _, _, _ = kspace.shape
    return mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)


def recon_from_aliased(aliased_image, smap, mask, cg_iter):
    output_cg = aliased_image
    batch_size = aliased_image.shape[0]
    smoothing = opt.smoothing

    for _ in range(cg_iter):
        if smoothing == 'none':
            output_nn = netG(output_cg)
        elif smoothing in ('SMUG', 'WeightedSMUG'):
            repeated = output_cg.repeat(opt.num_sample, 1, 1, 1)
            noise = torch.normal(0, opt.smoothing_epsilon, repeated.shape, device=repeated.device)
            noised_input = torch.clamp(repeated + noise, min=-1, max=1)
            denoised = netG(noised_input)
            if smoothing == 'SMUG':
                output_nn = average_repeated_batch(denoised, batch_size, opt.num_sample)
            else:
                weights = weight_encoder(noised_input)
                output_nn = weighted_average_repeated_batch(denoised, weights, batch_size, opt.num_sample)
        elif smoothing == 'SMUGv0':
            repeated = output_cg.repeat(opt.num_sample, 1, 1, 1)
            noise = torch.normal(0, opt.smoothing_epsilon, repeated.shape, device=repeated.device)
            noised_input = torch.clamp(repeated + noise, min=-1, max=1)
            output_nn = netG(noised_input)
            output_cg = cg_differentiable(output_nn, opt.CGtol, opt.Lambda, smap.repeat(opt.num_sample, 1, 1, 1, 1), mask.repeat(opt.num_sample, 1, 1, 1), aliased_image.repeat(opt.num_sample, 1, 1, 1))
            output_cg = average_repeated_batch(output_cg, batch_size, opt.num_sample)
            continue
        else:
            raise ValueError(f'Unsupported smoothing mode: {smoothing}')
        output_cg = cg_differentiable(output_nn, opt.CGtol, opt.Lambda, smap, mask, aliased_image)
    return output_cg


def recon_from_kspace(kspace, smap, mask, cg_iter):
    if opt.smoothing == 'RSE2E':
        sampled = kspace_mask_like(kspace, mask)
        repeated_kspace = kspace.repeat(opt.num_sample, 1, 1, 1, 1)
        repeated_smap = smap.repeat(opt.num_sample, 1, 1, 1, 1)
        repeated_mask = mask.repeat(opt.num_sample, 1, 1, 1)
        noise = torch.normal(0, opt.smoothing_epsilon, repeated_kspace.shape, device=repeated_kspace.device) * sampled.repeat(opt.num_sample, 1, 1, 1, 1)
        noisy_kspace = repeated_kspace + noise
        aliased = adjoint_from_kspace(noisy_kspace, repeated_smap, repeated_mask)
        old_smoothing = opt.smoothing
        opt.smoothing = 'none'
        outputs = recon_from_aliased(aliased, repeated_smap, repeated_mask, cg_iter)
        opt.smoothing = old_smoothing
        return average_repeated_batch(outputs, kspace.shape[0], opt.num_sample)

    aliased = adjoint_from_kspace(kspace, smap, mask)
    return recon_from_aliased(aliased, smap, mask, cg_iter)


def pgd_kspace(kspace, target, smap, mask, cg_iter):
    netG.requires_grad_(False)
    if weight_encoder is not None:
        weight_encoder.requires_grad_(False)

    sampled = kspace_mask_like(kspace, mask)
    original = kspace.detach()
    delta = torch.normal(0, PAPER_PGD_EPSILON, original.shape, device=original.device)
    delta = torch.clamp(delta, min=-PAPER_PGD_EPSILON, max=PAPER_PGD_EPSILON) * sampled
    adversarial_kspace = (original + delta).detach()
    alpha = PAPER_PGD_EPSILON / 3.0

    for _ in range(opt.pgd_steps):
        adversarial_kspace.requires_grad_(True)
        output = recon_from_kspace(adversarial_kspace, smap, mask, cg_iter)
        loss = mse_loss(output, target)
        netG.zero_grad(set_to_none=True)
        if weight_encoder is not None:
            weight_encoder.zero_grad(set_to_none=True)
        loss.backward()
        adversarial_kspace = adversarial_kspace + alpha * adversarial_kspace.grad.sign()
        delta = torch.clamp(adversarial_kspace - original, min=-PAPER_PGD_EPSILON, max=PAPER_PGD_EPSILON) * sampled
        adversarial_kspace = (original + delta).detach()

    netG.requires_grad_(True)
    if weight_encoder is not None:
        weight_encoder.requires_grad_(True)
    return adversarial_kspace


_, _, _ = opt.dataroot, opt.mask_dataroot, opt.netGpath
test_dataset = KspaceDataset(opt.dataroot, opt.mask_dataroot, opt.train_valiSize, opt.testSize)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

clean_psnr = [[] for _ in PAPER_UNROLLING_STEPS]
clean_ssim = [[] for _ in PAPER_UNROLLING_STEPS]
robust_psnr = [[] for _ in PAPER_UNROLLING_STEPS]
robust_ssim = [[] for _ in PAPER_UNROLLING_STEPS]

print(f'Paper unrolling-step test: steps=1..16, PGD epsilon={PAPER_PGD_EPSILON}, PGD steps={opt.pgd_steps}')
print(f'Test split: train_valiSize={opt.train_valiSize}, testSize={opt.testSize}, smoothing={opt.smoothing}')

for i, (kspace, _, target, smap, mask) in enumerate(test_loader):
    kspace = kspace.to(device).float()
    target = target.to(device).float()
    smap = smap.to(device).float()
    mask = mask.to(device).float()

    for j, step in enumerate(PAPER_UNROLLING_STEPS):
        torch.manual_seed(0)
        with torch.no_grad():
            test_result = recon_from_kspace(kspace, smap, mask, step)
            clean_psnr[j].append(float(PSNR(target, test_result)))
            clean_ssim[j].append(float(ssim_loss(target, test_result)))

        torch.manual_seed(0)
        adversarial_kspace = pgd_kspace(kspace, target, smap, mask, step)
        torch.manual_seed(0)
        with torch.no_grad():
            adv_result = recon_from_kspace(adversarial_kspace, smap, mask, step)
            robust_psnr[j].append(float(PSNR(target, adv_result)))
            robust_ssim[j].append(float(ssim_loss(target, adv_result)))

    if opt.visualize and i == 2:
        image = test_result.detach().cpu().numpy().squeeze(0)
        image = image[0] + image[1] * 1j
        plt.imshow(np.abs(image), cmap='gray')
        plt.axis('off')
        plt.savefig(opt.netGpath[:-4] + '_16_steps.pdf', dpi=600)
        plt.close()

message = ''
message += f'Paper setting: 4x mask, unrolling steps 1..16, k-space PGD epsilon={PAPER_PGD_EPSILON}, PGD steps={opt.pgd_steps}\n'
message += f'smoothing: {opt.smoothing}, num_sample={opt.num_sample}, smoothing_epsilon={opt.smoothing_epsilon}\n'
message += f'train_valiSize={opt.train_valiSize}, testSize={opt.testSize}\n\n'
for j, step in enumerate(PAPER_UNROLLING_STEPS):
    message += f'{step} steps:\n'
    message += f'Clean PSNR: {np.average(clean_psnr[j]):.4f} ± {np.std(clean_psnr[j]):.4f}\n'
    message += f'Clean SSIM: {np.average(clean_ssim[j]):.4f} ± {np.std(clean_ssim[j]):.4f}\n'
    message += f'Robust PSNR: {np.average(robust_psnr[j]):.4f} ± {np.std(robust_psnr[j]):.4f}\n'
    message += f'Robust SSIM: {np.average(robust_ssim[j]):.4f} ± {np.std(robust_ssim[j]):.4f}\n\n'

file_name = opt.netGpath[:-4] + '_paper_steps_test.out'
with open(file_name, 'w') as result_file:
    result_file.write(message)
print(message)
