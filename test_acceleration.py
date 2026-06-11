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

from models import networks
from models.didn import DIDN
from options.test_options import TestOptions
from util.metrics import PSNR
from util.util import complex_conj, complex_matmul, fft2, ifft2


opt = TestOptions().parse()
opt.batchSize = 1

PAPER_DEFAULT_TRAIN_VALI_SIZE = 3032
PAPER_DEFAULT_TEST_SIZE = 64
PAPER_ACCELERATIONS = [2, 3, 4, 5, 6, 7, 8]
MASK_SEED = int(os.environ.get('SMUG_ACCEL_MASK_SEED', '0'))

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


def make_vdrs_mask(nx, ny, acceleration, seed):
    rng = np.random.default_rng(seed)
    target_lines = max(1, int(round(ny / acceleration)))
    low_freqs = max(1, int(round(ny * 0.32 / acceleration)))
    low_freqs = min(low_freqs, target_lines)

    mask_1d = np.zeros(ny, dtype=np.bool_)
    center_start = (ny - low_freqs) // 2
    center_end = center_start + low_freqs
    mask_1d[center_start:center_end] = True

    remaining = target_lines - low_freqs
    if remaining > 0:
        candidates = np.concatenate([np.arange(0, center_start), np.arange(center_end, ny)])
        chosen = rng.choice(candidates, size=remaining, replace=False)
        mask_1d[chosen] = True
    return np.repeat(mask_1d[np.newaxis, :], nx, axis=0)


def adjoint_from_kspace(kspace, smap, mask):
    _, num_coils, _, _, _ = kspace.shape
    mask_coil = mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    k_under = kspace * mask_coil
    im_u = ifft2(k_under.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    return complex_matmul(im_u, complex_conj(smap)).sum(1)


class AccelerationDataset(Dataset):
    def __init__(self, dataroot, mask_root, start, count, acceleration):
        dataroot = Path(dataroot)
        self.paths = sorted(p for p in dataroot.glob('*.npz'))[start:start + count]
        self.base_masks = load_masks_by_shape(mask_root)
        self.acceleration = acceleration
        if len(self.paths) == 0:
            raise RuntimeError(f'No test files found from start={start}, count={count} in {dataroot}')

    def __len__(self):
        return len(self.paths)

    def mask_for_shape(self, nx, ny):
        if self.acceleration == 4 and (nx, ny) in self.base_masks:
            return self.base_masks[(nx, ny)]
        return make_vdrs_mask(nx, ny, self.acceleration, MASK_SEED + int(self.acceleration * 1000) + ny)

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
        mask_np = self.mask_for_shape(nx, ny)

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


def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)


def average_repeated_batch(tensor, batch_size, num_sample):
    return tensor.reshape(num_sample, batch_size, *tensor.shape[1:]).mean(dim=0)


def weighted_average_repeated_batch(values, weights, batch_size, num_sample):
    values = values.reshape(num_sample, batch_size, *values.shape[1:])
    weights = weights.reshape(num_sample, batch_size, 1, 1, 1)
    return (weights * values).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-8)


def kspace_mask_like(kspace, mask):
    _, num_coils, _, _, _ = kspace.shape
    return mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)


def recon_from_aliased(aliased_image, smap, mask, smoothing):
    output_cg = aliased_image
    batch_size = aliased_image.shape[0]

    for _ in range(opt.blockIter):
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
            output_cg = CG(output_nn, opt.CGtol, opt.Lambda, smap.repeat(opt.num_sample, 1, 1, 1, 1), mask.repeat(opt.num_sample, 1, 1, 1), aliased_image.repeat(opt.num_sample, 1, 1, 1))
            output_cg = average_repeated_batch(output_cg, batch_size, opt.num_sample)
            continue
        else:
            raise ValueError(f'Unsupported smoothing mode: {smoothing}')
        output_cg = CG(output_nn, opt.CGtol, opt.Lambda, smap, mask, aliased_image)
    return output_cg


def recon_from_kspace(kspace, smap, mask):
    if opt.smoothing == 'RSE2E':
        sampled = kspace_mask_like(kspace, mask)
        repeated_kspace = kspace.repeat(opt.num_sample, 1, 1, 1, 1)
        repeated_smap = smap.repeat(opt.num_sample, 1, 1, 1, 1)
        repeated_mask = mask.repeat(opt.num_sample, 1, 1, 1)
        noise = torch.normal(0, opt.smoothing_epsilon, repeated_kspace.shape, device=repeated_kspace.device) * sampled.repeat(opt.num_sample, 1, 1, 1, 1)
        noisy_kspace = repeated_kspace + noise
        aliased = adjoint_from_kspace(noisy_kspace, repeated_smap, repeated_mask)
        outputs = recon_from_aliased(aliased, repeated_smap, repeated_mask, 'none')
        return average_repeated_batch(outputs, kspace.shape[0], opt.num_sample)

    aliased = adjoint_from_kspace(kspace, smap, mask)
    return recon_from_aliased(aliased, smap, mask, opt.smoothing)


message = ''
message += 'Paper acceleration test: models trained at 4x, evaluated at accelerations 2x..8x.\n'
message += f'smoothing: {opt.smoothing}, blockIter={opt.blockIter}, num_sample={opt.num_sample}, smoothing_epsilon={opt.smoothing_epsilon}\n'
message += f'train_valiSize={opt.train_valiSize}, testSize={opt.testSize}, mask_seed={MASK_SEED}\n\n'

print('Paper acceleration test: accelerations=2..8')
print(f'Test split: train_valiSize={opt.train_valiSize}, testSize={opt.testSize}, smoothing={opt.smoothing}')

for acceleration in PAPER_ACCELERATIONS:
    dataset = AccelerationDataset(opt.dataroot, opt.mask_dataroot, opt.train_valiSize, opt.testSize, acceleration)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_psnr = []
    test_ssim = []

    for i, (kspace, _, target, smap, mask) in enumerate(loader):
        kspace = kspace.to(device).float()
        target = target.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()

        torch.manual_seed(0)
        with torch.no_grad():
            result = recon_from_kspace(kspace, smap, mask)
            test_psnr.append(float(PSNR(target, result)))
            test_ssim.append(float(ssim_loss(target, result)))

        if opt.visualize and i == 2:
            image = result.detach().cpu().numpy().squeeze(0)
            image = image[0] + image[1] * 1j
            plt.imshow(np.abs(image), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(opt.netGpath[:-4] + f'_{acceleration}x_acceleration.pdf', dpi=600)
            plt.close()

    message += f'{acceleration}x acceleration:\n'
    message += f'PSNR: {np.average(test_psnr):.4f} ± {np.std(test_psnr):.4f}\n'
    message += f'SSIM: {np.average(test_ssim):.4f} ± {np.std(test_ssim):.4f}\n\n'
    print(f'{acceleration}x: PSNR {np.average(test_psnr):.4f} ± {np.std(test_psnr):.4f}, SSIM {np.average(test_ssim):.4f} ± {np.std(test_ssim):.4f}')

file_name = opt.netGpath[:-4] + '_paper_acceleration_test.out'
with open(file_name, 'w') as result_file:
    result_file.write(message)
print(message)
