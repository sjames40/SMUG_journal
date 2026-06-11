"""
Fine-tunes Weighted-SMUG by adding a trainable weight encoder that predicts one smoothing weight for each noisy denoiser input. 
It jointly trains the DIDN denoiser and weight encoder, 
saving a combined checkpoint plus separate netG and weight_encoder weight files.
"""

import os

import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

import global_network_dataset
from models import networks
from models.didn import DIDN
from options.tune_options import TuneOptions
from util.metrics import PSNR


opt = TuneOptions().parse()
opt.smoothing = 'WeightedSMUG'

device = torch.device("cuda:" + str(opt.gpu_ids[0]) if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else "cpu")


class WeightEncoder(nn.Module):
    """E_phi from the Weighted SMUG paper: 5 Conv-BN-ReLU blocks + Linear + Sigmoid."""

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


def load_denoiser(path):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'netG' in checkpoint:
        return checkpoint['netG']
    return checkpoint


netG = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=opt.n_res_blocks)
netG.load_state_dict(load_denoiser(opt.netGpath))
netG = netG.float()

weight_channels = int(os.environ.get('WEIGHT_ENCODER_CHANNELS', '16'))
weight_encoder = WeightEncoder(in_channels=2, hidden_channels=weight_channels).float()

if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
    netG = nn.DataParallel(netG, device_ids=opt.gpu_ids)
    weight_encoder = nn.DataParallel(weight_encoder, device_ids=opt.gpu_ids)

netG = netG.to(device)
weight_encoder = weight_encoder.to(device)


def lambda_rule(epoch):
    return 1.0 - max(0, epoch + 1 - 20) / float(opt.epoch + 1 - 20)


mse_loss = nn.MSELoss().to(device)
ssim_loss = pytorch_msssim.SSIM(data_range=2.0, channel=2).to(device)
optimG = torch.optim.Adam(
    list(netG.parameters()) + list(weight_encoder.parameters()),
    lr=opt.lr,
    betas=[0.5, 0.999],
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimG, lr_lambda=lambda_rule)
GRAD_CLIP_NORM = float(os.environ.get('WEIGHTED_SMUG_GRAD_CLIP', '1.0'))


def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)


def weighted_average_repeated_batch(values, weights, batch_size, num_sample):
    values = values.reshape(num_sample, batch_size, *values.shape[1:])
    weights = weights.reshape(num_sample, batch_size, 1, 1, 1)
    numerator = (weights * values).sum(dim=0)
    denominator = weights.sum(dim=0).clamp_min(1e-8)
    return numerator / denominator


def Recon(cg_iter, smap, mask, input, label, num_sample=10, epsilon=0.01, is_train=False):
    output_CG = input
    loss = torch.zeros((), device=device)
    target_denoised = netG(label) if is_train else None
    batch_size = input.shape[0]

    for _ in range(cg_iter):
        output_CG_i = output_CG.repeat(num_sample, 1, 1, 1)
        noises = torch.normal(0, epsilon, output_CG_i.shape).to(device)
        noised_input = torch.clamp(output_CG_i + noises, min=-1, max=1)

        if is_train:
            output_NN = checkpoint(netG, noised_input, use_reentrant=False)
            weights = weight_encoder(noised_input)
        else:
            output_NN = netG(noised_input)
            weights = weight_encoder(noised_input)
        output_NN_final = weighted_average_repeated_batch(output_NN, weights, batch_size, num_sample)

        if is_train:
            loss += loss_fn(output_NN, target_denoised.repeat(num_sample, 1, 1, 1))

        output_CG = CG(output_NN_final, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)

    if is_train:
        loss += loss_fn(output_CG, label) * opt.LossLambda

    return output_CG, loss


def loss_fn(outputs, labels):
    return mse_loss(outputs, labels)


def module_state_dict(module):
    return module.module.state_dict() if isinstance(module, nn.DataParallel) else module.state_dict()


train_rmse = []
vali_rmse = []
vali_rmse_min = None
train_psnr = []
vali_psnr = []
train_ssim = []
vali_ssim = []

train_loader, test_loader = global_network_dataset.loadData(
    opt.dataroot,
    opt.mask_dataroot,
    opt.trainSize,
    opt.valiSize,
    opt.batchSize,
)

train_size = len(train_loader.dataset)
vali_size = len(test_loader.dataset)
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

for epoch in tqdm(range(opt.epoch), desc='Epoch'):
    train_rmse_total = 0.
    train_psnr_total = 0.
    train_ssim_total = 0.

    netG.train()
    weight_encoder.train()
    train_bar = tqdm(train_loader, desc=f'Train {epoch + 1}/{opt.epoch}', leave=False)
    for direct, target, smap, mask in train_bar:
        input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()

        output, loss_G = Recon(
            cg_iter=opt.blockIter,
            smap=smap,
            mask=mask,
            input=input,
            label=label,
            num_sample=opt.num_sample,
            epsilon=opt.smoothing_epsilon,
            is_train=True,
        )

        if not torch.isfinite(loss_G):
            raise FloatingPointError(f'Non-finite Weighted-SMUG training loss at epoch {epoch}: {float(loss_G.detach())}')

        optimG.zero_grad(set_to_none=True)
        loss_G.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                list(netG.parameters()) + list(weight_encoder.parameters()),
                max_norm=GRAD_CLIP_NORM,
            )
        optimG.step()

        with torch.no_grad():
            psnr_train = PSNR(label, output)
            ssim_train = ssim_loss(label, output)
            batch_rmse = np.sqrt(float(mse_loss(output, label)))
            batch_psnr = float(psnr_train)
            batch_ssim = float(ssim_train)
            train_rmse_total += batch_rmse
            train_psnr_total += batch_psnr
            train_ssim_total += batch_ssim
            train_bar.set_postfix(
                loss=f'{float(loss_G.detach()):.4f}',
                rmse=f'{batch_rmse:.4f}',
                psnr=f'{batch_psnr:.2f}',
                ssim=f'{batch_ssim:.4f}',
            )

    vali_rmse_total = 0.
    vali_psnr_total = 0.
    vali_ssim_total = 0.

    netG.eval()
    weight_encoder.eval()
    vali_bar = tqdm(test_loader, desc=f'Vali {epoch + 1}/{opt.epoch}', leave=False)
    for vali_direct, vali_target, vali_smap, vali_mask in vali_bar:
        vali_input = vali_direct.to(device).float()
        vali_smap = vali_smap.to(device).float()
        vali_mask = vali_mask.to(device).float()
        vali_label = vali_target.to(device).float()

        with torch.no_grad():
            vali_result, _ = Recon(
                cg_iter=opt.blockIter,
                smap=vali_smap,
                mask=vali_mask,
                input=vali_input,
                label=vali_label,
                num_sample=opt.num_sample,
                epsilon=opt.smoothing_epsilon,
            )

            psnr_vali = PSNR(vali_label, vali_result)
            ssim_vali = ssim_loss(vali_label, vali_result)
            batch_vali_rmse = np.sqrt(float(mse_loss(vali_result, vali_label)))
            batch_vali_psnr = float(psnr_vali)
            batch_vali_ssim = float(ssim_vali)
            vali_rmse_total += batch_vali_rmse
            vali_psnr_total += batch_vali_psnr
            vali_ssim_total += batch_vali_ssim
            vali_bar.set_postfix(
                rmse=f'{batch_vali_rmse:.4f}',
                psnr=f'{batch_vali_psnr:.2f}',
                ssim=f'{batch_vali_ssim:.4f}',
            )

    scheduler.step()
    curr_lr = optimG.param_groups[0]['lr']
    print(f'learning rate: {curr_lr:.6f}')

    if vali_rmse_min is None or vali_rmse_total < vali_rmse_min:
        vali_rmse_min = vali_rmse_total
        checkpoint_payload = {
            'netG': module_state_dict(netG),
            'weight_encoder': module_state_dict(weight_encoder),
            'epoch': epoch,
            'vali_rmse': vali_rmse_min,
            'weighted_smug': True,
        }
        torch.save(checkpoint_payload, os.path.join(expr_dir, 'vali_best.pth'))
        torch.save(module_state_dict(netG), os.path.join(expr_dir, 'vali_best_netG.pth'))
        torch.save(module_state_dict(weight_encoder), os.path.join(expr_dir, 'vali_best_weight_encoder.pth'))
        print(f'saving weighted vali best model at epoch {epoch}')

    train_rmse.append(train_rmse_total / train_size * opt.batchSize)
    vali_rmse.append(vali_rmse_total / vali_size * opt.batchSize)
    train_psnr.append(train_psnr_total / train_size * opt.batchSize)
    vali_psnr.append(vali_psnr_total / vali_size * opt.batchSize)
    train_ssim.append(train_ssim_total / train_size * opt.batchSize)
    vali_ssim.append(vali_ssim_total / vali_size * opt.batchSize)

    print(f'Epoch {epoch}:')
    print(f'Train RMSE: {train_rmse[epoch]:.4f} \tTrain PSNR: {train_psnr[epoch]:.4f} \tTrain SSIM: {train_ssim[epoch]:.4f}')
    print(f'Vali RMSE: {vali_rmse[epoch]:.4f} \tVali RSNR: {vali_psnr[epoch]:.4f} \tVali SSIM: {vali_ssim[epoch]:.4f}')

    np.save(os.path.join(expr_dir, 'train_rmse.npy'), np.array(train_rmse))
    np.save(os.path.join(expr_dir, 'vali_rmse.npy'), np.array(vali_rmse))
    np.save(os.path.join(expr_dir, 'train_psnr.npy'), np.array(train_psnr))
    np.save(os.path.join(expr_dir, 'vali_psnr.npy'), np.array(vali_psnr))
    np.save(os.path.join(expr_dir, 'train_ssim.npy'), np.array(train_ssim))
    np.save(os.path.join(expr_dir, 'vali_ssim.npy'), np.array(vali_ssim))
