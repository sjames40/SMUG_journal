"""
Fine-tunes SMUG by adding Gaussian noise at each unrolled denoiser input, and averaging the denoiser outputs. 
Its loss combines the unrolled stability denoising term with final reconstruction MSE, and it saves the best checkpoint as vali_best.pth.
"""

import torch
import torch.nn as nn
import numpy as np
from models.didn import DIDN
from models import networks
from util.metrics import PSNR
import pytorch_msssim
import global_network_dataset
from tqdm import tqdm
import os
from options.tune_options import TuneOptions

opt = TuneOptions().parse()
opt.smoothing = 'SMUG'

device = torch.device("cuda:" + str(opt.gpu_ids[0]) if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else "cpu")

netG = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=opt.n_res_blocks)
netG.load_state_dict(torch.load(opt.netGpath, map_location=device))
netG = netG.float()
if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
    netG = nn.DataParallel(netG, device_ids=opt.gpu_ids)
netG = netG.to(device)


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 - 20) / float(opt.epoch + 1 - 20)
    return lr_l

# loss and optimizer
mse_loss = nn.MSELoss().to(device)
ssim_loss = pytorch_msssim.SSIM(data_range=2.0, channel=2).to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=[0.5, 0.999])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimG, lr_lambda=lambda_rule)

def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)


def average_repeated_batch(tensor, batch_size, num_sample):
    return tensor.reshape(num_sample, batch_size, *tensor.shape[1:]).mean(dim=0)


def Recon(cg_iter, smap, mask, input, label, smoothing=False, num_sample=10, epsilon=0.01, is_train=False):
    output_CG = input
    if smoothing == 'none':
        for _ in range(cg_iter):
            output_NN = netG(output_CG)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)
        return output_CG, None
    else:
        batch_size = input.shape[0]
        loss = torch.zeros((), device=device)
        target_denoised = netG(label) if is_train else None

        for _ in range(cg_iter):
            output_CG_i = output_CG.repeat(num_sample, 1, 1, 1)
            noises = torch.normal(0, epsilon, output_CG_i.shape).to(device)
            noised_input = torch.clamp(noises + output_CG_i, min=-1, max=1)
            output_NN = netG(noised_input)

            if is_train:
                loss += loss_fn(output_NN, target_denoised.repeat(num_sample, 1, 1, 1))

            output_NN_final = average_repeated_batch(output_NN, batch_size, num_sample)

            output_CG = CG(output_NN_final, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)

        if is_train:
            loss += loss_fn(output_CG, label) * opt.LossLambda

        return output_CG, loss

def PGD(pgd_steps, cg_iter, smap, mask, input, label, crition, eps, alpha, norm='linfty'):
    clamp_fn = l2_clamp if norm =='l2' else linfty_clamp

    orig_input = input.detach()
    input = input.clone().detach()

    input = clamp_fn(input + torch.normal(0, eps, input.shape).to(device), input, eps)
    input = torch.clamp(input, min=-1, max=1)

    for i in range(pgd_steps):
        input.requires_grad = True
        output, _ = Recon(cg_iter, smap, mask, input, label)
        loss = crition(output, label)
        netG.zero_grad()
        loss.backward()
        adv_images = input + alpha * input.grad.sign()
        input = clamp_fn(adv_images, orig_input, eps)
        input = torch.clamp(input, min=-1, max=1).detach()
    return input

def linfty_clamp(input, center, epsilon):
    input = torch.clamp(input, min=center-epsilon, max=center+epsilon)
    return input

def l2_clamp(input, center, epsilon):
    delta = (input - center).flatten(1)
    delta_len = torch.linalg.vector_norm(delta, ord=2, dim=1)
    delta_len = delta_len.repeat(delta.shape[1], 1).T
    delta[delta_len > epsilon] = delta[delta_len > epsilon] / delta_len[delta_len > epsilon] * epsilon
    input = center + delta.reshape(input.shape)
    return input

def loss_fn(outputs, labels):
    loss = mse_loss(outputs, labels)
    return loss

train_rmse = []
vali_rmse = []
vali_rmse_min = None
train_psnr = []
vali_psnr = []
train_ssim = []
vali_ssim = []

train_loader, test_loader = global_network_dataset.loadData(opt.dataroot, opt.mask_dataroot, opt.trainSize, opt.valiSize, opt.batchSize)

train_size = len(train_loader.dataset)
vali_size = len(test_loader.dataset)

expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

for epoch in tqdm(range(opt.epoch)):
    train_rmse_total = 0.
    train_psnr_total = 0.
    train_ssim_total = 0.
    for direct, target, smap, mask in train_loader:
        input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()
        clean_input = input

        output, loss_G = Recon(cg_iter=opt.blockIter, smap=smap, mask=mask, input=clean_input, label=label, smoothing=opt.smoothing, num_sample=opt.num_sample, epsilon=opt.smoothing_epsilon, is_train=True)

        optimG.zero_grad()
        loss_G.backward()
        optimG.step()

        psnr_train = PSNR(label, output)
        ssim_train = ssim_loss(label, output)
        train_rmse_total += np.sqrt(float(mse_loss(output, label)))
        train_psnr_total += float(psnr_train)
        train_ssim_total += float(ssim_train)

    vali_rmse_total = 0.
    vali_psnr_total = 0.
    vali_ssim_total = 0.
    for vali_direct, vali_target, vali_smap, vali_mask in test_loader:
        vali_input = vali_direct.to(device).float()
        vali_smap = vali_smap.to(device).float()
        vali_mask = vali_mask.to(device).float()
        vali_label = vali_target.to(device).float()
        clean_vali_input = vali_input

        with torch.no_grad():
            vali_result, _ = Recon(cg_iter=opt.blockIter, smap=vali_smap, mask=vali_mask, input=clean_vali_input, label=vali_label, smoothing=opt.smoothing, num_sample=opt.num_sample, epsilon=opt.smoothing_epsilon)

            psnr_vali = PSNR(vali_label, vali_result)
            ssim_vali = ssim_loss(vali_label, vali_result)
            vali_rmse_total += np.sqrt(float(mse_loss(vali_result, vali_label)))
            vali_psnr_total += float(psnr_vali)
            vali_ssim_total += float(ssim_vali)
        
    scheduler.step()
    curr_lr = optimG.param_groups[0]['lr']
    print(f'learning rate: {curr_lr:.6f}')

    if vali_rmse_min is None or vali_rmse_total < vali_rmse_min:
        vali_rmse_min = vali_rmse_total
        state_dict = netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict()
        torch.save(state_dict, os.path.join(expr_dir, 'vali_best.pth'))
        print(f'saving vali best model at epoch {epoch}')

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
