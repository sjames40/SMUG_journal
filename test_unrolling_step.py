import torch
import torch.nn as nn
import numpy as np
from models.didn import DIDN
from models import networks
from util.metrics import PSNR
import pytorch_msssim
import global_network_dataset
import matplotlib.pyplot as plt
from options.test_options import TestOptions

opt = TestOptions().parse()

opt.batchSize = 1

device = torch.device(('cuda:' + str(opt.gpu_ids[0])) if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')

netG = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=opt.n_res_blocks)
netG.load_state_dict(torch.load(opt.netGpath, map_location=device))
netG = netG.float().to(device)

epsilons = [0.2/255, 0.5/255, 1/255, 2/255]
unrolling_steps = [0, 2, 4, 6, 8, 10, 12, 14, 16]

# Loss and optimizer
mse_loss = nn.MSELoss().to(device)
ssim_loss = pytorch_msssim.SSIM(data_range=2.0, channel=2).to(device)

def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)


def average_repeated_batch(tensor, batch_size, num_sample):
    return tensor.reshape(num_sample, batch_size, *tensor.shape[1:]).mean(dim=0)


def Recon(cg_iter, smap, mask, input, smoothing, num_sample, epsilon):
    output_CG = input
    if smoothing == 'none': # vanilla MoDL
        for _ in range(cg_iter):
            output_NN = netG(output_CG)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)
    elif smoothing == 'RSE2E':
        input_i = input.repeat(num_sample, 1, 1, 1)
        noises = torch.normal(0, epsilon, input_i.shape).to(device)
        noised_input = torch.clamp(noises + input_i, min=-1, max=1)
        output_CG = noised_input
        
        for _ in range(cg_iter):
            output_NN = netG(output_CG)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap.repeat(num_sample, 1, 1, 1, 1), mask=mask.repeat(num_sample, 1, 1, 1), alised_image=input_i)
        
        output_CG = average_repeated_batch(output_CG, input.shape[0], num_sample)
    elif smoothing == 'SMUGv0':
        for _ in range(cg_iter):
            output_CG = output_CG.repeat(num_sample, 1, 1, 1)
            noises = torch.normal(0, epsilon, output_CG.shape).to(device)
            noised_input = torch.clamp(noises + output_CG, min=-1, max=1)

            output_NN = netG(noised_input)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap.repeat(num_sample, 1, 1, 1, 1), mask=mask.repeat(num_sample, 1, 1, 1), alised_image=input.repeat(num_sample, 1, 1, 1))

            output_CG = average_repeated_batch(output_CG, input.shape[0], num_sample)
    elif smoothing == 'SMUG':
        for _ in range(cg_iter):
            output_CG = output_CG.repeat(num_sample, 1, 1, 1)
            noises = torch.normal(0, epsilon, output_CG.shape).to(device)
            noised_input = torch.clamp(noises + output_CG, min=-1, max=1)

            output_NN = netG(noised_input)
            output_NN_final = average_repeated_batch(output_NN, input.shape[0], num_sample)

            output_CG = CG(output_NN_final, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)

    return output_CG

def PGD(pgd_steps, cg_iter, smap, mask, input, label, crition, eps, alpha, smoothing, num_sample, smoothing_epsilon, norm='linfty'):
    clamp_fn = l2_clamp if norm == 'l2' else linfty_clamp

    netG.requires_grad_(False)

    orig_input = input.detach()
    input = input.clone().detach()

    input = clamp_fn(input + torch.normal(0, eps, input.shape).to(device), input, eps)
    input = torch.clamp(input, min=-1, max=1)

    for _ in range(pgd_steps):
        input.requires_grad = True
        output = Recon(cg_iter, smap, mask, input, smoothing, num_sample, smoothing_epsilon)
        loss = crition(output, label)
        loss.backward()
        adv_images = input + alpha * input.grad.sign()
        input = clamp_fn(adv_images, orig_input, eps)
        input = torch.clamp(input, min=-1, max=1).detach()

    netG.requires_grad_(True)
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

_, test_loader = global_network_dataset.loadData(opt.dataroot, opt.mask_dataroot, opt.train_valiSize, opt.testSize, opt.batchSize)

test_psnr = [[] for _ in range(len(unrolling_steps))]
test_ssim = [[] for _ in range(len(unrolling_steps))]
adv_test_psnr = [[[] for _ in range(len(unrolling_steps))] for _ in range(len(epsilons))]
adv_test_ssim = [[[] for _ in range(len(unrolling_steps))] for _ in range(len(epsilons))]

for i, (test_direct, test_target, test_smap, test_mask) in enumerate(test_loader):
    # adv_vis_img = None
    vis = opt.visualize and i == 2

    test_input = test_direct.to(device).float()
    test_smap = test_smap.to(device).float()
    test_mask = test_mask.to(device).float()
    test_label = test_target.to(device).float()
    clean_test_input = test_input

    for j, step in enumerate(unrolling_steps):
        with torch.no_grad():
            test_result = Recon(step, test_smap, test_mask, clean_test_input, opt.smoothing, opt.num_sample, opt.smoothing_epsilon)
            psnr_test = PSNR(test_label, test_result)
            ssim_test = ssim_loss(test_label, test_result)
            test_psnr[j].append(float(psnr_test))
            test_ssim[j].append(float(ssim_test))
        
        for ii, epsilon in enumerate(epsilons):
            adv_test_input = test_input.clone()
            adv_test_input = PGD(opt.pgd_steps, step, test_smap, test_mask, adv_test_input, test_label, loss_fn, epsilon, epsilon / 3, opt.smoothing, opt.num_sample, opt.smoothing_epsilon)
            with torch.no_grad():
                adv_test_result = Recon(step, test_smap, test_mask, adv_test_input, opt.smoothing, opt.num_sample, opt.smoothing_epsilon)
                adv_psnr_test = PSNR(test_label, adv_test_result)
                adv_ssim_test = ssim_loss(test_label, adv_test_result)
                adv_test_psnr[ii][j].append(float(adv_psnr_test))
                adv_test_ssim[ii][j].append(float(adv_ssim_test))
                # if vis and ii == 3:
                #     adv_vis_img = adv_test_result
    
        if vis and j == len(unrolling_steps) - 1:
            img_out = test_result.detach().cpu().numpy()
            img_out = img_out.squeeze(0)
            img_out = img_out[0] + img_out[1] * 1j

            plt.imshow(np.abs(img_out), cmap='gray')
            plt.axis('off')

            img_name = opt.netGpath[:-4] +  f'_{step}_steps.pdf'
            plt.savefig(img_name, dpi=600)
            plt.close()
    

message = ''
for j in range(len(unrolling_steps)):
    message += f'{unrolling_steps[j]} steps:\n'
    message += f'test RSNR: {np.average(test_psnr[j]):.4f} ± {np.std(test_psnr[j]):.4f}\n'
    message += f'test SSIM: {np.average(test_ssim[j]):.4f} ± {np.std(test_ssim[j]):.4f}\n'
    message += '\n'
    for ii in range(len(epsilons)):
        message += f'epsilon = {epsilons[ii] * 255:.1f}/255\n'
        message += f'Robust PSNR: {np.average(adv_test_psnr[ii][j]):.4f} ± {np.std(adv_test_psnr[ii][j]):.4f}\n'
        message += f'Robust SSIM: {np.average(adv_test_ssim[ii][j]):.4f} ± {np.std(adv_test_ssim[ii][j]):.4f}\n'
        message += '\n'

file_name = opt.netGpath[:-4] + '_steps_test.out'
with open(file_name, 'w') as result_file:
    result_file.write(message)

print(message)