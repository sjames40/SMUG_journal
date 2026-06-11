"""
This script follows the paper's measurement-space attack.
Here the PGD variable is the normalized coil k-space tensor. The adversarial image-domain model input and
the data-consistency term are both recomputed as A^H(y + delta).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SMUG_ROOT = Path("/SMUG")
REPO_ROOT = SMUG_ROOT / "SMUG_journal-main"
DEFAULT_INPUT = REPO_ROOT / "data_path"
DEFAULT_MASK_ROOT = REPO_ROOT / "data/MASK_4X"
DEFAULT_VANILLA = REPO_ROOT / "checkpoints/path/vali_best.pth"
DEFAULT_SMUG = REPO_ROOT / "checkpoints/smug_modl/vali_best.pth"
DEFAULT_WEIGHTED = REPO_ROOT / "checkpoints/weighted_smug_modl_fixed/vali_best.pth"
DEFAULT_OUTPUT_DIR = SMUG_ROOT

sys.path.insert(0, str(REPO_ROOT))
from models.didn import DIDN  
from util.metrics import PSNR  
from util.util import complex_conj, complex_matmul, fft2, ifft2 


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


def load_mask(mask_root, shape):
    path = mask_root / f"mask_{shape[0]}x{shape[1]}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing mask for shape {shape}: {path}")
    return np.load(path, "r").astype(np.bool_)


def adjoint_from_kspace(kspace, smap, mask):
    """Compute A^H y for batched coil k-space.

    kspace: [B, coils, 2, H, W]
    smap:   [B, coils, 2, H, W]
    mask:   [B, 2, H, W]
    """
    _, num_coils, _, _, _ = kspace.shape
    mask_coil = mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    k_under = kspace * mask_coil
    im_u = ifft2(k_under.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    return complex_matmul(im_u, complex_conj(smap)).sum(1)


def load_single_sample(npz_path, mask_root, crop_size=(320, 320), device="cuda"):
    with np.load(npz_path, "r") as data:
        s_r = data["s_r"].astype(np.float32) / 32767.0
        s_i = data["s_i"].astype(np.float32) / 32767.0
        k_r = data["k_r"].astype(np.float32) / 32767.0
        k_i = data["k_i"].astype(np.float32) / 32767.0

    _, nx, ny = s_r.shape
    crop_h, crop_w = crop_size
    top = nx // 2 - crop_h // 2
    left = ny // 2 - crop_w // 2
    mask_np = load_mask(mask_root, (nx, ny))

    k_np = np.stack((k_r, k_i), axis=0)
    s_np = np.stack((
        s_r[:, top:top + crop_h, left:left + crop_w],
        s_i[:, top:top + crop_h, left:left + crop_w],
    ), axis=0)
    mask_crop = mask_np[top:top + crop_h, left:left + crop_w]

    with torch.no_grad():
        mask = torch.tensor(np.repeat(mask_crop[np.newaxis], 2, axis=0), dtype=torch.float32, device=device)
        k_full = torch.tensor(k_np, dtype=torch.float32, device=device).permute(1, 0, 2, 3)
        image = ifft2(k_full.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        image = image[:, :, top:top + crop_h, left:left + crop_w]
        smap = torch.tensor(s_np, dtype=torch.float32, device=device).permute(1, 0, 2, 3)

        sos = torch.sum(complex_matmul(image, complex_conj(smap)), dim=0)
        image = image / torch.max(torch.abs(sos))
        kspace = fft2(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        kspace = kspace.unsqueeze(0)
        smap = smap.unsqueeze(0)
        mask = mask.unsqueeze(0)
        direct = adjoint_from_kspace(kspace, smap, mask)
        target = adjoint_from_kspace(kspace, smap, torch.ones_like(mask))

    return kspace, direct, target, smap, mask


def load_didn(path, n_res_blocks, device):
    model = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=n_res_blocks)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "netG" in state:
        state = state["netG"]
    model.load_state_dict(state)
    return model.float().to(device).eval()


def load_weighted(path, n_res_blocks, device):
    if path.is_dir():
        path = path / "vali_best.pth"

    split_netg_path = path.with_name("vali_best_netG.pth")
    split_encoder_path = path.with_name("vali_best_weight_encoder.pth")

    if path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    else:
        checkpoint = None

    if isinstance(checkpoint, dict) and "netG" in checkpoint and "weight_encoder" in checkpoint:
        denoiser_state = checkpoint["netG"]
        encoder_state = checkpoint["weight_encoder"]
    elif split_netg_path.exists() and split_encoder_path.exists():
        denoiser_state = torch.load(split_netg_path, map_location=device, weights_only=False)
        encoder_state = torch.load(split_encoder_path, map_location=device, weights_only=False)
    else:
        raise ValueError(
            "Use the combined Weighted-SMUG checkpoint containing netG and weight_encoder, "
            f"or keep split checkpoints next to it: {path}"
        )

    denoiser = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=n_res_blocks)
    denoiser.load_state_dict(denoiser_state)
    hidden_channels = encoder_state["features.0.weight"].shape[0]
    encoder = WeightEncoder(in_channels=2, hidden_channels=hidden_channels)
    encoder.load_state_dict(encoder_state)
    return denoiser.float().to(device).eval(), encoder.float().to(device).eval()


def apply_ata(image, smap, mask, lam):
    _, num_coils, _, _, _ = smap.shape
    image_coil = image.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    image_s = complex_matmul(image_coil, smap)
    k_full = fft2(image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    mask_coil = mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)
    k_under = k_full * mask_coil
    im_u = ifft2(k_under.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
    ata = complex_matmul(im_u, complex_conj(smap)).sum(1)
    return ata + lam * image


def cg_differentiable(dn, tol, lam, smap, mask, aliased_image, max_iter):
    lam_tensor = torch.as_tensor(lam, dtype=dn.dtype, device=dn.device)
    b0 = dn * lam_tensor + aliased_image
    xk = dn
    rk = b0 - apply_ata(xk, smap, mask, lam_tensor)
    pk = rk

    for _ in range(max_iter):
        if float(torch.norm(rk).detach().cpu()) <= tol:
            break
        rtr = torch.pow(torch.norm(rk), 2)
        apk = apply_ata(pk, smap, mask, lam_tensor)
        p_ap = torch.sum(complex_matmul(complex_conj(pk), apk))
        alpha = rtr / (p_ap + 1e-12)
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


def recon(denoiser, smap, mask, aliased_image, method, args, weight_encoder=None):
    output_cg = aliased_image
    batch_size = aliased_image.shape[0]

    for _ in range(args.block_iter):
        if method == "Vanilla MoDL":
            output_nn = denoiser(output_cg)
        else:
            repeated = output_cg.repeat(args.num_sample, 1, 1, 1)
            noise = torch.normal(0, args.smoothing_epsilon, repeated.shape, device=repeated.device)
            noised_input = torch.clamp(repeated + noise, min=-1, max=1)
            denoised = denoiser(noised_input)
            if method == "SMUG":
                output_nn = average_repeated_batch(denoised, batch_size, args.num_sample)
            elif method == "Weighted-SMUG":
                weights = weight_encoder(noised_input)
                output_nn = weighted_average_repeated_batch(denoised, weights, batch_size, args.num_sample)
            else:
                raise ValueError(f"Unsupported method: {method}")

        output_cg = cg_differentiable(
            output_nn,
            tol=args.cg_tol,
            lam=args.lam,
            smap=smap,
            mask=mask,
            aliased_image=aliased_image,
            max_iter=args.cg_max_iter,
        )

    return output_cg


def kspace_mask_like(kspace, mask):
    _, num_coils, _, _, _ = kspace.shape
    return mask.unsqueeze(1).repeat(1, num_coils, 1, 1, 1)


def pgd_kspace(denoiser, kspace, target, smap, mask, method, args, weight_encoder=None):
    """L-infinity PGD on sampled k-space real/imag values."""
    modules = [denoiser] + ([weight_encoder] if weight_encoder is not None else [])
    for module in modules:
        module.requires_grad_(False)

    sampled = kspace_mask_like(kspace, mask)
    original = kspace.detach()
    delta = torch.normal(0, args.pgd_epsilon, original.shape, device=original.device)
    delta = torch.clamp(delta, min=-args.pgd_epsilon, max=args.pgd_epsilon) * sampled
    adversarial_kspace = (original + delta).detach()
    mse_loss = nn.MSELoss().to(kspace.device)

    for step in range(args.pgd_steps):
        print(f"  k-space PGD {method}: step {step + 1}/{args.pgd_steps}", flush=True)
        adversarial_kspace.requires_grad_(True)
        aliased_image = adjoint_from_kspace(adversarial_kspace, smap, mask)
        output = recon(denoiser, smap, mask, aliased_image, method, args, weight_encoder)
        loss = mse_loss(output, target)

        for module in modules:
            module.zero_grad(set_to_none=True)
        loss.backward()

        grad = adversarial_kspace.grad
        adversarial_kspace = adversarial_kspace + args.pgd_alpha * grad.sign()
        delta = torch.clamp(adversarial_kspace - original, min=-args.pgd_epsilon, max=args.pgd_epsilon)
        delta = delta * sampled
        adversarial_kspace = (original + delta).detach()

    for module in modules:
        module.requires_grad_(True)
    return adversarial_kspace


def magnitude(image_tensor):
    array = image_tensor.detach().cpu().numpy()[0]
    return np.sqrt(array[0] ** 2 + array[1] ** 2)


def display_image(image):
    low = np.percentile(image, 0.1)
    high = np.percentile(image, 99.9)
    return np.clip((image - low) / (high - low + 1e-8), 0, 1)


def add_zoom_boxes(ax, color="lime"):
    for x, y, width, height in [(12, 0, 72, 72), (182, 126, 88, 88)]:
        ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor=color, facecolor="none", linewidth=1.6))


def save_figure(output_path, target, results, psnrs, epsilon):
    panels = [("Ground Truth", magnitude(target), "PSNR = inf dB")]
    for method in ["Vanilla MoDL", "SMUG", "Weighted-SMUG"]:
        panels.append((method, magnitude(results[method]), f"PSNR = {psnrs[method]:.2f} dB"))

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.2))
    for ax, (title, image, subtitle) in zip(axes, panels):
        ax.imshow(display_image(image), cmap="gray", vmin=0, vmax=1)
        add_zoom_boxes(ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"4x undersampling, k-space PGD, epsilon={epsilon}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluation with paper-style k-space PGD.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--vanilla-ckpt", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--smug-ckpt", type=Path, default=DEFAULT_SMUG)
    parser.add_argument("--weighted-ckpt", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-res-blocks", type=int, default=3)
    parser.add_argument("--block-iter", type=int, default=8)
    parser.add_argument("--num-sample", type=int, default=10)
    parser.add_argument("--smoothing-epsilon", type=float, default=0.01)
    parser.add_argument("--pgd-epsilon", type=float, default=0.02)
    parser.add_argument("--pgd-alpha", type=float, default=None, help="Defaults to epsilon/3.")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--cg-max-iter", type=int, default=30)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.pgd_alpha is None:
        args.pgd_alpha = args.pgd_epsilon / 3.0

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kspace, direct, target, smap, mask = load_single_sample(args.input, args.mask_root, device=device)
    vanilla = load_didn(args.vanilla_ckpt, args.n_res_blocks, device)
    smug = load_didn(args.smug_ckpt, args.n_res_blocks, device)
    weighted_denoiser, weight_encoder = load_weighted(args.weighted_ckpt, args.n_res_blocks, device)
    models = {
        "Vanilla MoDL": (vanilla, None),
        "SMUG": (smug, None),
        "Weighted-SMUG": (weighted_denoiser, weight_encoder),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {args.input}")
    print(f"Device: {device}")
    print(f"Checkpoints: Vanilla={args.vanilla_ckpt}, SMUG={args.smug_ckpt}, Weighted={args.weighted_ckpt}")
    print(f"Settings: 4x, blockIter={args.block_iter}, sigma={args.smoothing_epsilon}, MC={args.num_sample}")
    print(f"k-space PGD: epsilon={args.pgd_epsilon}, alpha={args.pgd_alpha}, steps={args.pgd_steps}")
    print(f"Differentiable CG: tol={args.cg_tol}, max_iter={args.cg_max_iter}")

    summary_lines = [
        f"input: {args.input}",
        f"vanilla_checkpoint: {args.vanilla_ckpt}",
        f"smug_checkpoint: {args.smug_ckpt}",
        f"weighted_checkpoint: {args.weighted_ckpt}",
        f"settings: 4x, blockIter={args.block_iter}, sigma={args.smoothing_epsilon}, MC={args.num_sample}",
        f"PGD: k-space measurement attack, epsilon={args.pgd_epsilon}, alpha={args.pgd_alpha}, steps={args.pgd_steps}",
        f"CG: differentiable, tol={args.cg_tol}, max_iter={args.cg_max_iter}",
        "",
        "Clean reconstruction:",
    ]

    for method, (denoiser, encoder) in models.items():
        torch.manual_seed(args.seed)
        with torch.no_grad():
            clean_result = recon(denoiser, smap, mask, direct, method, args, encoder)
        clean_psnr = float(PSNR(target, clean_result))
        summary_lines.append(f"{method}: {clean_psnr:.4f} dB")
        print(f"{method} clean PSNR: {clean_psnr:.4f} dB")

    results = {}
    psnrs = {}
    summary_lines.extend(["", f"{args.pgd_steps}-step k-space PGD:"])
    for method, (denoiser, encoder) in models.items():
        torch.manual_seed(args.seed)
        print(f"Running k-space PGD for {method}...", flush=True)
        adversarial_kspace = pgd_kspace(denoiser, kspace, target, smap, mask, method, args, encoder)
        adversarial_input = adjoint_from_kspace(adversarial_kspace, smap, mask)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            result = recon(denoiser, smap, mask, adversarial_input, method, args, encoder)
        results[method] = result
        psnrs[method] = float(PSNR(target, result))
        summary_lines.append(f"{method}: {psnrs[method]:.4f} dB")
        print(f"{method} PSNR ({args.pgd_steps}-step k-space PGD): {psnrs[method]:.4f} dB")

    output_path = args.output_dir / f"kspace_pgd{args.pgd_steps}_{args.input.stem}.png"
    save_figure(output_path, target, results, psnrs, args.pgd_epsilon)
    print(f"Saved figure: {output_path}")

    summary_path = args.output_dir / f"fig6_kspace_pgd{args.pgd_steps}_results_{args.input.stem}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
