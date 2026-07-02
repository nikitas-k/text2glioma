"""Fit a per-channel whitening transform on stage-1 training latents.

For a stage-1 encoder that produces [B, C, D, H, W] latents, we estimate the
channel-wise mean `mu` (length C) and covariance `Sigma` (C x C) over ~200
training volumes, then form the ZCA whitening transform:

    W       = U @ diag(1 / sqrt(lambda + eps)) @ U.T
    W_inv   = U @ diag(sqrt(lambda + eps))      @ U.T

    z_white(b, c, d, h, w) = sum_c' W[c, c'] * (z(b, c', d, h, w) - mu[c'])
    z(b, c, d, h, w)       = mu[c] + sum_c' W_inv[c, c'] * z_white(b, c', d, h, w)

ZCA is preferred over PCA-whitening (Λ^{-1/2} U^T) because it stays close to
the identity — the whitened latent is as similar as possible to the original
in an L2 sense while having unit isotropic channel covariance. That preserves
whatever spatial structure the stage-1 encoder produced.

The output file stores {"mu": mu, "W": W, "W_inv": W_inv, "eps": eps,
"latent_channels": C, "n_samples": N} and can be consumed by
``train_stage2_ddp.py --latent_whitening_path`` and
``offline_sample_stage2_compare.py --latent_whitening_path``.

Usage:
    python scripts/fit_latent_whitening.py \\
        --datalist  ~/text2glioma/datalist_N1510.json \\
        --stage1_config configs/stage1.yaml \\
        --stage1_uri /g/data/vp06/$USER/text2glioma_train/runs/stage1_overfit_ablate_kl1e6/autoencoder_stage1/checkpoint.pth \\
        --num_subjects 200 \\
        --out /g/data/vp06/$USER/text2glioma_train/runs/stage1_overfit_ablate_kl1e6/autoencoder_stage1/latent_whitening.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai import transforms as T

from text2glioma.utils import get_model, load_config, stage1_ify


MSD_TO_T2G = [1, 2, 3, 0]


def _val_transform(channel_reorder: bool) -> T.Compose:
    xforms = [
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1, channel_wise=True
        ),
        T.ToTensord(keys=["image"]),
    ])
    return T.Compose(xforms)


def _infer_latent_channels(ckpt_path: Path) -> int | None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt if isinstance(ckpt, dict) else None
    for key in ("state_dict", "model", "autoencoder", "vae"):
        if isinstance(ckpt.get(key), dict) and ckpt[key]:
            state = ckpt[key]
            break
    for k in ("quant_conv_mu.conv.weight",
              "module.quant_conv_mu.conv.weight",
              "model.quant_conv_mu.conv.weight"):
        w = state.get(k) if isinstance(state, dict) else None
        if torch.is_tensor(w):
            return int(w.shape[0])
    return None


def _load_stage1(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_config(str(config_path))
    params = cfg.setdefault("model", {}).setdefault("params", {})
    inferred = _infer_latent_channels(ckpt_path)
    if inferred is not None:
        params["latent_channels"] = inferred
    if device.type != "cuda":
        params["use_flash_attention"] = False
    stage1 = stage1_ify(get_model("AutoencoderKL", cfg, from_file=str(ckpt_path)))
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad = False
    return stage1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=Path, required=True)
    ap.add_argument("--stage1_config", type=Path, required=True)
    ap.add_argument("--stage1_uri", type=Path, required=True)
    ap.add_argument("--split", type=str, default="training",
                    choices=["training", "validation"])
    ap.add_argument("--num_subjects", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="Numerical floor for eigenvalues before inverse sqrt.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no_channel_reorder", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    with open(args.datalist) as f:
        dl = json.load(f)
    items = dl[args.split][: args.num_subjects]
    print(f"fitting whitening on {len(items)} subjects from split '{args.split}'")

    stage1 = _load_stage1(args.stage1_config, args.stage1_uri, device)
    transform = _val_transform(channel_reorder=not args.no_channel_reorder)

    # Streaming mean and covariance across all voxels of all volumes.
    # x = latent voxel vector in R^C; we accumulate sum_x, sum_xxT, count.
    sum_x = None
    sum_xxT = None
    n = 0
    C = None
    with torch.no_grad():
        for i, item in enumerate(items):
            batch = transform(dict(item))
            x = batch["image"].unsqueeze(0).to(device)
            z = stage1(x).float()  # [1, C, D, H, W]
            _, C_, D, H, W = z.shape
            if C is None:
                C = C_
                sum_x = torch.zeros(C, dtype=torch.float64, device=device)
                sum_xxT = torch.zeros(C, C, dtype=torch.float64, device=device)
            zf = z.permute(0, 2, 3, 4, 1).reshape(-1, C).to(torch.float64)  # [N_v, C]
            sum_x += zf.sum(dim=0)
            sum_xxT += zf.T @ zf
            n += zf.shape[0]
            if (i + 1) % 20 == 0:
                print(f"  processed {i + 1}/{len(items)} subjects, {n:,} voxels")

    mu = sum_x / n
    cov = sum_xxT / n - torch.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)  # symmetrise

    print(f"\nlatent channels     : {C}")
    print(f"voxels used         : {n:,}")
    print(f"mu (per-channel)    : {mu.cpu().numpy()}")
    print(f"cov diag (per-chan) : {torch.diag(cov).cpu().numpy()}")

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=args.eps)
    inv_sqrt = torch.diag(1.0 / eigvals.sqrt())
    sqrt_ = torch.diag(eigvals.sqrt())
    W = eigvecs @ inv_sqrt @ eigvecs.T
    W_inv = eigvecs @ sqrt_ @ eigvecs.T

    print(f"eigenvalue range    : [{float(eigvals.min()):.4e}, {float(eigvals.max()):.4e}]")
    print(f"condition number    : {float(eigvals.max() / eigvals.min()):.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mu":               mu.cpu().float(),
        "W":                W.cpu().float(),
        "W_inv":            W_inv.cpu().float(),
        "eps":              float(args.eps),
        "latent_channels":  int(C),
        "n_samples":        int(n),
        "eigvals":          eigvals.cpu().float(),
        "kind":             "zca_channelwise",
    }, str(args.out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
