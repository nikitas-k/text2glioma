"""Compare latent-space statistics of BrainLDM-FT vs MaxFeat stage-1 encoders.

The two stage-1 checkpoints differ in KL weight, channel count, and pretraining
history. Their downstream diffusion trainability also differs sharply
(BrainLDM-FT's stage-2 val loss converges below 1.0; MaxFeat's plateaus near
the zero-output floor). This script quantifies *what* differs about the two
latents so we can pick a targeted fix rather than a blind KL sweep.

Metrics computed (over N validation subjects, per model):
    - Per-channel mean and std.
    - Per-dim KL to N(0, I) (voxel-wise, averaged).
    - Off-diagonal channel covariance strength (correlation between channels
      averaged over voxels; near-zero = decorrelated).
    - Spatial autocorrelation at lag 1 (mean |z(x,y,z) - z(x+1,y,z)| divided
      by std; low = smooth latent, high = noisy latent).
    - Effective per-channel rank via singular value spectrum on the flattened
      voxels-by-channels matrix.
    - Radial power spectrum on a random slice (rough frequency-content hint).

Writes:
    - CSV with all numerical statistics per (model, channel).
    - A short console report highlighting BrainLDM-FT vs MaxFeat differences.

Usage:
    python scripts/compare_latent_statistics.py \\
        --datalist /home/575/nk9793/text2glioma/datalist_N1510.json \\
        --runs_root /g/data/vp06/$USER/text2glioma_train/runs \\
        --num_subjects 32 \\
        --out paper/tables/latent_statistics.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai import transforms as T

from text2glioma.utils import get_model, load_config, stage1_ify


MSD_TO_T2G = [1, 2, 3, 0]

MODELS = {
    "BrainLDM-FT": {
        "stage1_config": "configs/stage1_pinaya_decoder_only.yaml",
        "stage1_ckpt":   "pinaya_decoder_only_v5_no_disc/autoencoder_stage1/final_model.pth",
    },
    "MaxFeat_LC=6": {
        "stage1_config": "configs/stage1.yaml",
        "stage1_ckpt":   "stage1_overfit_ablate_kl1e6/autoencoder_stage1/checkpoint.pth",
    },
    "MaxFeat_LC=3": {
        "stage1_config": "configs/stage1.yaml",
        "stage1_ckpt":   "stage1_kl1e6_lc3/autoencoder_stage1/checkpoint.pth",
    },
    "MaxFeat_FB_LC=6": {
        "stage1_config": "configs/stage1.yaml",
        "stage1_ckpt":   "stage1_kl1e6_freebits_lc6/autoencoder_stage1/checkpoint.pth",
    },
    "MaxFeat_FB_LC=3": {
        "stage1_config": "configs/stage1.yaml",
        "stage1_ckpt":   "stage1_kl1e6_freebits_lc3/autoencoder_stage1/checkpoint.pth",
    },
}


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
    if isinstance(ckpt, dict):
        state = ckpt
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


def _lag1_autocorr(z: torch.Tensor) -> float:
    """Mean absolute per-voxel diff along the X axis, normalised by std.

    z is [B, C, D, H, W]. Returns a scalar (averaged over B, C, and voxels).
    Lower = smoother latent (spatial autocorrelation), higher = noisier.
    """
    z_ = z.float()
    diff = (z_[..., 1:, :, :] - z_[..., :-1, :, :]).abs().mean()
    std = z_.std().clamp(min=1e-6)
    return float((diff / std).item())


def _channel_correlation(z: torch.Tensor) -> tuple[float, np.ndarray]:
    """Return (mean |off-diag corr|, full C×C correlation matrix)."""
    z_ = z.float()
    B, C, D, H, W = z_.shape
    flat = z_.permute(1, 0, 2, 3, 4).reshape(C, -1)  # [C, N_voxels]
    corr = torch.corrcoef(flat).cpu().numpy()
    off_diag = corr - np.eye(C)
    mean_abs = float(np.abs(off_diag).sum() / max(C * (C - 1), 1))
    return mean_abs, corr


def _effective_rank(z: torch.Tensor) -> float:
    """Shannon effective rank of the channel-flattened latent.

    exp(H) where H = -sum p_i log p_i, p_i = s_i^2 / sum(s_i^2).
    Ranges from 1 (all variance in one direction) to C (uniform).
    """
    z_ = z.float()
    C = z_.shape[1]
    flat = z_.permute(1, 0, 2, 3, 4).reshape(C, -1)
    # Subtract mean before SVD
    flat = flat - flat.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(flat)
    p = (s ** 2) / (s ** 2).sum().clamp(min=1e-12)
    H = -(p * (p.clamp(min=1e-12).log())).sum()
    return float(torch.exp(H).item())


def _radial_power_spectrum(z: torch.Tensor, n_bins: int = 32) -> np.ndarray:
    """1D radial power spectrum on the middle axial slice, averaged over C, B.

    Returns array length n_bins with mean power per radial-frequency bin
    (log-scale friendly). Higher energy at high k = noisier latent.
    """
    z_ = z.float()
    B, C, D, H, W = z_.shape
    mid = D // 2
    slabs = z_[:, :, mid, :, :]  # [B, C, H, W]
    fft = torch.fft.fftshift(torch.fft.fftn(slabs, dim=(-2, -1)), dim=(-2, -1))
    power = (fft.abs() ** 2).mean(dim=(0, 1))  # [H, W]
    Hs, Ws = power.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, Hs, device=power.device),
        torch.linspace(-1, 1, Ws, device=power.device),
        indexing="ij",
    )
    r = torch.sqrt(x * x + y * y)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=power.device)
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            out[i] = float(power[mask].mean().item())
    return out


def _per_channel_kl_to_std_normal(z: torch.Tensor) -> np.ndarray:
    """Analytic KL(q(z_c) || N(0,1)) assuming q(z_c) is Gaussian N(mu_c, sig_c).

    Returns length-C array of per-channel nats.
    """
    z_ = z.float()
    C = z_.shape[1]
    kls = np.zeros(C, dtype=np.float64)
    for c in range(C):
        chan = z_[:, c].reshape(-1)
        mu = float(chan.mean().item())
        var = float(chan.var(unbiased=False).clamp(min=1e-12).item())
        kls[c] = 0.5 * (var + mu * mu - 1.0 - np.log(var))
    return kls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=Path, required=True)
    ap.add_argument("--runs_root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd(),
                    help="Repo root (for --stage1_config relative paths).")
    ap.add_argument("--split", type=str, default="training",
                    choices=["training", "validation"])
    ap.add_argument("--num_subjects", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=Path("paper/tables/latent_statistics.csv"))
    ap.add_argument("--no_channel_reorder", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    with open(args.datalist) as f:
        dl = json.load(f)
    items = dl[args.split][: args.num_subjects]
    transform = _val_transform(channel_reorder=not args.no_channel_reorder)

    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for model_name, spec in MODELS.items():
        cfg_path = args.repo / spec["stage1_config"]
        ckpt_path = args.runs_root / spec["stage1_ckpt"]
        if not ckpt_path.is_file():
            print(f"[skip] {model_name}: {ckpt_path} missing")
            continue

        print(f"\n=== {model_name} ===")
        print(f"  cfg  : {cfg_path}")
        print(f"  ckpt : {ckpt_path}")
        stage1 = _load_stage1(cfg_path, ckpt_path, device)

        latents = []
        with torch.no_grad():
            for i, item in enumerate(items):
                batch = transform(dict(item))
                x = batch["image"].unsqueeze(0).to(device)
                z = stage1(x)  # [1, C, D, H, W]
                latents.append(z.detach().cpu())
        Z = torch.cat(latents, dim=0)  # [N, C, D, H, W]
        Z = Z.to(device)
        C = Z.shape[1]

        # Global statistics.
        overall_std = float(Z.std().item())
        overall_mean = float(Z.mean().item())
        lag1 = _lag1_autocorr(Z)
        mean_abs_offdiag_corr, corr_mat = _channel_correlation(Z)
        eff_rank = _effective_rank(Z)
        kls = _per_channel_kl_to_std_normal(Z)
        rps = _radial_power_spectrum(Z)

        summary[model_name] = {
            "n_channels": C,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "lag1_autocorr": lag1,
            "mean_abs_offdiag_corr": mean_abs_offdiag_corr,
            "effective_rank": eff_rank,
            "mean_kl_to_N01": float(kls.mean()),
            "max_kl_to_N01": float(kls.max()),
            "radial_power_hi_over_lo": float(rps[-4:].mean() / max(rps[:4].mean(), 1e-12)),
        }

        for c in range(C):
            zc = Z[:, c]
            rows.append({
                "model": model_name,
                "channel": c,
                "mean": float(zc.mean().item()),
                "std": float(zc.std().item()),
                "kl_to_N01": float(kls[c]),
            })

        del Z, latents
        torch.cuda.empty_cache() if device.type == "cuda" else None

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")

    print("\n=== per-model summary ===")
    for name, s in summary.items():
        print(f"\n[{name}]  ({s['n_channels']} channels)")
        print(f"  overall mean/std             : {s['overall_mean']:+.4f} / {s['overall_std']:.4f}")
        print(f"  lag-1 autocorr (rough noise) : {s['lag1_autocorr']:.4f}   (lower = smoother)")
        print(f"  mean |off-diag corr|         : {s['mean_abs_offdiag_corr']:.4f}   (lower = more decorrelated)")
        print(f"  effective rank               : {s['effective_rank']:.3f} / {s['n_channels']}")
        print(f"  mean KL to N(0,1)            : {s['mean_kl_to_N01']:.4f}   (per-channel, nats)")
        print(f"  max  KL to N(0,1)            : {s['max_kl_to_N01']:.4f}")
        print(f"  hi/lo radial power ratio     : {s['radial_power_hi_over_lo']:.3f}   (>1 = high-freq dominated)")

    if len(summary) == 2:
        a, b = list(summary.keys())
        print("\n=== deltas (MaxFeat - BrainLDM-FT) ===")
        for k in ("lag1_autocorr", "mean_abs_offdiag_corr",
                  "mean_kl_to_N01", "max_kl_to_N01",
                  "radial_power_hi_over_lo"):
            diff = summary[b][k] - summary[a][k]
            direction = "MaxFeat higher" if diff > 0 else "MaxFeat lower "
            print(f"  {k:35s}: {diff:+.4f}   ({direction})")


if __name__ == "__main__":
    main()
