#!/usr/bin/env python3
"""Offline per-channel L1 & SSIM evaluation on the val set.

Loads the latest checkpoint from a training run, reconstructs the full
val set, and prints per-channel L1 + 3D SSIM without modifying the
running job.  Runs on a single GPU (no DDP needed).

Usage (local or Gadi interactive)::

    python scripts/offline_val_metrics.py \
        --run_dir /g/data/vp06/$USER/text2glioma_train/runs/bf16_adaptive_v2_pw_1.0_r1_1.0_adv_weight_0.1 \
        --data_dir /g/data/vp06/$USER/text2glioma_train/data

Or with a datalist::

    python scripts/offline_val_metrics.py \
        --run_dir ... --datalist datalist.json --no_channel_reorder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from tqdm import tqdm

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]
CH_NAMES = ["T1", "T1CE", "T2", "FLAIR"]


def parse_args():
    p = argparse.ArgumentParser(description="Offline per-channel L1 & SSIM eval")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Root run dir (contains autoencoder_stage1/)")
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Root for DecathlonDataset")
    p.add_argument("--datalist", type=str, default=None,
                   help="JSON datalist (overrides --data_dir)")
    p.add_argument("--no_channel_reorder", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--save_samples", type=int, default=5,
                   help="Number of val samples to save as NIfTI (0 to skip)")
    return p.parse_args()


def get_val_transform(channel_reorder: bool = True) -> T.Compose:
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
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.ToTensord(keys=["image"]),
    ])
    return T.Compose(xforms)


def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Mean SSIM for single-channel 3D volumes (N,1,D,H,W)."""
    kernel = torch.ones(1, 1, window_size, window_size, window_size,
                        device=pred.device, dtype=pred.dtype)
    kernel = kernel / kernel.numel()
    pad = window_size // 2

    mu_p = F.conv3d(pred, kernel, padding=pad)
    mu_t = F.conv3d(target, kernel, padding=pad)
    mu_pp, mu_tt, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_pp = F.conv3d(pred * pred, kernel, padding=pad) - mu_pp
    sigma_tt = F.conv3d(target * target, kernel, padding=pad) - mu_tt
    sigma_pt = F.conv3d(pred * target, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))
    return ssim_map.mean()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load config from run dir ──
    run_dir = Path(args.run_dir) / "autoencoder_stage1"
    config_path = run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config_snapshot.yaml in {run_dir}")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Load model ──
    from text2glioma.utils import get_model
    model_type = config["model"]["name"]
    model = get_model(model_type, config)
    model = model.to(device)

    ckpt_path = run_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint.pth in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Handle DDP state_dict (keys prefixed with "module.")
    state_dict = ckpt["state_dict"]
    clean_sd = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_sd[key] = v
    model.load_state_dict(clean_sd)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {epoch}")

    model.eval()

    # ── Dataset ──
    channel_reorder = not args.no_channel_reorder
    if args.datalist:
        with open(args.datalist) as f:
            datalist = json.load(f)
        val_data = datalist["validation"]
        val_ds = Dataset(data=val_data, transform=get_val_transform(channel_reorder))
    else:
        from monai.utils import set_determinism
        set_determinism(args.seed)
        val_ds = DecathlonDataset(
            root_dir=args.data_dir,
            task="Task01_BrainTumour",
            section="validation",
            download=False,
            seed=args.seed,
            val_frac=args.val_frac,
            transform=get_val_transform(channel_reorder),
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Val set: {len(val_ds)} samples")

    # ── Optional NIfTI sample saving ──
    saver = None
    if args.save_samples > 0:
        from text2glioma.inference.saver import NiftiSaver
        samples_dir = run_dir / "samples"
        saver = NiftiSaver(output_dir=str(samples_dir), rescale=True)
        print(f"Will save {args.save_samples} sample pairs to {samples_dir}")

    # ── Evaluate ──
    # Accumulators
    per_ch_l1 = {cn: 0.0 for cn in CH_NAMES}
    per_ch_ssim = {cn: 0.0 for cn in CH_NAMES}
    total_l1 = 0.0
    n_samples = 0
    n_saved = 0

    with torch.no_grad():
        for x in tqdm(val_loader, desc="Evaluating"):
            images = x["image"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                reconstruction, z_mu, z_sigma = model(x=images)

            rec_f = reconstruction.float()
            img_f = images.float()
            bs = images.shape[0]

            total_l1 += F.l1_loss(rec_f, img_f).item() * bs

            for ci, cn in enumerate(CH_NAMES):
                r = rec_f[:, ci:ci+1]
                t = img_f[:, ci:ci+1]
                per_ch_l1[cn] += F.l1_loss(r, t).item() * bs
                per_ch_ssim[cn] += ssim_3d(r, t).item() * bs

            # Save sample pairs as NIfTI
            if saver is not None and n_saved < args.save_samples:
                for i in range(bs):
                    if n_saved >= args.save_samples:
                        break
                    saver.save(img_f[i], f"ep{epoch}_sample{n_saved:02d}_original.nii.gz")
                    saver.save(rec_f[i], f"ep{epoch}_sample{n_saved:02d}_recon.nii.gz")
                    n_saved += 1

            n_samples += bs

    # ── Report ──
    total_l1 /= n_samples
    for cn in CH_NAMES:
        per_ch_l1[cn] /= n_samples
        per_ch_ssim[cn] /= n_samples

    mean_ssim = sum(per_ch_ssim.values()) / len(CH_NAMES)

    print(f"\n{'='*60}")
    print(f"  Offline Val Metrics  (epoch {epoch}, N={n_samples})")
    print(f"{'='*60}")
    print(f"\n  Overall L1:   {total_l1:.6f}")
    print(f"  Overall SSIM: {mean_ssim:.6f}")
    print(f"\n  {'Channel':>8s}  {'L1':>10s}  {'SSIM':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
    for cn in CH_NAMES:
        print(f"  {cn:>8s}  {per_ch_l1[cn]:>10.6f}  {per_ch_ssim[cn]:>10.6f}")

    # Identify weakest channel
    worst_l1 = max(per_ch_l1, key=per_ch_l1.get)
    worst_ssim = min(per_ch_ssim, key=per_ch_ssim.get)
    print(f"\n  Weakest channel (L1):   {worst_l1} ({per_ch_l1[worst_l1]:.6f})")
    print(f"  Weakest channel (SSIM): {worst_ssim} ({per_ch_ssim[worst_ssim]:.6f})")

    # ── Save JSON ──
    metrics = {
        "epoch": epoch,
        "n_samples": n_samples,
        "overall_l1": total_l1,
        "overall_ssim": mean_ssim,
        "per_channel_l1": per_ch_l1,
        "per_channel_ssim": per_ch_ssim,
        "weakest_l1": worst_l1,
        "weakest_ssim": worst_ssim,
    }
    out_path = run_dir / "offline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved to {out_path}")

    if n_saved > 0:
        print(f"  Saved {n_saved} sample pairs to {run_dir / 'samples'}")


if __name__ == "__main__":
    main()
