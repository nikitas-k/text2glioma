#!/usr/bin/env python3
"""Offline per-channel L1 & SSIM evaluation on train + val sets.

Loads the latest checkpoint from a training run, reconstructs the full
val set and a subset of the training set, and prints per-channel L1 +
3D SSIM without modifying the running job.  Runs on a single GPU
(no DDP needed).

Usage (local or Gadi interactive)::

    python scripts/offline_val_metrics.py \
        --run_dir /g/data/vp06/$USER/text2glioma_train/runs/bf16_adaptive_v2_pw_1.0_r1_1.0_adv_weight_0.1 \
        --data_dir /g/data/vp06/$USER/text2glioma_train/data

Or with a datalist::

    python scripts/offline_val_metrics.py \
        --run_dir ... --datalist datalist.json --no_channel_reorder

By default evaluates all val samples and the same number of training
samples (randomly drawn).  Use --train_samples N to control.
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
                   help="Number of samples to save as NIfTI per split (0 to skip)")
    p.add_argument("--train_samples", type=int, default=-1,
                   help="Number of training samples to evaluate (-1 = same as val set size)")
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


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    epoch: int | str,
    saver=None,
    max_save: int = 0,
) -> dict:
    """Evaluate a single data split and return metrics dict."""
    per_ch_l1 = {cn: 0.0 for cn in CH_NAMES}
    per_ch_ssim = {cn: 0.0 for cn in CH_NAMES}
    total_l1 = 0.0
    n_samples = 0
    n_saved = 0

    with torch.no_grad():
        for x in tqdm(loader, desc=f"Evaluating {split_name}"):
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
            if saver is not None and n_saved < max_save:
                for i in range(bs):
                    if n_saved >= max_save:
                        break
                    prefix = f"ep{epoch}_{split_name}"
                    saver.save(img_f[i], f"{prefix}_sample{n_saved:02d}_original.nii.gz")
                    saver.save(rec_f[i], f"{prefix}_sample{n_saved:02d}_recon.nii.gz")
                    n_saved += 1

            n_samples += bs

    # Aggregate
    total_l1 /= n_samples
    for cn in CH_NAMES:
        per_ch_l1[cn] /= n_samples
        per_ch_ssim[cn] /= n_samples
    mean_ssim = sum(per_ch_ssim.values()) / len(CH_NAMES)

    worst_l1 = max(per_ch_l1, key=per_ch_l1.get)
    worst_ssim = min(per_ch_ssim, key=per_ch_ssim.get)

    return {
        "split": split_name,
        "n_samples": n_samples,
        "n_saved": n_saved,
        "overall_l1": total_l1,
        "overall_ssim": mean_ssim,
        "per_channel_l1": per_ch_l1,
        "per_channel_ssim": per_ch_ssim,
        "weakest_l1": worst_l1,
        "weakest_ssim": worst_ssim,
    }


def print_split_report(m: dict, epoch: int | str) -> None:
    """Pretty-print metrics for one split."""
    split = m["split"]
    n = m["n_samples"]
    print(f"\n{'='*60}")
    print(f"  {split.upper()} Metrics  (epoch {epoch}, N={n})")
    print(f"{'='*60}")
    print(f"\n  Overall L1:   {m['overall_l1']:.6f}")
    print(f"  Overall SSIM: {m['overall_ssim']:.6f}")
    print(f"\n  {'Channel':>8s}  {'L1':>10s}  {'SSIM':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
    for cn in CH_NAMES:
        print(f"  {cn:>8s}  {m['per_channel_l1'][cn]:>10.6f}  {m['per_channel_ssim'][cn]:>10.6f}")
    print(f"\n  Weakest channel (L1):   {m['weakest_l1']} ({m['per_channel_l1'][m['weakest_l1']]:.6f})")
    print(f"  Weakest channel (SSIM): {m['weakest_ssim']} ({m['per_channel_ssim'][m['weakest_ssim']]:.6f})")


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

    # ── Datasets ──
    channel_reorder = not args.no_channel_reorder
    xform = get_val_transform(channel_reorder)

    if args.datalist:
        with open(args.datalist) as f:
            datalist = json.load(f)

        val_data = datalist["validation"]
        val_ds = Dataset(data=val_data, transform=xform)

        # Training subset
        train_data = datalist["training"]
        n_train = args.train_samples if args.train_samples > 0 else len(val_data)
        n_train = min(n_train, len(train_data))
        # Deterministic random subset
        import random
        rng = random.Random(args.seed)
        train_subset = rng.sample(train_data, n_train)
        train_ds = Dataset(data=train_subset, transform=xform)
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
            transform=xform,
            num_workers=args.num_workers,
        )
        train_ds = DecathlonDataset(
            root_dir=args.data_dir,
            task="Task01_BrainTumour",
            section="training",
            download=False,
            seed=args.seed,
            val_frac=args.val_frac,
            transform=xform,
            num_workers=args.num_workers,
        )
        # Subsample training set
        n_train = args.train_samples if args.train_samples > 0 else len(val_ds)
        n_train = min(n_train, len(train_ds))
        import random
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(train_ds)), n_train)
        train_ds = torch.utils.data.Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train subset: {len(train_ds)} samples")
    print(f"Val set:      {len(val_ds)} samples")

    # ── Optional NIfTI sample saving ──
    saver = None
    if args.save_samples > 0:
        from text2glioma.inference.saver import NiftiSaver
        samples_dir = run_dir / "samples"
        saver = NiftiSaver(output_dir=str(samples_dir), rescale=False)
        print(f"Will save {args.save_samples} sample pairs per split to {samples_dir}")

    # ── Evaluate both splits ──
    train_metrics = evaluate_split(
        model, train_loader, device, "train", epoch,
        saver=saver, max_save=args.save_samples,
    )
    val_metrics = evaluate_split(
        model, val_loader, device, "val", epoch,
        saver=saver, max_save=args.save_samples,
    )

    # ── Report ──
    print_split_report(train_metrics, epoch)
    print_split_report(val_metrics, epoch)

    # ── Generalisation gap ──
    gap_l1 = val_metrics["overall_l1"] - train_metrics["overall_l1"]
    gap_ssim = val_metrics["overall_ssim"] - train_metrics["overall_ssim"]
    print(f"\n{'='*60}")
    print(f"  Generalisation Gap (val − train)")
    print(f"{'='*60}")
    print(f"  ΔL1:   {gap_l1:+.6f}")
    print(f"  ΔSSIM: {gap_ssim:+.6f}")

    # ── Save JSON ──
    combined = {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
        "generalisation_gap": {"l1": gap_l1, "ssim": gap_ssim},
    }
    out_path = run_dir / "offline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Saved to {out_path}")

    total_saved = train_metrics["n_saved"] + val_metrics["n_saved"]
    if total_saved > 0:
        print(f"  Saved {total_saved} sample pairs to {run_dir / 'samples'}")


if __name__ == "__main__":
    main()
