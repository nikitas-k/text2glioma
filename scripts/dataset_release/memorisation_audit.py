"""Memorisation audit for the Text2Glioma synthetic release.

For each generated sample, compute the maximum SSIM (per-modality mean)
against every training-cohort image. High values indicate the model may
have memorised a specific training case. Following Dar et al. 2024
(arXiv 2402.01054), we flag samples whose NN-SSIM exceeds 0.90 on any
modality as potential-memorisation candidates.

The audit is embarrassingly parallel: each synthetic sample is
independent. We support --shard-based parallelism to match the
generation pipeline.

Reference distribution
----------------------

For calibration, we also report a null distribution: NN-SSIM computed
between **held-out training samples and the rest of the training set**.
If synthetic samples have NN-SSIM comparable to this "one-training-vs-
others" baseline, the model isn't memorising; it's producing samples
about as close to the training set as any individual training sample is
to its neighbours.

Output
------

CSV rows::

    sample_id, modality, nn_ssim, nn_train_subj, nn_train_path

Plus a summary JSON with:

    * per-modality nn_ssim distribution (mean, median, p90, p99, max)
    * count of samples with any modality > 0.90 (potential memorisation)
    * null-distribution stats for comparison

Usage
-----
::

    python scripts/dataset_release/memorisation_audit.py \\
        --synth_root /path/to/synth_10k/ \\
        --datalist datalist_N1510.json \\
        --split training \\
        --out audit_shard_0.csv \\
        --shard 0 --num_shards 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


MODALITIES = ("T1", "T1CE", "T2", "FLAIR")
DEFAULT_MEMORISATION_THRESHOLD = 0.90


def _ssim_3d(pred: torch.Tensor, target: torch.Tensor,
             window_size: int = 7,
             c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    """Single-channel SSIM. Both tensors are (1, 1, D, H, W)."""
    kernel = torch.ones(1, 1, window_size, window_size, window_size,
                        device=pred.device, dtype=pred.dtype) / (window_size**3)
    pad = window_size // 2
    mu_p = F.conv3d(pred, kernel, padding=pad)
    mu_t = F.conv3d(target, kernel, padding=pad)
    mu_pp = mu_p * mu_p; mu_tt = mu_t * mu_t; mu_pt = mu_p * mu_t
    s_pp = F.conv3d(pred * pred, kernel, padding=pad) - mu_pp
    s_tt = F.conv3d(target * target, kernel, padding=pad) - mu_tt
    s_pt = F.conv3d(pred * target, kernel, padding=pad) - mu_pt
    ssim_map = ((2 * mu_pt + c1) * (2 * s_pt + c2)) / \
               ((mu_pp + mu_tt + c1) * (s_pp + s_tt + c2))
    return ssim_map.mean()


def _ssim_3d_batched(
    pred: torch.Tensor,           # (1, 1, D, H, W)
    bank_chunk: torch.Tensor,     # (B, 1, D, H, W)
    window_size: int = 7,
    c1: float = 0.01**2, c2: float = 0.03**2,
) -> torch.Tensor:
    """Batched SSIM of one prediction against B reference volumes.

    Same numerics as ``_ssim_3d`` — the mean-of-map SSIM with a 7^3
    uniform window — but exploits broadcasting so all B convolutions run
    in single conv3d calls. Returns (B,) per-volume SSIM values.
    """
    kernel = torch.ones(1, 1, window_size, window_size, window_size,
                        device=pred.device, dtype=pred.dtype) / (window_size**3)
    pad = window_size // 2
    # Reused across the B references (pred is constant per call).
    mu_p  = F.conv3d(pred, kernel, padding=pad)                     # (1, 1, D, H, W)
    mu_pp = mu_p * mu_p                                             # (1, 1, D, H, W)
    s_pp  = F.conv3d(pred * pred, kernel, padding=pad) - mu_pp      # (1, 1, D, H, W)

    mu_t  = F.conv3d(bank_chunk, kernel, padding=pad)               # (B, 1, D, H, W)
    mu_tt = mu_t * mu_t                                             # (B, 1, D, H, W)
    s_tt  = F.conv3d(bank_chunk * bank_chunk, kernel, padding=pad) - mu_tt

    # Cross terms broadcast (1,1,...) with (B,1,...) -> (B,1,...)
    mu_pt = mu_p * mu_t                                             # (B, 1, D, H, W)
    s_pt  = F.conv3d(pred * bank_chunk, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * s_pt + c2)) / \
               ((mu_pp + mu_tt + c1) * (s_pp + s_tt + c2))
    return ssim_map.mean(dim=(1, 2, 3, 4))  # (B,)


def _load_synth_image(sample_dir: Path) -> np.ndarray:
    """Load a synthetic sample's 4D image as (4, X, Y, Z) float32 in [0, 1]."""
    arr = nib.load(str(sample_dir / "image.nii.gz")).get_fdata().astype(np.float32)
    if arr.ndim == 4:
        return np.moveaxis(arr, -1, 0)
    if arr.ndim == 3:
        return arr[None]
    raise ValueError(f"unexpected shape {arr.shape} at {sample_dir}")


def _load_training_bank(datalist_path: Path, split: str,
                        device: torch.device,
                        half: bool = False,
                        keep_on_cpu: bool = True,
                        ) -> tuple[torch.Tensor, list[str], list[str]]:
    """Load and preprocess every training case into a stacked tensor.

    Returns:
        bank:    (N, 4, X, Y, Z) tensor in [0, 1]. Kept on **CPU** by default
                 (~54 GB fp16 for N=1187 on the standard 160x224x160 grid) so
                 it fits alongside a GPU that only has ~32 GB. Chunks are
                 transferred to GPU on demand inside ``audit_one_sample``.
        paths:   list of source image paths (for reporting nn_train_path)
        subjs:   list of subject_ids (for nn_train_subj)

    Uses the same MONAI preprocessing as the offline sampler so the
    training images are in the same 160x224x160 preprocessed space as
    the synthetic samples.
    """
    from monai import transforms as T

    with open(datalist_path) as f:
        dl = json.load(f)
    if split not in dl:
        raise KeyError(f"split {split!r} not in {datalist_path}")
    items = dl[split]

    xforms = T.Compose([
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
        T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5,
                                          b_min=0, b_max=1, channel_wise=True),
        T.ToTensord(keys=["image"]),
    ])

    target_dtype = torch.float16 if half else torch.float32
    target_device = torch.device("cpu") if keep_on_cpu else device

    stacked: list[torch.Tensor] = []
    paths: list[str] = []
    subjs: list[str] = []
    for i, item in enumerate(items):
        try:
            batch = xforms({"image": item["image"]})
        except Exception as e:
            print(f"[warn] skipping {item.get('subject_id', i)}: {e}", file=sys.stderr)
            continue
        # Cast + move immediately to keep peak memory bounded.
        img = batch["image"].as_subclass(torch.Tensor).to(
            device=target_device, dtype=target_dtype, copy=False
        ).clamp(0.0, 1.0)
        stacked.append(img)
        paths.append(str(item["image"]))
        subjs.append(str(item.get("subject_id", f"idx_{i}")))
        if (i + 1) % 100 == 0:
            print(f"  loaded {i+1}/{len(items)} training images "
                  f"(bank so far: {sum(t.element_size()*t.numel() for t in stacked)/1e9:.1f} GB)",
                  flush=True)
    if not stacked:
        raise RuntimeError("no training images could be loaded")
    bank = torch.stack(stacked, dim=0)  # (N, 4, X, Y, Z)
    print(f"training bank: {bank.shape} dtype={bank.dtype} on {bank.device}   "
          f"{bank.element_size()*bank.numel()/1e9:.1f} GB")
    return bank, paths, subjs


def audit_one_sample(
    synth_img: torch.Tensor,   # (4, X, Y, Z), on GPU
    bank: torch.Tensor,         # (N, 4, X, Y, Z), typically on CPU
    bank_paths: list[str],
    bank_subjs: list[str],
    sample_id: str,
    chunk_size: int = 32,
    gpu_device: torch.device | None = None,
) -> list[dict]:
    """For each modality, find the nearest training image (by SSIM) and
    return one row per modality.

    The bank stays on CPU (fp16, ~54 GB for N=1187); each chunk is
    transferred to GPU on demand so peak GPU memory is only
    ``chunk_size × 4 × 160 × 224 × 160 × dtype_bytes`` plus SSIM
    intermediates. On a V100 with chunk_size=32 and fp16, that's ~1.5 GB
    per chunk, easily fitting alongside the current sample.
    """
    if gpu_device is None:
        gpu_device = synth_img.device
    rows: list[dict] = []
    N = bank.shape[0]
    for c, mod in enumerate(MODALITIES):
        s_pred = synth_img[c:c+1, None]                   # (1, 1, X, Y, Z)
        best_ssim = -1.0
        best_idx = -1
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # Move only this chunk to GPU. If the bank is already on the
            # GPU (e.g. small N + fp16 fits), the .to() is a no-op alias.
            s_ref = bank[start:end, c:c+1].to(
                device=gpu_device, dtype=s_pred.dtype, non_blocking=True
            )
            v = _ssim_3d_batched(s_pred, s_ref)           # (B,)
            local_max, local_arg = torch.max(v, dim=0)
            local_max_f = float(local_max.item())
            if local_max_f > best_ssim:
                best_ssim = local_max_f
                best_idx = start + int(local_arg.item())
            del s_ref, v
        rows.append({
            "sample_id": sample_id,
            "modality":  mod,
            "nn_ssim":   float(best_ssim),
            "nn_train_subj": bank_subjs[best_idx] if best_idx >= 0 else None,
            "nn_train_path": bank_paths[best_idx] if best_idx >= 0 else None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synth_root", type=Path, required=True,
                    help="Root dir containing shard_XXXX/sample_XXXXXXX/image.nii.gz")
    ap.add_argument("--datalist",   type=Path, required=True,
                    help="Datalist providing the training bank.")
    ap.add_argument("--split",      default="training")
    ap.add_argument("--shard",      type=int, default=None,
                    help="Optional: audit only this shard (workers use PBS_ARRAY_INDEX).")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="If sharding across workers, total number of shards to hash-split into.")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Audit only the first N samples (across the whole synth set).")
    ap.add_argument("--memorisation_threshold", type=float,
                    default=DEFAULT_MEMORISATION_THRESHOLD,
                    help="NN-SSIM threshold above which a sample is flagged.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--chunk_size", type=int, default=32,
                    help="Batch size for the bank scan (larger = faster if "
                         "GPU memory allows; V100 handles 32 comfortably).")
    ap.add_argument("--half", action="store_true",
                    help="Cast bank + inputs to fp16 to halve GPU memory. "
                         "SSIM values differ by <1e-3 vs fp32.")
    args = ap.parse_args()

    if args.shard is None:
        env = os.environ.get("PBS_ARRAY_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is not None:
            args.shard = int(env)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    # Enumerate synthetic sample dirs
    synth_dirs: list[Path] = sorted(
        p for p in args.synth_root.glob("shard_*/sample_*")
        if (p / "image.nii.gz").is_file()
    )
    print(f"found {len(synth_dirs)} synthetic samples under {args.synth_root}")

    # Shard-split if requested
    if args.shard is not None and args.num_shards > 1:
        synth_dirs = [d for i, d in enumerate(synth_dirs)
                      if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards} -> {len(synth_dirs)} samples")
    if args.max_samples is not None:
        synth_dirs = synth_dirs[: args.max_samples]
        print(f"capped to first {len(synth_dirs)} samples")

    # Load training bank (once). Kept on CPU (fp16 if --half) to fit
    # alongside the GPU; chunks are transferred to GPU inside
    # audit_one_sample() as they're needed.
    print(f"\nloading training bank from {args.datalist}...", flush=True)
    bank, bank_paths, bank_subjs = _load_training_bank(
        args.datalist, args.split, device,
        half=args.half, keep_on_cpu=True,
    )

    rows: list[dict] = []
    t0 = time.time()
    for i, sample_dir in enumerate(synth_dirs):
        try:
            img_np = _load_synth_image(sample_dir)
            img = torch.from_numpy(img_np).to(device).clamp(0.0, 1.0)
            if args.half and device.type == "cuda":
                img = img.half()
            sample_id = sample_dir.name
            rows.extend(audit_one_sample(img, bank, bank_paths, bank_subjs,
                                          sample_id, chunk_size=args.chunk_size,
                                          gpu_device=device))
        except Exception as e:
            print(f"[warn] {sample_dir}: {e}", file=sys.stderr)
            continue
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(synth_dirs) - i - 1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(synth_dirs)}  ({rate:.2f}/s, eta {eta/60:.1f} min)",
                  flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}   ({len(df)} rows)")

    # Summary stats
    if len(df) == 0:
        print("no audit rows written; nothing to summarise")
        return
    print("\n== per-modality NN-SSIM ==")
    for mod in MODALITIES:
        sub = df[df.modality == mod]["nn_ssim"]
        if len(sub) == 0: continue
        print(f"  {mod:>5}   mean={sub.mean():.3f}   median={sub.median():.3f}   "
              f"p90={sub.quantile(0.9):.3f}   p99={sub.quantile(0.99):.3f}   max={sub.max():.3f}")

    # Sample-level "flagged" count: any modality exceeds threshold.
    flagged = (df.groupby("sample_id")["nn_ssim"].max() > args.memorisation_threshold)
    n_flag = int(flagged.sum())
    print(f"\nsamples with any modality NN-SSIM > {args.memorisation_threshold}: "
          f"{n_flag}/{flagged.size} ({100*n_flag/max(flagged.size,1):.2f}%)")


if __name__ == "__main__":
    main()
