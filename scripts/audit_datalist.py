#!/usr/bin/env python3
"""Audit spatial metadata and effective resolution for every image in a datalist JSON.

Reports per-subject:
  - Voxel resolution (mm)
  - Volume dimensions (voxels)
  - Slice thickness (if anisotropic)
  - Brain bounding box after foreground crop (voxels)
  - Per-channel spectral sharpness (high-freq energy ratio) — detects upsampled data
  - Per-channel Laplacian variance — proxy for edge sharpness
  - Flags outliers (resolution, dimensions, low effective resolution)

Usage (on Gadi or wherever the NIfTIs live)::

    python scripts/audit_datalist.py --datalist datalist_N1511.json
    python scripts/audit_datalist.py --datalist datalist_N1511.json --split training
    python scripts/audit_datalist.py --datalist datalist_N1511.json --max-subjects 50

    # Full sharpness audit with CSV output:
    python scripts/audit_datalist.py --datalist datalist_N1511.json --sharpness --csv audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import laplace


# ── Helpers ──────────────────────────────────────────────────────────────────

def brain_bbox(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tight bounding box of non-zero voxels.

    Returns
    -------
    starts : array of int, shape (ndim,)
        Start index of bounding box per axis.
    stops  : array of int, shape (ndim,)
        Stop index (exclusive) per axis.
    extents : array of int, shape (ndim,)
        Size of bounding box per axis.
    """
    # Collapse to 3D if multi-channel (e.g. 4-ch MRI stored as 4th dim)
    if data.ndim == 4:
        mask = np.any(data != 0, axis=-1)  # last dim is channels
    else:
        mask = data != 0

    nonzero = np.argwhere(mask)
    if len(nonzero) == 0:
        # Entirely zero — shouldn't happen but handle gracefully
        return np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    starts = nonzero.min(axis=0)
    stops = nonzero.max(axis=0) + 1
    extents = stops - starts
    return starts, stops, extents


def is_outlier_resolution(pixdim: np.ndarray, tol: float = 0.05) -> bool:
    """Flag if any voxel dimension deviates from 1.0 mm by more than tol."""
    return bool(np.any(np.abs(pixdim[:3] - 1.0) > tol))


def is_anisotropic(pixdim: np.ndarray, tol: float = 0.05) -> bool:
    """Flag if voxel dimensions are not isotropic within tol."""
    return bool(np.ptp(pixdim[:3]) > tol)


MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]


def spectral_sharpness(vol: np.ndarray) -> float:
    """Ratio of high-frequency to total energy in 3D FFT.

    A genuinely high-resolution volume has more high-frequency energy.
    Upsampled (interpolated) volumes concentrate energy near DC.

    Returns a scalar in [0, 1]; higher = sharper.
    """
    fft = np.fft.fftn(vol)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)

    shape = np.array(vol.shape)
    centre = shape // 2
    lf_frac = 0.15  # fraction of each axis counted as "low frequency"
    slices = tuple(
        slice(int(c - lf_frac * s), int(c + lf_frac * s))
        for c, s in zip(centre, shape)
    )
    total = np.sum(mag ** 2)
    if total == 0:
        return 0.0
    lf_mask = np.zeros_like(mag)
    lf_mask[slices] = 1.0
    hf_energy = np.sum((mag * (1 - lf_mask)) ** 2)
    return float(hf_energy / total)


def laplacian_variance(vol: np.ndarray) -> float:
    """Variance of the Laplacian — higher = sharper edges."""
    lap = laplace(vol.astype(np.float32))
    return float(np.var(lap))


def per_channel_sharpness(data: np.ndarray) -> list[dict]:
    """Compute spectral sharpness and Laplacian variance per channel.

    Parameters
    ----------
    data : array, shape (X, Y, Z, C) or (X, Y, Z)

    Returns
    -------
    List of dicts with keys: modality, hf_ratio, lap_var.
    """
    if data.ndim == 3:
        data = data[..., np.newaxis]

    n_ch = data.shape[-1]
    results = []
    for ch in range(min(n_ch, 4)):
        vol = data[:, :, :, ch]
        mod = MODALITY_NAMES[ch] if ch < len(MODALITY_NAMES) else f"ch{ch}"
        hf = spectral_sharpness(vol)
        lv = laplacian_variance(vol)
        results.append({"modality": mod, "hf_ratio": hf, "lap_var": lv})
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit spatial metadata of datalist images.")
    parser.add_argument("--datalist", type=str, required=True, help="Path to datalist JSON.")
    parser.add_argument("--split", type=str, default=None,
                        help="Which split to audit (training, validation, testing). "
                             "Default: all splits with entries.")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects to process (for quick checks).")
    parser.add_argument("--crop-target", type=int, nargs=3, default=[160, 224, 160],
                        metavar=("D", "H", "W"),
                        help="Target crop size for flagging (default: 160 224 160).")
    parser.add_argument("--include-labels", action="store_true",
                        help="Also report label bounding box (tumour extent).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write per-subject CSV with all metrics.")
    parser.add_argument("--sharpness", action="store_true", default=False,
                        help="Compute per-channel spectral sharpness and Laplacian "
                             "variance (slower — loads full image data for FFT).")
    args = parser.parse_args()

    crop_target = np.array(args.crop_target)

    with open(args.datalist) as f:
        datalist = json.load(f)

    # Determine which splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = [k for k in ("training", "validation", "testing") if datalist.get(k)]

    # Collect all entries
    entries: List[dict] = []
    for split in splits:
        for item in datalist.get(split, []):
            item["_split"] = split
            entries.append(item)

    if args.max_subjects:
        entries = entries[: args.max_subjects]

    n = len(entries)
    print(f"Auditing {n} subjects across {splits} ...\n")

    # Storage for summary statistics
    all_pixdims = []
    all_shapes = []
    all_bbox_extents = []
    all_sharpness = []  # list of per-subject channel sharpness dicts
    csv_rows = []
    flags: List[str] = []

    # Per-subject header
    header = (
        f"{'#':>5s}  {'subject_id':<20s}  {'split':<10s}  "
        f"{'shape':>20s}  {'pixdim (mm)':>18s}  "
        f"{'bbox extent':>20s}  {'flags'}"
    )
    print(header)
    print("-" * len(header))

    for i, item in enumerate(entries):
        img_path = item["image"]
        subj_id = item.get("subject_id", Path(img_path).stem)
        split = item["_split"]

        # --- Load header only first (fast) ---
        try:
            nii = nib.load(img_path)
        except Exception as exc:
            flag_str = f"LOAD_ERROR: {exc}"
            flags.append(flag_str)
            print(f"{i+1:5d}  {subj_id:<20s}  {split:<10s}  {'---':>20s}  {'---':>18s}  {'---':>20s}  {flag_str}")
            continue

        hdr = nii.header
        shape = np.array(nii.shape[:3])  # spatial dims only
        pixdim = np.abs(hdr.get_zooms()[:3])
        full_shape_str = "x".join(str(s) for s in nii.shape)

        # --- Load data for bbox (slower but necessary) ---
        data = np.asarray(nii.dataobj)
        _, _, bbox_ext = brain_bbox(data)

        all_pixdims.append(pixdim)
        all_shapes.append(shape)
        all_bbox_extents.append(bbox_ext)

        # --- Per-channel sharpness (optional, slower) ---
        ch_sharp = []
        if args.sharpness:
            ch_sharp = per_channel_sharpness(data)
            all_sharpness.append(ch_sharp)

        # --- Flag outliers ---
        subj_flags = []

        # Resolution
        if is_outlier_resolution(pixdim):
            subj_flags.append(f"RESOLUTION({pixdim[0]:.3f},{pixdim[1]:.3f},{pixdim[2]:.3f})")

        if is_anisotropic(pixdim):
            subj_flags.append("ANISOTROPIC")

        # Dimensions — BraTS is typically 240x240x155
        if np.any(shape[:2] != 240) or shape[2] != 155:
            subj_flags.append(f"NON_STANDARD_DIM({full_shape_str})")

        # Brain bbox vs crop target
        for ax, (ext, tgt, ax_name) in enumerate(
            zip(bbox_ext, crop_target, ["LR", "AP", "SI"])
        ):
            if ext > tgt:
                excess = ext - tgt
                subj_flags.append(f"BRAIN_EXCEEDS_{ax_name}({ext}>{tgt}, clip={excess}vox)")
            elif ext < tgt * 0.6:
                subj_flags.append(f"SMALL_BRAIN_{ax_name}({ext}<{int(tgt*0.6)})")

        # Zero volume
        if np.all(bbox_ext == 0):
            subj_flags.append("EMPTY_VOLUME")

        # Effective resolution (spectral sharpness)
        if ch_sharp:
            for cs in ch_sharp:
                if cs["hf_ratio"] < 0.35:
                    subj_flags.append(
                        f"LOW_HF_{cs['modality']}({cs['hf_ratio']:.3f})")

        flag_str = ", ".join(subj_flags) if subj_flags else "ok"
        if subj_flags:
            flags.extend(subj_flags)

        pixdim_str = f"{pixdim[0]:.3f}x{pixdim[1]:.3f}x{pixdim[2]:.3f}"
        bbox_str = "x".join(str(e) for e in bbox_ext)

        sharp_str = ""
        if ch_sharp:
            sharp_str = "  " + "  ".join(
                f"{cs['modality']}:hf={cs['hf_ratio']:.3f},lap={cs['lap_var']:.1f}"
                for cs in ch_sharp
            )

        print(
            f"{i+1:5d}  {subj_id:<20s}  {split:<10s}  "
            f"{full_shape_str:>20s}  {pixdim_str:>18s}  "
            f"{bbox_str:>20s}  {flag_str}{sharp_str}"
        )

        # Accumulate for CSV
        if args.csv:
            csv_row = {
                "subject_id": subj_id,
                "split": split,
                "image": img_path,
                "shape": full_shape_str,
                "spacing_x": f"{pixdim[0]:.4f}",
                "spacing_y": f"{pixdim[1]:.4f}",
                "spacing_z": f"{pixdim[2]:.4f}",
                "bbox_LR": bbox_ext[0],
                "bbox_AP": bbox_ext[1],
                "bbox_SI": bbox_ext[2],
                "flag": flag_str,
            }
            if ch_sharp:
                for cs in ch_sharp:
                    csv_row[f"{cs['modality']}_hf_ratio"] = f"{cs['hf_ratio']:.4f}"
                    csv_row[f"{cs['modality']}_lap_var"] = f"{cs['lap_var']:.4f}"
            csv_rows.append(csv_row)

        # Optional: label bbox (tumour extent)
        if args.include_labels and "label" in item:
            try:
                lbl_nii = nib.load(item["label"])
                lbl_data = np.asarray(lbl_nii.dataobj)
                _, _, lbl_ext = brain_bbox(lbl_data)
                lbl_bbox_str = "x".join(str(e) for e in lbl_ext)
                print(f"       {'':20s}  {'':10s}  {'':>20s}  {'':>18s}  {lbl_bbox_str:>20s}  tumour_bbox")
            except Exception as exc:
                print(f"       {'':20s}  {'':10s}  {'':>20s}  {'':>18s}  {'---':>20s}  LABEL_ERROR: {exc}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not all_shapes:
        print("No subjects processed.")
        sys.exit(1)

    shapes_arr = np.array(all_shapes)
    pixdims_arr = np.array(all_pixdims)
    bbox_arr = np.array(all_bbox_extents)

    print(f"\nSubjects processed: {len(all_shapes)}")

    # Dimensions
    print(f"\nVolume dimensions (voxels):")
    for ax, name in enumerate(["dim0 (LR)", "dim1 (AP)", "dim2 (SI)"]):
        vals = shapes_arr[:, ax]
        print(f"  {name}: min={vals.min()}, max={vals.max()}, "
              f"mean={vals.mean():.1f}, unique={sorted(set(vals.tolist()))}")

    # Resolution
    print(f"\nVoxel resolution (mm):")
    for ax, name in enumerate(["LR", "AP", "SI"]):
        vals = pixdims_arr[:, ax]
        print(f"  {name}: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Brain bounding box
    print(f"\nBrain bounding box extents (voxels):")
    for ax, (name, tgt) in enumerate(zip(["LR", "AP", "SI"], crop_target)):
        vals = bbox_arr[:, ax]
        n_exceeds = int(np.sum(vals > tgt))
        pct_exceeds = 100.0 * n_exceeds / len(vals)
        print(f"  {name}: min={vals.min()}, max={vals.max()}, "
              f"mean={vals.mean():.1f}, std={vals.std():.1f}, "
              f"median={int(np.median(vals))}, "
              f"exceeds_{tgt}={n_exceeds}/{len(vals)} ({pct_exceeds:.1f}%)")

    # Percentiles for the brain bbox
    print(f"\nBrain bbox percentiles (voxels):")
    for pctl in [1, 5, 25, 50, 75, 95, 99]:
        vals = np.percentile(bbox_arr, pctl, axis=0).astype(int)
        print(f"  P{pctl:02d}: {vals[0]:4d} x {vals[1]:4d} x {vals[2]:4d}")

    # Flags summary
    unique_flags = {}
    for f in flags:
        key = f.split("(")[0]
        unique_flags[key] = unique_flags.get(key, 0) + 1
    if unique_flags:
        print(f"\nFlag counts:")
        for k, v in sorted(unique_flags.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    else:
        print(f"\nNo flags raised — all subjects look standard.")

    # Sharpness summary
    if all_sharpness:
        print(f"\nPer-channel sharpness statistics:")
        for ch_idx, mod in enumerate(MODALITY_NAMES):
            hf_vals = [s[ch_idx]["hf_ratio"] for s in all_sharpness if ch_idx < len(s)]
            lv_vals = [s[ch_idx]["lap_var"] for s in all_sharpness if ch_idx < len(s)]
            if hf_vals:
                hf_arr = np.array(hf_vals)
                lv_arr = np.array(lv_vals)
                n_low = int(np.sum(hf_arr < 0.35))
                print(f"  {mod}:")
                print(f"    HF ratio:  min={hf_arr.min():.4f}  P25={np.percentile(hf_arr,25):.4f}  "
                      f"median={np.median(hf_arr):.4f}  P75={np.percentile(hf_arr,75):.4f}  "
                      f"max={hf_arr.max():.4f}  low(<0.35)={n_low}")
                print(f"    Lap var:   min={lv_arr.min():.2f}  P25={np.percentile(lv_arr,25):.2f}  "
                      f"median={np.median(lv_arr):.2f}  P75={np.percentile(lv_arr,75):.2f}  "
                      f"max={lv_arr.max():.2f}")

    # Write CSV
    if args.csv and csv_rows:
        all_keys = set()
        for r in csv_rows:
            all_keys.update(r.keys())
        base_cols = ["subject_id", "split", "image", "shape",
                     "spacing_x", "spacing_y", "spacing_z",
                     "bbox_LR", "bbox_AP", "bbox_SI"]
        mod_cols = sorted(k for k in all_keys
                          if any(k.startswith(m + "_") for m in MODALITY_NAMES))
        other_cols = ["flag"]
        columns = [c for c in base_cols + mod_cols + other_cols if c in all_keys]

        with open(args.csv, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV written to: {args.csv}")

    print()


if __name__ == "__main__":
    main()
