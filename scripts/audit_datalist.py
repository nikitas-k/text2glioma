#!/usr/bin/env python3
"""Audit spatial metadata for every image in a datalist JSON.

Reports per-subject:
  - Voxel resolution (mm)
  - Volume dimensions (voxels)
  - Slice thickness (if anisotropic)
  - Brain bounding box after foreground crop (voxels)
  - Flags outliers (resolution != 1mm iso, unusual dimensions, tight/large brain bbox)

Usage (on Gadi or wherever the NIfTIs live):
    python scripts/audit_datalist.py --datalist datalist_N1511.json
    python scripts/audit_datalist.py --datalist datalist_N1511.json --split training
    python scripts/audit_datalist.py --datalist datalist_N1511.json --max-subjects 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np


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

        flag_str = ", ".join(subj_flags) if subj_flags else "ok"
        if subj_flags:
            flags.extend(subj_flags)

        pixdim_str = f"{pixdim[0]:.3f}x{pixdim[1]:.3f}x{pixdim[2]:.3f}"
        bbox_str = "x".join(str(e) for e in bbox_ext)

        print(
            f"{i+1:5d}  {subj_id:<20s}  {split:<10s}  "
            f"{full_shape_str:>20s}  {pixdim_str:>18s}  "
            f"{bbox_str:>20s}  {flag_str}"
        )

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

    print()


if __name__ == "__main__":
    main()
