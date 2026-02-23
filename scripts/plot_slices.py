#!/usr/bin/env python3
"""Plot axial slices for every Nth subject, all 4 modalities side-by-side.

For each selected subject, shows every ~15th axial slice (Z-direction)
as columns, with modalities (T1, T1CE, T2, FLAIR) as rows.

Usage::

    # Plot every 20th subject, save to plots/ directory:
    python scripts/plot_slices.py --datalist datalist_N1511.json

    # Customise stride and output:
    python scripts/plot_slices.py --datalist datalist_N1511.json \
        --subject-stride 50 --slice-stride 10 --outdir qc_slices

    # Specific split only:
    python scripts/plot_slices.py --datalist datalist_N1511.json --split training
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]


def extract_nnunet_id(image_path: str) -> str:
    """Extract nnUNetv2-XXXXX identifier from the file path."""
    stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")
    # e.g. "nnUNetv2-01064" → keep as-is
    return stem


def plot_subject(image_path: str, subject_id: str, slice_stride: int,
                 outdir: Path) -> Path:
    """Plot axial slices for one subject and save to outdir.

    Layout: rows = modalities (T1, T1CE, T2, FLAIR)
            cols = axial slices every `slice_stride` voxels in Z
    """
    nii = nib.load(image_path)
    data = np.asarray(nii.dataobj, dtype=np.float32)  # (X, Y, Z, C) or (X, Y, Z)

    if data.ndim == 3:
        data = data[..., np.newaxis]

    n_x, n_y, n_z, n_ch = data.shape
    n_mod = min(n_ch, 4)

    # Select slice indices (every slice_stride, skip first/last few empty slices)
    margin = max(5, n_z // 20)  # skip ~5% from edges
    slice_indices = list(range(margin, n_z - margin, slice_stride))
    if not slice_indices:
        slice_indices = list(range(0, n_z, slice_stride))
    n_slices = len(slice_indices)

    nnunet_id = extract_nnunet_id(image_path)

    fig, axes = plt.subplots(
        n_mod, n_slices,
        figsize=(2.2 * n_slices, 2.5 * n_mod),
        squeeze=False,
    )
    fig.suptitle(f"{nnunet_id}  ({subject_id})", fontsize=14, fontweight="bold",
                 y=1.01)

    for row, ch in enumerate(range(n_mod)):
        mod_name = MODALITY_NAMES[ch] if ch < len(MODALITY_NAMES) else f"ch{ch}"
        vol = data[:, :, :, ch]

        # Compute display range from non-zero voxels (robust windowing)
        nz_vals = vol[vol > 0]
        if len(nz_vals) > 0:
            vmin = 0.0
            vmax = float(np.percentile(nz_vals, 99.5))
        else:
            vmin, vmax = 0.0, 1.0

        for col, z_idx in enumerate(slice_indices):
            ax = axes[row, col]
            # Axial slice: take (X, Y) at Z=z_idx, transpose for display
            slc = vol[:, :, z_idx].T  # (Y, X) for imshow
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax,
                      origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Column title (slice index) on top row only
            if row == 0:
                ax.set_title(f"z={z_idx}", fontsize=8)

            # Row label (modality) on leftmost column only
            if col == 0:
                ax.set_ylabel(mod_name, fontsize=11, fontweight="bold",
                              rotation=0, labelpad=35, va="center")

    plt.tight_layout()
    out_path = outdir / f"{nnunet_id}_{subject_id}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot axial slices per modality for sampled subjects."
    )
    parser.add_argument("--datalist", type=str, required=True,
                        help="Path to datalist JSON.")
    parser.add_argument("--split", type=str, default=None,
                        help="Which split (training/validation). Default: all.")
    parser.add_argument("--subject-stride", type=int, default=20,
                        help="Plot every Nth subject (default: 20).")
    parser.add_argument("--slice-stride", type=int, default=15,
                        help="Show every Nth axial slice (default: 15).")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="Output directory for PNGs (default: plots/).")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Maximum number of subjects to plot.")
    args = parser.parse_args()

    with open(args.datalist) as f:
        datalist = json.load(f)

    if args.split:
        splits = [args.split]
    else:
        splits = [k for k in ("training", "validation", "testing")
                  if k in datalist and isinstance(datalist.get(k), list)
                  and len(datalist[k]) > 0]

    entries = []
    for split in splits:
        for item in datalist.get(split, []):
            entries.append(item)

    # Sample every Nth subject
    selected = entries[:: args.subject_stride]
    if args.max_subjects:
        selected = selected[: args.max_subjects]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {len(selected)} subjects (every {args.subject_stride}th "
          f"of {len(entries)}) → {outdir}/")
    print(f"Slice stride: every {args.slice_stride} voxels in Z\n")

    for i, item in enumerate(selected):
        img_path = item["image"]
        subj_id = item.get("subject_id", Path(img_path).stem)
        print(f"  [{i+1}/{len(selected)}] {subj_id} ...", end=" ", flush=True)
        try:
            out = plot_subject(img_path, subj_id, args.slice_stride, outdir)
            print(f"→ {out}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\nDone. {len(selected)} plots saved to {outdir}/")


if __name__ == "__main__":
    main()
