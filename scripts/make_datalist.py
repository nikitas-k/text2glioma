#!/usr/bin/env python
"""Generate a MONAI-style JSON datalist from paired image/label NIfTI files.

Example — create datalist for Task03_BrainTumourDx::

    python scripts/make_datalist.py \\
        --images '/g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr/nnUNetv2-0*_full.nii' \\
        --labels '/g/data/hl36/mhf/monai/Task03_BrainTumourDx/labelsTr/nnUNetv2-0*.nii.gz' \\
        --val_frac 0.2 \\
        --seed 42 \\
        -o datalist_task03.json

The output JSON has the format expected by ``train_stage1_ddp --datalist``::

    {
      "training": [{"image": "...", "label": "..."}, ...],
      "validation": [{"image": "...", "label": "..."}, ...]
    }

Image–label matching is done by deriving a *subject ID* from each
filename: NIfTI extensions are stripped (``.nii.gz``, ``.nii``), then
image files have ``_full`` removed.  For example::

    nnUNetv2-00001_full.nii   →  nnUNetv2-00001
    nnUNetv2-00001.nii.gz     →  nnUNetv2-00001

Override with ``--image_suffix`` / ``--label_suffix`` if your naming
convention differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _strip_nifti_ext(name: str) -> str:
    """Remove .nii.gz or .nii extension from a filename."""
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def subject_id(path: Path, suffix: str = "") -> str:
    """Derive a subject ID from a NIfTI path.

    1. Strip ``.nii.gz`` / ``.nii``.
    2. If *suffix* is non-empty, strip that trailing string as well.

    Examples (suffix="_full")::

        nnUNetv2-00001_full.nii  →  nnUNetv2-00001
        nnUNetv2-00001.nii.gz   →  nnUNetv2-00001
    """
    stem = _strip_nifti_ext(path.name)
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def main():
    p = argparse.ArgumentParser(
        description="Generate MONAI datalist JSON from image/label NIfTI files.",
    )
    p.add_argument("--images", type=str, required=True,
                   help="Glob pattern for image files (quote it!).")
    p.add_argument("--labels", type=str, default=None,
                   help="Glob pattern for label files (optional; omit for stage-1 only).")
    p.add_argument("--image_suffix", type=str, default="_full",
                   help="Suffix to strip from image stems before matching "
                        "(default: '_full').  Set to '' to disable.")
    p.add_argument("--label_suffix", type=str, default="",
                   help="Suffix to strip from label stems before matching "
                        "(default: none).")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of cases for validation (default: 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    p.add_argument("-o", "--output", type=str, default="datalist.json",
                   help="Output JSON path.")
    args = p.parse_args()

    # ── Collect images ──────────────────────────────────────────────
    from glob import glob
    image_paths = sorted(glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {args.images}")
    print(f"Found {len(image_paths)} images.")

    # ── Collect labels (if provided) & match ────────────────────────
    if args.labels:
        label_paths = sorted(glob(args.labels))
        if not label_paths:
            raise FileNotFoundError(f"No labels found matching: {args.labels}")
        print(f"Found {len(label_paths)} labels.")

        # Build ID → path maps
        img_by_id = {}
        for p_img in image_paths:
            sid = subject_id(Path(p_img), args.image_suffix)
            img_by_id[sid] = p_img

        lbl_by_id = {}
        for p_lbl in label_paths:
            sid = subject_id(Path(p_lbl), args.label_suffix)
            lbl_by_id[sid] = p_lbl

        # Show sample IDs for debugging
        sample_img = list(img_by_id.items())[:3]
        sample_lbl = list(lbl_by_id.items())[:3]
        print(f"  Sample image IDs: {sample_img}")
        print(f"  Sample label IDs: {sample_lbl}")
        print(f"  Unique image IDs: {len(img_by_id)}, unique label IDs: {len(lbl_by_id)}")

        # Match
        common_ids = sorted(set(img_by_id) & set(lbl_by_id))
        if not common_ids:
            raise ValueError("No matching case IDs between images and labels.\n"
                             f"  Image IDs sample: {list(img_by_id)[:5]}\n"
                             f"  Label IDs sample: {list(lbl_by_id)[:5]}")
        unmatched_img = set(img_by_id) - set(lbl_by_id)
        unmatched_lbl = set(lbl_by_id) - set(img_by_id)
        if unmatched_img:
            print(f"  Warning: {len(unmatched_img)} images without matching labels (skipped).")
        if unmatched_lbl:
            print(f"  Warning: {len(unmatched_lbl)} labels without matching images (skipped).")

        pairs = [{"image": img_by_id[cid], "label": lbl_by_id[cid]}
                 for cid in common_ids]
        print(f"Matched {len(pairs)} image-label pairs.")
    else:
        pairs = [{"image": p_img} for p_img in image_paths]
        print(f"No labels provided — image-only datalist ({len(pairs)} entries).")

    # ── Train/val split ─────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_idx = set(indices[:n_val].tolist())

    training = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    validation = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    datalist = {"training": training, "validation": validation}

    # ── Write ───────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(datalist, f, indent=2)
    print(f"Wrote {len(training)} training + {len(validation)} validation "
          f"entries to {out}")


if __name__ == "__main__":
    main()
