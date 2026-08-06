"""Split 4-channel stacked NIfTIs into nnU-Net v2 per-modality files."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.in_dir.glob(args.pattern)):
        img = nib.load(str(src))
        data = np.asarray(img.dataobj)
        if data.ndim == 4 and data.shape[0] == 4:
            data = np.moveaxis(data, 0, -1)
        if data.ndim != 4 or data.shape[-1] != 4:
            raise SystemExit(f"expected 4-channel volume, got shape {data.shape} for {src}")
        stem = src.name.split(".nii")[0]
        for c in range(4):
            out = args.out_dir / f"{stem}_{c:04d}.nii.gz"
            nib.save(nib.Nifti1Image(data[..., c].astype(np.float32), img.affine, img.header), str(out))


if __name__ == "__main__":
    main()
