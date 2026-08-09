"""Split 4-channel stacked NIfTIs into nnU-Net v2 per-modality files.

Two input modes:
  * --in-dir DIR              : stage every *.nii.gz in DIR
  * --from-datalist FILE      : stage entries from datalist JSON under --split
                                (default "validation") and copy matching labels
                                to --out-labels
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def _split_one(src: Path, out_dir: Path, stem: str) -> None:
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[0] == 4:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 4 or data.shape[-1] != 4:
        raise SystemExit(f"expected 4-channel volume, got shape {data.shape} for {src}")
    for c in range(4):
        out = out_dir / f"{stem}_{c:04d}.nii.gz"
        nib.save(nib.Nifti1Image(data[..., c].astype(np.float32), img.affine, img.header), str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path)
    ap.add_argument("--from-datalist", type=Path)
    ap.add_argument("--split", default="validation",
                    help="Which datalist split to stage (default: validation).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination for per-modality NIfTIs.")
    ap.add_argument("--out-labels", type=Path,
                    help="Destination for label copies (datalist mode only).")
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()

    if bool(args.in_dir) == bool(args.from_datalist):
        raise SystemExit("provide exactly one of --in-dir or --from-datalist")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.in_dir is not None:
        sources = sorted(args.in_dir.glob(args.pattern))
        if not sources:
            raise SystemExit(f"no files matching {args.pattern!r} under {args.in_dir}")
        for src in sources:
            _split_one(src, args.out_dir, src.name.split(".nii")[0])
        return

    entries = json.loads(args.from_datalist.read_text()).get(args.split, [])
    if not entries:
        raise SystemExit(f"datalist {args.from_datalist} has no '{args.split}' split")
    if args.out_labels is None:
        raise SystemExit("--out-labels is required in --from-datalist mode")
    args.out_labels.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        sid = entry["subject_id"]
        _split_one(Path(entry["image"]), args.out_dir, sid)
        # nnU-Net predictions are named <sid>.nii.gz; match for eval.
        shutil.copy(entry["label"], args.out_labels / f"{sid}.nii.gz")


if __name__ == "__main__":
    main()
