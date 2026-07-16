"""Audit the enhancing-label fraction across the training-cohort masks.

For each label file in the datalist, compute:
    - voxel count for label 2 (oedema) and label 3 (enhancing)
    - enhancing fraction f3 = n3 / (n2 + n3)

Writes a CSV with one row per subject. Also prints a summary.

Run on Gadi (where all mask files are accessible):

    python scripts/dataset_release/audit_mask_labels.py \
        --datalist datalist_N1510.json \
        --split training \
        --out mask_label_audit.csv

Downstream: pass the CSV to `filter_manifest_by_mask_quality.py` to drop
prompt-mask pairs where a "strongly enhancing" prompt is paired with a
mask that has too little enhancing tissue.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _iter_rows(datalist_path: Path, split: str):
    dl = json.loads(datalist_path.read_text())
    rows = dl.get(split) or dl.get({"training": "train", "train": "training"}.get(split, split), [])
    for r in rows:
        if isinstance(r, dict):
            yield r
        else:
            yield {"label": r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=Path, required=True)
    ap.add_argument("--split", default="training")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n_ok = 0
    n_miss = 0
    fracs: list[float] = []

    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subj", "mask_path", "n_label2", "n_label3", "n_tumour",
                    "enhancing_fraction", "has_enhancing", "has_meaningful_enhancing"])
        for row in _iter_rows(args.datalist, args.split):
            mp = Path(row["label"])
            subj = row.get("subj") or row.get("subject") or mp.stem.replace(".nii", "")
            if not mp.exists():
                n_miss += 1
                w.writerow([subj, str(mp), "", "", "", "", "", ""])
                continue
            m = nib.load(str(mp)).get_fdata()
            n2 = int((m == 2).sum())
            n3 = int((m == 3).sum())
            nt = n2 + n3
            f3 = float(n3) / nt if nt > 0 else 0.0
            has_enh = int(n3 > 0)
            # "Meaningful" = enough voxels + fraction not vanishingly small.
            has_meaningful = int(n3 >= 100 and f3 >= 0.05)
            w.writerow([subj, str(mp), n2, n3, nt, f"{f3:.4f}", has_enh, has_meaningful])
            fracs.append(f3)
            n_ok += 1

    a = np.array(fracs)
    print(f"processed: {n_ok}   missing files: {n_miss}", file=sys.stderr)
    if len(a):
        print(f"label-3 fraction over {len(a)} masks:", file=sys.stderr)
        print(f"  mean={a.mean():.3f}  median={np.median(a):.3f}", file=sys.stderr)
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            print(f"  p{int(q*100):02d}={np.quantile(a, q):.3f}", file=sys.stderr)
        print(f"  n_no_label3 (n3==0):        {int((a == 0).sum())}  ({(a == 0).mean()*100:.1f}%)", file=sys.stderr)
        print(f"  n_fraction<0.05:            {int((a < 0.05).sum())}  ({(a < 0.05).mean()*100:.1f}%)", file=sys.stderr)
        print(f"  n_fraction<0.10:            {int((a < 0.10).sum())}  ({(a < 0.10).mean()*100:.1f}%)", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
