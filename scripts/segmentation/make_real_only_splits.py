"""Write nnU-Net v2 splits_final.json with a real-only validation fold.

Model selection during nnU-Net training is driven by the "val" fold Dice. If the val
fold contains synthetic cases, the best-checkpoint criterion is biased toward the
synth distribution -- which is not what we want when reporting downstream performance
on a real held-out test set. This script enforces:

  * val fold  = a fixed random subset of REAL_* cases (identical across doses)
  * train fold = the remaining REAL_* cases + every SYNTH_* case present in the dataset

The same real subjects are held out for every Dataset510..514, giving a paired
comparison at model-selection time.

Usage (after nnUNetv2_plan_and_preprocess for each dataset):

    python scripts/segmentation/make_real_only_splits.py \
        --nnUNet-preprocessed "$nnUNet_preprocessed" \
        --datasets 510 511 512 513 514 \
        --val-frac 0.2 --seed 42

nnU-Net expects a 5-entry list even when only fold 0 is trained; folds 1..4 are
written with the same split so any accidental multi-fold run stays well-defined.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _find_dataset_dir(root: Path, did: int) -> Path:
    matches = sorted(root.glob(f"Dataset{did}_*"))
    if not matches:
        raise FileNotFoundError(f"No Dataset{did}_* under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous Dataset{did}_* under {root}: {matches}")
    return matches[0]


def _case_ids(dataset_dir: Path) -> list[str]:
    # Preferred: preprocessed data identifier folders contain <case>.b2nd / .npz / .npy.
    # Fall back to nnUNet_raw/../imagesTr/<case>_0000.nii.gz if needed.
    candidates: set[str] = set()
    for cfg_dir in dataset_dir.glob("nnUNet*"):
        if not cfg_dir.is_dir():
            continue
        for f in cfg_dir.iterdir():
            stem = f.name
            for suf in (".b2nd", ".npz", ".npy", ".pkl"):
                if stem.endswith(suf):
                    case = stem[: -len(suf)]
                    # Skip sibling segmentation files (e.g. CASE_seg.b2nd) that share the case stem.
                    if case.endswith("_seg"):
                        break
                    candidates.add(case)
                    break
    if candidates:
        return sorted(candidates)
    # Fallback via dataset.json numTraining + scanning raw imagesTr sibling.
    raw_sibling = dataset_dir.parent.parent / "nnUNet_raw" / dataset_dir.name / "imagesTr"
    if raw_sibling.is_dir():
        for f in raw_sibling.glob("*_0000.nii.gz"):
            candidates.add(f.name[: -len("_0000.nii.gz")])
    return sorted(candidates)


def _build_splits(cases: list[str], val_real: set[str]) -> list[dict]:
    train = sorted(c for c in cases if c not in val_real)
    val = sorted(c for c in cases if c in val_real)
    fold = {"train": train, "val": val}
    return [fold for _ in range(5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnUNet-preprocessed", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", type=int, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite splits_final.json if it already exists.")
    args = ap.parse_args()

    # Discover the real case pool from the smallest dataset (real-only baseline).
    baseline_dir = _find_dataset_dir(args.nnUNet_preprocessed, min(args.datasets))
    baseline_cases = _case_ids(baseline_dir)
    real_cases = sorted(c for c in baseline_cases if c.startswith("REAL_"))
    if not real_cases:
        raise RuntimeError(f"No REAL_* cases found in {baseline_dir}")

    rng = random.Random(args.seed)
    shuffled = real_cases[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(real_cases) * args.val_frac)))
    val_real = set(shuffled[:n_val])
    print(f"[splits] real total={len(real_cases)}  val={n_val}  train={len(real_cases) - n_val}")

    for did in args.datasets:
        dset_dir = _find_dataset_dir(args.nnUNet_preprocessed, did)
        out = dset_dir / "splits_final.json"
        if out.exists() and not args.force:
            print(f"[skip] {out} already exists (use --force to overwrite)")
            continue
        cases = _case_ids(dset_dir)
        if not cases:
            raise RuntimeError(f"No case IDs discovered under {dset_dir}")
        missing_val = val_real - set(cases)
        if missing_val:
            raise RuntimeError(
                f"{len(missing_val)} val real cases missing from {dset_dir}; "
                "datasets must share the same REAL_* indexing."
            )
        splits = _build_splits(cases, val_real)
        out.write_text(json.dumps(splits, indent=2))
        n_synth = sum(1 for c in cases if c.startswith("SYNTH_"))
        print(f"[wrote] {out}  train={len(splits[0]['train'])} (real+{n_synth} synth)  val={len(splits[0]['val'])} real")


if __name__ == "__main__":
    main()
