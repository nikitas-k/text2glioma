"""Build longitudinal pairs from a BraTS-GLI 2025 dataset directory.

Naming convention (from challenge spec):
  BraTS-GLI-SSSSS-TTT
    SSSSS = 5-digit subject id
    TTT   = 3-digit timepoint/treatment code:
              000 = pre-op,  first baseline
              001 = pre-op,  second baseline
              100 = post-op, first follow-up
              101 = post-op, second follow-up

For each subject with ≥2 timepoints, emit chronologically ordered pairs (A, B).
For each pair, classify the trajectory from segmentation deltas, and stamp the
pre-/post-op treatment status from the codes.

Pair-enumeration modes (``--pair_mode``):
  consecutive       Only adjacent timepoints in chronological order.
                    A subject with K timepoints contributes K-1 pairs.
                    Default. Recommended for trajectory modelling because each
                    pair spans a single inter-scan interval, so the volume
                    delta corresponds to one treatment / observation window.
  baseline_anchored Pair the earliest timepoint with every later one
                    (K-1 pairs per subject, always sharing tps[0] as A).
                    Useful for "distance from baseline" experiments.
  all               Enumerate every ordered pair (i < j).  K*(K-1)/2 pairs.
                    Matches the previous behaviour of this script.  Inflates
                    pair counts on subjects with many timepoints (e.g. one
                    10-timepoint subject → 45 pairs) and conflates multiple
                    intervals into a single delta — use only when explicitly
                    needed.

Usage (run on Gadi where the dataset lives):

  python scripts/build_brats_gli_pairs.py \
      --dataset_root /g/data/hl36/.../BraTS2025-GLI \
      --image_pattern "{sid_tp}/{sid_tp}-{mod}.nii.gz" \
      --label_pattern "{sid_tp}/{sid_tp}-seg.nii.gz" \
      --pair_mode consecutive \
      --out datalist_brats_gli_2025_pairs.json

Then split with `split_validation_80_20.py` analogue, or merge with N1510.

The output schema (per pair):
  {
    "subject_id":          "BraTS-GLI-00001",
    "timepoint_a":         "000",
    "timepoint_b":         "100",
    "image_a":             "/abs/path/BraTS-GLI-00001-000/...-t1.nii.gz",  ...
    "label_a":             "/abs/path/BraTS-GLI-00001-000/...-seg.nii.gz",
    "image_b":             "...",
    "label_b":             "...",
    "treatment_status_a":  "pre"  | "post",
    "treatment_status_b":  "pre"  | "post",
    "trajectory":          "response" | "stable" | "progression" | "novel",
    "vol_a":               int,
    "vol_b":               int,
    "rel_change":          float
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text2glioma.preprocessing.inpainting_masks import (  # noqa: E402
    PROGRESSION_DV_THRESHOLD,
    RESPONSE_DV_THRESHOLD,
)


SID_TP_RE = re.compile(r"BraTS-GLI-(\d{5})-(\d{3})")
MODALITIES = ("t1n", "t1c", "t2w", "t2f")  # BraTS-2025 channel names


def _treatment_status(tp: str) -> str:
    """Map timepoint code to treatment status: '0xx' = pre, '1xx' = post."""
    return "pre" if tp[0] == "0" else "post"


def _classify_from_volumes(va: int, vb: int) -> tuple[str, float]:
    if va == 0 and vb == 0:
        return "stable", 0.0
    if va == 0 and vb > 0:
        return "novel", float("inf")
    rel = (vb - va) / max(va, 1)
    if rel <= RESPONSE_DV_THRESHOLD:
        return "response", rel
    if rel >= PROGRESSION_DV_THRESHOLD:
        return "progression", rel
    return "stable", rel


def _label_voxel_count(label_path: Path) -> int:
    if nib is None:
        raise RuntimeError("nibabel is required to compute trajectory classes")
    return int((np.asarray(nib.load(label_path).dataobj) > 0).sum())


def _resolve_paths(
    root: Path,
    sid_tp: str,
    image_pattern: str,
    label_pattern: str,
) -> tuple[list[Path], Path]:
    images = [root / image_pattern.format(sid_tp=sid_tp, mod=m) for m in MODALITIES]
    label = root / label_pattern.format(sid_tp=sid_tp)
    return images, label


def _discover_subjects(root: Path) -> dict[str, list[str]]:
    """Walk dataset root, return {subject_id: [sorted timepoint codes]}."""
    by_subject: dict[str, list[str]] = defaultdict(list)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = SID_TP_RE.match(child.name)
        if not m:
            continue
        sid, tp = m.group(1), m.group(2)
        by_subject[f"BraTS-GLI-{sid}"].append(tp)
    for sid in by_subject:
        by_subject[sid].sort()  # 000, 001, 100, 101 are lexicographically chronological
    return dict(by_subject)


def _enumerate_pairs(tps: list[str], mode: str) -> list[tuple[str, str]]:
    """Return ordered (tp_a, tp_b) pairs for one subject under the given mode."""
    if len(tps) < 2:
        return []
    if mode == "consecutive":
        return list(zip(tps[:-1], tps[1:]))
    if mode == "baseline_anchored":
        return [(tps[0], tp) for tp in tps[1:]]
    if mode == "all":
        return [(tps[i], tps[j]) for i in range(len(tps) - 1) for j in range(i + 1, len(tps))]
    raise ValueError(f"Unknown pair_mode={mode!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True,
                   help="Root containing per-subject-per-timepoint folders "
                        "named BraTS-GLI-SSSSS-TTT.")
    p.add_argument("--image_pattern", type=str,
                   default="BraTS-GLI-{sid_tp}/BraTS-GLI-{sid_tp}-{mod}.nii.gz",
                   help="Path template under dataset_root for each modality. "
                        "Substitutions: {sid_tp} (e.g. '00001-000'), {mod}.")
    p.add_argument("--label_pattern", type=str,
                   default="BraTS-GLI-{sid_tp}/BraTS-GLI-{sid_tp}-seg.nii.gz")
    p.add_argument("--out", type=Path,
                   default=ROOT / "datalist_brats_gli_2025_pairs.json")
    p.add_argument("--pair_mode", type=str, default="consecutive",
                   choices=("consecutive", "baseline_anchored", "all"),
                   help="How to enumerate (A, B) pairs per subject. "
                        "See module docstring for semantics. Default: consecutive.")
    p.add_argument("--no_traj", action="store_true",
                   help="Skip trajectory classification (no nibabel reads). "
                        "Useful for a fast structure-only audit.")
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Limit for smoke testing.")
    args = p.parse_args()

    if not args.dataset_root.is_dir():
        sys.exit(f"dataset_root not a directory: {args.dataset_root}")

    by_subject = _discover_subjects(args.dataset_root)
    print(f"Found {len(by_subject)} subjects under {args.dataset_root}")

    if args.max_subjects:
        keep = list(by_subject)[:args.max_subjects]
        by_subject = {k: by_subject[k] for k in keep}
        print(f"  limited to {len(by_subject)} for smoke test")

    pairs: list[dict] = []
    timepoint_counts = Counter()
    skipped_no_pair = 0
    skipped_missing_file = 0

    for sid, tps in by_subject.items():
        timepoint_counts[len(tps)] += 1
        if len(tps) < 2:
            skipped_no_pair += 1
            continue
        for tp_a, tp_b in _enumerate_pairs(tps, args.pair_mode):
            sid_tp_a = f"{sid.removeprefix('BraTS-GLI-')}-{tp_a}"
            sid_tp_b = f"{sid.removeprefix('BraTS-GLI-')}-{tp_b}"
            imgs_a, lbl_a = _resolve_paths(args.dataset_root, sid_tp_a,
                                           args.image_pattern, args.label_pattern)
            imgs_b, lbl_b = _resolve_paths(args.dataset_root, sid_tp_b,
                                           args.image_pattern, args.label_pattern)
            if any(not p.exists() for p in (*imgs_a, *imgs_b, lbl_a, lbl_b)):
                skipped_missing_file += 1
                continue
            if args.no_traj:
                traj, rel, va, vb = "unknown", 0.0, 0, 0
            else:
                va = _label_voxel_count(lbl_a)
                vb = _label_voxel_count(lbl_b)
                traj, rel = _classify_from_volumes(va, vb)
            pairs.append({
                "subject_id":         sid,
                "timepoint_a":        tp_a,
                "timepoint_b":        tp_b,
                "image_a":            [str(p) for p in imgs_a],
                "label_a":            str(lbl_a),
                "image_b":            [str(p) for p in imgs_b],
                "label_b":            str(lbl_b),
                "treatment_status_a": _treatment_status(tp_a),
                "treatment_status_b": _treatment_status(tp_b),
                "trajectory":         traj,
                "vol_a":              va,
                "vol_b":              vb,
                "rel_change":         rel if rel != float("inf") else None,
            })

    treatment_direction_counts = dict(Counter(
        f"{p['treatment_status_a']}->{p['treatment_status_b']}" for p in pairs
    ))
    timepoint_pair_counts = dict(Counter(
        f"{p['timepoint_a']}->{p['timepoint_b']}" for p in pairs
    ).most_common(20))

    out = {
        "pairs":  pairs,
        "n_pairs": len(pairs),
        "n_subjects_total":      len(by_subject),
        "n_subjects_with_pairs": len(by_subject) - skipped_no_pair,
        "skipped_missing_file":  skipped_missing_file,
        "pair_mode":             args.pair_mode,
        "trajectory_counts":     dict(Counter(p["trajectory"] for p in pairs)),
        "treatment_direction_counts":         treatment_direction_counts,
        "timepoint_pair_counts_top20":        timepoint_pair_counts,
        "timepoints_per_subject_histogram":   dict(timepoint_counts),
        "_thresholds": {
            "response_dv":    RESPONSE_DV_THRESHOLD,
            "progression_dv": PROGRESSION_DV_THRESHOLD,
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(pairs)} pairs from {out['n_subjects_with_pairs']} subjects (mode={args.pair_mode})")
    print(f"  trajectory:          {out['trajectory_counts']}")
    print(f"  treatment direction: {out['treatment_direction_counts']}")
    print(f"  timepoints/subject:  {out['timepoints_per_subject_histogram']}")
    print(f"  skipped (singleton subject): {skipped_no_pair}")
    print(f"  skipped (missing file):      {skipped_missing_file}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
