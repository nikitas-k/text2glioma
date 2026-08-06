"""Lesion-wise Dice + HD95 for BraTS-2024-style evaluation.

Computes per-connected-component metrics on the three BraTS regions
(WT = 1+2+3, TC = 1+3, ET = 3), with the BraTS-2024 <=50-voxel small-lesion
threshold. Ground-truth lesions below the threshold are dropped from the
denominator; predicted false-positive components below the threshold are
also dropped. Matched components are paired by highest IoU; unmatched GT
components score Dice=0 and HD95=penalty; unmatched predicted components
score Dice=0 and HD95=penalty.

Aggregation follows the BraTS-2024 challenge protocol:
    per-case metric = mean over that case's regions and lesions
    cohort metric   = mean over cases

Outputs a per-case CSV and a summary CSV per dataset. Runs on CPU; expects
predictions already saved as int-label NIfTIs.

Usage:
    python scripts/segmentation/eval_lesion_wise.py \
        --pred-dir /path/to/predictions \
        --gt-dir   /path/to/labelsTs \
        --out-csv  eval/nnunet_synth500_internal.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial.distance import cdist

REGIONS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}
SMALL_LESION_VOXELS = 50
HD95_PENALTY_MM = 374.0  # BraTS-2024 default penalty for false positives / misses


def _region_mask(seg: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(seg, labels)


def _connected_components(binary: np.ndarray, min_voxels: int) -> list[np.ndarray]:
    lab, n = cc_label(binary)
    out = []
    for i in range(1, n + 1):
        mask = lab == i
        if int(mask.sum()) >= min_voxels:
            out.append(mask)
    return out


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    denom = float(a.sum() + b.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


def _hd95(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Two-sided 95th percentile Hausdorff distance in mm."""
    if not a.any() or not b.any():
        return HD95_PENALTY_MM
    ap = np.argwhere(a).astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    bp = np.argwhere(b).astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    # Cap point clouds so cdist stays tractable on huge lesions.
    def _cap(pts: np.ndarray, cap: int = 20000) -> np.ndarray:
        if pts.shape[0] > cap:
            idx = np.random.default_rng(0).choice(pts.shape[0], cap, replace=False)
            return pts[idx]
        return pts
    ap, bp = _cap(ap), _cap(bp)
    d_ab = cdist(ap, bp).min(axis=1)
    d_ba = cdist(bp, ap).min(axis=1)
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def _match_lesions(gt: list[np.ndarray], pr: list[np.ndarray]) -> list[tuple[int, int]]:
    """Greedy IoU matching between GT and predicted components."""
    pairs: list[tuple[int, int]] = []
    if not gt or not pr:
        return pairs
    iou = np.zeros((len(gt), len(pr)), dtype=np.float64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            inter = float(np.logical_and(g, p).sum())
            union = float(np.logical_or(g, p).sum())
            iou[i, j] = inter / union if union > 0 else 0.0
    used_g, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, pj in order:
        gi, pj = int(gi), int(pj)
        if iou[gi, pj] <= 0.0:
            break
        if gi in used_g or pj in used_p:
            continue
        pairs.append((gi, pj))
        used_g.add(gi); used_p.add(pj)
    return pairs


def evaluate_case(gt_seg: np.ndarray, pr_seg: np.ndarray,
                  spacing: tuple[float, float, float]) -> dict[str, float]:
    row: dict[str, float] = {}
    for region, labels in REGIONS.items():
        gt_cc = _connected_components(_region_mask(gt_seg, labels), SMALL_LESION_VOXELS)
        pr_cc = _connected_components(_region_mask(pr_seg, labels), SMALL_LESION_VOXELS)
        if not gt_cc and not pr_cc:
            row[f"{region}_Dice"] = 1.0
            row[f"{region}_HD95"] = 0.0
            row[f"{region}_nLesGT"] = 0
            row[f"{region}_nLesPR"] = 0
            continue
        pairs = _match_lesions(gt_cc, pr_cc)
        dices, hds = [], []
        for gi, pj in pairs:
            dices.append(_dice(gt_cc[gi], pr_cc[pj]))
            hds.append(_hd95(gt_cc[gi], pr_cc[pj], spacing))
        matched_g = {gi for gi, _ in pairs}
        matched_p = {pj for _, pj in pairs}
        # Missed GT lesions
        for gi in range(len(gt_cc)):
            if gi not in matched_g:
                dices.append(0.0); hds.append(HD95_PENALTY_MM)
        # False-positive predicted lesions
        for pj in range(len(pr_cc)):
            if pj not in matched_p:
                dices.append(0.0); hds.append(HD95_PENALTY_MM)
        row[f"{region}_Dice"] = float(np.mean(dices)) if dices else 0.0
        row[f"{region}_HD95"] = float(np.mean(hds)) if hds else HD95_PENALTY_MM
        row[f"{region}_nLesGT"] = len(gt_cc)
        row[f"{region}_nLesPR"] = len(pr_cc)
    return row


def _load(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    data = np.asarray(img.dataobj).astype(np.int16)
    data[data == 4] = 3
    return data, zooms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--gt-dir",   required=True, type=Path)
    ap.add_argument("--out-csv",  required=True, type=Path)
    ap.add_argument("--case-glob", default="*.nii.gz")
    args = ap.parse_args()

    rows: list[dict[str, float | str]] = []
    for pred_path in sorted(args.pred_dir.glob(args.case_glob)):
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.is_file():
            print(f"[skip] no GT for {pred_path.name}")
            continue
        pr_seg, spacing = _load(pred_path)
        gt_seg, _       = _load(gt_path)
        m = evaluate_case(gt_seg, pr_seg, spacing)
        m["case"] = pred_path.stem
        rows.append(m)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case"] + [k for k in rows[0] if k != "case"] if rows else ["case"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Cohort summary
    if rows:
        summary_path = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std", "n"])
            for k in cols[1:]:
                vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]
                if vals:
                    w.writerow([k, float(np.mean(vals)), float(np.std(vals)), len(vals)])
        print(f"[wrote] {args.out_csv} and {summary_path}")


if __name__ == "__main__":
    main()
