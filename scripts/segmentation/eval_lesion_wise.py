"""Lesion-wise Dice + HD95 for BraTS-2024-style evaluation.

Computes per-connected-component metrics on the BraTS regions
(WT = 1+2+3, TC = 1+3, ET = 3, ED = 2), with the BraTS-2024 <=50-voxel small-lesion
threshold. Ground-truth lesions below the threshold are dropped from the
denominator; predicted false-positive components below the threshold are
also dropped. Matched components are paired by highest IoU; unmatched GT
components score Dice=0 and HD95=penalty; unmatched predicted components
score Dice=0 and HD95=penalty.

Aggregation follows the BraTS-2024 challenge protocol:
    per-case metric = mean over that case's regions and lesions
    cohort metric   = mean over cases

Outputs a per-case CSV and a summary CSV per dataset. Runs on CPU, one
process per case; expects predictions already saved as int-label NIfTIs.

Usage:
    python scripts/segmentation/eval_lesion_wise.py \
        --pred-dir /path/to/predictions \
        --gt-dir   /path/to/labelsTs \
        --out-csv  eval/nnunet_synth500_internal.csv \
        --workers 12
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects, label as cc_label
from scipy.spatial.distance import cdist

REGIONS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
    "ED": (2,),
}
SMALL_LESION_VOXELS = 50
HD95_PENALTY_MM = 374.0
_HD95_POINT_CAP = 20000  # match legacy point cap for reproducibility with existing CSVs.


def _region_mask(seg: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(seg, labels)


def _connected_components(binary: np.ndarray, min_voxels: int) -> list[dict]:
    """Return one dict per surviving component: {slices, mask (bbox-cropped bool)}."""
    lab, n = cc_label(binary)
    if n == 0:
        return []
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    slices = find_objects(lab)
    out: list[dict] = []
    for i in range(1, n + 1):
        if int(counts[i]) < min_voxels:
            continue
        sl = slices[i - 1]
        if sl is None:
            continue
        sub = lab[sl] == i
        out.append({"slices": sl, "mask": sub, "size": int(counts[i])})
    return out


def _pair_iou(a: dict, b: dict, ndim: int) -> tuple[float, tuple[slice, ...] | None]:
    """IoU between two components using bounding-box intersection only."""
    inter_sl: list[slice] = []
    for d in range(ndim):
        sa, sb = a["slices"][d], b["slices"][d]
        lo = max(sa.start, sb.start)
        hi = min(sa.stop,  sb.stop)
        if lo >= hi:
            return 0.0, None
        inter_sl.append(slice(lo, hi))
    tup = tuple(inter_sl)
    # Extract intersecting sub-volumes from each cropped mask.
    a_slices = tuple(slice(tup[d].start - a["slices"][d].start,
                           tup[d].stop  - a["slices"][d].start) for d in range(ndim))
    b_slices = tuple(slice(tup[d].start - b["slices"][d].start,
                           tup[d].stop  - b["slices"][d].start) for d in range(ndim))
    inter = int(np.logical_and(a["mask"][a_slices], b["mask"][b_slices]).sum())
    if inter == 0:
        return 0.0, tup
    union = a["size"] + b["size"] - inter
    return inter / union, tup


def _dice_from_components(a: dict, b: dict, ndim: int) -> float:
    inter_sl: list[slice] = []
    for d in range(ndim):
        sa, sb = a["slices"][d], b["slices"][d]
        lo = max(sa.start, sb.start)
        hi = min(sa.stop,  sb.stop)
        if lo >= hi:
            return 0.0
        inter_sl.append(slice(lo, hi))
    tup = tuple(inter_sl)
    a_slices = tuple(slice(tup[d].start - a["slices"][d].start,
                           tup[d].stop  - a["slices"][d].start) for d in range(ndim))
    b_slices = tuple(slice(tup[d].start - b["slices"][d].start,
                           tup[d].stop  - b["slices"][d].start) for d in range(ndim))
    inter = float(np.logical_and(a["mask"][a_slices], b["mask"][b_slices]).sum())
    denom = float(a["size"] + b["size"])
    return 2.0 * inter / denom if denom > 0 else 1.0


def _hd95_from_components(a: dict, b: dict, spacing: tuple[float, float, float]) -> float:
    ap = np.argwhere(a["mask"]).astype(np.float32)
    bp = np.argwhere(b["mask"]).astype(np.float32)
    if ap.size == 0 or bp.size == 0:
        return HD95_PENALTY_MM
    # Add back the bbox origin, then scale by spacing.
    ap += np.array([a["slices"][d].start for d in range(ap.shape[1])], dtype=np.float32)
    bp += np.array([b["slices"][d].start for d in range(bp.shape[1])], dtype=np.float32)
    ap *= np.asarray(spacing, dtype=np.float32)
    bp *= np.asarray(spacing, dtype=np.float32)
    if ap.shape[0] > _HD95_POINT_CAP:
        idx = np.random.default_rng(0).choice(ap.shape[0], _HD95_POINT_CAP, replace=False)
        ap = ap[idx]
    if bp.shape[0] > _HD95_POINT_CAP:
        idx = np.random.default_rng(0).choice(bp.shape[0], _HD95_POINT_CAP, replace=False)
        bp = bp[idx]
    d_ab = cdist(ap, bp).min(axis=1)
    d_ba = cdist(bp, ap).min(axis=1)
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def _match_lesions(gt: list[dict], pr: list[dict], ndim: int) -> list[tuple[int, int]]:
    """Greedy IoU matching using bbox-cropped component sub-volumes."""
    if not gt or not pr:
        return []
    iou = np.zeros((len(gt), len(pr)), dtype=np.float64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            iou[i, j], _ = _pair_iou(g, p, ndim)
    pairs: list[tuple[int, int]] = []
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
    ndim = gt_seg.ndim
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
        pairs = _match_lesions(gt_cc, pr_cc, ndim)
        dices, hds = [], []
        for gi, pj in pairs:
            dices.append(_dice_from_components(gt_cc[gi], pr_cc[pj], ndim))
            hds.append(_hd95_from_components(gt_cc[gi], pr_cc[pj], spacing))
        matched_g = {gi for gi, _ in pairs}
        matched_p = {pj for _, pj in pairs}
        for gi in range(len(gt_cc)):
            if gi not in matched_g:
                dices.append(0.0); hds.append(HD95_PENALTY_MM)
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


def _worker(pred_path: str, gt_dir: str) -> dict | None:
    pp = Path(pred_path)
    gp = Path(gt_dir) / pp.name
    if not gp.is_file():
        return None
    pr_seg, spacing = _load(pp)
    gt_seg, _ = _load(gp)
    m = evaluate_case(gt_seg, pr_seg, spacing)
    m["case"] = pp.stem
    return m


def _iter_cases(pred_dir: Path, pattern: str, gt_dir: Path) -> Iterable[tuple[str, str]]:
    for pred_path in sorted(pred_dir.glob(pattern)):
        yield (str(pred_path), str(gt_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--gt-dir",   required=True, type=Path)
    ap.add_argument("--out-csv",  required=True, type=Path)
    ap.add_argument("--case-glob", default="*.nii.gz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    pred_paths = sorted(str(p) for p in args.pred_dir.glob(args.case_glob))
    print(f"[eval] {len(pred_paths)} predictions in {args.pred_dir}  |  workers={args.workers}")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, p, str(args.gt_dir)): p for p in pred_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is None:
                print(f"[skip] no GT for {Path(futures[fut]).name}")
                continue
            rows.append(r)
            if i % 25 == 0 or i == len(pred_paths):
                print(f"  [{i}/{len(pred_paths)}] done")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["case"])
    cols = ["case"] + [k for k in rows[0] if k != "case"] if rows else ["case"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

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
