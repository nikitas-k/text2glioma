#!/usr/bin/env python3
"""Compute WM/GM contrast-to-noise ratio (CNR) for T1 images.

For each subject:
  1. Load T1 channel (ch 0) and label file
  2. Mask out whole tumour (label > 0) to get normal brain tissue
  3. Fit a 3-component GMM to the non-tumour brain voxels (CSF, GM, WM)
  4. Compute CNR = |mu_WM - mu_GM| / sqrt(0.5 * (sigma_WM^2 + sigma_GM^2))

Outputs a CSV with columns: nnunet_id, T1_wm_mean, T1_gm_mean, T1_csf_mean,
T1_wm_std, T1_gm_std, T1_csf_std, T1_wmgm_cnr

Usage (on Gadi):
  python scripts/compute_cnr.py \
      --datalist datalist_N1510.json \
      --csv cnr_audit.csv \
      --workers 16

  # Quick test on a few subjects:
  python scripts/compute_cnr.py \
      --datalist datalist_N1510.json --max-subjects 10 --csv cnr_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture


# ── per-subject computation ──────────────────────────────────────────
def compute_cnr(image_path: str, label_path: str,
                t1_channel: int = 0) -> dict:
    """Compute WM/GM CNR from a 3-component GMM fit on non-tumour brain.

    Parameters
    ----------
    image_path : str
        Path to multi-channel NIfTI (H, W, D, C).
    label_path : str
        Path to label NIfTI (H, W, D). BraTS convention:
        0=background, 1=necrosis, 2=edema, 4=enhancing.
    t1_channel : int
        Channel index for T1 (default 0).

    Returns
    -------
    dict with keys: nnunet_id, T1_wm_mean, T1_gm_mean, T1_csf_mean,
    T1_wm_std, T1_gm_std, T1_csf_std, T1_wmgm_cnr, n_brain_voxels,
    gmm_converged, gmm_bic_3, gmm_bic_2
    """
    nnunet_id = Path(image_path).name.replace(".nii.gz", "")

    img = nib.load(image_path)
    vol = img.get_fdata()
    t1 = vol[..., t1_channel].astype(np.float32)

    lab = nib.load(label_path).get_fdata()
    tumour_mask = lab > 0  # whole tumour

    # brain mask: non-zero T1 voxels that are not tumour
    brain_mask = (t1 > 0) & ~tumour_mask
    brain_voxels = t1[brain_mask]

    result = {
        "nnunet_id": nnunet_id,
        "n_brain_voxels": int(brain_voxels.size),
    }

    if brain_voxels.size < 1000:
        # not enough voxels — return NaN
        for k in ("T1_wm_mean", "T1_gm_mean", "T1_csf_mean",
                   "T1_wm_std", "T1_gm_std", "T1_csf_std",
                   "T1_wmgm_cnr", "gmm_converged", "gmm_bic_3", "gmm_bic_2"):
            result[k] = float("nan")
        return result

    # Robust clipping: remove extreme outliers (>99.5th percentile)
    clip_hi = np.percentile(brain_voxels, 99.5)
    brain_voxels = brain_voxels[brain_voxels <= clip_hi]

    X = brain_voxels.reshape(-1, 1)

    # Fit 3-component GMM (CSF, GM, WM)
    gmm3 = GaussianMixture(n_components=3, random_state=42,
                            max_iter=200, n_init=5,
                            covariance_type="full")
    gmm3.fit(X)

    # Also fit 2-component for BIC comparison
    gmm2 = GaussianMixture(n_components=2, random_state=42,
                            max_iter=200, n_init=3,
                            covariance_type="full")
    gmm2.fit(X)

    # Sort components by mean intensity: CSF < GM < WM
    means = gmm3.means_.ravel()
    stds = np.sqrt(gmm3.covariances_.ravel())
    order = np.argsort(means)

    csf_mean, gm_mean, wm_mean = means[order]
    csf_std, gm_std, wm_std = stds[order]

    # CNR = |mu_WM - mu_GM| / sqrt(0.5 * (sigma_WM^2 + sigma_GM^2))
    denom = np.sqrt(0.5 * (wm_std**2 + gm_std**2))
    cnr = abs(wm_mean - gm_mean) / denom if denom > 0 else 0.0

    result.update({
        "T1_csf_mean": round(float(csf_mean), 2),
        "T1_gm_mean": round(float(gm_mean), 2),
        "T1_wm_mean": round(float(wm_mean), 2),
        "T1_csf_std": round(float(csf_std), 2),
        "T1_gm_std": round(float(gm_std), 2),
        "T1_wm_std": round(float(wm_std), 2),
        "T1_wmgm_cnr": round(float(cnr), 4),
        "gmm_converged": int(gmm3.converged_),
        "gmm_bic_3": round(float(gmm3.bic(X)), 1),
        "gmm_bic_2": round(float(gmm2.bic(X)), 1),
    })
    return result


def _worker(args):
    """Wrapper for ProcessPoolExecutor."""
    image_path, label_path, t1_channel = args
    try:
        return compute_cnr(image_path, label_path, t1_channel)
    except Exception as e:
        nnunet_id = Path(image_path).name.replace(".nii.gz", "")
        print(f"  ERROR {nnunet_id}: {e}", file=sys.stderr)
        return {"nnunet_id": nnunet_id, "error": str(e)}


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datalist", required=True,
                   help="Path to datalist JSON (must have 'image' and 'label')")
    p.add_argument("--csv", default="cnr_audit.csv",
                   help="Output CSV path (default: cnr_audit.csv)")
    p.add_argument("--t1-channel", type=int, default=0,
                   help="Channel index for T1 (default 0)")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Limit number of subjects (for testing)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers (default 8)")
    p.add_argument("--split", choices=["training", "validation", "all"],
                   default="all",
                   help="Which split(s) to process (default: all)")
    args = p.parse_args()

    # Load datalist
    with open(args.datalist) as f:
        datalist = json.load(f)

    subjects = []
    if args.split == "all":
        for split in ("training", "validation"):
            subjects.extend(datalist.get(split, []))
    else:
        subjects = datalist.get(args.split, [])

    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    print(f"Processing {len(subjects)} subjects with {args.workers} workers")

    # Build task list
    tasks = [
        (s["image"], s["label"], args.t1_channel)
        for s in subjects
    ]

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] done")

    # Sort by nnunet_id
    results.sort(key=lambda r: r.get("nnunet_id", ""))

    # Write CSV
    fieldnames = [
        "nnunet_id", "n_brain_voxels",
        "T1_csf_mean", "T1_gm_mean", "T1_wm_mean",
        "T1_csf_std", "T1_gm_std", "T1_wm_std",
        "T1_wmgm_cnr", "gmm_converged", "gmm_bic_3", "gmm_bic_2",
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nWrote {len(results)} rows to {args.csv}")

    # Quick summary
    cnr_vals = [r["T1_wmgm_cnr"] for r in results
                if "T1_wmgm_cnr" in r and not np.isnan(r.get("T1_wmgm_cnr", float("nan")))]
    if cnr_vals:
        cnr_arr = np.array(cnr_vals)
        print(f"\nT1 WM/GM CNR summary (n={len(cnr_arr)}):")
        print(f"  mean={cnr_arr.mean():.3f}  std={cnr_arr.std():.3f}")
        print(f"  median={np.median(cnr_arr):.3f}")
        print(f"  min={cnr_arr.min():.3f}  max={cnr_arr.max():.3f}")
        print(f"  Q1={np.percentile(cnr_arr, 25):.3f}  "
              f"Q3={np.percentile(cnr_arr, 75):.3f}")

    n_err = sum(1 for r in results if "error" in r)
    if n_err:
        print(f"\n{n_err} subjects had errors (see stderr)")


if __name__ == "__main__":
    main()
