"""Cross-patient W1 / KS ceiling for tumour-region intensities.

For each modality and each of `--n_pairs` random pairs of *different* patients,
compute the 1-D Wasserstein and Kolmogorov-Smirnov distance between the
intensity distributions inside the respective expert tumour masks.

Outputs a long-format CSV with columns:
    pair_idx, subject_a, subject_b, modality, w1, ks, n_a, n_b

Usage
-----
    python scripts/compute_cross_patient_ceiling.py \
        --datalist datalist_N1510_val3.json \
        --split validation \
        --n_pairs 200 \
        --seed 0 \
        --out /Users/nk233/mhf/projects/text2glioma/runs/pinaya_decoder_only_v5_no_disc/data/cfg_sweep_text_only/tumour_region_real_reference.csv

Notes
-----
* Uses ``_build_val_transform`` from ``offline_sample_stage2_compare`` with
  ``channel_reorder=False`` to match the pre-processing used in the CFG sweep.
* Pairs are drawn without replacement (a, b) with a != b. Each pair contributes
  one row per modality.
* Voxel selection requires both masks to contain >= ``--min_voxels`` foreground
  voxels (default 500), matching the test-retest floor cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from offline_sample_stage2_compare import _build_val_transform  # noqa: E402
from text2glioma.utils import MODALITY_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datalist", type=Path, default=REPO / "datalist_N1510_val3.json")
    p.add_argument("--split", default="validation")
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--min_voxels", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_channel_reorder", action="store_true", default=True,
                   help="Match v5_no_disc training (default: True).")
    p.add_argument("--channel_reorder", dest="no_channel_reorder", action="store_false")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_case(item: dict, channel_reorder: bool):
    """Run val transform and return (image[CxHxWxD], mask[HxWxD] bool, subj_id)."""
    item = dict(item)
    has_label = bool(item.get("label"))
    if not has_label:
        return None
    t = _build_val_transform(channel_reorder=channel_reorder, has_label=True)(item)
    image = t["image"]  # tensor [C, H, W, D]
    label = t["label"]  # tensor [1, H, W, D] or [H, W, D]
    if label.ndim == 4:
        label = label[0]
    mask = (label > 0).bool().numpy()
    return image.numpy(), mask, item.get("subject_id", "")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.datalist) as f:
        datalist = json.load(f)
    if args.split not in datalist:
        raise KeyError(f"Split {args.split!r} not in datalist")
    data = [d for d in datalist[args.split] if d.get("label")]
    n = len(data)
    print(f"Split={args.split}  cases with labels: {n}")
    if n < 2:
        raise SystemExit("Need at least two labelled cases")

    # Sample distinct pairs (a, b) with a < b, no duplicates
    seen = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < args.n_pairs:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    channel_reorder = not args.no_channel_reorder

    # Cache transforms per index (each case appears in multiple pairs)
    cache: dict[int, tuple[np.ndarray, np.ndarray, str] | None] = {}

    def get(idx: int):
        if idx not in cache:
            cache[idx] = load_case(data[idx], channel_reorder=channel_reorder)
        return cache[idx]

    rows: list[dict] = []
    for pi, (a, b) in enumerate(tqdm(pairs, desc="Pairs")):
        A = get(a)
        B = get(b)
        if A is None or B is None:
            continue
        img_a, mask_a, sa = A
        img_b, mask_b, sb = B
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        if n_a < args.min_voxels or n_b < args.min_voxels:
            continue
        for c, mod in enumerate(MODALITY_NAMES):
            va = img_a[c][mask_a].ravel()
            vb = img_b[c][mask_b].ravel()
            if va.size < args.min_voxels or vb.size < args.min_voxels:
                continue
            w1 = float(wasserstein_distance(va, vb))
            ks = float(ks_2samp(va, vb).statistic)
            rows.append({
                "pair_idx": pi,
                "subject_a": sa,
                "subject_b": sb,
                "modality": mod,
                "w1": w1,
                "ks": ks,
                "n_a": n_a,
                "n_b": n_b,
            })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    print()
    print("Per-modality W1 summary:")
    print(df.groupby("modality")["w1"].agg(["mean", "median", "std", "count"]).round(4).to_string())
    print()
    print("Per-modality KS summary:")
    print(df.groupby("modality")["ks"].agg(["mean", "median", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
