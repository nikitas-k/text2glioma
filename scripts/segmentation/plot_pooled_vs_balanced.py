"""Supplementary Figure S6: nnU-Net Dice by synthetic dose, pooled vs balanced sampler.

Reads per-case CSVs from paper/tables/nnunet_eval/{internal,lumiere}_synth{n}.csv
(pooled sampler) and paper/tables/nnunet_eval_balanced/... (balanced sampler),
builds three-panel Dice-vs-dose plot for WT/TC/ET on the internal held-out cohort,
overlays the real-only baseline, and writes both PNG + PDF to
paper/figures/supplementary/fig_seg_pooled_vs_balanced.{png,pdf}.

Bootstrap 95% CIs (2000 resamples, seed 42) are drawn on top of point estimates
so the mechanism (pooled degrades at high dose; balanced flat) is legible at a
glance. Only n_synth doses present in both samplers are shown; the baseline
comes from Dataset510 (real-only) shared across both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIONS = ["WT", "TC", "ET", "ED"]
DOSES = [500, 1000, 5000, 10000]
POOLED_DIR = Path("paper/tables/nnunet_eval")
BALANCED_DIR = Path("paper/tables/nnunet_eval_balanced")
OUT_DIR = Path("paper/figures/supplementary")


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_percase(directory: Path, split: str, dose: int) -> pd.DataFrame | None:
    p = directory / f"{split}_synth{dose}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _region_dice(df: pd.DataFrame, region: str) -> np.ndarray:
    col = f"{region}_Dice"
    if col not in df.columns:
        return np.array([])
    return df[col].to_numpy(dtype=float)


def _dose_curve(directory: Path, split: str, region: str) -> list[tuple[int, float, float, float]]:
    out: list[tuple[int, float, float, float]] = []
    for dose in DOSES:
        df = _load_percase(directory, split, dose)
        if df is None:
            continue
        vals = _region_dice(df, region)
        m, lo, hi = _bootstrap_mean_ci(vals)
        out.append((dose, m, lo, hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="internal", choices=["internal", "lumiere"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Drop regions whose *_Dice column isn't present in the baseline CSV yet.
    base_probe = _load_percase(POOLED_DIR, args.split, 0)
    assert base_probe is not None, f"missing baseline {args.split}_synth0.csv"
    regions = [r for r in REGIONS if f"{r}_Dice" in base_probe.columns]

    fig, axes = plt.subplots(1, len(regions), figsize=(3.6 * len(regions), 3.4), sharey=True)
    if len(regions) == 1:
        axes = [axes]
    for ax, region in zip(axes, regions):
        # Baseline (Dataset510, real-only) is shared; read from pooled dir.
        base_df = _load_percase(POOLED_DIR, args.split, 0)
        assert base_df is not None
        base_m, base_lo, base_hi = _bootstrap_mean_ci(_region_dice(base_df, region))
        ax.axhline(base_m, color="0.35", linestyle="--", linewidth=1.2, label=f"real-only baseline")
        ax.axhspan(base_lo, base_hi, color="0.85", alpha=0.6)

        for directory, colour, label in [
            (POOLED_DIR, "#c1272d", "pooled sampler"),
            (BALANCED_DIR, "#0088cc", "balanced sampler"),
        ]:
            curve = _dose_curve(directory, args.split, region)
            if not curve:
                continue
            xs = [c[0] for c in curve]
            ys = [c[1] for c in curve]
            lo = [c[2] for c in curve]
            hi = [c[3] for c in curve]
            ax.plot(xs, ys, marker="o", color=colour, label=label, linewidth=1.8)
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.18, linewidth=0)

        ax.set_xscale("log")
        ax.set_xticks(DOSES)
        ax.set_xticklabels([str(d) for d in DOSES])
        ax.set_xlabel(r"$n_{\mathrm{synth}}$")
        ax.set_title(region)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Lesion-wise Dice")
    axes[-1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"nnU-Net Dice vs synthetic dose ({args.split} test cohort)", fontsize=11)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = args.out_dir / f"fig_seg_pooled_vs_balanced_{args.split}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
