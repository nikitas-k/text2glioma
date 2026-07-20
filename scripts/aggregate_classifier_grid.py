"""Aggregate metrics.json outputs from the classifier grid.

Walks ``<cls_root>/<task>/n_synth_<N>/seed_<S>/metrics.json`` for every
(task, N, S) combination that ran and produces:

    * summary.csv \u2014 flat per-run table (task, n_synth, seed, best_auroc, ...)
    * summary_by_condition.csv \u2014 aggregated (task, n_synth): mean / std / n
    * Optionally a matplotlib figure of AUROC vs n_synth per task with error bars

Usage
-----
::

    python scripts/aggregate_classifier_grid.py \\
        --cls_root /g/data/vp06/nk9793/text2glioma_classify \\
        --out_dir  ./results/cls_grid \\
        --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_metric_files(cls_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(cls_root.glob("*/n_synth_*/seed_*/metrics.json")):
        try:
            with metrics_path.open() as fh:
                m = json.load(fh)
        except Exception as e:
            print(f"[warn] skipping {metrics_path}: {e}", file=sys.stderr)
            continue
        fm = m.get("final_metrics", {})
        rows.append({
            "task":            m.get("task"),
            "n_synthetic":     int(m.get("n_synthetic", 0)),
            "seed":            int(m.get("seed", -1)),
            "best_epoch":      int(m.get("best_epoch", -1)),
            "best_auroc":      float(m.get("best_auroc", float("nan"))),
            "final_auroc":     float(fm.get("auroc", float("nan"))),
            "final_f1":        float(fm.get("f1", float("nan"))),
            "final_bal_acc":   float(fm.get("balanced_accuracy", float("nan"))),
            "n_train_real":    int(m.get("n_train_real", 0)),
            "n_train_synth":   int(m.get("n_train_synth", 0)),
            "n_val":           int(m.get("n_val", 0)),
            "metrics_path":    str(metrics_path),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cls_root", type=Path, required=True,
                    help="Root containing <task>/n_synth_<N>/seed_<S>/metrics.json")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--plot", action="store_true",
                    help="Also emit an AUROC-vs-n_synth line plot per task.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_metric_files(args.cls_root)
    if not rows:
        raise SystemExit(f"no metrics.json files found under {args.cls_root}")

    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} runs", file=sys.stderr)
    print(df[["task", "n_synthetic", "seed", "best_auroc", "final_auroc"]]
          .to_string(index=False), file=sys.stderr)

    df.to_csv(args.out_dir / "summary.csv", index=False)

    # Aggregate across seeds
    agg = (df.groupby(["task", "n_synthetic"])
             .agg(auroc_mean=("best_auroc", "mean"),
                  auroc_std =("best_auroc", "std"),
                  f1_mean   =("final_f1",   "mean"),
                  bal_acc_mean=("final_bal_acc", "mean"),
                  n_seeds   =("best_auroc", "size"),
             )
             .reset_index()
             .sort_values(["task", "n_synthetic"]))
    agg.to_csv(args.out_dir / "summary_by_condition.csv", index=False)
    print("\nAggregated by condition:", file=sys.stderr)
    print(agg.to_string(index=False), file=sys.stderr)

    # Optional Wilcoxon vs real-only baseline
    from scipy.stats import wilcoxon
    stats_rows: list[dict] = []
    for task, sub in df.groupby("task"):
        baseline = sub[sub.n_synthetic == 0]["best_auroc"].values
        if len(baseline) < 2:
            continue
        for n_synth, aug in sub[sub.n_synthetic > 0].groupby("n_synthetic"):
            aug_v = aug["best_auroc"].values
            if len(aug_v) != len(baseline):
                # Wilcoxon needs paired samples; only run if same seed count.
                p = float("nan")
                stat = float("nan")
            else:
                try:
                    stat, p = wilcoxon(aug_v, baseline)
                    p = float(p)
                    stat = float(stat)
                except ValueError:
                    stat, p = float("nan"), float("nan")
            stats_rows.append({
                "task": task,
                "n_synthetic": int(n_synth),
                "wilcoxon_stat": stat,
                "wilcoxon_p":    p,
            })
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(args.out_dir / "wilcoxon_vs_baseline.csv", index=False)
        print("\nWilcoxon vs real-only baseline:", file=sys.stderr)
        print(stats_df.to_string(index=False), file=sys.stderr)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for task, sub in agg.groupby("task"):
            xs = sub["n_synthetic"].values
            ys = sub["auroc_mean"].values
            es = sub["auroc_std"].values
            ax.errorbar(np.maximum(xs, 1), ys, yerr=es,
                        marker="o", capsize=3, label=task.upper())
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Synthetic training samples added (log)")
        ax.set_ylabel("Best validation AUROC")
        ax.set_title("Downstream classifier: real + N synthetic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "auroc_vs_n_synth.png", dpi=150)
        print(f"\nwrote {args.out_dir / 'auroc_vs_n_synth.png'}", file=sys.stderr)

    print(f"\nwrote summary CSVs under {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
