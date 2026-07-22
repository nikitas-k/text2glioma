"""Aggregate the diagnostic classifier grid (baseline vs weight/batch fixes).

Walks ``<cls_diag_root>/<task>/n_synth_<N>/<config>/seed_<S>/metrics.json``
and produces:

    * diag_summary.csv           - per-run
    * diag_summary_by_config.csv - mean/std AUROC per config, plus paired
                                    Wilcoxon vs baseline
    * diag_bars.png              - bar chart of AUROC per config with error bars

Usage
-----
::

    python scripts/aggregate_classifier_grid_diag.py \\
        --cls_diag_root /g/data/vp06/nk9793/text2glioma_classify_diag \\
        --out_dir       ./results/cls_diag \\
        --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_runs(root: Path) -> list[dict]:
    rows: list[dict] = []
    for mp in sorted(root.glob("*/n_synth_*/*/seed_*/metrics.json")):
        try:
            with mp.open() as fh:
                m = json.load(fh)
        except Exception as e:
            print(f"[warn] skipping {mp}: {e}", file=sys.stderr)
            continue
        # path: <root>/<task>/n_synth_<N>/<config>/seed_<S>/metrics.json
        parts = mp.parts
        config = parts[-3]
        fm = m.get("final_metrics", {})
        rows.append({
            "task":               m.get("task"),
            "n_synthetic":        int(m.get("n_synthetic", 0)),
            "config":             config,
            "seed":               int(m.get("seed", -1)),
            "class_weight_source": m.get("class_weight_source", "all"),
            "balance_batches":    bool(m.get("balance_batches", False)),
            "best_epoch":         int(m.get("best_epoch", -1)),
            "best_auroc":         float(m.get("best_auroc", float("nan"))),
            "final_auroc":        float(fm.get("auroc", float("nan"))),
            "final_f1":           float(fm.get("f1", float("nan"))),
            "final_bal_acc":      float(fm.get("balanced_accuracy", float("nan"))),
            "n_train_real":       int(m.get("n_train_real", 0)),
            "n_train_synth":      int(m.get("n_train_synth", 0)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cls_diag_root", type=Path, required=True)
    ap.add_argument("--out_dir",       type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_runs(args.cls_diag_root)
    if not rows:
        raise SystemExit(f"no metrics.json under {args.cls_diag_root}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "diag_summary.csv", index=False)
    print(f"loaded {len(df)} diagnostic runs", file=sys.stderr)

    # Aggregate across seeds per config.
    agg = (df.groupby(["task", "n_synthetic", "config"])
             .agg(auroc_mean=("best_auroc", "mean"),
                  auroc_std =("best_auroc", "std"),
                  best_epoch_mean=("best_epoch", "mean"),
                  n_seeds  =("best_auroc", "size"))
             .reset_index())

    # Wilcoxon vs baseline (per task/n_synthetic).
    from scipy.stats import wilcoxon
    stats_rows: list[dict] = []
    for (task, n_synth), sub in df.groupby(["task", "n_synthetic"]):
        baseline = sub[sub.config == "baseline"]["best_auroc"].values
        for cfg in sub["config"].unique():
            if cfg == "baseline":
                continue
            aug = sub[sub.config == cfg]["best_auroc"].values
            if len(baseline) == len(aug) and len(baseline) >= 2:
                try:
                    stat, p = wilcoxon(aug, baseline)
                    p_val = float(p); stat_val = float(stat)
                except ValueError:
                    p_val = float("nan"); stat_val = float("nan")
            else:
                p_val = float("nan"); stat_val = float("nan")
            stats_rows.append({
                "task": task, "n_synthetic": int(n_synth), "config": cfg,
                "wilcoxon_stat": stat_val, "wilcoxon_p": p_val,
                "delta_vs_baseline": float(aug.mean() - baseline.mean()),
            })
    stats_df = pd.DataFrame(stats_rows)
    agg = agg.merge(stats_df, on=["task", "n_synthetic", "config"], how="left")

    agg.to_csv(args.out_dir / "diag_summary_by_config.csv", index=False)
    print("\nAggregated by config:", file=sys.stderr)
    print(agg.to_string(index=False), file=sys.stderr)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        config_order = ["baseline", "weight_real", "balance_batches", "both"]
        # colour-scheme: red for degradation risk, green for improvement, grey for baseline
        colors = {"baseline": "#888888", "weight_real": "#2ca02c",
                  "balance_batches": "#1f77b4", "both": "#9467bd"}
        for task in agg["task"].unique():
            sub = agg[agg.task == task]
            xs = np.arange(len(config_order))
            ys = []
            es = []
            for cfg in config_order:
                r = sub[sub.config == cfg]
                if len(r) == 0:
                    ys.append(float("nan")); es.append(0)
                else:
                    ys.append(float(r["auroc_mean"].iloc[0]))
                    es.append(float(r["auroc_std"].iloc[0]) if not np.isnan(r["auroc_std"].iloc[0]) else 0)
            bars = ax.bar([f"{cfg}" for cfg in config_order], ys, yerr=es,
                          capsize=4, color=[colors[c] for c in config_order],
                          alpha=0.85, edgecolor="black", linewidth=0.5,
                          label=task.upper())
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.005, f"{y:.3f}", ha="center", va="bottom",
                        fontsize=9)
        ax.set_ylabel("Best validation AUROC")
        ax.set_title(f"Diagnostic: real+{int(agg['n_synthetic'].iloc[0])} synthetic\n"
                     f"class-weight source × batch-balance combinations")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "diag_bars.png", dpi=150)
        print(f"\nwrote {args.out_dir / 'diag_bars.png'}", file=sys.stderr)

    print(f"\nCSVs in {args.out_dir}")


if __name__ == "__main__":
    main()
