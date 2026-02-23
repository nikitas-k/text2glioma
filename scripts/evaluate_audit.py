#!/usr/bin/env python3
"""Evaluate the text2glioma_audit.csv — summary statistics, outlier
detection, and distribution plots."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "text2glioma_audit.csv"
OUTDIR = Path(__file__).resolve().parent.parent / "audit_figures"
MODS = ["T1", "T1CE", "T2", "FLAIR"]


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def basic_info(df: pd.DataFrame) -> None:
    section("BASIC INFO")
    print(f"Subjects : {len(df)}")
    print(f"Splits   : {df['split'].value_counts().to_dict()}")
    uniq = df["shape"].value_counts()
    print(f"Shapes   : {uniq.to_dict()}")
    for ax in ["spacing_x", "spacing_y", "spacing_z"]:
        v = df[ax]
        print(f"{ax}: nunique={v.nunique()}, range=[{v.min():.4f}, {v.max():.4f}]")


def fwhm_stats(df: pd.DataFrame) -> None:
    section("FWHM STATISTICS (mm)")
    rows = []
    for mod in MODS:
        for axis in ["mean", "x", "y", "z"]:
            col = f"{mod}_fwhm_{axis}"
            if col not in df.columns:
                continue
            v = df[col].dropna()
            rows.append({
                "column": col,
                "n": len(v),
                "mean": v.mean(),
                "std": v.std(),
                "median": v.median(),
                "min": v.min(),
                "p5": v.quantile(0.05),
                "p25": v.quantile(0.25),
                "p75": v.quantile(0.75),
                "p95": v.quantile(0.95),
                "max": v.max(),
            })
    tbl = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.max_columns", 20,
                           "display.width", 200):
        print(tbl.to_string(index=False))


def sharpness_stats(df: pd.DataFrame) -> None:
    section("SHARPNESS STATISTICS")
    rows = []
    for mod in MODS:
        for metric in ["hf_ratio", "lap_var"]:
            col = f"{mod}_{metric}"
            if col not in df.columns:
                continue
            v = df[col].dropna()
            rows.append({
                "column": col,
                "n": len(v),
                "mean": v.mean(),
                "std": v.std(),
                "median": v.median(),
                "min": v.min(),
                "p5": v.quantile(0.05),
                "p95": v.quantile(0.95),
                "max": v.max(),
            })
    tbl = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_columns", 20,
                           "display.width", 200):
        print(tbl.to_string(index=False))


def bbox_stats(df: pd.DataFrame) -> None:
    section("BOUNDING BOX (voxels)")
    for bb in ["bbox_LR", "bbox_AP", "bbox_SI"]:
        v = df[bb].dropna()
        print(f"  {bb}: mean={v.mean():.1f}  std={v.std():.1f}  "
              f"min={v.min():.0f}  max={v.max():.0f}")


def flag_analysis(df: pd.DataFrame) -> None:
    section("FLAG ANALYSIS")
    flagged = df[df["flag"].notna() & (df["flag"] != "")]
    n_total = len(df)
    n_flagged = len(flagged)
    print(f"Subjects with any flag: {n_flagged} / {n_total} "
          f"({100*n_flagged/n_total:.1f}%)")

    ctr = Counter()
    for flags_str in flagged["flag"]:
        for f in str(flags_str).split(", "):
            key = f.split("(")[0].strip()
            if key:
                ctr[key] += 1

    print(f"\n{'Flag':30s}  {'Count':>6s}  {'%':>6s}")
    print("-" * 46)
    for flag, count in ctr.most_common():
        print(f"  {flag:28s}  {count:6d}  {100*count/n_total:5.1f}%")


def outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outliers at mean+2σ on FWHM-mean per modality."""
    section("OUTLIER DETECTION (mean + 2σ on fwhm_mean)")
    outlier_mask = pd.Series(False, index=df.index)
    for mod in MODS:
        col = f"{mod}_fwhm_mean"
        if col not in df.columns:
            continue
        v = df[col].dropna()
        thr = v.mean() + 2 * v.std()
        is_out = df[col] > thr
        n_out = is_out.sum()
        outlier_mask |= is_out
        print(f"  {mod:6s}: threshold={thr:.2f} mm  → {n_out} outliers")

    n_any = outlier_mask.sum()
    print(f"\n  Union (any modality): {n_any} outlier subjects")

    outliers = df[outlier_mask].copy()
    return outliers


def correlation_analysis(df: pd.DataFrame) -> None:
    """Correlations between FWHM-mean and sharpness metrics."""
    section("CORRELATIONS (FWHM-mean vs sharpness)")
    for mod in MODS:
        fwhm_col = f"{mod}_fwhm_mean"
        hf_col = f"{mod}_hf_ratio"
        lap_col = f"{mod}_lap_var"
        mask = df[fwhm_col].notna() & df[hf_col].notna()
        sub = df[mask]
        if len(sub) < 10:
            continue
        r_hf = sub[fwhm_col].corr(sub[hf_col])
        r_lap = sub[fwhm_col].corr(sub[lap_col])
        print(f"  {mod:6s}: FWHM vs hf_ratio r={r_hf:+.3f}   "
              f"FWHM vs lap_var r={r_lap:+.3f}")


# ── Plotting ────────────────────────────────────────────────────────

def plot_fwhm_distributions(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, mod in enumerate(MODS):
        ax = axes[i]
        for axis, color, ls in [("x", "tab:blue", "-"),
                                 ("y", "tab:orange", "-"),
                                 ("z", "tab:green", "-"),
                                 ("mean", "black", "--")]:
            col = f"{mod}_fwhm_{axis}"
            if col not in df.columns:
                continue
            v = df[col].dropna()
            ax.hist(v, bins=50, alpha=0.4, label=axis,
                    color=color if axis != "mean" else "gray",
                    histtype="stepfilled" if axis != "mean" else "step",
                    linewidth=2 if axis == "mean" else 1)
        ax.set_title(f"{mod} FWHM", fontsize=13, fontweight="bold")
        ax.set_xlabel("FWHM (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        # threshold line
        col_m = f"{mod}_fwhm_mean"
        if col_m in df.columns:
            v = df[col_m].dropna()
            thr = v.mean() + 2 * v.std()
            ax.axvline(thr, color="red", ls="--", lw=1.5,
                       label=f"μ+2σ = {thr:.1f}")
            ax.legend(fontsize=9)
    fig.suptitle("FWHM Distributions (corrected parser)", fontsize=15,
                 fontweight="bold")
    plt.tight_layout()
    path = outdir / "fwhm_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sharpness_distributions(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, mod in enumerate(MODS):
        ax = axes[i]
        col = f"{mod}_hf_ratio"
        if col not in df.columns:
            continue
        v = df[col].dropna()
        ax.hist(v, bins=60, alpha=0.6, color="steelblue", edgecolor="white")
        ax.axvline(v.mean(), color="red", ls="-", lw=1.5, label=f"mean={v.mean():.4f}")
        ax.axvline(v.mean() - 2*v.std(), color="red", ls="--", lw=1,
                   label=f"μ-2σ={v.mean()-2*v.std():.4f}")
        ax.set_title(f"{mod} HF Ratio", fontsize=13, fontweight="bold")
        ax.set_xlabel("HF energy ratio")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    fig.suptitle("Spectral Sharpness (HF ratio) Distributions", fontsize=15,
                 fontweight="bold")
    plt.tight_layout()
    path = outdir / "hf_ratio_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fwhm_vs_hf(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, mod in enumerate(MODS):
        ax = axes[i]
        fc = f"{mod}_fwhm_mean"
        hc = f"{mod}_hf_ratio"
        if fc not in df.columns or hc not in df.columns:
            continue
        mask = df[fc].notna() & df[hc].notna()
        x = df.loc[mask, fc]
        y = df.loc[mask, hc]
        ax.scatter(x, y, s=8, alpha=0.5, c="steelblue", edgecolors="none")
        ax.set_xlabel("FWHM mean (mm)")
        ax.set_ylabel("HF ratio")
        ax.set_title(f"{mod}: FWHM vs HF ratio  (r={x.corr(y):+.3f})",
                     fontsize=12, fontweight="bold")
    fig.suptitle("FWHM vs Spectral Sharpness (scatter)", fontsize=15,
                 fontweight="bold")
    plt.tight_layout()
    path = outdir / "fwhm_vs_hf_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_modality_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """Box plots comparing FWHM-mean across modalities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for mod in MODS:
        col = f"{mod}_fwhm_mean"
        if col in df.columns:
            data.append(df[col].dropna().values)
            labels.append(mod)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.3))
    colours = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("FWHM mean (mm)")
    ax.set_title("FWHM Distribution by Modality", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = outdir / "fwhm_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    csv_path = CSV
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])

    df = pd.read_csv(csv_path)

    basic_info(df)
    fwhm_stats(df)
    sharpness_stats(df)
    bbox_stats(df)
    flag_analysis(df)
    outliers = outlier_analysis(df)
    correlation_analysis(df)

    # Save outlier list
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    outlier_path = outdir / "fwhm_outliers.csv"
    cols_out = (["subject_id", "split", "image"]
                + [f"{m}_fwhm_mean" for m in MODS]
                + [f"{m}_hf_ratio" for m in MODS])
    cols_out = [c for c in cols_out if c in outliers.columns]
    outliers[cols_out].to_csv(outlier_path, index=False)
    print(f"\n  Outlier list saved: {outlier_path}")

    section("PLOTS")
    plot_fwhm_distributions(df, outdir)
    plot_sharpness_distributions(df, outdir)
    plot_fwhm_vs_hf(df, outdir)
    plot_modality_comparison(df, outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
