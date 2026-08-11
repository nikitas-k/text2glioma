"""Paired Wilcoxon signed-rank test of per-case Dice vs the real-only baseline
for both the pooled and balanced samplers.

Multiple-comparisons correction (Benjamini-Hochberg) is applied across the full
set of (sampler x dose x region) tests. Writes paper/tables/nnunet_paired_stats.csv
(long form) and paper/tables/nnunet_paired_stats.tex (compact three-region table
for the supplementary).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

REGIONS = ["WT", "TC", "ET", "ED"]
DOSES = [500, 1000, 5000, 10000]
POOLED_DIR = Path("paper/tables/nnunet_eval")
BALANCED_DIR = Path("paper/tables/nnunet_eval_balanced")


def _load(directory: Path, split: str, dose: int) -> pd.DataFrame | None:
    p = directory / f"{split}_synth{dose}.csv"
    return pd.read_csv(p) if p.exists() else None


def _paired_test(base: pd.DataFrame, treat: pd.DataFrame, region: str) -> tuple[int, float, float]:
    col = f"{region}_Dice"
    merged = base[["case", col]].merge(treat[["case", col]], on="case", suffixes=("_b", "_t"))
    d = merged[f"{col}_t"] - merged[f"{col}_b"]
    d = d.dropna().to_numpy()
    if d.size == 0 or np.allclose(d, 0):
        return d.size, float("nan"), float("nan")
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return int(d.size), float(d.mean()), float(res.pvalue)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="internal", choices=["internal", "lumiere", "brats_ssa"])
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-tex", type=Path, default=None)
    args = ap.parse_args()

    out_csv = args.out_csv or Path(f"paper/tables/nnunet_paired_stats_{args.split}.csv") \
        if args.split != "internal" else Path("paper/tables/nnunet_paired_stats.csv")
    out_tex = args.out_tex or Path(f"paper/tables/nnunet_paired_stats_{args.split}.tex") \
        if args.split != "internal" else Path("paper/tables/nnunet_paired_stats.tex")

    base = _load(POOLED_DIR, args.split, 0)
    assert base is not None, f"missing {args.split}_synth0.csv (real-only baseline)"

    # Only test regions actually present in the baseline CSV.
    regions = [r for r in REGIONS if f"{r}_Dice" in base.columns]

    rows: list[dict] = []
    for sampler, directory in [("pooled", POOLED_DIR), ("balanced", BALANCED_DIR)]:
        for dose in DOSES:
            treat = _load(directory, args.split, dose)
            if treat is None:
                continue
            for region in regions:
                n, delta, p = _paired_test(base, treat, region)
                rows.append({
                    "sampler": sampler, "dose": dose, "region": region,
                    "n": n, "delta_dice": delta, "p_raw": p,
                })

    df = pd.DataFrame(rows)
    valid = df["p_raw"].notna()
    corrected = multipletests(df.loc[valid, "p_raw"].to_numpy(), method="fdr_bh")
    df["p_bh"] = np.nan
    df.loc[valid, "p_bh"] = corrected[1]
    df["sig_bh05"] = df["p_bh"] < 0.05

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}")

    n_reg = len(regions)
    n_tests = int(df["p_raw"].notna().sum())
    n_cases = int(df["n"].dropna().iloc[0]) if not df["n"].dropna().empty else 0
    cohort_desc = {
        "internal": f"internal held-out cohort ($N={n_cases}$)",
        "lumiere": f"LUMIERE baseline pre-op cohort ($N={n_cases}$)",
        "brats_ssa": f"BraTS-Africa 2023 external cohort ($N={n_cases}$)",
    }[args.split]
    col_spec = "ll" + "rr" * n_reg
    region_hdr = " & ".join(rf"\multicolumn{{2}}{{c}}{{{r}}}" for r in regions)
    cmid = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(n_reg))
    subhdr = " & ".join([r"$\Delta$ & $p_\mathrm{BH}$"] * n_reg)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Paired Wilcoxon signed-rank test of per-case Dice vs. the real-only baseline for the pooled and balanced samplers on the {cohort_desc}. $\Delta$ is the mean per-case Dice change; $p_\mathrm{{BH}}$ is Benjamini--Hochberg-corrected across all {n_tests} tests. Bold entries are significant at $p_\mathrm{{BH}}<0.05$.}}",
        rf"\label{{tab:seg_paired_stats_{args.split}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Sampler & $n_\mathrm{{synth}}$ & {region_hdr} \\",
        cmid,
        rf" & & {subhdr} \\",
        r"\midrule",
    ]

    def _fmt(val: float, is_p: bool = False) -> str:
        if np.isnan(val):
            return "--"
        if is_p:
            return f"{val:.3f}" if val >= 0.001 else "$<$0.001"
        return f"{val:+.3f}"

    for sampler in ["pooled", "balanced"]:
        for dose in DOSES:
            sub = df[(df.sampler == sampler) & (df.dose == dose)]
            if sub.empty:
                continue
            cells: list[str] = []
            for region in regions:
                row = sub[sub.region == region]
                if row.empty:
                    cells.extend(["--", "--"])
                    continue
                d = float(row["delta_dice"].iloc[0])
                p = float(row["p_bh"].iloc[0])
                d_str = _fmt(d)
                p_str = _fmt(p, is_p=True)
                if row["sig_bh05"].iloc[0]:
                    d_str = r"\textbf{" + d_str + r"}"
                cells.extend([d_str, p_str])
            lines.append(f"{sampler} & {dose} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_tex}")

    # Console summary
    print("\n=== Significant results (p_BH < 0.05) ===")
    sig = df[df["sig_bh05"]].sort_values(["sampler", "region", "dose"])
    if sig.empty:
        print("  none")
    else:
        print(sig[["sampler", "dose", "region", "delta_dice", "p_bh"]].to_string(index=False))


if __name__ == "__main__":
    main()
