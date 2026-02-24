#!/usr/bin/env python3
"""Scatter plot of T1 FWHM vs WM/GM CNR with example axial slices inset,
highlighting subjects from the tails of good and poor quality clusters.

Usage (on Gadi or wherever NIfTIs live):
  python scripts/plot_scatter_with_slices.py \
      --image_dir /g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr \
      --n_examples 5 \
      --output audit_figures/scatter_with_slices.png

Requires: nibabel, scikit-learn, matplotlib, pandas, numpy
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit_csv", default="text2glioma_audit.csv")
    p.add_argument("--cnr_csv", default="cnr_audit.csv")
    p.add_argument("--meta_csv", default="2025_cancer_dataset_v3.csv")
    p.add_argument("--image_dir", required=True,
                   help="Directory containing nnUNetv2-XXXXX.nii.gz files")
    p.add_argument("--n_examples", type=int, default=5,
                   help="Number of tail examples per cluster (default: 5)")
    p.add_argument("--channel", type=int, default=0,
                   help="Volume channel to display (0=T1,1=T1CE,2=T2,3=FLAIR)")
    p.add_argument("--output", default="audit_figures/scatter_with_slices.png")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────
def load_slice(nifti_path: pathlib.Path, channel: int = 0) -> np.ndarray:
    """Load a representative axial slice from a 4-channel NIfTI.

    Picks the axial slice with the maximum foreground area to show a
    representative view of the brain (avoiding near-empty slices).
    """
    import nibabel as nib
    img = nib.load(str(nifti_path))
    vol = img.get_fdata()                       # (H, W, D, C)
    ch = vol[..., channel]                      # (H, W, D)

    # pick axial slice with most brain voxels
    mask = ch > (ch.max() * 0.05)
    counts = mask.sum(axis=(0, 1))              # per-slice foreground count
    z = int(np.argmax(counts))
    sl = ch[:, :, z].T                          # transpose for imshow (rows=AP)

    # robust windowing (1st–99th percentile)
    lo, hi = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
    sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
    return sl


def draw_gmm_ellipses(ax, gm, scaler, good_idx, bad_idx):
    """Draw GMM 1σ/2σ ellipses and cluster centres."""
    for k in range(2):
        mean_std = gm.means_[k]
        cov_std = gm.covariances_[k]
        s_diag = np.diag(scaler.scale_)
        cov_orig = s_diag @ cov_std @ s_diag
        mean_orig = scaler.inverse_transform(mean_std.reshape(1, -1)).ravel()
        eigvals, eigvecs = np.linalg.eigh(cov_orig)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ec = "#2ca02c" if k == good_idx else "#d62728"
        for n_std in [1, 2]:
            w, h = 2 * n_std * np.sqrt(eigvals)
            lbl = ("good" if k == good_idx else "poor") if n_std == 1 else None
            ell = Ellipse(mean_orig, w, h, angle=angle,
                          facecolor="none", edgecolor=ec,
                          linestyle="--", linewidth=1.2,
                          alpha=0.7 if n_std == 1 else 0.35,
                          label=f"GMM {lbl} ($\\mu\\pm{n_std}\\sigma$)" if lbl else None)
            ax.add_patch(ell)
        ax.plot(mean_orig[0], mean_orig[1], "*", markersize=14,
                markeredgecolor="k", markeredgewidth=0.8, color=ec)


def draw_decision_boundary(ax, gm, scaler, good_idx, bad_idx, x_range, y_range):
    """Draw the GMM decision boundary contour."""
    xx = np.linspace(*x_range, 300)
    yy = np.linspace(*y_range, 300)
    XX, YY = np.meshgrid(xx, yy)
    grid_std = scaler.transform(np.column_stack([XX.ravel(), YY.ravel()]))
    prob = gm.predict_proba(grid_std)
    ZZ = (prob[:, bad_idx] - prob[:, good_idx]).reshape(XX.shape)
    ax.contour(XX, YY, ZZ, levels=[0], colors="k",
               linewidths=1.2, linestyles="--")


# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    image_dir = pathlib.Path(args.image_dir)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load & merge ──────────────────────────────────────────────────
    audit = pd.read_csv(args.audit_csv)
    cnr = pd.read_csv(args.cnr_csv)
    meta = pd.read_csv(args.meta_csv)
    audit["nnunet_id"] = audit["image"].str.extract(r"(nnUNetv2-\d+)")

    merged = (
        meta[["Collection", "DataID"]]
        .merge(cnr, left_on="DataID", right_on="nnunet_id", how="inner")
        .merge(audit[["nnunet_id", "T1_fwhm_mean", "T1_hf_ratio"]],
               on="nnunet_id", how="inner")
    )
    merged["Collection"] = merged["Collection"].fillna("Unknown")
    print(f"Merged: {len(merged)} subjects")

    # ── GMM on FWHM + CNR ────────────────────────────────────────────
    X = merged[["T1_fwhm_mean", "T1_wmgm_cnr"]].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    gm = GaussianMixture(n_components=2, random_state=42,
                          covariance_type="full", n_init=5)
    gm.fit(X_std)
    labels = gm.predict(X_std)

    centres = scaler.inverse_transform(gm.means_)
    # "good" = higher CNR
    good_idx = int(np.argmax(centres[:, 1]))
    bad_idx = int(np.argmin(centres[:, 1]))
    merged["cluster"] = np.where(labels == good_idx, "good", "poor")

    n_good = int((labels == good_idx).sum())
    n_poor = int((labels == bad_idx).sum())
    print(f"Good:  FWHM={centres[good_idx, 0]:.2f}, "
          f"CNR={centres[good_idx, 1]:.3f}, n={n_good}")
    print(f"Poor:  FWHM={centres[bad_idx, 0]:.2f}, "
          f"CNR={centres[bad_idx, 1]:.3f}, n={n_poor}")

    # ── select tail subjects ──────────────────────────────────────────
    # Good tail: highest CNR subjects
    good_tail = merged[merged["cluster"] == "good"].nlargest(
        args.n_examples, "T1_wmgm_cnr"
    )
    # Poor tail: lowest CNR subjects
    poor_tail = merged[merged["cluster"] == "poor"].nsmallest(
        args.n_examples, "T1_wmgm_cnr"
    )

    print(f"\nGood tail ({len(good_tail)} subjects, highest CNR):")
    for _, r in good_tail.iterrows():
        print(f"  {r['nnunet_id']}  CNR={r['T1_wmgm_cnr']:.3f}  "
              f"FWHM={r['T1_fwhm_mean']:.2f}  [{r['Collection']}]")
    print(f"Poor tail ({len(poor_tail)} subjects, lowest CNR):")
    for _, r in poor_tail.iterrows():
        print(f"  {r['nnunet_id']}  CNR={r['T1_wmgm_cnr']:.3f}  "
              f"FWHM={r['T1_fwhm_mean']:.2f}  [{r['Collection']}]")

    # ── load slices ───────────────────────────────────────────────────
    def get_slice(row):
        nii = image_dir / f"{row['nnunet_id']}.nii.gz"
        if not nii.exists():
            print(f"  WARNING: {nii} not found, skipping")
            return None
        return load_slice(nii, channel=args.channel)

    good_slices = [(r, get_slice(r)) for _, r in good_tail.iterrows()]
    poor_slices = [(r, get_slice(r)) for _, r in poor_tail.iterrows()]
    good_slices = [(r, s) for r, s in good_slices if s is not None]
    poor_slices = [(r, s) for r, s in poor_slices if s is not None]

    if not good_slices and not poor_slices:
        print("ERROR: No NIfTI files found. Check --image_dir.")
        return

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 4, 1.2],
                          hspace=0.08)
    gs_top = gs[0].subgridspec(1, max(len(good_slices), 1), wspace=0.05)
    gs_bot = gs[2].subgridspec(1, max(len(poor_slices), 1), wspace=0.05)
    ax_scatter = fig.add_subplot(gs[1])

    # ── scatter ───────────────────────────────────────────────────────
    COLOURS = {
        "UCSF-PDGM": "#1f77b4", "UPENN-GBM": "#ff7f0e",
        "TCGA-LGG": "#2ca02c", "TCGA-GBM": "#d62728", "Unknown": "#7f7f7f",
    }
    MARKERS = {
        "UCSF-PDGM": "o", "UPENN-GBM": "s",
        "TCGA-LGG": "^", "TCGA-GBM": "D", "Unknown": "x",
    }
    for col in ["UCSF-PDGM", "UPENN-GBM", "TCGA-LGG", "TCGA-GBM", "Unknown"]:
        sub = merged[merged["Collection"] == col]
        ax_scatter.scatter(
            sub["T1_fwhm_mean"], sub["T1_wmgm_cnr"],
            c=COLOURS[col], marker=MARKERS[col],
            s=16, alpha=0.45, linewidths=0.2, edgecolors="k",
            label=f"{col} (n={len(sub)})",
        )

    # GMM ellipses & decision boundary
    draw_gmm_ellipses(ax_scatter, gm, scaler, good_idx, bad_idx)
    draw_decision_boundary(
        ax_scatter, gm, scaler, good_idx, bad_idx,
        x_range=(merged["T1_fwhm_mean"].min() - 0.5,
                 merged["T1_fwhm_mean"].max() + 0.5),
        y_range=(merged["T1_wmgm_cnr"].min() - 0.1,
                 merged["T1_wmgm_cnr"].max() + 0.1),
    )

    # highlight tail subjects on scatter
    for row, _ in good_slices:
        ax_scatter.scatter(row["T1_fwhm_mean"], row["T1_wmgm_cnr"],
                           s=80, facecolors="none", edgecolors="#2ca02c",
                           linewidths=2, zorder=5)
    for row, _ in poor_slices:
        ax_scatter.scatter(row["T1_fwhm_mean"], row["T1_wmgm_cnr"],
                           s=80, facecolors="none", edgecolors="#d62728",
                           linewidths=2, zorder=5)

    ax_scatter.set_xlabel("T1 FWHM mean (mm)", fontsize=12)
    ax_scatter.set_ylabel("T1 WM/GM CNR", fontsize=12)
    ax_scatter.set_title("T1 Image Quality: FWHM vs CNR with Example Slices",
                         fontsize=14, fontweight="bold")
    ax_scatter.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_scatter.grid(True, alpha=0.2)

    # ── slice strips ──────────────────────────────────────────────────
    # Top row = good tail (high CNR)
    for i, (row, sl) in enumerate(good_slices):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(sl, cmap="gray", aspect="equal", interpolation="none")
        ax.set_title(f"{row['nnunet_id']}\n"
                     f"CNR={row['T1_wmgm_cnr']:.2f}  FWHM={row['T1_fwhm_mean']:.1f}",
                     fontsize=7, color="#2ca02c", fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#2ca02c")
            spine.set_linewidth(2)

    if good_slices:
        fig.text(0.02, 0.92, f"HIGH CNR tail\n(n={len(good_slices)})",
                 fontsize=10, fontweight="bold", color="#2ca02c",
                 ha="left", va="center")

    # Bottom row = poor tail (low CNR)
    for i, (row, sl) in enumerate(poor_slices):
        ax = fig.add_subplot(gs_bot[0, i])
        ax.imshow(sl, cmap="gray", aspect="equal", interpolation="none")
        ax.set_title(f"{row['nnunet_id']}\n"
                     f"CNR={row['T1_wmgm_cnr']:.2f}  FWHM={row['T1_fwhm_mean']:.1f}",
                     fontsize=7, color="#d62728", fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#d62728")
            spine.set_linewidth(2)

    if poor_slices:
        fig.text(0.02, 0.08, f"LOW CNR tail\n(n={len(poor_slices)})",
                 fontsize=10, fontweight="bold", color="#d62728",
                 ha="left", va="center")

    # save
    for ext in ("png", "pdf"):
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
