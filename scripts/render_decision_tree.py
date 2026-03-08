#!/usr/bin/env python3
"""Render the Stage-1 VAE diagnostic decision tree as a publication-ready figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTDIR = Path("/Users/nk233/text2glioma/audit_figures")


def rounded_box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333",
                lw=1.5, fontsize=8, fontweight="normal", text_color="#222",
                alpha=1.0, style="round,pad=0.02", ha="center"):
    """Draw a rounded rectangle with centered text."""
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style, facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight,
            color=text_color, ha=ha, va="center", zorder=3,
            linespacing=1.4, family="sans-serif",
            wrap=False)


def diamond(ax, x, y, w, h, text, fc="#fff3cd", ec="#856404",
            fontsize=7.5, text_color="#333"):
    """Draw a diamond (decision node)."""
    hw, hh = w/2, h/2
    verts = [(x, y+hh), (x+hw, y), (x, y-hh), (x-hw, y), (x, y+hh)]
    poly = plt.Polygon(verts, closed=True, facecolor=fc, edgecolor=ec,
                       linewidth=1.5, zorder=2)
    ax.add_patch(poly)
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold",
            color=text_color, ha="center", va="center", zorder=3,
            linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label="", color="#555", lw=1.2,
          fontsize=7, label_color="#444"):
    """Draw an arrow with optional label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0"),
                zorder=1)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=fontsize, color=label_color,
                ha="left", va="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 20))
    ax.set_xlim(-3.5, 21)
    ax.set_ylim(-5, 19)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(10, 18.2, "Stage-1 VAE Training: Diagnostic Decision Tree",
            fontsize=18, fontweight="bold", ha="center", va="center",
            color="#2c3e50")
    ax.text(10, 17.7, "Why is training taking so long?",
            fontsize=12, ha="center", va="center", color="#7f8c8d",
            style="italic")

    # Context note
    ax.text(10, 17.25,
            "This is one iteration of many \u2014 each cycle requires diagnosing the failure mode,\n"
            "implementing a fix, waiting 1\u20132 days in the Gadi HPC queue, and evaluating results.",
            fontsize=9, ha="center", va="center", color="#555",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec="#d4ac0d",
                      lw=1.2, alpha=0.9))

    # ── Row 0: Problem ──
    rounded_box(ax, 10, 16.8, 5.5, 0.8,
                "Reconstructions too smooth\nVAE outputs blurry after 1000 epochs",
                fc="#ff6b6b", ec="#c0392b", fontsize=9, fontweight="bold",
                text_color="white")
    ax.text(13.2, 16.8, "Fri 20 Feb",
            fontsize=8.5, fontweight="bold", color="#8e44ad", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5eef8", ec="#8e44ad", lw=1))

    arrow(ax, 10, 16.4, 10, 15.8)

    # ── Row 1: First decision ──
    diamond(ax, 10, 15.3, 3.5, 1.0,
            "Is it a loss\nweighting issue?")

    arrow(ax, 10, 14.8, 10, 14.2, label="Hypothesis: L1 dominates")

    # ── Row 2: First fix attempt ──
    rounded_box(ax, 10, 13.7, 5.5, 0.9,
                "Attempt 1: Increased weights 5\u00d7\n"
                "perceptual: 0.002 \u2192 0.01\n"
                "adversarial: 0.005 \u2192 0.025",
                fc="#dfe6e9", ec="#636e72", fontsize=8)
    ax.text(13.2, 13.7, "Sat 21 Feb",
            fontsize=8.5, fontweight="bold", color="#8e44ad", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5eef8", ec="#8e44ad", lw=1))
    ax.text(13.2, 13.25, "~1\u20132 day Gadi queue wait",
            fontsize=7.5, color="#7f8c8d", ha="left", va="center", style="italic")

    arrow(ax, 10, 13.25, 10, 12.7, label="Retrained ~1000 epochs on Gadi H200 \u00d74")

    # ── Row 3: Still smooth? ──
    diamond(ax, 10, 12.2, 3.0, 0.9,
            "Still smooth?")

    # ── LEFT BRANCH: Data quality audit ──
    arrow(ax, 8.5, 12.2, 5, 11.5, label="Yes \u2192 check data")

    rounded_box(ax, 5, 10.9, 5.0, 0.9,
                "Built audit pipeline (audit_datalist.py)\n"
                "\u2022 3D FFT spectral sharpness\n"
                "\u2022 Laplacian variance\n"
                "\u2022 FWHM via Connectome Workbench",
                fc="#dfe6e9", ec="#636e72", fontsize=7.5)

    arrow(ax, 5, 10.45, 5, 9.7, label="Audited N=1510 subjects")

    rounded_box(ax, 5, 9.15, 5.0, 0.9,
                "FWHM Results (mm):\n"
                "T1=11.6  T1CE=7.7  T2=7.2  FLAIR=10.0\n"
                "Wide variation across collections",
                fc="#e8f8f5", ec="#1abc9c", fontsize=8)

    # ── RIGHT BRANCH: Discriminator collapse ──
    arrow(ax, 11.5, 12.2, 15, 11.5, label="Re-examine logs")

    diamond(ax, 15, 11.0, 3.5, 0.9,
            "Discriminator\ncollapsed?")

    arrow(ax, 15, 10.55, 15, 9.9, label="d_loss \u2192 0.003 \u2717")

    rounded_box(ax, 15, 9.4, 5.0, 0.8,
                "Disc wins too easily \u2192 zero gradient\n"
                "\u2192 generator defaults to pure L1 smoothing\n"
                "Edge sharpness dropped 83% over 25k steps",
                fc="#f39c12", ec="#e67e22", fontsize=7.5, fontweight="bold",
                text_color="white")

    arrow(ax, 15, 9.0, 15, 8.3)

    rounded_box(ax, 15, 7.85, 4.5, 0.7,
                "Increased adv_weight 10\u00d7\n0.025 \u2192 0.25",
                fc="#ebf5fb", ec="#2980b9", fontsize=8.5, fontweight="bold")

    # ── Attempt 2: retrain with adv_weight fix ──
    arrow(ax, 5, 8.7, 10, 7.5)
    arrow(ax, 15, 7.5, 10, 7.5)

    rounded_box(ax, 10, 7.0, 6.0, 0.8,
                "Attempt 2: Retrained with adv_weight = 0.25\n"
                "\u2022 perceptual_weight = 0.01  \u2022 N=1510 (full dataset)",
                fc="#3498db", ec="#2471a3", fontsize=8.5, fontweight="bold",
                text_color="white")
    ax.text(13.5, 7.0, "Sun 23 Feb",
            fontsize=8.5, fontweight="bold", color="#8e44ad", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5eef8", ec="#8e44ad", lw=1))
    ax.text(13.5, 6.55, "~1\u20132 day Gadi queue wait",
            fontsize=7.5, color="#7f8c8d", ha="left", va="center", style="italic")

    arrow(ax, 10, 6.6, 10, 6.1)

    # ── Converge: deeper diagnosis ──
    diamond(ax, 10, 5.6, 4.0, 1.0,
            "Root cause of\nblur?")

    # ── LEFT: CNR audit (deferred) ──
    arrow(ax, 8.0, 5.6, 4.5, 4.9, label="Data quality?")

    rounded_box(ax, 4.5, 4.3, 5.0, 1.0,
                "CNR Audit: WM/GM tissue contrast\n"
                "\u2022 3-comp GMM on non-tumour T1 voxels\n"
                "\u2022 CNR = |\u03bcWM \u2212 \u03bcGM| / \u221a(0.5(\u03c3\u00b2WM + \u03c3\u00b2GM))\n"
                "\u2022 CNR \u2265 1.75 \u2192 1323 subjects kept",
                fc="#e8f8f5", ec="#1abc9c", fontsize=7.5)

    arrow(ax, 4.5, 3.8, 4.5, 3.1)

    rounded_box(ax, 4.5, 2.7, 4.5, 0.7,
                "CNR-filtered datalist ready\n1510 \u2192 1323 subjects",
                fc="#fef9e7", ec="#d4ac0d", fontsize=8, fontweight="bold")

    ax.text(4.5, 2.15, "Available if architecture\nchange is insufficient",
            fontsize=7.5, color="#7f8c8d", ha="center", va="top",
            style="italic", linespacing=1.3)

    # ── RIGHT: Architecture (primary path) ──
    arrow(ax, 12.0, 5.6, 15.5, 4.9, label="Architecture too weak?")

    rounded_box(ax, 15.5, 4.3, 5.0, 1.0,
                "Bottleneck has no attention \u2192\ncannot model long-range dependencies\n"
                "\u2022 Channels: [32,64,128,128] (narrow)\n"
                "\u2022 attention_levels: all False",
                fc="#fce4ec", ec="#c0392b", fontsize=7.5)

    arrow(ax, 15.5, 3.8, 15.5, 3.1)

    rounded_box(ax, 15.5, 2.5, 5.5, 1.0,
                "Architecture upgrade:\n"
                "\u2022 Channels: [64, 128, 256, 512]\n"
                "\u2022 attention_levels: [F, F, F, True]\n"
                "\u2022 encoder + decoder nonlocal attention\n"
                "\u2022 xformers flash attention (O(N) memory)",
                fc="#3498db", ec="#2471a3", fontsize=8, fontweight="bold",
                text_color="white")
    ax.text(18.7, 2.5, "Mon 24 Feb",
            fontsize=8.5, fontweight="bold", color="#8e44ad", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5eef8", ec="#8e44ad", lw=1))

    arrow(ax, 15.5, 2.0, 15.5, 1.2)

    # Memory analysis note
    rounded_box(ax, 15.5, 0.8, 5.0, 0.7,
                "Memory analysis: +4 GB for attention\n"
                "at 20\u00d728\u00d720 (\u224811k tokens) \u2192 fits H200 80 GB",
                fc="#dfe6e9", ec="#636e72", fontsize=7.5)

    # ── Retrain box ──
    arrow(ax, 15.5, 0.45, 10, -0.4)

    rounded_box(ax, 10, -0.9, 7.0, 0.9,
                "Attempt 3: Architecture fix:\n"
                "\u2022 Wider encoder [64,128,256,512]  \u2022 Attention at bottleneck\n"
                "\u2022 Flash attention via xformers  \u2022 N=1510 (full dataset)",
                fc="#3498db", ec="#2471a3", fontsize=8.5, fontweight="bold",
                text_color="white")
    ax.text(14.0, -0.9, "Tue 25 Feb",
            fontsize=8.5, fontweight="bold", color="#8e44ad", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5eef8", ec="#8e44ad", lw=1))
    ax.text(14.0, -1.35, "~1\u20132 day Gadi queue wait",
            fontsize=7.5, color="#7f8c8d", ha="left", va="center", style="italic")

    arrow(ax, 10, -1.35, 10, -1.9)

    # ── Monitor outcomes ──
    diamond(ax, 10, -2.3, 3.5, 0.8,
            "Monitor\nconvergence")

    # Three outcomes
    arrow(ax, 8.25, -2.3, 4, -3.0, label="Sharper recons")
    rounded_box(ax, 4, -3.4, 4.5, 0.6,
                "[OK] Attention captures edges\nProceed to Stage 2 LDM",
                fc="#2ecc71", ec="#27ae60", fontsize=8, fontweight="bold",
                text_color="white")

    arrow(ax, 10, -2.7, 10, -3.3, label="Still smooth")
    rounded_box(ax, 10, -3.7, 4.5, 0.7,
                "[Next] Apply CNR filter (N=1323)\n+ frequency loss (FFL / Sobel)",
                fc="#e74c3c", ec="#c0392b", fontsize=7.5, fontweight="bold",
                text_color="white")

    arrow(ax, 11.75, -2.3, 16, -3.0, label="OOM / diverges")
    rounded_box(ax, 16, -3.4, 4.5, 0.6,
                "[!] Drop attention or\nreduce batch size to 1",
                fc="#e74c3c", ec="#c0392b", fontsize=7.5, fontweight="bold",
                text_color="white")

    # ── TIMELINE SIDEBAR ──────────────────────────────────────────
    tl_x = -2.0
    tl_events = [
        (16.8, "Fri 20 Feb",  "Problem identified"),
        (13.7, "Sat 21 Feb",  "Attempt 1: loss weights\n+1\u20132 day queue"),
        (12.2, "Sun\u2013Mon",    "Analysis: data audit\n& TensorBoard logs"),
        (7.0,  "Sun 23 Feb",  "Attempt 2: adv_weight 0.25\n+1\u20132 day queue"),
        (5.6,  "Mon 24 Feb",  "Deeper diagnosis:\nCNR audit + arch review"),
        (-0.9, "Tue 25 Feb",  "Attempt 3: architecture\n+1\u20132 day queue"),
    ]
    ax.plot([tl_x, tl_x], [tl_events[-1][0], tl_events[0][0]],
            color="#8e44ad", lw=2.5, solid_capstyle="round", zorder=1)

    for y, date, desc in tl_events:
        ax.plot(tl_x, y, "o", color="#8e44ad", markersize=10, zorder=3)
        ax.plot(tl_x, y, "o", color="white", markersize=6, zorder=4)
        ax.text(tl_x - 0.15, y + 0.25, date,
                fontsize=9, fontweight="bold", color="#8e44ad",
                ha="right", va="bottom")
        ax.text(tl_x - 0.15, y - 0.15, desc,
                fontsize=7.5, color="#555", ha="right", va="top",
                linespacing=1.3)
        connector_target = 3.0 if y > 5 else 4.0
        ax.plot([tl_x + 0.15, connector_target], [y, y],
                color="#8e44ad", lw=0.8, ls=":", alpha=0.5, zorder=1)

    plt.tight_layout()
    out_path = OUTDIR / "decision_tree.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save as PDF for presentations
    out_pdf = OUTDIR / "decision_tree.pdf"
    fig2, ax2 = plt.subplots(figsize=(24, 20))
    ax2.set_xlim(-1, 21)
    ax2.set_ylim(-5, 19)
    ax2.set_aspect("equal")
    ax2.axis("off")
    img = plt.imread(out_path)
    ax2.imshow(img)
    ax2.axis("off")
    fig2.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0)
    plt.close(fig2)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
