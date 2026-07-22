"""Tier-1 sanity check for the trained MolecularClassConditioning head.

Loads ``best_molecular_head.pth`` and quantifies whether the six embedding
rows moved meaningfully from their Gaussian initialisation, and whether
the three states within each field (IDH / MGMT) developed distinct
representations. Purely a post-hoc analysis on the saved checkpoint —
no GPU required, runs in seconds.

Emits:

    * A JSON summary (``molecular_head_report.json``) with per-row L2
      norms, pairwise cosine-similarity matrices, and Δ-from-init
      distances.
    * A 3-panel PNG (``molecular_head_report.png``):
        (i)   bar chart of per-row L2 norms with init reference,
        (ii)  IDH 3x3 cosine-similarity heatmap,
        (iii) MGMT 3x3 cosine-similarity heatmap.

Usage
-----
::

    python scripts/analyse_molecular_head.py \\
        --ckpt   /path/to/best_molecular_head.pth \\
        --hidden_dim 768 \\
        --init_seed 42 \\
        --init_std 0.02 \\
        --out_dir results/mol_head_report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Local import
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from text2glioma.training.molecular_conditioning import (  # noqa: E402
    MolecularClassConditioning,
    IDH_WILDTYPE, IDH_MUTANT, IDH_UNKNOWN,
    MGMT_UNMETHYLATED, MGMT_METHYLATED, MGMT_UNKNOWN,
)


_IDH_LABELS  = ["wildtype", "mutant", "unknown"]
_MGMT_LABELS = ["unmethylated", "methylated", "unknown"]


def _pairwise_cosine(w: torch.Tensor) -> np.ndarray:
    """(N, D) tensor -> (N, N) cosine-similarity matrix."""
    return F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=-1).cpu().numpy()


def _per_row_norms(w: torch.Tensor) -> np.ndarray:
    return w.norm(dim=1).cpu().numpy()


def _delta_from_init(trained: torch.Tensor, init: torch.Tensor) -> np.ndarray:
    return (trained - init).norm(dim=1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to best_molecular_head.pth")
    ap.add_argument("--hidden_dim", type=int, default=768)
    ap.add_argument("--init_seed", type=int, default=42,
                    help="Torch seed used at training time. Ensures the "
                         "reconstructed init matches the actual training init "
                         "bit-for-bit. Use the value of --seed passed to "
                         "train_stage2_ddp.py; defaults to 42.")
    ap.add_argument("--init_std", type=float, default=0.02,
                    help="Standard deviation used at init (matches "
                         "MolecularClassConditioning default).")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load trained weights ────────────────────────────────────────
    trained_head = MolecularClassConditioning(
        hidden_dim=args.hidden_dim,
        dropout_to_unknown_p=0.0,
        init_std=args.init_std,
    )
    state = torch.load(str(args.ckpt), map_location="cpu")
    trained_head.load_state_dict(state, strict=True)
    trained_head.eval()

    trained_idh  = trained_head.idh_embedding.weight.detach()
    trained_mgmt = trained_head.mgmt_embedding.weight.detach()

    # ── Reconstruct init (same seed) ────────────────────────────────
    torch.manual_seed(int(args.init_seed))
    init_head = MolecularClassConditioning(
        hidden_dim=args.hidden_dim,
        dropout_to_unknown_p=0.0,
        init_std=args.init_std,
    )
    init_idh  = init_head.idh_embedding.weight.detach()
    init_mgmt = init_head.mgmt_embedding.weight.detach()

    # ── Metrics ─────────────────────────────────────────────────────
    expected_init_norm = args.init_std * math.sqrt(args.hidden_dim)

    report = {
        "hidden_dim":         args.hidden_dim,
        "init_seed":          args.init_seed,
        "init_std":           args.init_std,
        "expected_init_norm": float(expected_init_norm),
        "idh": {
            "labels":         _IDH_LABELS,
            "trained_norms":  _per_row_norms(trained_idh).tolist(),
            "init_norms":     _per_row_norms(init_idh).tolist(),
            "delta_from_init":_delta_from_init(trained_idh, init_idh).tolist(),
            "trained_cosine": _pairwise_cosine(trained_idh).tolist(),
            "init_cosine":    _pairwise_cosine(init_idh).tolist(),
        },
        "mgmt": {
            "labels":         _MGMT_LABELS,
            "trained_norms":  _per_row_norms(trained_mgmt).tolist(),
            "init_norms":     _per_row_norms(init_mgmt).tolist(),
            "delta_from_init":_delta_from_init(trained_mgmt, init_mgmt).tolist(),
            "trained_cosine": _pairwise_cosine(trained_mgmt).tolist(),
            "init_cosine":    _pairwise_cosine(init_mgmt).tolist(),
        },
    }

    out_json = args.out_dir / "molecular_head_report.json"
    with out_json.open("w") as fh:
        json.dump(report, fh, indent=2)

    # ── Pretty-print to stdout ──────────────────────────────────────
    def _fmt_matrix(m, labels):
        header = "        " + "  ".join(f"{l:>12}" for l in labels)
        lines = [header]
        for i, l in enumerate(labels):
            row = "  ".join(f"{m[i][j]:>12.4f}" for j in range(len(labels)))
            lines.append(f"{l:>7}  {row}")
        return "\n".join(lines)

    print(f"Loaded {args.ckpt}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Expected Gaussian-init norm: {expected_init_norm:.4f}  (std={args.init_std}, dim={args.hidden_dim})")
    print()
    for name, block, labels in [("IDH", report["idh"], _IDH_LABELS),
                                 ("MGMT", report["mgmt"], _MGMT_LABELS)]:
        print(f"── {name} embeddings ──")
        for i, l in enumerate(labels):
            tn = block["trained_norms"][i]
            init_n = block["init_norms"][i]
            dl = block["delta_from_init"][i]
            print(f"  {l:>13}: trained_norm={tn:6.4f}  init_norm={init_n:6.4f}  "
                  f"‖trained-init‖={dl:6.4f}  norm_ratio={tn/init_n:5.2f}x")
        print(f"\n  Trained pairwise cosine similarity:")
        print(_fmt_matrix(block["trained_cosine"], labels))
        print(f"\n  Init pairwise cosine similarity (for reference; expected ~0):")
        print(_fmt_matrix(block["init_cosine"], labels))
        print()

    # ── Figure ──────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # (i) Norm bar chart
        ax = axes[0]
        all_labels = [f"IDH-{l}" for l in _IDH_LABELS] + [f"MGMT-{l}" for l in _MGMT_LABELS]
        trained_norms = report["idh"]["trained_norms"] + report["mgmt"]["trained_norms"]
        init_norms    = report["idh"]["init_norms"]    + report["mgmt"]["init_norms"]
        x = np.arange(len(all_labels))
        width = 0.4
        ax.bar(x - width/2, init_norms,    width, label="init",    color="#bbbbbb")
        ax.bar(x + width/2, trained_norms, width, label="trained", color="#1f77b4")
        ax.axhline(expected_init_norm, color="red", linestyle="--", linewidth=0.8,
                   label=f"E[‖w‖] at init ≈ {expected_init_norm:.2f}")
        ax.set_xticks(x); ax.set_xticklabels(all_labels, rotation=30, ha="right")
        ax.set_ylabel("L2 norm")
        ax.set_title("Embedding vector norms")
        ax.legend(fontsize=8, loc="upper left")

        # (ii, iii) Cosine heatmaps
        for ax, block, labels, title in [
            (axes[1], report["idh"],  _IDH_LABELS,  "IDH pairwise cosine (trained)"),
            (axes[2], report["mgmt"], _MGMT_LABELS, "MGMT pairwise cosine (trained)"),
        ]:
            m = np.array(block["trained_cosine"])
            im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center",
                            color="white" if abs(m[i][j]) > 0.5 else "black",
                            fontsize=9)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.7)

        fig.tight_layout()
        out_png = args.out_dir / "molecular_head_report.png"
        fig.savefig(str(out_png), dpi=150)
        print(f"\nWrote figure: {out_png}")
    except ImportError:
        print("\n[warn] matplotlib not available, skipping figure")

    print(f"Wrote JSON:   {out_json}")


if __name__ == "__main__":
    main()
