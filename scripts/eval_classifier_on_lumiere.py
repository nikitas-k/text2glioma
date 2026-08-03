"""Evaluate trained IDH classifiers on the LUMIERE external test cohort.

Tests the domain-generalisation hypothesis: does training on more
diverse samples (real + synth via the principled `balance_batches` +
`real` class weights recipe) shrink the internal-vs-external
performance gap relative to real-only training?

Data
----
* **Model checkpoints** — one or more ``best_model.pth`` files from the
  ``train_molecular_classifier.py`` grid. Each is evaluated on the same
  LUMIERE cohort so the results are directly comparable.
* **LUMIERE datalist** — produced by ``scripts/ingest_lumiere.py``.
  Contains one entry per (patient, session); we deduplicate to the
  earliest session per patient so a subject with 5 timepoints doesn't
  dominate the specificity estimate.
* **LUMIERE demographics CSV** — the standard release table with
  ``Patient`` and ``IDH (WT: wild type)`` columns. We join on the
  ``Patient`` field.

Label handling
--------------
The LUMIERE IDH column has five values:

    * ``WT`` / ``wt``                         -> 0 (wildtype), high confidence
    * ``R132H mut``                           -> 1 (mutant),   high confidence
    * ``IDH1 neg, Sequencing required``       -> ambiguous. IHC-negative for
      the canonical R132H mutation but non-R132H mutations (~10% of glioma
      IDH mutations) have not been sequenced. In routine glioma workups
      these are conventionally reported as *presumed WT* since R132H
      accounts for ~90% of IDH mutations in glioma. Controlled via
      ``--include_presumed_wt`` (default: exclude; strict mode).
    * ``na``                                  -> dropped from evaluation.

At the time of writing the strict-label cohort is 57 WT + 1 MUT (=58);
the "presumed WT" mode expands the WT half to 67.

Metrics
-------
Because the labelled cohort is 57:1 WT-to-MUT-heavy, AUROC has minimal
statistical power (a single positive point). We therefore report:

    * **Predicted probability distribution** on WT and on the single MUT
      subject — the primary read-out, plotted as a strip plot.
    * **Specificity @ threshold 0.5** — proportion of WT subjects
      correctly predicted <0.5. This is the direct domain-generalisation
      metric: if the model over-predicts MUT on external WT, that's an
      internal-domain-fitting failure.
    * **Mean predicted P(MUT)** on WT subjects — a continuous specificity
      analogue; lower is better.
    * **AUROC** — reported for completeness but flagged as underpowered.

Usage
-----
::

    python scripts/eval_classifier_on_lumiere.py \\
        --model_ckpts /g/data/vp06/$USER/text2glioma_classify_idh_only_cfg70_realw_balbat/idh/n_synth_*/seed_*/best_model.pth \\
        --lumiere_datalist /g/data/vp06/$USER/text2glioma_train/data/lumiere_ingested/datalist_lumiere.json \\
        --lumiere_csv data/LUMIERE-Demographics_Pathology.csv \\
        --out_dir results/lumiere_eval \\
        --device cuda

Add ``--include_presumed_wt`` to include the 10 IHC-negative cases in the WT
denominator. Add ``--baseline_ckpts <paths>`` to overlay a real-only baseline
grid for direct comparison. When multiple checkpoints share the same
``n_synth`` (across seeds) they are averaged in the summary CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai import transforms as T
from monai.data import Dataset
from monai.networks import nets
from torch.utils.data import DataLoader


_TARGET_SPATIAL = (160, 224, 160)
_N_SYNTH_RE = re.compile(r"n_synth_(\d+)")
_SEED_RE    = re.compile(r"seed_(\d+)")


# ---------------------------------------------------------------------
# CSV -> per-subject IDH label
# ---------------------------------------------------------------------

def _load_lumiere_labels(csv_path: Path, include_presumed_wt: bool
                         ) -> dict[str, int]:
    """Return a dict ``{Patient-XXX: 0|1}`` for subjects with a resolvable
    IDH status. Subjects with ``na`` are dropped; the 10 ``IDH1 neg,
    Sequencing required`` cases are included as WT only when
    ``include_presumed_wt`` is True.
    """
    df = pd.read_csv(csv_path)
    col_idh = "IDH (WT: wild type)"
    if col_idh not in df.columns:
        raise SystemExit(f"CSV missing expected column {col_idh!r}")
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        pat = str(row["Patient"]).strip()
        raw = str(row[col_idh]).strip()
        if raw.lower() in ("wt",):
            labels[pat] = 0
        elif raw.lower() in ("r132h mut", "mut", "mutant"):
            labels[pat] = 1
        elif "sequencing required" in raw.lower() or "idh1 neg" in raw.lower():
            if include_presumed_wt:
                labels[pat] = 0
        # else: 'na' or unrecognised -> drop
    return labels


# ---------------------------------------------------------------------
# Datalist -> first session per subject
# ---------------------------------------------------------------------

def _load_lumiere_items(datalist_path: Path,
                          labels: dict[str, int]) -> list[dict]:
    """Return items ``{image, subject, session, idh}`` for labelled patients,
    keeping only the earliest session per patient."""
    with datalist_path.open() as fh:
        dl = json.load(fh)
    # The ingest script writes the cohort under 'validation'.
    entries = dl.get("validation", []) or dl.get("training", []) or []
    if not entries:
        raise SystemExit(f"no entries found in {datalist_path}")
    # Sort by (subject, session) and keep first per subject.
    def _session_key(e):
        s = str(e.get("session", ""))
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 10**9
    entries_sorted = sorted(entries, key=lambda e: (e["subject"], _session_key(e)))
    seen: dict[str, dict] = {}
    for e in entries_sorted:
        subj = e["subject"]
        if subj in seen:
            continue
        if subj not in labels:
            continue
        seen[subj] = {
            "image":   e["image"],
            "subject": subj,
            "session": e.get("session", ""),
            "idh":     labels[subj],
        }
    return list(seen.values())


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def _build_transforms() -> T.Compose:
    """Match the training-time transforms used by
    ``train_molecular_classifier.py`` (skips CropForeground; that
    matters when synth samples were preprocessed to (160, 224, 160)
    but here we're on raw LUMIERE geometry that has already been
    registered to SRI24 upstream — same 160x224x160 target)."""
    return T.Compose([
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.SpatialPadd(keys=["image"], spatial_size=_TARGET_SPATIAL, mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=_TARGET_SPATIAL),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0,
            channel_wise=True, clip=True,
        ),
        T.ToTensord(keys=["image"]),
    ])


@torch.no_grad()
def _predict(model_ckpt: Path, items: list[dict], device: torch.device,
              dropout_prob: float, batch_size: int) -> np.ndarray:
    """Return P(MUT) per item in the same order."""
    model = nets.DenseNet121(
        spatial_dims=3, in_channels=4, out_channels=2,
        dropout_prob=dropout_prob,
    ).to(device)
    state = torch.load(str(model_ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    if not items:
        return np.zeros((0,), dtype=np.float32)
    ds = Dataset(data=items, transform=_build_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[float] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        p_mut = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p_mut.tolist())
    return np.asarray(probs, dtype=np.float32)


# ---------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------

def _summarise(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Return a dict of scalar metrics."""
    wt_mask  = labels == 0
    mut_mask = labels == 1
    n_wt  = int(wt_mask.sum())
    n_mut = int(mut_mask.sum())

    p_mut_on_wt  = float(preds[wt_mask].mean())  if n_wt  else float("nan")
    p_mut_on_mut = float(preds[mut_mask].mean()) if n_mut else float("nan")
    specificity  = float((preds[wt_mask] < 0.5).mean()) if n_wt else float("nan")

    # AUROC only if both classes present.
    auroc = float("nan")
    if n_wt >= 1 and n_mut >= 1:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(labels, preds))
        except Exception:
            pass

    return {
        "n_wt":         n_wt,
        "n_mut":        n_mut,
        "mean_p_mut_on_wt":  p_mut_on_wt,
        "mean_p_mut_on_mut": p_mut_on_mut,
        "specificity_at_0.5": specificity,
        "auroc":        auroc,
    }


def _parse_ckpt_id(path: Path) -> dict:
    """Extract ``n_synth``, ``seed`` from a checkpoint path."""
    ns  = _N_SYNTH_RE.search(str(path))
    sd  = _SEED_RE.search(str(path))
    return {
        "n_synth": int(ns.group(1)) if ns else -1,
        "seed":    int(sd.group(1)) if sd else -1,
    }


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_ckpts", type=Path, nargs="+", required=True,
                    help="One or more best_model.pth files. Path is parsed for "
                         "n_synth_<N>/seed_<S> tags; unrecognised paths get "
                         "n_synth=-1, seed=-1.")
    ap.add_argument("--baseline_ckpts", type=Path, nargs="+", default=None,
                    help="Optional real-only baseline checkpoints to overlay.")
    ap.add_argument("--lumiere_datalist", type=Path, required=True,
                    help="datalist_lumiere.json from scripts/ingest_lumiere.py.")
    ap.add_argument("--lumiere_csv", type=Path, required=True,
                    help="Path to LUMIERE-Demographics_Pathology.csv.")
    ap.add_argument("--include_presumed_wt", action="store_true",
                    help="Include 10 IHC-negative-but-not-sequenced cases as "
                         "WT. Default: strict, exclude them.")
    ap.add_argument("--dropout_prob", type=float, default=0.3)
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--device",       type=str, default="cuda")
    ap.add_argument("--out_dir",      type=Path, required=True)
    ap.add_argument("--label",        type=str, default="principled",
                    help="Display label for the main --model_ckpts group.")
    ap.add_argument("--baseline_label", type=str, default="real-only",
                    help="Display label for --baseline_ckpts.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load labels + items ------------------------------------
    labels = _load_lumiere_labels(args.lumiere_csv, args.include_presumed_wt)
    items  = _load_lumiere_items(args.lumiere_datalist, labels)
    y      = np.asarray([it["idh"] for it in items], dtype=int)
    print(f"[cohort] {len(items)} labelled subjects: "
          f"WT={int((y == 0).sum())}  MUT={int((y == 1).sum())}  "
          f"(include_presumed_wt={args.include_presumed_wt})",
          file=sys.stderr, flush=True)

    if len(items) == 0:
        raise SystemExit("no labelled LUMIERE subjects — check CSV / datalist "
                          "Patient ID matching.")

    # ---- Inference across all checkpoints -----------------------
    per_subject_rows: list[dict] = []
    per_run_rows:     list[dict] = []

    def _run_group(ckpts: list[Path], group: str) -> None:
        for ckpt in ckpts:
            tags = _parse_ckpt_id(ckpt)
            print(f"[run] {group}  n_synth={tags['n_synth']}  seed={tags['seed']}  "
                  f"ckpt={ckpt}", file=sys.stderr, flush=True)
            preds = _predict(ckpt, items, device, args.dropout_prob, args.batch_size)
            summary = _summarise(preds, y)
            per_run_rows.append({
                "group":   group,
                "n_synth": tags["n_synth"],
                "seed":    tags["seed"],
                "ckpt":    str(ckpt),
                **summary,
            })
            for it, p in zip(items, preds):
                per_subject_rows.append({
                    "group":   group,
                    "n_synth": tags["n_synth"],
                    "seed":    tags["seed"],
                    "subject": it["subject"],
                    "session": it["session"],
                    "idh":     it["idh"],
                    "p_mut":   float(p),
                })

    _run_group(list(args.model_ckpts), args.label)
    if args.baseline_ckpts:
        _run_group(list(args.baseline_ckpts), args.baseline_label)

    per_subject_df = pd.DataFrame(per_subject_rows)
    per_run_df     = pd.DataFrame(per_run_rows)

    # Aggregate per (group, n_synth): mean over seeds of the per-run summaries.
    agg = (per_run_df
           .groupby(["group", "n_synth"], as_index=False)
           .agg(n_seeds            =("seed", "nunique"),
                mean_p_mut_on_wt   =("mean_p_mut_on_wt",   "mean"),
                mean_p_mut_on_wt_sd=("mean_p_mut_on_wt",   "std"),
                specificity_mean   =("specificity_at_0.5", "mean"),
                specificity_sd     =("specificity_at_0.5", "std"),
                auroc_mean         =("auroc",              "mean"),
                auroc_sd           =("auroc",              "std"),
                n_wt               =("n_wt",               "first"),
                n_mut              =("n_mut",              "first"),
                mean_p_mut_on_mut  =("mean_p_mut_on_mut",  "mean"))
           .sort_values(["group", "n_synth"]))

    # ---- Save + report ------------------------------------------
    per_subject_df.to_csv(args.out_dir / "predictions_per_subject.csv", index=False)
    per_run_df.to_csv    (args.out_dir / "predictions_per_run.csv",     index=False)
    agg.to_csv           (args.out_dir / "summary_by_condition.csv",    index=False)

    print()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(agg[["group", "n_synth", "n_seeds",
                    "mean_p_mut_on_wt", "specificity_mean", "auroc_mean"]]
              .to_string(index=False))
    print()
    print(f"wrote {args.out_dir}/predictions_per_subject.csv")
    print(f"wrote {args.out_dir}/predictions_per_run.csv")
    print(f"wrote {args.out_dir}/summary_by_condition.csv")

    # ---- Figure -------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1: specificity vs n_synth per group
    ax = axes[0]
    for group, sub in agg.groupby("group"):
        # Drop the "unknown-tag" -1 rows.
        sub = sub[sub.n_synth >= 0].sort_values("n_synth")
        if sub.empty:
            continue
        ax.errorbar(
            sub.n_synth.replace(0, 0.5),   # log-friendly baseline placement
            sub.specificity_mean,
            yerr=sub.specificity_sd,
            marker="o", markersize=6, linewidth=1.6, capsize=3,
            label=group,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.4, 15000)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$n_{\rm synth}$ (0 shown at 0.5 for log axis)")
    ax.set_ylabel("Specificity (P(MUT) < 0.5 on WT)")
    ax.set_title("LUMIERE WT specificity")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: P(MUT) distribution per subject, one strip per (group, n_synth)
    ax = axes[1]
    conds = agg[["group", "n_synth"]].drop_duplicates()
    conds = conds[conds.n_synth >= 0].sort_values(["group", "n_synth"])
    labels_axis: list[str] = []
    for i, (_, r) in enumerate(conds.iterrows()):
        sub = per_subject_df[(per_subject_df.group == r.group)
                             & (per_subject_df.n_synth == r.n_synth)]
        # WT strip
        wt_p = sub.loc[sub.idh == 0, "p_mut"].to_numpy()
        mut_p = sub.loc[sub.idh == 1, "p_mut"].to_numpy()
        x = np.full_like(wt_p, i, dtype=float) + np.random.default_rng(0).uniform(
            -0.15, 0.15, size=len(wt_p),
        )
        ax.scatter(x, wt_p, s=14, alpha=0.35, color="#1f77b4",
                    edgecolors="white", linewidths=0.4,
                    label="WT" if i == 0 else None)
        if len(mut_p):
            ax.scatter(np.full_like(mut_p, i), mut_p, s=60, marker="^",
                        color="#d62728", edgecolors="black", linewidths=0.6,
                        label="MUT" if i == 0 else None, zorder=4)
        labels_axis.append(f"{r.group}\nn={int(r.n_synth)}")

    ax.set_xticks(range(len(labels_axis)))
    ax.set_xticklabels(labels_axis, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(MUT)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="threshold")
    ax.set_title(f"LUMIERE per-subject predictions (n_seeds averaged over seeds)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(args.out_dir / "lumiere_eval.pdf"), dpi=200)
    fig.savefig(str(args.out_dir / "lumiere_eval.png"), dpi=200)
    plt.close(fig)
    print(f"wrote {args.out_dir}/lumiere_eval.pdf")


if __name__ == "__main__":
    main()
