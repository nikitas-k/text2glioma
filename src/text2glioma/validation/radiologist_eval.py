"""Radiologist evaluation prep: Turing test sets, quality rating forms."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slice rendering
# ---------------------------------------------------------------------------

def _vol_to_png(
    vol: np.ndarray,
    label: Optional[np.ndarray],
    slice_idx: Optional[int],
    output_path: str,
    channels: Tuple[int, ...] = (1, 3),  # T1CE + FLAIR
    overlay_alpha: float = 0.3,
) -> str:
    """Render mid-axial slices of selected channels + optional label overlay.

    Saves a horizontal concatenation of the channels as PNG.
    """
    from PIL import Image, ImageDraw

    ch_slices: List[np.ndarray] = []
    s = slice_idx if slice_idx is not None else vol.shape[1] // 2 if vol.ndim == 4 else vol.shape[0] // 2

    for ch in channels:
        sl = vol[ch, s] if vol.ndim == 4 else vol[s]
        vmin, vmax = float(sl.min()), float(sl.max())
        if vmax - vmin > 1e-8:
            sl = (sl - vmin) / (vmax - vmin)
        ch_slices.append((sl * 255).astype(np.uint8))

    # Label overlay (if available)
    if label is not None:
        lbl_sl = label[s] if label.ndim == 3 else label[0, s]
        # Colour map: 1=green (oedema), 2=yellow (nCET), 3=red (enhancing)
        overlay = np.zeros((*lbl_sl.shape, 3), dtype=np.uint8)
        overlay[lbl_sl == 1] = [0, 200, 0]
        overlay[lbl_sl == 2] = [200, 200, 0]
        overlay[lbl_sl == 3] = [200, 0, 0]
    else:
        overlay = None

    panels: List[Image.Image] = []
    for sl_arr in ch_slices:
        img = Image.fromarray(sl_arr, mode="L").convert("RGB")
        if overlay is not None:
            ov_img = Image.fromarray(overlay, mode="RGB")
            img = Image.blend(img, ov_img, overlay_alpha)
        panels.append(img)

    # Concatenate horizontally
    total_w = sum(p.width for p in panels)
    canvas = Image.new("RGB", (total_w, panels[0].height))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# §7.1 — Visual Turing test set
# ---------------------------------------------------------------------------

def prepare_turing_test(
    real_dir: str,
    synth_dir: str,
    real_label_dir: Optional[str],
    synth_label_dir: Optional[str],
    output_dir: str,
    n_each: int = 50,
    channels: Tuple[int, ...] = (1, 3),
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a randomised Turing test image set.

    Selects ``n_each`` real & synthetic volumes, renders mid-axial PNG
    triplets, and produces a shuffled assignment CSV for blinded review.

    Returns
    -------
    dict with paths to output images and assignment file
    """
    import nibabel as nib

    rng = random.Random(seed)
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    real_paths = sorted(Path(real_dir).glob("*.nii*"))
    synth_paths = sorted(Path(synth_dir).glob("*.nii*"))

    if len(real_paths) < n_each or len(synth_paths) < n_each:
        logger.warning(
            "Not enough volumes (real=%d, synth=%d). Using available.",
            len(real_paths), len(synth_paths),
        )
    real_sel = rng.sample(real_paths, min(n_each, len(real_paths)))
    synth_sel = rng.sample(synth_paths, min(n_each, len(synth_paths)))

    def _load(p: Path) -> np.ndarray:
        return nib.load(str(p)).get_fdata().astype(np.float32)

    def _load_label(label_dir: Optional[str], name: str) -> Optional[np.ndarray]:
        if label_dir is None:
            return None
        candidates = list(Path(label_dir).glob(f"{name}*"))
        if candidates:
            return nib.load(str(candidates[0])).get_fdata().astype(np.int32)
        return None

    # Build items: (path, source_tag, stem)
    items: List[Tuple[Path, str, str]] = []
    for p in real_sel:
        items.append((p, "real", p.stem.split(".")[0]))
    for p in synth_sel:
        items.append((p, "synthetic", p.stem.split(".")[0]))

    rng.shuffle(items)

    # Render and record
    assignment: List[Dict[str, str]] = []
    for idx, (vol_path, tag, stem) in enumerate(items):
        vol = _load(vol_path)
        # Ensure [C, D, H, W]
        if vol.ndim == 3:
            vol = vol[np.newaxis]
        elif vol.ndim == 4 and vol.shape[-1] <= 4:
            vol = vol.transpose(3, 0, 1, 2)

        label_dir = real_label_dir if tag == "real" else synth_label_dir
        label = _load_label(label_dir, stem)

        img_name = f"{idx:04d}.png"
        _vol_to_png(
            vol, label,
            slice_idx=None,
            output_path=str(out / "images" / img_name),
            channels=channels,
        )
        assignment.append({
            "id": f"{idx:04d}",
            "filename": img_name,
            "ground_truth": tag,
        })

    # Write assignment CSV (ground truth column hidden from raters)
    answer_csv = out / "answer_key.csv"
    with open(answer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "ground_truth"])
        writer.writeheader()
        writer.writerows(assignment)

    # Rater form CSV (no ground truth)
    rater_csv = out / "rater_form.csv"
    with open(rater_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "rater_response"])
        writer.writeheader()
        for row in assignment:
            writer.writerow({"id": row["id"], "filename": row["filename"], "rater_response": ""})

    logger.info("Turing test prepared: %d images in %s", len(assignment), out / "images")
    return {
        "image_dir": str(out / "images"),
        "answer_key": str(answer_csv),
        "rater_form": str(rater_csv),
        "n_items": len(assignment),
    }


# ---------------------------------------------------------------------------
# §7.2 — Quality rating form
# ---------------------------------------------------------------------------

QUALITY_CRITERIA = [
    "anatomical_plausibility",
    "tumour_realism",
    "modality_consistency",
    "artefact_free",
]


def prepare_quality_rating(
    synth_dir: str,
    synth_label_dir: Optional[str],
    output_dir: str,
    n_samples: int = 50,
    channels: Tuple[int, ...] = (1, 3),
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a set of synthetic images for 5-point Likert quality rating.

    Renders PNGs and generates a CSV form with columns for each criterion.
    """
    import nibabel as nib

    rng = random.Random(seed)
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    synth_paths = sorted(Path(synth_dir).glob("*.nii*"))
    synth_sel = rng.sample(synth_paths, min(n_samples, len(synth_paths)))

    def _load(p: Path) -> np.ndarray:
        return nib.load(str(p)).get_fdata().astype(np.float32)

    rows: List[Dict[str, str]] = []
    for idx, vol_path in enumerate(synth_sel):
        vol = _load(vol_path)
        if vol.ndim == 3:
            vol = vol[np.newaxis]
        elif vol.ndim == 4 and vol.shape[-1] <= 4:
            vol = vol.transpose(3, 0, 1, 2)

        label = None
        if synth_label_dir:
            stem = vol_path.stem.split(".")[0]
            candidates = list(Path(synth_label_dir).glob(f"{stem}*"))
            if candidates:
                label = nib.load(str(candidates[0])).get_fdata().astype(np.int32)

        img_name = f"synth_{idx:04d}.png"
        _vol_to_png(
            vol, label,
            slice_idx=None,
            output_path=str(out / "images" / img_name),
            channels=channels,
        )
        row: Dict[str, str] = {"id": f"{idx:04d}", "filename": img_name}
        for crit in QUALITY_CRITERIA:
            row[crit] = ""  # rater fills 1–5
        rows.append(row)

    form_csv = out / "quality_rating_form.csv"
    with open(form_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename"] + QUALITY_CRITERIA)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Quality rating form: %d items in %s", len(rows), form_csv)
    return {
        "image_dir": str(out / "images"),
        "form_csv": str(form_csv),
        "n_items": len(rows),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_turing_test(
    answer_key_csv: str,
    rater_csvs: List[str],
) -> Dict[str, Any]:
    """Analyse Turing test results from rater response CSVs.

    Returns mean accuracy and inter-rater Cohen's kappa.
    """
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    answers = pd.read_csv(answer_key_csv)
    gt = answers.set_index("id")["ground_truth"]

    rater_accuracies: List[float] = []
    rater_responses: List[pd.Series] = []

    for rcsv in rater_csvs:
        df = pd.read_csv(rcsv)
        df = df.set_index("id")
        resp = df["rater_response"].dropna()
        merged = gt.loc[resp.index]

        correct = (resp.str.lower() == merged.str.lower()).sum()
        acc = correct / max(len(merged), 1)
        rater_accuracies.append(float(acc))
        rater_responses.append(resp)

    result: Dict[str, Any] = {
        "rater_accuracies": rater_accuracies,
        "mean_accuracy": float(np.mean(rater_accuracies)),
    }

    if len(rater_responses) >= 2:
        r1 = rater_responses[0]
        r2 = rater_responses[1]
        common = r1.index.intersection(r2.index)
        if len(common) >= 3:
            kappa = cohen_kappa_score(
                r1.loc[common].values, r2.loc[common].values
            )
            result["inter_rater_kappa"] = float(kappa)

    return result


def analyse_quality_rating(rating_csvs: List[str]) -> Dict[str, Any]:
    """Analyse Likert quality ratings, returning per-criterion statistics."""
    import pandas as pd

    all_dfs = [pd.read_csv(rc) for rc in rating_csvs]
    combined = pd.concat(all_dfs, ignore_index=True)

    results: Dict[str, Any] = {}
    for crit in QUALITY_CRITERIA:
        vals = pd.to_numeric(combined[crit], errors="coerce").dropna()
        results[crit] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "pct_gte_3": float(100 * (vals >= 3).mean()) if len(vals) else float("nan"),
        }

    # Overall
    all_vals = pd.to_numeric(
        combined[QUALITY_CRITERIA].values.flatten(), errors="coerce"
    )
    all_vals = all_vals[~np.isnan(all_vals)]
    results["overall"] = {
        "mean": float(all_vals.mean()) if len(all_vals) else float("nan"),
        "pct_gte_3": float(100 * (all_vals >= 3).mean()) if len(all_vals) else float("nan"),
    }
    return results
