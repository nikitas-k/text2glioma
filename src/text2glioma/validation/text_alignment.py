"""Text–image alignment: VASARI feature recovery and CLIP text–image score."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# VASARI features we can evaluate automatically
CATEGORICAL_FEATURES = {
    "F1 Tumour Location": "location",
    "F2 Side of Tumour Epicenter": "laterality",
    "F4 Enhancement Quality": "enhancement",
    "F9 Multifocal or Multicentric": "multifocal",
}
ORDINAL_FEATURES = {
    "F5 Proportion Enhancing": "proportion_enh",
    "F14 Proportion of Oedema": "proportion_oedema",
}


# ---------------------------------------------------------------------------
# VASARI feature recovery
# ---------------------------------------------------------------------------


def extract_vasari(
    label_path: str,
    atlas_dir: str,
    enhancing_label: int = 3,
    nonenhancing_label: int = 2,
    oedema_label: int = 1,
) -> pd.DataFrame:
    """Run VASARI-auto on a single segmentation label file.

    Returns the single-row DataFrame from get_vasari_features().
    """
    from text2glioma.preprocessing.vasari_auto import get_vasari_features

    return get_vasari_features(
        file=label_path,
        atlases=atlas_dir + "/" if not atlas_dir.endswith("/") else atlas_dir,
        enhancing_label=enhancing_label,
        nonenhancing_label=nonenhancing_label,
        oedema_label=oedema_label,
        verbose=False,
    )


def accuracy(pred: list, gt: list) -> float:
    """Top-1 accuracy (ignoring NaN predictions)."""
    correct, total = 0, 0
    for p, g in zip(pred, gt):
        if pd.isna(p) or pd.isna(g):
            continue
        total += 1
        correct += int(p == g)
    return correct / max(total, 1)


def ordinal_kappa(pred: list, gt: list) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal VASARI scores."""
    from sklearn.metrics import cohen_kappa_score

    p_clean, g_clean = [], []
    for p, g in zip(pred, gt):
        if pd.isna(p) or pd.isna(g):
            continue
        p_clean.append(int(p))
        g_clean.append(int(g))
    if len(p_clean) < 3:
        return float("nan")
    return float(cohen_kappa_score(g_clean, p_clean, weights="quadratic"))


def vasari_feature_recovery(
    gt_labels: List[str],
    synth_labels: List[str],
    atlas_dir: str,
    enhancing_label: int = 3,
    nonenhancing_label: int = 2,
    oedema_label: int = 1,
) -> Dict[str, float]:
    """Compare VASARI features between ground-truth and synthetic labels.

    Parameters
    ----------
    gt_labels : list of paths to real segmentation NIfTI files
    synth_labels : list of paths to round-trip segmentation NIfTI files
                   (predicted from synthetic images — same ordering as gt_labels)
    atlas_dir : path to atlas masks for VASARI location derivation

    Returns
    -------
    dict with accuracy / kappa per VASARI feature
    """
    gt_rows, synth_rows = [], []

    for gt_p, syn_p in tqdm(
        zip(gt_labels, synth_labels), total=len(gt_labels), desc="VASARI extraction"
    ):
        gt_rows.append(
            extract_vasari(gt_p, atlas_dir, enhancing_label, nonenhancing_label, oedema_label)
        )
        synth_rows.append(
            extract_vasari(syn_p, atlas_dir, enhancing_label, nonenhancing_label, oedema_label)
        )

    gt_df = pd.concat(gt_rows, ignore_index=True)
    synth_df = pd.concat(synth_rows, ignore_index=True)

    results: Dict[str, float] = {}

    # Categorical → accuracy
    for col, short in CATEGORICAL_FEATURES.items():
        if col in gt_df.columns and col in synth_df.columns:
            acc = accuracy(synth_df[col].tolist(), gt_df[col].tolist())
            results[f"{short}_accuracy"] = acc
            logger.info("%s accuracy = %.2f %%", short, acc * 100)

    # Ordinal → quadratic kappa
    for col, short in ORDINAL_FEATURES.items():
        if col in gt_df.columns and col in synth_df.columns:
            kappa = ordinal_kappa(synth_df[col].tolist(), gt_df[col].tolist())
            results[f"{short}_kappa"] = kappa
            logger.info("%s κ = %.3f", short, kappa)

    return results


# ---------------------------------------------------------------------------
# CLIP text–image score
# ---------------------------------------------------------------------------


def compute_clip_score(
    texts: List[str],
    volumes: List[np.ndarray],
    device: str = "cpu",
    slice_idx: Optional[int] = None,
) -> Dict[str, float]:
    """Compute mean CLIP cosine similarity between text prompts and images.

    Uses mid-axial slices of the T1CE channel (index 1) as image input.
    Requires ``open-clip-torch`` (optional dependency in ``[eval]``).
    """
    try:
        import open_clip
    except ImportError:
        logger.warning("open-clip-torch not installed. Skipping CLIP score.")
        return {"clip_score_mean": float("nan")}

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    cos_sims = []
    for text, vol in tqdm(zip(texts, volumes), total=len(texts), desc="CLIP score"):
        # extract mid-axial T1CE slice
        ch = vol[1] if vol.ndim == 4 else vol  # [D, H, W]
        s = slice_idx if slice_idx is not None else ch.shape[0] // 2
        sl = ch[min(s, ch.shape[0] - 1)]
        # normalise
        vmin, vmax = float(sl.min()), float(sl.max())
        if vmax - vmin > 1e-8:
            sl = (sl - vmin) / (vmax - vmin)
        # convert to 3-channel PIL-like tensor
        import torch as _torch
        from PIL import Image

        sl_uint8 = (sl * 255).astype(np.uint8)
        img = Image.fromarray(sl_uint8, mode="L").convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)

        tok = tokenizer([text]).to(device)

        with _torch.no_grad():
            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()
        cos_sims.append(sim)

    return {
        "clip_score_mean": float(np.mean(cos_sims)),
        "clip_score_std": float(np.std(cos_sims)),
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_text_alignment(
    gt_label_dir: str,
    synth_label_dir: str,
    atlas_dir: str,
    prompts_json: str,
    synth_image_dir: str,
    output_json: str,
    enhancing_label: int = 3,
    nonenhancing_label: int = 2,
    oedema_label: int = 1,
    device: str = "cpu",
    max_n: Optional[int] = None,
) -> dict:
    """End-to-end text–image alignment evaluation."""
    gt_label_paths = sorted(Path(gt_label_dir).glob("*.nii*"))[:max_n]
    synth_label_paths = sorted(Path(synth_label_dir).glob("*.nii*"))[:max_n]

    n = min(len(gt_label_paths), len(synth_label_paths))
    gt_label_paths = gt_label_paths[:n]
    synth_label_paths = synth_label_paths[:n]

    results: dict = {}

    # VASARI recovery
    results["vasari"] = vasari_feature_recovery(
        gt_labels=[str(p) for p in gt_label_paths],
        synth_labels=[str(p) for p in synth_label_paths],
        atlas_dir=atlas_dir,
        enhancing_label=enhancing_label,
        nonenhancing_label=nonenhancing_label,
        oedema_label=oedema_label,
    )

    # CLIP score
    with open(prompts_json) as f:
        prompts_data = json.load(f)
    if isinstance(prompts_data, dict):
        prompts_data = prompts_data.get("training", []) + prompts_data.get("validation", [])
    texts = [p.get("impression", p.get("findings", "")) for p in prompts_data][:n]

    from text2glioma.validation.image_quality import load_nifti_volumes
    synth_vols = load_nifti_volumes(synth_image_dir, max_n=n)
    results["clip"] = compute_clip_score(texts, synth_vols, device=device)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Text alignment results saved to %s", out_path)
    return results
