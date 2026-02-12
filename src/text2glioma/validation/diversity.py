"""Diversity & memorisation checks: NN distance and intra-prompt diversity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction helpers (reuses image_quality infrastructure)
# ---------------------------------------------------------------------------

def _extract_features(volumes: List[np.ndarray], device: str = "cpu") -> np.ndarray:
    """Extract flattened 2048-d MedicalNet features for each volume.

    Returns [N, D] feature matrix (averaged over axial slices and channels).
    """
    from text2glioma.validation.image_quality import (
        _load_medicalnet_extractor,
        extract_slice_features,
    )

    extractor = _load_medicalnet_extractor(device=device)
    all_feats: List[np.ndarray] = []

    for vol in tqdm(volumes, desc="Extracting features"):
        ch_feats = []
        n_channels = vol.shape[0] if vol.ndim == 4 else 1
        for ch in range(n_channels):
            ch_data = vol[ch] if vol.ndim == 4 else vol
            feats = extract_slice_features(ch_data, extractor, device=device)
            ch_feats.append(feats.mean(axis=0))  # avg over slices
        all_feats.append(np.mean(ch_feats, axis=0))  # avg over channels

    return np.stack(all_feats)  # [N, 2048]


# ---------------------------------------------------------------------------
# §6.1 — Nearest-neighbour distance (memorisation check)
# ---------------------------------------------------------------------------

def compute_nn_distances(
    real_features: np.ndarray,
    synth_features: np.ndarray,
) -> Dict[str, float]:
    """For each synthetic, find L2-nearest real sample and return stats.

    Parameters
    ----------
    real_features : [N_real, D]
    synth_features : [N_synth, D]

    Returns
    -------
    dict with median_nn_dist, mean_inter_real_dist, ratio, pct_near_copies
    """
    from scipy.spatial.distance import cdist

    # Pairwise distances: synth → real
    D_sr = cdist(synth_features, real_features, metric="euclidean")  # [N_synth, N_real]
    nn_dists = D_sr.min(axis=1)  # [N_synth]

    # Inter-real distances
    D_rr = cdist(real_features, real_features, metric="euclidean")
    np.fill_diagonal(D_rr, np.inf)
    inter_real_nn = D_rr.min(axis=1)
    mean_inter_real = float(inter_real_nn.mean())
    min_inter_real = float(inter_real_nn.min())

    median_nn = float(np.median(nn_dists))
    ratio = median_nn / max(mean_inter_real, 1e-10)
    # "copies" threshold: within 5 % of minimum inter-real distance
    threshold = 0.05 * min_inter_real
    pct_copies = float(100 * (nn_dists < threshold).mean())

    return {
        "median_nn_dist": median_nn,
        "mean_inter_real_dist": mean_inter_real,
        "min_inter_real_dist": min_inter_real,
        "nn_ratio": ratio,
        "copy_threshold": threshold,
        "pct_near_copies": pct_copies,
    }


def run_memorisation_check(
    real_dir: str,
    synth_dir: str,
    device: str = "cpu",
    max_n: Optional[int] = None,
) -> Dict[str, float]:
    """End-to-end memorisation check from NIfTI directories."""
    from text2glioma.validation.image_quality import load_nifti_volumes

    real_vols = load_nifti_volumes(real_dir, max_n=max_n)
    synth_vols = load_nifti_volumes(synth_dir, max_n=max_n)

    real_feats = _extract_features(real_vols, device=device)
    synth_feats = _extract_features(synth_vols, device=device)

    return compute_nn_distances(real_feats, synth_feats)


# ---------------------------------------------------------------------------
# §6.2 — Intra-prompt diversity
# ---------------------------------------------------------------------------

def compute_intra_prompt_diversity(
    volumes: List[np.ndarray],
) -> Dict[str, float]:
    """Compute pairwise MS-SSIM among a set of samples from the same prompt.

    Parameters
    ----------
    volumes : list of [C, D, H, W] arrays (all from the same prompt)

    Returns
    -------
    dict with mean, std, and all pairwise MS-SSIM values
    """
    from text2glioma.validation.image_quality import compute_ms_ssim_3d

    n = len(volumes)
    if n < 2:
        return {"intra_ms_ssim_mean": float("nan"), "intra_ms_ssim_std": float("nan")}

    pairwise: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_ms_ssim_3d(volumes[i], volumes[j])
            pairwise.append(sim)

    return {
        "intra_ms_ssim_mean": float(np.mean(pairwise)),
        "intra_ms_ssim_std": float(np.std(pairwise)),
        "intra_ms_ssim_min": float(np.min(pairwise)),
        "intra_ms_ssim_max": float(np.max(pairwise)),
        "n_pairs": len(pairwise),
    }


def run_intra_prompt_diversity(
    prompt_dirs: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Evaluate intra-prompt diversity for multiple prompts.

    Parameters
    ----------
    prompt_dirs : mapping from prompt id → directory of NIfTI files
                  (each dir should contain multiple samples from one prompt)

    Returns
    -------
    dict mapping prompt_id → diversity metrics
    """
    from text2glioma.validation.image_quality import load_nifti_volumes

    results: Dict[str, Dict[str, float]] = {}
    for pid, pdir in prompt_dirs.items():
        vols = load_nifti_volumes(pdir)
        results[pid] = compute_intra_prompt_diversity(vols)
        logger.info(
            "Prompt %s: MS-SSIM = %.3f ± %.3f (n_pairs=%d)",
            pid,
            results[pid]["intra_ms_ssim_mean"],
            results[pid]["intra_ms_ssim_std"],
            results[pid]["n_pairs"],
        )
    return results


# ---------------------------------------------------------------------------
# Combined CLI runner
# ---------------------------------------------------------------------------

def run_diversity(
    real_dir: str,
    synth_dir: str,
    prompt_dirs: Optional[Dict[str, str]] = None,
    device: str = "cpu",
    max_n: Optional[int] = None,
    output_json: str = "diversity_results.json",
) -> Dict[str, Any]:
    """End-to-end diversity and memorisation evaluation.

    Parameters
    ----------
    real_dir : directory of real NIfTI volumes
    synth_dir : directory of synthetic NIfTI volumes (for NN check)
    prompt_dirs : optional dict mapping prompt_id → sample dir (for intra-prompt)
    """
    results: Dict[str, Any] = {}

    # §6.1 memorisation check
    results["memorisation"] = run_memorisation_check(
        real_dir=real_dir, synth_dir=synth_dir, device=device, max_n=max_n
    )
    logger.info(
        "Memorisation: median NN dist = %.4f  (ratio = %.3f, copies = %.1f %%)",
        results["memorisation"]["median_nn_dist"],
        results["memorisation"]["nn_ratio"],
        results["memorisation"]["pct_near_copies"],
    )

    # §6.2 intra-prompt diversity
    if prompt_dirs:
        results["intra_prompt"] = run_intra_prompt_diversity(prompt_dirs)
    else:
        logger.info("No prompt_dirs provided — skipping intra-prompt diversity")

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Diversity results saved to %s", out_path)
    return results
