"""Inpainting mask sampler for Paper B (text-conditioned tumour counterfactuals).

Five mask scenarios cover the three inference modes:

  - `tumor_dilated`  -> I3 (modification): tumour mask dilated by 1-20 mm
  - `tumor_eroded`   -> I3 (modification): tumour mask eroded by 1-5 mm
  - `tumor_exact`    -> I2 (removal):       tumour mask itself; text -> null
  - `healthy_blob`   -> I1 (insertion):     random ellipsoid in non-tumour brain
  - `random_blob`    -> regulariser:        random ellipsoid anywhere in brain

A scenario is drawn per sample from ``DEFAULT_WEIGHTS``; if the scenario requires
a tumour and the case has none, it falls back to ``healthy_blob``.

All operations assume the volume is already on the canonical Stage-1 grid
(1 mm isotropic, (D, H, W) = (160, 224, 160)). The sampler returns a binary
mask at full resolution; downsample to latent space outside this module.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


DEFAULT_WEIGHTS: dict[str, float] = {
    "tumor_dilated": 0.40,
    "tumor_eroded":  0.15,
    "tumor_exact":   0.10,
    "healthy_blob":  0.30,
    "random_blob":   0.05,
}

NULL_TEXT_TEMPLATES: list[str] = [
    "Normal brain parenchyma. No mass lesion identified.",
    "Unremarkable MRI brain. No focal parenchymal abnormality.",
    "No intracranial mass. Grey-white differentiation preserved.",
    "Normal brain MRI. No abnormal enhancement, mass effect, or midline shift.",
    "No focal abnormality identified. Ventricles normal in size and configuration.",
    "Normal-appearing brain parenchyma without focal lesion.",
    "No mass, haemorrhage, or abnormal enhancement.",
    "Normal study. No tumour, mass effect, or midline shift.",
    "Unremarkable brain MRI. No tumour identified.",
    "No intracranial mass lesion. Ventricular system within normal limits.",
    "Brain parenchyma normal in signal. No focal lesion or oedema.",
    "Normal MRI of the brain. No abnormality detected.",
    "No focal mass identified. Sulcation and gyration unremarkable.",
    "Normal cerebral parenchyma. No restricted diffusion or abnormal enhancement.",
    "Unremarkable parenchymal signal. No mass effect.",
    "Brain MRI within normal limits.",
    "No focal abnormal signal. No enhancing lesion identified.",
    "Normal-appearing supratentorial and infratentorial structures.",
    "No mass lesion. Cortical thickness preserved.",
    "Negative study. No tumour or mass.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _brain_mask(image: torch.Tensor, threshold: float = 0.01) -> np.ndarray:
    """Boolean (D, H, W) brain mask = any channel above threshold."""
    arr = _to_numpy(image)
    return np.any(arr > threshold, axis=0)


def _label_mask(label: torch.Tensor) -> np.ndarray:
    """Binary tumour mask (D, H, W)."""
    if label.dim() == 4:
        label = label[0]
    return _to_numpy(label) > 0


def _random_ellipsoid(
    shape: Sequence[int],
    center: Sequence[int],
    radii_vox: Sequence[float],
) -> np.ndarray:
    D, H, W = shape
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = center
    rz, ry, rx = radii_vox
    return (
        ((zz - cz) / rz) ** 2
        + ((yy - cy) / ry) ** 2
        + ((xx - cx) / rx) ** 2
    ) <= 1.0


def _sample_blob(
    region_mask: np.ndarray,
    rng: np.random.Generator,
    radii_range_mm: tuple[float, float] = (5.0, 40.0),
    elongation_max: float = 1.5,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Place a random ellipsoid whose centre lies inside ``region_mask``."""
    if not region_mask.any():
        return np.zeros(region_mask.shape, dtype=bool)
    coords = np.argwhere(region_mask)
    cz, cy, cx = coords[rng.integers(len(coords))]
    base_r_mm = rng.uniform(*radii_range_mm)
    e = rng.uniform(1.0 / elongation_max, elongation_max, size=3)
    radii_mm = base_r_mm * e
    radii_vox = tuple(r / v for r, v in zip(radii_mm, voxel_size_mm))
    return _random_ellipsoid(region_mask.shape, (cz, cy, cx), radii_vox)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_inpainting_mask(
    image: torch.Tensor,
    label: torch.Tensor,
    scenario_weights: Optional[dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    healthy_margin_mm: float = 5.0,
) -> tuple[torch.Tensor, str]:
    """Sample one inpainting mask and return ``(mask, scenario)``.

    Parameters
    ----------
    image : (C, D, H, W) float tensor in [0, 1] (intensity-normalised).
    label : (1, D, H, W) or (D, H, W) integer tensor; 0 = background.
    scenario_weights : custom mixture; defaults to ``DEFAULT_WEIGHTS``.
    rng : numpy Generator for reproducibility.
    voxel_size_mm : physical spacing; default 1 mm isotropic.
    healthy_margin_mm : minimum distance from existing tumour for
        ``healthy_blob`` placement.

    Returns
    -------
    mask : (1, D, H, W) float32 tensor in {0, 1}.
    scenario : str — the realised scenario (may differ from the drawn one
        if a tumour-dependent scenario was requested for a tumour-free case).
    """
    rng = rng or np.random.default_rng()
    weights = dict(scenario_weights or DEFAULT_WEIGHTS)
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    probs /= probs.sum()
    scenario = str(rng.choice(keys, p=probs))

    brain = _brain_mask(image)
    tumor = _label_mask(label)
    has_tumor = tumor.any()

    if scenario in ("tumor_dilated", "tumor_eroded", "tumor_exact") and not has_tumor:
        scenario = "healthy_blob"

    if scenario == "tumor_dilated":
        iters = int(rng.integers(1, 21))
        mask = binary_dilation(tumor, iterations=iters)
    elif scenario == "tumor_eroded":
        iters = int(rng.integers(1, 6))
        mask = binary_erosion(tumor, iterations=iters)
        if not mask.any():
            mask = tumor.copy()
            scenario = "tumor_exact"
    elif scenario == "tumor_exact":
        mask = tumor.copy()
    elif scenario == "healthy_blob":
        if has_tumor:
            forbidden = binary_dilation(tumor, iterations=int(healthy_margin_mm))
            region = brain & ~forbidden
        else:
            region = brain
        mask = _sample_blob(region, rng, voxel_size_mm=voxel_size_mm)
    elif scenario == "random_blob":
        mask = _sample_blob(brain, rng, voxel_size_mm=voxel_size_mm)
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    mask = mask & brain
    out = torch.from_numpy(mask.astype(np.float32))[None]
    return out, scenario


def mix_text_for_scenario(
    report: str,
    scenario: str,
    null_templates: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Choose the text conditioning given the realised scenario.

    Policy:
      tumor_exact  -> null text (the inpainted region must look healthy).
      everything else -> original report.

    For ``healthy_blob`` the original report describes a tumour: the model
    learns to *generate* one inside the empty ROI from the text — this is
    the I1 (insertion) capability.
    """
    rng = rng or np.random.default_rng()
    templates = list(null_templates or NULL_TEXT_TEMPLATES)
    if scenario == "tumor_exact":
        return str(rng.choice(templates))
    return report


def apply_inpainting_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Zero (or fill) the masked region of ``image``.

    image : (C, D, H, W) float tensor.
    mask  : (1, D, H, W) or (C, D, H, W) tensor in {0, 1}.
    """
    if mask.shape[0] == 1 and image.shape[0] > 1:
        m = mask.expand_as(image)
    else:
        m = mask
    return image * (1.0 - m) + fill_value * m
