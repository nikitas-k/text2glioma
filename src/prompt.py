"""Prompt generation utilities for glioma labels and healthy scans.

This module combines morphology and location analysis functions to produce
simple textual prompts describing a tumour based on its segmentation labels
or returns a generic description for healthy MRI volumes.
"""

from __future__ import annotations

from pathlib import Path
import random

import nibabel as nib
import numpy as np

from .functions import (
    detect_ring_enhancement,
    edema_severity_from_labels,
    tumour_lobe_hemisphere,
)


__all__ = ["generate_prompt", "generate_healthy_prompt"]


def generate_healthy_prompt() -> str:
    """Return a generic healthy brain MRI description.

    The function randomly chooses between two fixed phrases describing a
    normal scan with no abnormalities.

    Returns
    -------
    str
        Either ``"A MRI with nil finding"`` or
        ``"A MRI with no intracranial abnormalities"``.
    """

    return random.choice(
        ["A MRI with nil finding", "A MRI with no intracranial abnormalities"]
    )


def generate_prompt(label_file: str | Path, healthy: bool = False) -> str:
    """Generate a text prompt from a NIfTI label file.

    Parameters
    ----------
    label_file:
        Path to the tumour label volume in NIfTI format.
    healthy:
        When ``True``, bypass label parsing and return a healthy scan prompt
        via :func:`generate_healthy_prompt`.

    Returns
    -------
    str
        A short textual description derived from morphology and location
        functions. ``"no tumour features detected"`` is returned when no
        information can be extracted.
    """

    if healthy:
        return generate_healthy_prompt()

    img = nib.load(str(label_file))
    lab = np.asanyarray(img.dataobj)
    vox = img.header.get_zooms()[:3]

    hemisphere, lobe = tumour_lobe_hemisphere(lab, vox)
    edema_sev, _ed_ml, _ed_frac = edema_severity_from_labels(lab, vox)
    ring, _ring_frac, _touch_frac = detect_ring_enhancement(lab)

    parts: list[str] = []
    if hemisphere and lobe:
        parts.append(f"{hemisphere} {lobe} lesion")
    elif hemisphere:
        parts.append(f"{hemisphere} lesion")

    if edema_sev != "none":
        parts.append(f"{edema_sev} edema")
    if ring:
        parts.append("ring enhancement")

    return ", ".join(parts) if parts else "no tumour features detected"
