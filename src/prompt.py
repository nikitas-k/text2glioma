"""Prompt generation utilities for glioma labels.

This module combines morphology and location analysis functions to produce
simple textual prompts describing a tumour based on its segmentation labels.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .functions import (
    detect_ring_enhancement,
    edema_severity_from_labels,
    tumour_lobe_hemisphere,
)


def generate_prompt(label_file: str | Path) -> str:
    """Generate a text prompt from a NIfTI label file.

    Parameters
    ----------
    label_file:
        Path to the tumour label volume in NIfTI format.

    Returns
    -------
    str
        A short textual description derived from morphology and location
        functions. ``"no tumour features detected"`` is returned when no
        information can be extracted.
    """

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
