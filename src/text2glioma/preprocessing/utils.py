"""Radiology prompt composer – builds text descriptions from VASARI-auto features.

All text is derived exclusively from VASARI-auto output.  If VASARI
feature extraction fails a ``RuntimeError`` is raised – there is no
heuristic fallback.
"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_random_state

from text2glioma.preprocessing.vasari_auto import get_vasari_features
from text2glioma.preprocessing.vasari_auto import _ATLAS_DIR_DEFAULT

from typing import Optional, Dict, Any


# ── VASARI code → human-readable look-up tables ──────────────────────────

VASARI_F1 = {
    1: "frontal lobe", 2: "temporal lobe", 3: "insula",
    4: "parietal lobe", 5: "occipital lobe", 6: "brainstem",
    7: "corpus callosum", 8: "thalamus",
}
VASARI_F2 = {1: "right", 2: "bilateral", 3: "left"}

VASARI_F4 = {1: "non-enhancing", 2: "mild enhancement", 3: "marked enhancement"}

VASARI_F5 = {3: "≤5%", 4: "5–33%", 5: "33–67%", 6: "67–100%"}

VASARI_F6 = {
    3: "≤5%", 4: "5–33%", 5: "33–67%",
    6: "67–95%", 7: "95–99.5%", 8: ">99.5%",
}

VASARI_F7 = {
    2: "no necrosis", 3: "≤5% necrosis",
    4: "5–33% necrosis", 5: "33–67% necrosis",
}

VASARI_F11 = {
    3: "thin irregular enhancing rim",
    4: "thick enhancing rim",
    5: "solid enhancement, no non-enhancing component",
}

VASARI_F14 = {
    2: "no oedema", 3: "mild oedema",
    4: "moderate oedema", 5: "extensive oedema",
}

_BOOL = {1: False, 2: True}   # F19, F20, F21, F24
_MID  = {2: False, 3: True}   # F22, F23

# Short names for mixed-location phrases (drop "lobe" for compactness)
_LOC_SHORT = {
    "frontal lobe": "frontal",
    "temporal lobe": "temporal",
    "parietal lobe": "parietal",
    "occipital lobe": "occipital",
    "insula": "insular",
    "brainstem": "brainstem",
    "thalamus": "thalamic",
    "corpus callosum": "callosal",
}


def _mixed_location(region_props: Dict[str, float], min_frac: float = 0.10) -> str:
    """Build a mixed location string from VASARI region proportions.

    Regions with proportion >= *min_frac* are included, sorted by
    descending proportion, joined with hyphens.
    e.g. ``"frontal-parietal"``.
    Falls back to the single top region, or ``"indeterminate location"``.
    """
    if not region_props:
        return "indeterminate location"
    significant = [
        (name, frac)
        for name, frac in region_props.items()
        if frac >= min_frac
    ]
    if not significant:
        # Nothing above threshold — pick the single highest
        top = max(region_props.items(), key=lambda x: x[1])
        return _LOC_SHORT.get(top[0], top[0])
    # Sort by descending proportion
    significant.sort(key=lambda x: x[1], reverse=True)
    parts = [_LOC_SHORT.get(name, name) for name, _ in significant]
    return "-".join(parts)


# ── helpers ───────────────────────────────────────────────────────────────

def _safe_int(v):
    """Convert a value to int, returning ``None`` for NaN / None."""
    try:
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
    except Exception:
        return None


def _get_vasari(file_path: str, atlases_dir: str, **kwargs) -> Dict[str, Any]:
    """Run VASARI-auto and return a parsed feature dict.

    Raises
    ------
    RuntimeError
        If VASARI feature extraction fails for any reason.
    """
    try:
        df = get_vasari_features(file_path, atlases=atlases_dir, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"VASARI feature extraction failed for {file_path}: {exc}"
        ) from exc

    row = df.iloc[0].to_dict()

    # Region proportion map (ROI name → fraction of tumour in that region)
    region_props = {
        "frontal lobe":     float(row.get("prop_frontal", 0) or 0),
        "temporal lobe":    float(row.get("prop_temporal", 0) or 0),
        "parietal lobe":    float(row.get("prop_parietal", 0) or 0),
        "occipital lobe":   float(row.get("prop_occipital", 0) or 0),
        "insula":           float(row.get("prop_insula", 0) or 0),
        "brainstem":        float(row.get("prop_brainstem", 0) or 0),
        "thalamus":         float(row.get("prop_thalamus", 0) or 0),
        "corpus callosum":  float(row.get("prop_corpus_callosum", 0) or 0),
    }

    return {
        "F1_loc":       _safe_int(row.get("F1 Tumour Location")),
        "F2_side":      _safe_int(row.get("F2 Side of Tumour Epicenter")),
        "F4_quality":   _safe_int(row.get("F4 Enhancement Quality")),
        "F5_enh":       _safe_int(row.get("F5 Proportion Enhancing")),
        "F6_nCET":      _safe_int(row.get("F6 Proportion nCET")),
        "F7_nec":       _safe_int(row.get("F7 Proportion Necrosis")),
        "F9_multi":     _safe_int(row.get("F9 Multifocal or Multicentric")),
        "F11_rim":      _safe_int(row.get("F11 Thickness of enhancing margin")),
        "F14_edema":    _safe_int(row.get("F14 Proportion of Oedema")),
        "F19_epend":    _safe_int(row.get("F19 Ependymal Invasion")),
        "F20_cortex":   _safe_int(row.get("F20 Cortical involvement")),
        "F21_deepwm":   _safe_int(row.get("F21 Deep WM invasion")),
        "F22_nCET_mid": _safe_int(row.get("F22 nCET Crosses Midline")),
        "F23_CET_mid":  _safe_int(row.get("F23 CET Crosses midline")),
        "F24_sats":     _safe_int(row.get("F24 satellites")),
        "region_props": region_props,
    }


# ── main entry point ─────────────────────────────────────────────────────

def compose_radiology_prompts(
    image_path: str,
    label_path: str,
    atlas_dir: Optional[str] = None,
    enhancing_label: int = 3,
    nonenhancing_label: int = 1,
    edema_label: int = 2,
    z_dim: int = -1,
    cf: int = 1,
    t_ependymal: int = 5000,
    t_wm: int = 100,
    resolution: int = 1,
    midline_thresh: int = 5,
    enh_quality_thresh: int = 15,
    cyst_thresh: int = 50,
    cortical_thresh: int = 1000,
    focus_thresh: int = 30000,
    num_components_bin_thresh: int = 10,
    num_components_cet_thresh: int = 15,
    include_diag: str = "",
    shuffle_order: bool = True,
    seed: int | np.random.RandomState | None = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build radiology prompts from VASARI-auto features.

    Returns ``{"short": <impression>, "long": <findings>}``.

    Raises ``RuntimeError`` if VASARI feature extraction fails.
    """
    # ── resolve atlas directory ──────────────────────────────────────────
    if atlas_dir is None:
        atlas_dir = _ATLAS_DIR_DEFAULT
    atlas_str = str(atlas_dir)
    if not atlas_str.endswith("/"):
        atlas_str += "/"

    config = {
        "enhancing_label": enhancing_label,
        "nonenhancing_label": nonenhancing_label,
        "oedema_label": edema_label,
        "z_dim": z_dim,
        "cf": cf,
        "t_ependymal": t_ependymal,
        "t_wm": t_wm,
        "resolution": resolution,
        "midline_thresh": midline_thresh,
        "enh_quality_thresh": enh_quality_thresh,
        "cyst_thresh": cyst_thresh,
        "cortical_thresh": cortical_thresh,
        "focus_thresh": focus_thresh,
        "num_components_bin_thresh": num_components_bin_thresh,
        "num_components_cet_thresh": num_components_cet_thresh,
        "verbose": verbose,
    }

    # ── run VASARI-auto (raises on failure) ──────────────────────────────
    va = _get_vasari(label_path, atlas_str, **config)

    if seed:
        rng = check_random_state(seed)

    # ── decode VASARI codes ──────────────────────────────────────────────
    location   = _mixed_location(va.get("region_props", {}), min_frac=0.10)
    side       = VASARI_F2.get(va["F2_side"], "")
    quality    = VASARI_F4.get(va["F4_quality"], "indeterminate enhancement")
    enh_pct    = VASARI_F5.get(va["F5_enh"])
    ncet_pct   = VASARI_F6.get(va["F6_nCET"])
    necrosis   = VASARI_F7.get(va["F7_nec"], "no necrosis")
    rim        = VASARI_F11.get(va["F11_rim"])
    oedema     = VASARI_F14.get(va["F14_edema"], "no oedema")

    ependymal  = _BOOL.get(va["F19_epend"], False)
    cortical   = _BOOL.get(va["F20_cortex"], False)
    deepwm     = _BOOL.get(va["F21_deepwm"], False)
    ncet_mid   = _MID.get(va["F22_nCET_mid"], False)
    cet_mid    = _MID.get(va["F23_CET_mid"], False)
    satellites = _BOOL.get(va["F24_sats"], False)
    multifocal = (va["F9_multi"] == 2)

    # ── location phrase ──────────────────────────────────────────────────
    if side == "bilateral":
        loc_phrase = f"bilateral {location}"
    elif side:
        loc_phrase = f"{side} {location}"
    else:
        loc_phrase = location

    # ── enhancement phrase ───────────────────────────────────────────────
    enh_parts = [quality]
    if enh_pct and quality != "non-enhancing":
        enh_parts.append(f"{enh_pct} enhancing")
    if rim:
        enh_parts.append(rim)

    # ── invasion / extension ─────────────────────────────────────────────
    inv_parts: list[str] = []
    if cet_mid or ncet_mid:
        inv_parts.append("crosses midline")
    if ependymal:
        inv_parts.append("ependymal invasion")
    if deepwm:
        inv_parts.append("deep white matter invasion")
    if cortical:
        inv_parts.append("cortical involvement")
    if satellites:
        inv_parts.append("satellite lesions")
    if multifocal:
        inv_parts.append("multifocal")

    # ── SHORT (Impression-style, one sentence) ───────────────────────────
    short_bits: list[str] = []
    short_bits.append(f"{loc_phrase} mass")
    short_bits.append(", ".join(enh_parts))
    if necrosis != "no necrosis":
        short_bits.append(necrosis)
    if oedema != "no oedema":
        short_bits.append(oedema)
    short_bits.extend(inv_parts)
    if include_diag:
        short_bits.append(include_diag)

    # ── LONG (Findings-style) ────────────────────────────────────────────
    findings: list[str] = []
    findings.append(f"Location: {loc_phrase}.")
    findings.append(f"Enhancement: {quality}.")
    if enh_pct:
        findings.append(f"Proportion enhancing: {enh_pct}.")
    if ncet_pct:
        findings.append(f"Proportion non-enhancing tumour: {ncet_pct}.")
    if rim:
        findings.append(f"Enhancing margin: {rim}.")
    findings.append(f"Necrosis: {necrosis}.")
    findings.append(f"Oedema: {oedema}.")
    if inv_parts:
        findings.append(f"Invasion: {', '.join(inv_parts)}.")
    if include_diag:
        findings.append(f"Diagnosis: {include_diag}.")

    if shuffle_order:
        rng.shuffle(short_bits)
        rng.shuffle(findings)

    short = ", ".join(short_bits)
    short = short[0].upper() + short[1:]

    return {"short": short, "long": " ".join(findings)}
