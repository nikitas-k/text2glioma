"""Prompt-swap ablation for text-conditioning fidelity.

The circular baseline in ``text_alignment.vasari_feature_recovery`` compares
VASARI(ground-truth mask) to VASARI(round-trip seg of generation conditioned on
the same mask). That measures mask recovery, not text conditioning.

This module implements the non-circular alternative: hold the mask fixed and
swap the prompt to one taken from a different subject whose ground-truth VASARI
attribute contrasts on a target dimension. If the recovered attribute follows
the prompt (not the mask), text conditioning is doing work.

Pipeline (each stage a separate CLI in ``scripts/run_prompt_swap.py``):
  1. ``build_swap_pairs`` — select (i, j) subject pairs contrasting on one
     target VASARI attribute; emit a pairs manifest.
  2. Generation (outside this module; uses ``GenericSampler``) — for each pair,
     emit a native (mask_i, prompt_i) and a swap (mask_i, prompt_j) 4-ch NIfTI.
  3. External nnU-Net over both sets, followed by ``vasari_auto`` extraction.
  4. ``analyse_swap_recovery`` — per-attribute shift statistics + paired tests.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Target VASARI attributes and the vasari-auto column that carries them.
CATEGORICAL_TARGETS = {
    "laterality": "F2 Side of Tumour Epicenter",
    "location": "F1 Tumour Location",
    "enhancement": "F4 Enhancement Quality",
    "multifocal": "F9 Multifocal or Multicentric",
}
ORDINAL_TARGETS = {
    "proportion_enh": "F5 Proportion Enhancing",
    "proportion_oedema": "F14 Proportion of Oedema",
}


# ---------------------------------------------------------------------------
# Impression parsing (template-generated VASARI prompts)
# ---------------------------------------------------------------------------

_LOBE_TOKENS = (
    "frontal", "temporal", "parietal", "occipital", "insular",
    "thalamic", "callosal", "brainstem", "cerebellar",
)
_LATERALITY_RE = re.compile(r"\b(left|right|bilateral)\b", re.IGNORECASE)
_LOBE_RE = re.compile(rf"\b({'|'.join(_LOBE_TOKENS)})\b", re.IGNORECASE)
_ENHANCEMENT_RE = re.compile(
    r"\b(non-enhancing|marked enhancement|moderate enhancement|mild enhancement|no enhancement)\b",
    re.IGNORECASE,
)
_PROPORTION_ENH_RE = re.compile(r"(<=?\s*5%|5-33%|33-67%|67-100%)\s*enhancing", re.IGNORECASE)
_PROPORTION_OEDEMA_RE = re.compile(
    r"\b(no oedema|mild oedema|moderate oedema|extensive oedema)\b", re.IGNORECASE,
)
_PROPORTION_ENH_MAP = {"<=5%": 0, "5-33%": 1, "33-67%": 2, "67-100%": 3}
_OEDEMA_MAP = {"no oedema": 0, "mild oedema": 1, "moderate oedema": 2, "extensive oedema": 3}


def parse_impression_attributes(impression: str) -> Dict[str, object]:
    """Extract canonical VASARI attributes from a text2glioma template prompt.

    Returns short-key values (``laterality``, ``location``, ``enhancement``,
    ``multifocal``, ``proportion_enh``, ``proportion_oedema``). ``location`` is
    the first lobe token in the prompt; ``locations_all`` carries the full list
    for richer contrast selection later. Missing attributes map to ``None``.
    """
    imp = (impression or "").lower()
    out: Dict[str, object] = {}

    m = _LATERALITY_RE.search(imp)
    out["laterality"] = m.group(1).lower() if m else None

    lobes = [x.lower() for x in _LOBE_RE.findall(imp)]
    out["location"] = lobes[0] if lobes else None
    out["locations_all"] = lobes

    m = _ENHANCEMENT_RE.search(imp)
    if m:
        token = m.group(1).lower().replace(" enhancement", "").strip()
        out["enhancement"] = "non-enhancing" if token in {"non-enhancing", "no"} else token
    else:
        out["enhancement"] = None

    out["multifocal"] = ("multifocal" in imp) or ("satellite lesions" in imp)

    m = _PROPORTION_ENH_RE.search(imp)
    out["proportion_enh"] = (
        _PROPORTION_ENH_MAP.get(re.sub(r"\s+", "", m.group(1))) if m else None
    )

    m = _PROPORTION_OEDEMA_RE.search(imp)
    out["proportion_oedema"] = _OEDEMA_MAP.get(m.group(1).lower()) if m else None

    return out


def _extract_impression_batch(
    subjects: Sequence[dict],
    text_field: str = "impression",
    label_field: str = "label",
) -> pd.DataFrame:
    """Parse impressions for every subject; one row per subject."""
    rows = []
    for s in subjects:
        parsed = parse_impression_attributes(s.get(text_field, ""))
        parsed["_subj"] = s.get("subject_id", Path(s[label_field]).stem)
        rows.append(parsed)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


@dataclass
class SwapPair:
    """One (subject_i, subject_j) swap pair contrasting on ``target``."""

    pair_id: str
    target: str
    subj_i: str
    subj_j: str
    mask_i: str
    prompt_i: str
    prompt_j: str
    attr_i: object
    attr_j: object
    seed: int = 0
    # Populated after the vasari-auto pre-extraction:
    vasari_i: Dict[str, object] = field(default_factory=dict)
    vasari_j: Dict[str, object] = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        # Flatten dicts to JSON strings for CSV.
        d["vasari_i"] = json.dumps(d["vasari_i"], default=str)
        d["vasari_j"] = json.dumps(d["vasari_j"], default=str)
        return d


def _extract_vasari_batch(
    subjects: Sequence[dict],
    atlas_dir: str,
    label_field: str = "label",
    enhancing_label: int = 3,
    nonenhancing_label: int = 2,
    oedema_label: int = 1,
) -> pd.DataFrame:
    """Run vasari-auto on each subject's ground-truth label; return one row per subject."""
    from text2glioma.preprocessing.vasari_auto import get_vasari_features

    atlas_dir = atlas_dir.rstrip("/") + "/"
    rows = []
    for s in subjects:
        row = get_vasari_features(
            file=s[label_field],
            atlases=atlas_dir,
            enhancing_label=enhancing_label,
            nonenhancing_label=nonenhancing_label,
            oedema_label=oedema_label,
            verbose=False,
        )
        row = row.iloc[0].to_dict() if isinstance(row, pd.DataFrame) else dict(row)
        row["_subj"] = s.get("subject_id", Path(s[label_field]).stem)
        rows.append(row)
    return pd.DataFrame(rows)


def _ordinal_distance(a, b) -> float:
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return float("nan")


def build_swap_pairs(
    datalist: Sequence[dict],
    target: str,
    n_pairs: int = 100,
    text_field: str = "impression",
    label_field: str = "label",
    seed: int = 42,
    min_ordinal_gap: int = 2,
    atlas_dir: Optional[str] = None,
    use_vasari_auto: bool = False,
) -> List[SwapPair]:
    """Select ``n_pairs`` maximally contrasting subject pairs on ``target``.

    By default the ground-truth attribute of each subject is parsed from the
    ``impression`` field of the datalist entry (fast; canonical short-key
    vocabulary). Pass ``use_vasari_auto=True`` with ``atlas_dir`` to fall back
    to re-extracting attributes from the segmentation labels.

    Categorical targets: pair subjects whose attribute values differ (e.g. L vs R).
    Ordinal targets: pair subjects at least ``min_ordinal_gap`` steps apart.
    """
    if target not in CATEGORICAL_TARGETS and target not in ORDINAL_TARGETS:
        raise ValueError(f"Unknown target attribute: {target}")

    if use_vasari_auto:
        if not atlas_dir:
            raise ValueError("use_vasari_auto=True requires atlas_dir.")
        logger.info("Extracting VASARI via vasari-auto for %d subjects", len(datalist))
        v_df = _extract_vasari_batch(datalist, atlas_dir=atlas_dir, label_field=label_field)
        col = CATEGORICAL_TARGETS.get(target) or ORDINAL_TARGETS[target]
    else:
        logger.info("Parsing impressions for %d subjects", len(datalist))
        v_df = _extract_impression_batch(
            datalist, text_field=text_field, label_field=label_field,
        )
        col = target

    if col not in v_df.columns:
        raise KeyError(f"Attribute column '{col}' not present in extracted table.")

    # Index subjects by attribute value and drop NaNs.
    v_df = v_df.dropna(subset=[col]).reset_index(drop=True)
    if v_df.empty:
        raise RuntimeError(f"No subjects with non-NaN {col}.")

    subj_lookup = {s.get("subject_id", Path(s[label_field]).stem): s for s in datalist}

    rng = random.Random(seed)
    pairs: List[SwapPair] = []
    attempts = 0
    max_attempts = n_pairs * 50

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(len(v_df)), 2)
        ai, aj = v_df.at[i, col], v_df.at[j, col]

        if target in CATEGORICAL_TARGETS:
            if ai == aj:
                continue
        else:
            if _ordinal_distance(ai, aj) < min_ordinal_gap:
                continue

        subj_i = str(v_df.at[i, "_subj"])
        subj_j = str(v_df.at[j, "_subj"])
        si, sj = subj_lookup[subj_i], subj_lookup[subj_j]

        pairs.append(
            SwapPair(
                pair_id=f"{target}_{len(pairs):04d}",
                target=target,
                subj_i=subj_i,
                subj_j=subj_j,
                mask_i=si[label_field],
                prompt_i=si.get(text_field, ""),
                prompt_j=sj.get(text_field, ""),
                attr_i=ai,
                attr_j=aj,
                seed=rng.randint(0, 2**31 - 1),
                vasari_i=v_df.iloc[i].to_dict(),
                vasari_j=v_df.iloc[j].to_dict(),
            )
        )

    if len(pairs) < n_pairs:
        logger.warning("Only %d of %d requested pairs satisfied contrast constraint.",
                       len(pairs), n_pairs)
    return pairs


def save_pairs(pairs: Sequence[SwapPair], out_path: str) -> None:
    df = pd.DataFrame([p.to_row() for p in pairs])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d pairs to %s", len(df), out_path)


def load_pairs(csv_path: str) -> List[SwapPair]:
    df = pd.read_csv(csv_path)
    out: List[SwapPair] = []
    for _, r in df.iterrows():
        d = r.to_dict()
        d["vasari_i"] = json.loads(d.get("vasari_i") or "{}")
        d["vasari_j"] = json.loads(d.get("vasari_j") or "{}")
        out.append(SwapPair(**{k: v for k, v in d.items() if k in SwapPair.__annotations__}))
    return out


# ---------------------------------------------------------------------------
# Recovery analysis
# ---------------------------------------------------------------------------


def _matches(value, target_value, target_kind: str, tol: float = 0.0) -> Optional[bool]:
    """True if ``value`` matches ``target_value`` under the appropriate metric."""
    if pd.isna(value) or pd.isna(target_value):
        return None
    if target_kind == "categorical":
        return value == target_value
    try:
        return abs(float(value) - float(target_value)) <= tol
    except (TypeError, ValueError):
        return None


def analyse_swap_recovery(
    pairs: Sequence[SwapPair],
    native_vasari: pd.DataFrame,
    swap_vasari: pd.DataFrame,
    ordinal_tol: float = 0.0,
) -> pd.DataFrame:
    """Per-attribute shift statistics.

    ``native_vasari`` / ``swap_vasari`` must be indexed by ``pair_id`` and carry
    the standard vasari-auto columns.
    """
    results = []
    for p in pairs:
        col = CATEGORICAL_TARGETS.get(p.target) or ORDINAL_TARGETS[p.target]
        kind = "categorical" if p.target in CATEGORICAL_TARGETS else "ordinal"

        if p.pair_id not in native_vasari.index or p.pair_id not in swap_vasari.index:
            continue
        native_val = native_vasari.at[p.pair_id, col]
        swap_val = swap_vasari.at[p.pair_id, col]

        results.append(
            {
                "pair_id": p.pair_id,
                "target": p.target,
                "attr_i": p.attr_i,
                "attr_j": p.attr_j,
                "native_recovered": native_val,
                "swap_recovered": swap_val,
                "native_matches_i": _matches(native_val, p.attr_i, kind, ordinal_tol),
                "native_matches_j": _matches(native_val, p.attr_j, kind, ordinal_tol),
                "swap_matches_i": _matches(swap_val, p.attr_i, kind, ordinal_tol),
                "swap_matches_j": _matches(swap_val, p.attr_j, kind, ordinal_tol),
            }
        )
    return pd.DataFrame(results)


def summarise_shift(recovery_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-pair recovery to per-target summary statistics.

    Reports (a) native fidelity P(recovered=i | native), (b) swap prompt-follow
    rate P(recovered=j | swap), (c) net prompt-driven shift (b) - P(recovered=j | native),
    and (d) McNemar exact test on the paired (native-matches-j, swap-matches-j)
    contingency.
    """
    from scipy import stats as sstats

    rows = []
    for target, g in recovery_df.groupby("target"):
        g = g.dropna(subset=["native_matches_j", "swap_matches_j"])
        if g.empty:
            continue
        n = len(g)
        native_fidelity = g["native_matches_i"].dropna().mean()
        native_follow_j = g["native_matches_j"].mean()
        swap_follow_j = g["swap_matches_j"].mean()
        net_shift = swap_follow_j - native_follow_j

        # McNemar on discordant (native≠j, swap=j) vs (native=j, swap≠j).
        b = int(((~g["native_matches_j"]) & (g["swap_matches_j"])).sum())
        c = int((g["native_matches_j"] & (~g["swap_matches_j"])).sum())
        if b + c > 0:
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                mcn = mcnemar([[0, b], [c, 0]], exact=True)
                p_val = float(mcn.pvalue)
            except ImportError:
                p_val = float(sstats.binomtest(b, n=b + c, p=0.5).pvalue)
        else:
            p_val = float("nan")

        rows.append(
            {
                "target": target,
                "n": int(n),
                "native_fidelity_to_i": float(native_fidelity),
                "native_match_to_j": float(native_follow_j),
                "swap_match_to_j": float(swap_follow_j),
                "net_prompt_driven_shift": float(net_shift),
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_p": p_val,
            }
        )
    return pd.DataFrame(rows)
