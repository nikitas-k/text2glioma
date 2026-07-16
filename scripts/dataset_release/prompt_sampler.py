"""Prompt-pool builder for the Text2Glioma synthetic dataset release.

Assembles two pools:

    * ``real`` \u2014 verbatim VASARI impressions from the training split, used
      as-is with light re-pairing to a different (deformed) mask.
    * ``novel`` \u2014 combinatorial recompositions: parse a base impression into
      component segments, then replace 1-3 segments with segments from other
      training impressions. Produces VASARI-linguistically-consistent but
      unseen feature combinations.

The 70/30 split is enforced at the manifest level (see
``prepare_manifest.py``); this module just provides the sampling
primitives.

Parsing heuristic
-----------------

VASARI impressions produced by ``text2glioma.preprocessing.utils.compose_radiology_prompts``
are comma-separated segments in a stable order. We classify each segment
into a component category based on keyword matches against the VASARI
vocabulary tables (VASARI_F2/F4/F5/F7/F11/F14 plus the invasion
booleans). Segments that don't classify are left in place; the recomposer
never invents new phrases, it only shuffles existing ones.

Deterministic RNG
-----------------

All sampling is driven by a ``numpy.random.default_rng(seed)`` so the
release manifest is reproducible from a single top-level seed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Component classifier
# ---------------------------------------------------------------------------

# Component categories, in the order VASARI compose_radiology_prompts emits.
COMPONENTS = (
    "location",     # e.g. "Left frontal-parietal mass" -- the ONLY segment ending in "mass"
    "enhancement",  # F4: non-enhancing / mild enhancement / marked enhancement
    "prop_enh",     # F5: ≤5% enhancing / 5-33% enhancing / ...
    "prop_ncet",    # F6: ≤5% non-enhancing tumour / ...
    "rim",          # F11: thin irregular enhancing rim / thick enhancing rim / solid enhancement
    "necrosis",     # F7: no necrosis / ≤5% necrosis / ...
    "oedema",       # F14: no/mild/moderate/extensive oedema
    "invasion",     # collated list of booleans: cortical involvement, deep WM invasion, etc.
)

_INVASION_PHRASES = {
    "cortical involvement",
    "ependymal invasion",
    "deep white matter invasion",
    "crosses midline",
    "satellite lesions",
    "multifocal",
    "multicentric",
    "no non-enhancing component",
    "no invasion",
}

_ENHANCEMENT_PHRASES = {"non-enhancing", "mild enhancement", "marked enhancement"}
_RIM_PHRASES = {
    "thin irregular enhancing rim",
    "thick enhancing rim",
    "solid enhancement",
}
_NECROSIS_KEYWORD = "necrosis"
_OEDEMA_KEYWORD = "oedema"
_ENH_PCT_RE = re.compile(r"(\u226410%|\u22645%|5\u201333%|33\u201367%|67\u2013100%|67\u201395%|95\u201399\\.5%|>99\\.5%)\\s*enhancing", re.IGNORECASE)
_NCET_PCT_RE = re.compile(r"(\u22645%|5\u201333%|33\u201367%|67\u201395%|95\u201399\\.5%|>99\\.5%)\\s*non-enhancing", re.IGNORECASE)


def _classify(segment: str) -> str:
    """Assign one of COMPONENTS to a single trimmed segment."""
    s = segment.strip().lower()

    # Location (first-segment "... mass")
    if s.endswith("mass"):
        return "location"

    # Enhancement quality
    if any(p in s for p in _ENHANCEMENT_PHRASES) and "%" not in s:
        return "enhancement"

    # Enhancing proportion (contains "% enhancing" but not "non-enhancing")
    if "%" in s and "enhancing" in s and "non-enhancing" not in s and "necrosis" not in s:
        return "prop_enh"

    # nCET proportion
    if "%" in s and "non-enhancing" in s and "necrosis" not in s:
        return "prop_ncet"

    # Rim / margin
    if any(p in s for p in _RIM_PHRASES):
        return "rim"

    # Necrosis
    if _NECROSIS_KEYWORD in s:
        return "necrosis"

    # Oedema
    if _OEDEMA_KEYWORD in s:
        return "oedema"

    # Invasion / boolean features
    if any(p in s for p in _INVASION_PHRASES):
        return "invasion"

    # Fallback: treat as invasion (rare; unmatched free-form text)
    return "invasion"


@dataclass
class ParsedImpression:
    """A parsed VASARI impression, split by component category."""
    raw: str                                    # original impression string
    by_component: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_all_required(self) -> bool:
        return all(k in self.by_component and self.by_component[k]
                   for k in ("location", "enhancement", "prop_enh", "oedema"))


def parse_impression(imp: str) -> ParsedImpression:
    """Split an impression by commas and classify each segment."""
    if not imp:
        return ParsedImpression(raw="")
    segments = [s.strip() for s in imp.split(",") if s.strip()]
    by_component: dict[str, list[str]] = {c: [] for c in COMPONENTS}
    for seg in segments:
        cat = _classify(seg)
        by_component[cat].append(seg)
    return ParsedImpression(raw=imp, by_component=by_component)


def recompose(parsed: ParsedImpression, order: Iterable[str] = COMPONENTS) -> str:
    """Recompose an impression from parsed segments in the canonical order."""
    parts: list[str] = []
    for cat in order:
        parts.extend(parsed.by_component.get(cat, []))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Novel VASARI combinator
# ---------------------------------------------------------------------------

class PromptSampler:
    """Sample from a training-derived pool of real + novel VASARI prompts."""

    # Which components are eligible for cross-case replacement in "novel"
    # mode. Location is not replaced by default because it constrains the
    # mask+prompt agreement (the tumour location in the mask should match
    # the descriptor); to preserve mask-text coherence we hold location
    # fixed and vary the qualitative descriptors.
    _SWAPPABLE = ("enhancement", "prop_enh", "prop_ncet", "rim",
                  "necrosis", "oedema", "invasion")

    def __init__(self, real_impressions: list[str], seed: int = 42):
        """`real_impressions` is a list of VASARI impression strings from the training split."""
        self.rng = np.random.default_rng(seed)
        self.raw_pool: list[str] = [imp for imp in real_impressions if imp]
        self.parsed_pool: list[ParsedImpression] = [
            parse_impression(imp) for imp in self.raw_pool
        ]
        # Pool of well-parsed impressions (with all four required fields).
        self.valid_pool = [p for p in self.parsed_pool if p.has_all_required]
        if not self.valid_pool:
            raise ValueError("no valid impressions parsed from the input pool")

    # -- Public API ------------------------------------------------------

    def sample_real(self) -> tuple[str, dict]:
        """Draw a verbatim impression from the pool.

        Returns ``(impression, meta)`` where ``meta`` records the pool
        index for provenance.
        """
        idx = int(self.rng.integers(len(self.valid_pool)))
        return self.valid_pool[idx].raw, {
            "source": "real",
            "pool_idx": idx,
        }

    def sample_novel(self, n_swaps: int | None = None) -> tuple[str, dict]:
        """Compose a novel impression by swapping 1-3 components across cases.

        Returns ``(impression, meta)`` where meta records which categories
        were swapped and from which pool indices.
        """
        if n_swaps is None:
            n_swaps = int(self.rng.integers(1, 4))   # 1, 2, or 3
        n_swaps = max(1, min(n_swaps, len(self._SWAPPABLE)))

        base_idx = int(self.rng.integers(len(self.valid_pool)))
        base = self.valid_pool[base_idx]
        new_components = {k: list(v) for k, v in base.by_component.items()}

        # Pick which categories to swap.
        swap_cats = self.rng.choice(self._SWAPPABLE, size=n_swaps, replace=False)

        donors: dict[str, int] = {}
        for cat in swap_cats:
            # Pick a donor case whose parsed impression has this category populated.
            candidates = [i for i, p in enumerate(self.valid_pool)
                          if p.by_component.get(cat)]
            if not candidates:
                continue
            donor_idx = int(self.rng.choice(candidates))
            new_components[cat] = list(self.valid_pool[donor_idx].by_component[cat])
            donors[str(cat)] = donor_idx

        # Recompose in canonical order.
        parts: list[str] = []
        for cat in COMPONENTS:
            parts.extend(new_components.get(cat, []))
        recomposed = ", ".join(parts)

        return recomposed, {
            "source": "novel",
            "base_pool_idx": base_idx,
            "swaps": donors,
            "n_swaps": len(donors),
        }

    def sample_batch(self, n: int, novel_fraction: float = 0.3
                     ) -> list[tuple[str, dict]]:
        """Sample a batch of ``n`` impressions with the given novel fraction."""
        n_novel = int(round(n * novel_fraction))
        n_real  = n - n_novel
        batch: list[tuple[str, dict]] = []
        batch.extend(self.sample_real() for _ in range(n_real))
        batch.extend(self.sample_novel() for _ in range(n_novel))
        # Shuffle so real/novel are interleaved.
        idx = np.arange(len(batch))
        self.rng.shuffle(idx)
        return [batch[i] for i in idx]


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_training_impressions(datalist_path: Path,
                              split: str = "training",
                              field: str = "impression") -> list[str]:
    """Read the VASARI impressions from a datalist JSON."""
    with open(datalist_path) as f:
        dl = json.load(f)
    if split not in dl:
        raise KeyError(f"split {split!r} not in {datalist_path}")
    return [item.get(field, "") for item in dl[split]]


# ---------------------------------------------------------------------------
# CLI: sanity check the parser + sampler on a datalist
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datalist", type=Path, required=True)
    ap.add_argument("--split", default="training")
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--novel_fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    impressions = load_training_impressions(args.datalist, args.split)
    print(f"loaded {len(impressions)} impressions from {args.datalist} [{args.split}]")

    # Parse-rate audit
    parsed = [parse_impression(i) for i in impressions if i]
    n_valid = sum(1 for p in parsed if p.has_all_required)
    print(f"parseable-with-required-fields: {n_valid}/{len(parsed)} "
          f"({100*n_valid/max(len(parsed),1):.1f}%)")

    # Show a real / novel pair
    sampler = PromptSampler(impressions, seed=args.seed)
    print("\n== REAL examples ==")
    for i in range(args.n_examples):
        imp, meta = sampler.sample_real()
        print(f"  [{i}] {imp}")
        print(f"      meta: {meta}")

    print("\n== NOVEL examples ==")
    for i in range(args.n_examples):
        imp, meta = sampler.sample_novel()
        print(f"  [{i}] {imp}")
        print(f"      meta: {meta}")

    # Batch stat
    batch = sampler.sample_batch(1000, novel_fraction=args.novel_fraction)
    n_real = sum(1 for _, m in batch if m["source"] == "real")
    n_nov  = sum(1 for _, m in batch if m["source"] == "novel")
    print(f"\nbatch of 1000: {n_real} real, {n_nov} novel  "
          f"(target novel fraction = {args.novel_fraction})")


if __name__ == "__main__":
    main()
