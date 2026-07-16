"""Filter the pre-generation manifest to enforce prompt-mask consistency.

The training cohort contains masks whose tumour label content (label 2 =
oedema, label 3 = enhancing) is inconsistent with their VASARI text
impressions. Common failure mode: text describes "67-100% enhancing,
solid enhancement" but the segmentation mask has essentially zero label-3
voxels — the model then produces an oedema-only tumour (mask wins at
CFG=1.0), silently violating the release's text-conditioning contract.

This script consumes:
    1. The pre-generation manifest (from prepare_manifest.py)
    2. The mask-label audit CSV (from audit_mask_labels.py)

and writes a filtered manifest that drops rows whose prompt-mask pair is
inconsistent according to configurable rules.

Default rule:
    If the prompt implies "≥33% enhancing" (VASARI 33-67% or 67-100%
    categories) then the mask must have `has_meaningful_enhancing == 1`
    (n_label3 >= 100 AND enhancing_fraction >= 0.05).

Rows that fail the check are either **dropped** (default) or **repaired**
(re-sample a fresh prompt for the same mask that is consistent with its
label content). Repair keeps the total sample count at N and the shard
sizes balanced.

Usage:
    python scripts/dataset_release/filter_manifest_by_mask_quality.py \\
        --manifest manifest.csv \\
        --audit    mask_label_audit.csv \\
        --out      manifest_filtered.csv \\
        --mode     repair  # or 'drop'

If --mode repair is used, the script needs access to the datalist and
seed used by prepare_manifest.py to re-run the prompt sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

# --- prompt classification ------------------------------------------------

# VASARI enhancement-percentage segments that imply "meaningful enhancement".
# These substrings appear in prompts emitted by prompt_sampler.py.
_STRONG_ENHANCEMENT_PATTERNS = (
    "33-67% enhancing",
    "33\u201367% enhancing",   # en-dash variant
    "67-100% enhancing",
    "67\u2013100% enhancing",  # en-dash variant
    "solid enhancement",
    "thick enhancing rim",
    "thin enhancing rim",
    "marked enhancement",
)


def prompt_implies_enhancement(prompt: str) -> bool:
    """True if the prompt implies the tumour should have meaningful enhancing tissue."""
    p = prompt.lower()
    return any(pat.lower() in p for pat in _STRONG_ENHANCEMENT_PATTERNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Pre-generation manifest from prepare_manifest.py")
    ap.add_argument("--audit", type=Path, required=True,
                    help="Mask-label audit CSV from audit_mask_labels.py")
    ap.add_argument("--out", type=Path, required=True,
                    help="Filtered manifest output path")
    ap.add_argument("--mode", choices=["drop", "repair"], default="drop",
                    help=("drop = remove inconsistent rows (release size shrinks); "
                          "repair = keep row, resample prompt so it's consistent"))
    ap.add_argument("--min_label3_vox", type=int, default=100,
                    help="Minimum label-3 voxel count to accept 'meaningful enhancement'")
    ap.add_argument("--min_label3_frac", type=float, default=0.05,
                    help="Minimum label-3 / (label2+label3) fraction to accept 'meaningful enhancement'")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional: write a diagnostic CSV of dropped/repaired rows here")
    args = ap.parse_args()

    print(f"loading manifest: {args.manifest}", file=sys.stderr)
    m = pd.read_csv(args.manifest)
    print(f"loading audit:    {args.audit}", file=sys.stderr)
    a = pd.read_csv(args.audit)

    # Index audit by mask path (canonical join key).
    a = a.set_index("mask_path")
    audit_lookup = a.to_dict(orient="index")

    def mask_is_meaningful(mask_path: str) -> bool:
        row = audit_lookup.get(mask_path)
        if row is None:
            # unknown mask -> keep (conservative)
            return True
        n3 = row.get("n_label3", 0)
        f3 = row.get("enhancing_fraction", 0.0)
        try:
            n3 = int(n3) if n3 not in ("", None) else 0
            f3 = float(f3) if f3 not in ("", None) else 0.0
        except (ValueError, TypeError):
            n3, f3 = 0, 0.0
        return (n3 >= args.min_label3_vox) and (f3 >= args.min_label3_frac)

    # Classify each row.
    n = len(m)
    prompt_wants_enh = m["prompt"].map(prompt_implies_enhancement)
    mask_ok_for_enh = m["mask_source_path"].map(mask_is_meaningful)
    inconsistent = prompt_wants_enh & (~mask_ok_for_enh)

    print(f"total rows:                          {n}", file=sys.stderr)
    print(f"prompts implying enhancement:        {int(prompt_wants_enh.sum())}  "
          f"({prompt_wants_enh.mean()*100:.1f}%)", file=sys.stderr)
    print(f"masks with meaningful enh:           {int(mask_ok_for_enh.sum())}  "
          f"({mask_ok_for_enh.mean()*100:.1f}%)", file=sys.stderr)
    print(f"INCONSISTENT (enh prompt + poor mask): {int(inconsistent.sum())}  "
          f"({inconsistent.mean()*100:.1f}%)", file=sys.stderr)

    if args.report is not None:
        rep = m.assign(prompt_wants_enh=prompt_wants_enh,
                       mask_ok_for_enh=mask_ok_for_enh,
                       inconsistent=inconsistent)
        rep.to_csv(args.report, index=False)
        print(f"wrote diagnostic report: {args.report}", file=sys.stderr)

    if args.mode == "drop":
        out = m[~inconsistent].reset_index(drop=True)
        # Renumber sample_id contiguously and rebalance shards.
        shards = out["shard"].nunique()
        n_out = len(out)
        out["sample_id"] = [f"sample_{i:07d}" for i in range(n_out)]
        # Re-shard uniformly.
        out["shard"] = [i % shards for i in range(n_out)]
        print(f"dropped {int(inconsistent.sum())} rows; kept {n_out}  "
              f"({shards} shards, ~{n_out // shards}/shard)", file=sys.stderr)
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print("REPAIR MODE: not implemented in this script — "
              "either extend it here to re-invoke prompt_sampler with a per-mask "
              "constraint, or use --mode drop for now.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
