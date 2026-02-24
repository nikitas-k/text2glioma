#!/usr/bin/env python3
"""Filter a datalist JSON by excluding or including subjects from a filter file.

The filter file should be a JSON file containing a list of subject IDs, e.g.::

    ["nnUNetv2-00001", "nnUNetv2-00002", ...]

Or a JSON object with an ``"exclude"`` / ``"include"`` key::

    {"exclude": ["subj2127", "subj2115", ...]}

Usage::

    # Exclude FWHM outliers from the datalist:
    python scripts/filter_datalist.py \\
        --datalist datalist_N1511.json \\
        --exclude filter_subjects.json \\
        --output datalist_filtered.json

    # Keep only high-CNR subjects:
    python scripts/filter_datalist.py \\
        --datalist datalist_N1510.json \\
        --include filter_subjects_high_cnr.json \\
        --output datalist_N1510_high_cnr.json

    # Dry-run (just print counts, don't write):
    python scripts/filter_datalist.py \\
        --datalist datalist_N1511.json \\
        --exclude filter_subjects.json \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_filter_ids(path: str) -> set[str]:
    """Load subject IDs from a JSON file (list or {exclude/include: [...]})."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        for key in ("exclude", "include"):
            if key in data:
                return set(data[key])
        # Try values that are lists of strings
        for v in data.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return set(v)
    raise ValueError(
        f"Cannot parse ID list from {path}. "
        "Expected a JSON list of subject IDs or {{\"exclude\"/\"include\": [...]}}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter a datalist JSON by excluding or including subjects."
    )
    parser.add_argument(
        "--datalist", type=str, required=True,
        help="Path to input datalist JSON.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exclude", type=str, default=None,
        help="Path to JSON file with subject IDs to exclude.",
    )
    group.add_argument(
        "--include", type=str, default=None,
        help="Path to JSON file with subject IDs to keep (exclude all others).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path for the filtered datalist JSON. "
             "Default: <datalist>_filtered.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print counts but don't write output.",
    )
    args = parser.parse_args()

    # Load datalist
    with open(args.datalist) as f:
        datalist = json.load(f)

    # Load filter set
    if args.exclude:
        filter_ids = load_filter_ids(args.exclude)
        mode = "exclude"
        print(f"Exclude list: {len(filter_ids)} subject IDs from {args.exclude}")
    else:
        filter_ids = load_filter_ids(args.include)
        mode = "include"
        print(f"Include list: {len(filter_ids)} subject IDs from {args.include}")

    import re

    def _get_id(item):
        """Extract subject ID, also checking nnunet_id from image path."""
        sid = item.get("subject_id", Path(item["image"]).stem)
        # Also extract nnUNetv2-XXXXX from image path for matching
        m = re.search(r"(nnUNetv2-\d+)", item.get("image", ""))
        nnunet_id = m.group(1) if m else None
        return sid, nnunet_id

    def _keep(item):
        sid, nnunet_id = _get_id(item)
        if mode == "exclude":
            return sid not in filter_ids and (nnunet_id is None or nnunet_id not in filter_ids)
        else:  # include
            return sid in filter_ids or (nnunet_id is not None and nnunet_id in filter_ids)

    # Filter each split
    splits = [k for k in ("training", "validation", "testing")
              if k in datalist and isinstance(datalist[k], list)]

    filtered = {}
    total_before = 0
    total_after = 0
    total_removed = 0

    for split in splits:
        before = datalist[split]
        after = [item for item in before if _keep(item)]
        removed = len(before) - len(after)
        total_before += len(before)
        total_after += len(after)
        total_removed += removed
        filtered[split] = after
        print(f"  {split}: {len(before)} → {len(after)}  "
              f"(removed {removed})")

    # Copy non-split metadata keys
    for k, v in datalist.items():
        if k not in splits:
            filtered[k] = v

    # Update count metadata if present
    if "n_subjs" in filtered:
        filtered["n_subjs"] = total_after
    for split in splits:
        count_key = f"n_{split}"
        if count_key in filtered:
            filtered[count_key] = len(filtered[split])

    print(f"\n  Total: {total_before} → {total_after}  "
          f"(removed {total_removed})")

    # Check for filter IDs not matched
    all_ids = set()
    for split in splits:
        for item in datalist[split]:
            sid, nnunet_id = _get_id(item)
            all_ids.add(sid)
            if nnunet_id:
                all_ids.add(nnunet_id)
    not_found = filter_ids - all_ids
    if not_found:
        print(f"\n  WARNING: {len(not_found)} {mode} IDs not found in datalist: "
              f"{sorted(not_found)[:10]}{'...' if len(not_found) > 10 else ''}")

    if args.dry_run:
        print("\n  Dry run — no file written.")
        return

    # Write output
    if args.output is None:
        stem = Path(args.datalist).stem
        out_path = str(Path(args.datalist).parent / f"{stem}_filtered.json")
    else:
        out_path = args.output

    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"\n  Written to: {out_path}")


if __name__ == "__main__":
    main()
