#!/usr/bin/env python3
"""Filter a datalist JSON by excluding subjects listed in a filter file.

The filter file should be a JSON file containing a list of subject IDs to
**exclude**, e.g.::

    ["subj2127", "subj2115", "subj3087", "subj556", ...]

Or a JSON object with an ``"exclude"`` key::

    {"exclude": ["subj2127", "subj2115", ...]}

Usage::

    # Exclude FWHM outliers from the datalist:
    python scripts/filter_datalist.py \\
        --datalist datalist_N1511.json \\
        --exclude filter_subjects.json \\
        --output datalist_filtered.json

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


def load_exclude_ids(path: str) -> set[str]:
    """Load subject IDs from a JSON file (list or {exclude: [...]})."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        if "exclude" in data:
            return set(data["exclude"])
        # Try values that are lists of strings
        for v in data.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return set(v)
    raise ValueError(
        f"Cannot parse exclude list from {path}. "
        "Expected a JSON list of subject IDs or {{\"exclude\": [...]}}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter a datalist JSON by excluding subjects."
    )
    parser.add_argument(
        "--datalist", type=str, required=True,
        help="Path to input datalist JSON.",
    )
    parser.add_argument(
        "--exclude", type=str, required=True,
        help="Path to JSON file with subject IDs to exclude.",
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

    # Load exclude set
    exclude_ids = load_exclude_ids(args.exclude)
    print(f"Exclude list: {len(exclude_ids)} subject IDs from {args.exclude}")

    # Filter each split
    splits = [k for k in ("training", "validation", "testing")
              if k in datalist and isinstance(datalist[k], list)]

    filtered = {}
    total_before = 0
    total_after = 0
    total_removed = 0

    for split in splits:
        before = datalist[split]
        after = [
            item for item in before
            if item.get("subject_id", Path(item["image"]).stem) not in exclude_ids
        ]
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

    # Check for exclude IDs not found
    all_ids = set()
    for split in splits:
        for item in datalist[split]:
            all_ids.add(item.get("subject_id", Path(item["image"]).stem))
    not_found = exclude_ids - all_ids
    if not_found:
        print(f"\n  WARNING: {len(not_found)} exclude IDs not found in datalist: "
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
