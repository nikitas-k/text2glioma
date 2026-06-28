#!/usr/bin/env python
"""Freeze a one-session-per-patient LUMIERE test split.

LUMIERE contains 91 patients with multiple longitudinal sessions each. For
the preprint OOD evaluation we report **patient-level disjoint** statistics,
so we must pick exactly one session per patient (deterministically) before
running any metric.

Strategy
--------
For each subject in the LUMIERE datalist produced by
``scripts/ingest_lumiere.py``:
  1. Drop sessions whose ``label`` file is missing or whose ``image`` file
     is missing (defensive — should not happen, but ingestion failures
     manifest here).
  2. Drop sessions whose mask has fewer than ``--min_tumour_voxels`` non-zero
     voxels (default 500). Tiny / empty masks make the W1 / SSIM-inside-mask
     metrics noisy.
  3. From the surviving sessions, pick one deterministically:
       * ``--pick first``    — the first session in encounter order (typical
                               baseline scan).
       * ``--pick largest``  — the session with the largest tumour mask
                               (worst-case visibility, hardest to fake).
       * ``--pick random``   — uniformly at random under ``--seed``
                               (default; matches a held-out test split).

The output is a new datalist JSON with the *same shape* as the input — all
chosen sessions live under the ``"validation"`` key so downstream code
(``offline_sample_stage2_compare.py``, the CFG-sweep notebooks, the LUMIERE
external eval notebook) Just Works without modification.

Usage
-----

On Gadi::

    python scripts/freeze_lumiere_test_split.py \\
        --input  /g/data/vp06/$USER/text2glioma_train/data/lumiere_ingested/datalist_lumiere.json \\
        --output /g/data/vp06/$USER/text2glioma_train/data/lumiere_ingested/datalist_lumiere_test.json \\
        --pick random --seed 42

The audit summary printed to stdout is also written next to the output as
``<output>.audit.json`` for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


log = logging.getLogger("freeze_lumiere_test_split")


def _entry_subject(entry: dict) -> str | None:
    for k in ("subject_id", "subject", "patient_id", "patient"):
        v = entry.get(k)
        if v:
            return str(v)
    img = entry.get("image")
    if not img:
        return None
    # Fallback: derive subject from filename prefix "Patient-XX_<session>.nii.gz"
    stem = Path(img).name.split(".")[0]
    return stem.split("_")[0] if "_" in stem else stem


def _tumour_voxels(label_path: str) -> int:
    try:
        arr = nib.load(label_path).get_fdata()
    except Exception as exc:  # corrupt / missing
        log.warning("could not load label %s: %s", label_path, exc)
        return 0
    return int((arr > 0).sum())


def _pick_one(entries: list[dict], strategy: str, rng: random.Random) -> dict:
    if strategy == "first":
        return entries[0]
    if strategy == "largest":
        return max(entries, key=lambda e: e.get("_tumour_voxels", 0))
    if strategy == "random":
        return rng.choice(entries)
    raise ValueError(f"unknown pick strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=Path, required=True,
                   help="Path to datalist_lumiere.json produced by ingest_lumiere.py.")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the frozen split JSON.")
    p.add_argument("--split", type=str, default="validation",
                   help="Which key of the input datalist to draw from "
                        "(ingest_lumiere.py emits everything under 'validation').")
    p.add_argument("--pick", choices=("first", "largest", "random"), default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_tumour_voxels", type=int, default=500,
                   help="Drop sessions whose mask has fewer voxels than this. "
                        "Default 500 matches the test-retest filter used internally.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.is_file():
        sys.exit(f"input datalist not found: {args.input}")

    with open(args.input) as f:
        datalist = json.load(f)
    if args.split not in datalist or not datalist[args.split]:
        sys.exit(f"split '{args.split}' missing or empty in {args.input}")

    entries: list[dict] = list(datalist[args.split])
    log.info("loaded %d sessions from %s", len(entries), args.input.name)

    # 1) Drop sessions with missing image / label files.
    survived: list[dict] = []
    dropped_missing = 0
    for e in entries:
        img = e.get("image"); lab = e.get("label")
        if not img or not Path(img).is_file():
            dropped_missing += 1; continue
        if not lab or not Path(lab).is_file():
            dropped_missing += 1; continue
        survived.append(e)
    log.info("dropped %d sessions with missing image/label files", dropped_missing)

    # 2) Drop sessions with tiny tumours; cache voxel count on each entry for
    #    --pick largest to reuse.
    sized: list[dict] = []
    dropped_small = 0
    for e in survived:
        nvox = _tumour_voxels(e["label"])
        e["_tumour_voxels"] = nvox
        if nvox < args.min_tumour_voxels:
            dropped_small += 1; continue
        sized.append(e)
    log.info("dropped %d sessions with < %d tumour voxels",
             dropped_small, args.min_tumour_voxels)

    # 3) Group by subject and pick one session each.
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for e in sized:
        subj = _entry_subject(e)
        if subj is None:
            log.warning("could not derive subject for entry; skipping: %s",
                        e.get("image"))
            continue
        by_subject[subj].append(e)

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for subj in sorted(by_subject.keys()):
        # Sort sessions within a subject deterministically (by image path) so
        # 'first' and 'random' are reproducible.
        sub_entries = sorted(by_subject[subj], key=lambda e: e["image"])
        chosen.append(_pick_one(sub_entries, args.pick, rng))

    # Strip the helper voxel-count from the output (keep the JSON clean).
    for e in chosen:
        e.pop("_tumour_voxels", None)

    # Carry the original metadata forward and stamp this freeze.
    meta = dict(datalist.get("_metadata", {}))
    meta.update({
        "frozen_split_pick": args.pick,
        "frozen_split_seed": args.seed,
        "frozen_split_min_tumour_voxels": args.min_tumour_voxels,
        "frozen_split_n_subjects": len(by_subject),
        "frozen_split_n_sessions_chosen": len(chosen),
        "frozen_split_n_sessions_input": len(entries),
        "frozen_split_source": str(args.input),
    })

    out = {
        "training":   [],
        "validation": chosen,    # downstream code reads from 'validation'
        "testing":    [],
        "_metadata":  meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %d chosen sessions (one per subject) to %s",
             len(chosen), args.output)

    # Reproducibility audit file.
    audit = {
        "input": str(args.input),
        "output": str(args.output),
        "n_sessions_input": len(entries),
        "n_missing_dropped": dropped_missing,
        "n_small_tumour_dropped": dropped_small,
        "n_subjects": len(by_subject),
        "n_chosen": len(chosen),
        "pick": args.pick,
        "seed": args.seed,
        "min_tumour_voxels": args.min_tumour_voxels,
        "chosen_subjects": sorted(by_subject.keys()),
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    log.info("audit written to %s", audit_path)

    print(json.dumps({k: v for k, v in audit.items() if k != "chosen_subjects"},
                     indent=2))


if __name__ == "__main__":
    main()
