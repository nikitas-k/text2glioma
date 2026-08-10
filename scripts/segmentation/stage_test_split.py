"""Split 4-channel stacked NIfTIs into nnU-Net v2 per-modality files.

Two input modes:
  * --in-dir DIR              : stage every *.nii.gz in DIR
  * --from-datalist FILE      : stage entries from datalist JSON under --split
                                (default "validation") and copy matching labels
                                to --out-labels. With --baseline-only, keep only
                                the earliest session per subject (LUMIERE
                                longitudinal cohort -> pre-op scan).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


_WEEK_RE = re.compile(r"(?:week|session)[-_]?(\d+)", re.IGNORECASE)


def _split_one(src: Path, out_dir: Path, stem: str) -> None:
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[0] == 4:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 4 or data.shape[-1] != 4:
        raise SystemExit(f"expected 4-channel volume, got shape {data.shape} for {src}")
    for c in range(4):
        out = out_dir / f"{stem}_{c:04d}.nii.gz"
        nib.save(nib.Nifti1Image(data[..., c].astype(np.float32), img.affine, img.header), str(out))


def _session_index(entry: dict) -> int:
    for key in ("session", "session_id", "week"):
        v = entry.get(key)
        if v is None:
            continue
        m = _WEEK_RE.search(str(v))
        if m:
            return int(m.group(1))
    # Fallback: try to parse the image filename (e.g. Patient-01_week-000_...).
    img = entry.get("image", "")
    m = _WEEK_RE.search(str(img))
    return int(m.group(1)) if m else 10**9


def _keep_baseline_per_subject(entries: list[dict]) -> list[dict]:
    by_subject: dict[str, dict] = {}
    for e in entries:
        sid = e.get("subject_id") or e.get("subject") or e.get("patient_id")
        if sid is None:
            raise SystemExit(f"datalist entry missing subject_id: {e}")
        if sid not in by_subject or _session_index(e) < _session_index(by_subject[sid]):
            by_subject[sid] = e
    return sorted(by_subject.values(), key=lambda e: e["subject_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path)
    ap.add_argument("--from-datalist", type=Path)
    ap.add_argument("--split", default="validation",
                    help="Which datalist split to stage (default: validation).")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Datalist mode: keep only the earliest session per subject.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination for per-modality NIfTIs.")
    ap.add_argument("--out-labels", type=Path,
                    help="Destination for label copies (datalist mode only).")
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()

    if bool(args.in_dir) == bool(args.from_datalist):
        raise SystemExit("provide exactly one of --in-dir or --from-datalist")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.in_dir is not None:
        sources = sorted(args.in_dir.glob(args.pattern))
        if not sources:
            raise SystemExit(f"no files matching {args.pattern!r} under {args.in_dir}")
        for src in sources:
            _split_one(src, args.out_dir, src.name.split(".nii")[0])
        return

    entries = json.loads(args.from_datalist.read_text()).get(args.split, [])
    if not entries:
        raise SystemExit(f"datalist {args.from_datalist} has no '{args.split}' split")
    if args.out_labels is None:
        raise SystemExit("--out-labels is required in --from-datalist mode")
    args.out_labels.mkdir(parents=True, exist_ok=True)

    if args.baseline_only:
        before = len(entries)
        entries = _keep_baseline_per_subject(entries)
        print(f"[baseline-only] kept {len(entries)}/{before} entries (one per subject)")

    for entry in entries:
        sid = entry["subject_id"]
        _split_one(Path(entry["image"]), args.out_dir, sid)
        # nnU-Net predictions are named <sid>.nii.gz; match for eval.
        shutil.copy(entry["label"], args.out_labels / f"{sid}.nii.gz")


if __name__ == "__main__":
    main()
