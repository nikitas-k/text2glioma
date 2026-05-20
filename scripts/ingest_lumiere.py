#!/usr/bin/env python
"""Ingest the LUMIERE longitudinal glioblastoma cohort as an external test set.

LUMIERE (Suter et al., 2022) contains 91 patients with 4-modality MRI
(T1, T1CE/CT1, T2, FLAIR) at multiple longitudinal timepoints, skull-stripped
and co-registered to a common space, with DeepBraTumIA-derived tumour
segmentations refined by an expert.  Reference:
  Suter Y et al., "The LUMIERE dataset", Scientific Data 2022.

This script walks a LUMIERE root directory and produces:
  1. ``images/<subject>_<session>.nii.gz`` — 4-channel NIfTI stacked in
     pipeline order T1 / T1CE / T2 / FLAIR (channel axis = last, as the
     val transform expects ``channel_dim=3``).
  2. ``labels/<subject>_<session>.nii.gz`` — single-channel segmentation
     mask copied verbatim (LUMIERE uses the BraTS 0/1/2/3 convention).
  3. ``datalist_lumiere.json`` — MONAI-style datalist with one entry per
     session under the ``"validation"`` key (matches the offline-sample
     CLI's ``--split`` choices).  Each entry has ``image``, ``label``,
     ``subject_id``, ``session``, ``impression``, ``findings``.

The text prompts are generated with the same VASARI-auto composer used to
build the in-distribution datalist, so the conditioning distribution is
directly comparable.

Patient-level disjoint test split (no patient appears in both training
and test) is guaranteed by construction — LUMIERE is fully held out.

Usage
-----
On Gadi (where the data lives)::

    python scripts/ingest_lumiere.py \\
        --lumiere_root /g/data/vp06/$USER/LUMIERE \\
        --output_dir   /g/data/vp06/$USER/text2glioma_train/data/lumiere_ingested \\
        --workers 8

Then to run the offline-sample pipeline on it::

    python scripts/offline_sample_stage2_compare.py \\
        --datalist /g/data/vp06/$USER/text2glioma_train/data/lumiere_ingested/datalist_lumiere.json \\
        --split validation \\
        --num_cases 30 \\
        --config       configs/ldm_radbert_pinaya_decoder_only.yaml \\
        --stage1_config configs/stage1_pinaya_decoder_only.yaml \\
        --stage1_uri   <path>/autoencoder_stage1/final_model.pth \\
        --model_ckpt   <path>/ldm_stage2/best_model.pth \\
        --no_channel_reorder \\
        --text_field findings \\
        --output_dir   <path>/lumiere_test_run

Notes on label conventions
--------------------------
LUMIERE's DeepBraTumIA segmentation uses the **BraTS** label convention:
``0=background, 1=non-enhancing tumour core (necrosis), 2=oedema, 4=enhancing``
(legacy BraTS) or 1/2/3 after the post-2020 remap.  Pass
``--enhancing_label / --nonenhancing_label / --oedema_label`` to override the
defaults if the LUMIERE release you have uses different integers.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np

# Allow running the script directly from a checkout without `pip install -e .`.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Pipeline modality order (matches `--no_channel_reorder` ingestion of N1510).
PIPELINE_MODALITIES = ("T1", "T1CE", "T2", "FLAIR")

# Filename patterns used by LUMIERE's DeepBraTumIA folder layout, with
# alternative names used in different release variants.
_FILENAME_PATTERNS = {
    "T1":    [r"(?i)(?:^|/|_)T1(?:[_.]|_skull_strip|_brain)",  r"(?i)(?:^|/|_)T1w",         r"(?i)(?:^|/|_)T1\.nii"],
    "T1CE":  [r"(?i)(?:^|/|_)CT1",                              r"(?i)(?:^|/|_)T1c(?:e)?",  r"(?i)(?:^|/|_)T1Gd",
              r"(?i)(?:^|/|_)T1ce",                             r"(?i)(?:^|/|_)T1_ce",      r"(?i)(?:^|/|_)T1post"],
    "T2":    [r"(?i)(?:^|/|_)T2(?:[_.]|_skull_strip|_brain)",   r"(?i)(?:^|/|_)T2w",         r"(?i)(?:^|/|_)T2\.nii"],
    "FLAIR": [r"(?i)(?:^|/|_)FLAIR",                            r"(?i)(?:^|/|_)Flair"],
}

_SEGMENTATION_PATTERNS = [
    r"(?i)Segmentation\.nii",
    r"(?i)_seg(?:mentation)?\.nii",
    r"(?i)tumor_segmentation\.nii",
    r"(?i)/seg/",
]

_SESSION_DIR_PATTERNS = [
    r"(?i)week[-_]?\d+",
    r"(?i)session[-_]?\d+",
    r"(?i)tp[-_]?\d+",
    r"(?i)visit[-_]?\d+",
]

_SUBJECT_DIR_PATTERN = r"(?i)patient[-_]?\d+"

log = logging.getLogger("ingest_lumiere")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _looks_like_subject_dir(p: Path) -> bool:
    return re.search(_SUBJECT_DIR_PATTERN, p.name) is not None


def _looks_like_session_dir(p: Path) -> bool:
    return any(re.search(pat, p.name) for pat in _SESSION_DIR_PATTERNS)


def _find_one(candidates: Iterable[Path], patterns: list[str]) -> Optional[Path]:
    """Return the first candidate whose path matches any of `patterns`."""
    hits: list[Path] = []
    for c in candidates:
        s = str(c)
        if any(re.search(pat, s) for pat in patterns):
            hits.append(c)
    if not hits:
        return None
    # Prefer skull-stripped / brain-extracted variants when both exist.
    for keyword in ("skull_strip", "brain", "ss"):
        for h in hits:
            if keyword in h.name.lower():
                return h
    # Otherwise prefer the shortest filename (typically the canonical one).
    hits.sort(key=lambda p: (len(p.name), str(p)))
    return hits[0]


def _discover_session(session_dir: Path) -> Optional[dict]:
    """Return a dict {modality: Path, 'label': Path, 'subject': str, 'session': str}
    or None if a modality or the segmentation cannot be located.
    """
    niftis = sorted(p for p in session_dir.rglob("*.nii*")
                    if p.is_file() and p.suffix in (".gz", ".nii"))
    if not niftis:
        return None

    mods: dict[str, Path] = {}
    for mod in PIPELINE_MODALITIES:
        path = _find_one(niftis, _FILENAME_PATTERNS[mod])
        if path is None:
            return None
        mods[mod] = path

    seg = _find_one(niftis, _SEGMENTATION_PATTERNS)
    if seg is None:
        return None

    subject_dir = session_dir
    # Walk up until we find a directory that looks like a patient folder.
    for _ in range(6):
        if _looks_like_subject_dir(subject_dir):
            break
        subject_dir = subject_dir.parent
    subject = subject_dir.name if _looks_like_subject_dir(subject_dir) else session_dir.parent.name

    return {
        **mods,
        "label": seg,
        "subject": subject,
        "session": session_dir.name,
    }


def discover_sessions(root: Path) -> list[dict]:
    """Walk the LUMIERE root and yield one dict per discovered session."""
    sessions: list[dict] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or not _looks_like_subject_dir(subject_dir):
            continue
        # First look for direct child session directories; if none, descend
        # into known LUMIERE intermediates (e.g. DeepBraTumIA-segmentation/atlas).
        candidates = [d for d in subject_dir.iterdir() if d.is_dir()]
        sess_dirs = [d for d in candidates if _looks_like_session_dir(d)]
        if not sess_dirs:
            # Fall back to a recursive scan; pick the deepest dirs containing NIfTIs.
            for nifti in subject_dir.rglob("*.nii*"):
                d = nifti.parent
                while d != subject_dir and not _looks_like_session_dir(d):
                    d = d.parent
                if _looks_like_session_dir(d) and d not in sess_dirs:
                    sess_dirs.append(d)

        for sd in sess_dirs:
            info = _discover_session(sd)
            if info is None:
                log.warning("Session %s missing one or more modalities/segmentation; skipped.", sd)
                continue
            info["subject"] = subject_dir.name  # canonical subject name
            sessions.append(info)
    return sessions


# ---------------------------------------------------------------------------
# Per-session ingestion
# ---------------------------------------------------------------------------

def _check_shape_consistency(arrs: dict[str, np.ndarray]) -> tuple[bool, str]:
    shapes = {k: a.shape for k, a in arrs.items()}
    if len({s for s in shapes.values()}) != 1:
        return False, f"Modality shape mismatch: {shapes}"
    return True, ""


def _ingest_one(info: dict, output_root: Path,
                gen_prompts: bool, vasari_kwargs: dict,
                enhancing_label: int, nonenhancing_label: int, edema_label: int) -> Optional[dict]:
    subject = info["subject"]
    session = info["session"]
    case_id = f"{subject}_{session}".replace(" ", "_")

    img_out = output_root / "images" / f"{case_id}.nii.gz"
    lbl_out = output_root / "labels" / f"{case_id}.nii.gz"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done.
    if img_out.exists() and lbl_out.exists():
        log.info("[%s] cached", case_id)
    else:
        # Load each modality, validate shape, stack as channel-last.
        ref_nii = nib.load(str(info["T1"]))
        ref_aff = ref_nii.affine
        ref_hdr = ref_nii.header
        arrs: dict[str, np.ndarray] = {"T1": np.asarray(ref_nii.get_fdata(), dtype=np.float32)}
        for mod in ("T1CE", "T2", "FLAIR"):
            n = nib.load(str(info[mod]))
            # Sanity check geometry — LUMIERE is registered, so affines should match.
            if not np.allclose(n.affine, ref_aff, atol=1e-3):
                log.warning("[%s] %s affine differs from T1; data is supposed to be co-registered.",
                            case_id, mod)
            arrs[mod] = np.asarray(n.get_fdata(), dtype=np.float32)

        ok, msg = _check_shape_consistency(arrs)
        if not ok:
            log.error("[%s] %s — skipping", case_id, msg)
            return None

        stacked = np.stack([arrs[m] for m in PIPELINE_MODALITIES], axis=-1)  # (H, W, D, 4)
        nib.Nifti1Image(stacked, ref_aff, ref_hdr).to_filename(str(img_out))

        # Copy label verbatim (uint8 to keep file small; LUMIERE values are small ints).
        seg = nib.load(str(info["label"]))
        seg_arr = np.asarray(seg.get_fdata()).round().astype(np.uint8)
        nib.Nifti1Image(seg_arr, seg.affine, seg.header).to_filename(str(lbl_out))

    # Prompt generation (lazy import — VASARI atlas may not be on the worker's PATH).
    impression, findings = "", ""
    if gen_prompts:
        try:
            from text2glioma.preprocessing.utils import compose_radiology_prompts
            out = compose_radiology_prompts(
                image_path=str(img_out),
                label_path=str(lbl_out),
                enhancing_label=enhancing_label,
                nonenhancing_label=nonenhancing_label,
                edema_label=edema_label,
                **vasari_kwargs,
            )
            impression = str(out.get("short", "")).strip()
            findings   = str(out.get("long", "")).strip()
        except Exception as exc:
            log.warning("[%s] prompt generation failed (%s); leaving impression/findings empty.",
                        case_id, exc)

    entry = {
        "image": str(img_out),
        "label": str(lbl_out),
        "subject_id": case_id,
        "subject": subject,
        "session": session,
        "impression": impression,
        "findings":   findings,
        "source_dataset": "LUMIERE",
    }
    log.info("[%s] OK%s", case_id, "" if impression else "  (no prompts)")
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest LUMIERE for use as an external test set.")
    p.add_argument("--lumiere_root", type=Path, required=True,
                   help="Root directory of the LUMIERE release (contains Patient-XX/ subfolders).")
    p.add_argument("--output_dir",   type=Path, required=True,
                   help="Where to write the stacked images, labels, and datalist JSON.")
    p.add_argument("--datalist_name", type=str, default="datalist_lumiere.json")
    p.add_argument("--workers", type=int, default=4, help="Parallel session ingestion (default 4).")
    p.add_argument("--no_prompts", action="store_true",
                   help="Skip VASARI-auto prompt generation (leaves impression/findings empty). "
                        "Useful for smoke testing.")
    p.add_argument("--enhancing_label",     type=int, default=3)
    p.add_argument("--nonenhancing_label",  type=int, default=1)
    p.add_argument("--edema_label",         type=int, default=2)
    p.add_argument("--atlas_dir", type=str, default=None,
                   help="Override VASARI atlas directory (default: package-bundled).")
    p.add_argument("--limit",   type=int, default=None,
                   help="Process at most N sessions (smoke test).")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--quiet",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = args.lumiere_root.expanduser().resolve()
    out  = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        log.error("Lumiere root not found: %s", root); sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    log.info("Discovering sessions under %s ...", root)
    sessions = discover_sessions(root)
    if args.limit:
        sessions = sessions[: args.limit]
    log.info("Found %d sessions across %d subjects.",
             len(sessions), len({s["subject"] for s in sessions}))
    if not sessions:
        sys.exit("No LUMIERE sessions discovered — check --lumiere_root layout.")

    vasari_kwargs = {}
    if args.atlas_dir is not None:
        vasari_kwargs["atlas_dir"] = args.atlas_dir

    entries: list[dict] = []
    if args.workers > 1 and not args.no_prompts:
        log.warning("Reducing workers to 1 — VASARI-auto is not always fork-safe. "
                    "Use --no_prompts to ingest in parallel and run prompt generation as a second pass.")
        args.workers = 1

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_ingest_one, info, out, not args.no_prompts, vasari_kwargs,
                            args.enhancing_label, args.nonenhancing_label, args.edema_label): info
                for info in sessions
            }
            for fut in as_completed(futures):
                entry = fut.result()
                if entry:
                    entries.append(entry)
    else:
        for info in sessions:
            entry = _ingest_one(info, out, not args.no_prompts, vasari_kwargs,
                                args.enhancing_label, args.nonenhancing_label, args.edema_label)
            if entry:
                entries.append(entry)

    # Deterministic ordering for reproducibility.
    entries.sort(key=lambda e: e["subject_id"])

    datalist = {
        "training":   [],
        "validation": entries,   # entire LUMIERE cohort exposed under the "validation" split
        "testing":    [],
        "_metadata": {
            "source_dataset": "LUMIERE",
            "modality_order": list(PIPELINE_MODALITIES),
            "channel_dim": 3,
            "channel_reorder_required": False,
            "label_values": {
                "background": 0,
                "nonenhancing": args.nonenhancing_label,
                "edema":         args.edema_label,
                "enhancing":     args.enhancing_label,
            },
            "n_subjects": len({e["subject"] for e in entries}),
            "n_sessions": len(entries),
            "seed": args.seed,
        },
    }
    out_json = out / args.datalist_name
    with open(out_json, "w") as f:
        json.dump(datalist, f, indent=2)
    log.info("Wrote %d entries to %s", len(entries), out_json)
    log.info("Summary: %d subjects, %d sessions, %d with empty prompts.",
             datalist["_metadata"]["n_subjects"],
             datalist["_metadata"]["n_sessions"],
             sum(1 for e in entries if not e["impression"]))


if __name__ == "__main__":
    main()
