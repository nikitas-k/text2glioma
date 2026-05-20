#!/usr/bin/env python
"""Register LUMIERE sessions to the SRI24 atlas with ANTs (rigid + affine).

VASARI-auto expects the segmentation to be in the SRI24/BraTS atlas grid
(240×240×155, 1 mm isotropic) so the bundled atlas-region masks broadcast
correctly against the input label.  LUMIERE is co-registered *within* a
session but not to any standard atlas, so we register each session's T1
to the SRI24 T1 template and re-apply that single transform to the other
three modalities (linear interpolation) and to the segmentation (nearest
neighbour, to preserve discrete labels).

Per session this script writes::

    <output_dir>/registered/<subject>_<session>/
        T1.nii.gz   T1CE.nii.gz   T2.nii.gz   FLAIR.nii.gz   label.nii.gz
        T1_to_SRI240GenericAffine.mat   (kept for provenance)

After this runs, re-invoke ``scripts/ingest_lumiere.py`` with
``--registered_dir <output_dir>/registered`` and it will stack the
registered NIfTIs instead of the raw ones.

Usage (Gadi)::

    module load ants/2.5.1   # or whatever module name your system uses
    python scripts/register_lumiere_to_sri24.py \\
        --lumiere_root /g/data/vp06/$USER/LUMIERE \\
        --output_dir   /g/data/vp06/$USER/text2glioma_train/data/lumiere \\
        --workers 8

If ANTs isn't on $PATH set ``--ants_bin /path/to/ants/bin``.

The registration is rigid+affine only.  Brain tissue is preserved
faithfully at this resolution and orientation; VASARI location lookups
depend on rough spatial agreement with the atlas, not on cortical-level
SyN warping.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Reuse the discovery logic from the ingestion script so registration sees
# exactly the same sessions and modality files.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for ingest_lumiere import
from ingest_lumiere import discover_sessions, PIPELINE_MODALITIES  # noqa: E402

log = logging.getLogger("register_lumiere")

_ATLAS_RELPATH = "src/text2glioma/preprocessing/atlas_masks/sri24/MNI152_in_SRI24_T1_1mm_brain.nii.gz"


def _run(cmd: list[str], log_prefix: str) -> None:
    """Run a CLI command, streaming stderr/stdout into the logger on failure."""
    log.debug("[%s] $ %s", log_prefix, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("[%s] command failed (rc=%d): %s",
                  log_prefix, res.returncode, " ".join(cmd))
        if res.stdout:
            log.error("[%s] stdout: %s", log_prefix, res.stdout[-2000:])
        if res.stderr:
            log.error("[%s] stderr: %s", log_prefix, res.stderr[-2000:])
        raise RuntimeError(f"{log_prefix}: command failed")


def _register_one(info: dict, atlas: Path, out_root: Path,
                  ants_bin: Optional[Path], overwrite: bool) -> Optional[dict]:
    subject = info["subject"]
    session = info["session"]
    case_id = f"{subject}_{session}".replace(" ", "_")

    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "T1":    case_dir / "T1.nii.gz",
        "T1CE":  case_dir / "T1CE.nii.gz",
        "T2":    case_dir / "T2.nii.gz",
        "FLAIR": case_dir / "FLAIR.nii.gz",
        "label": case_dir / "label.nii.gz",
    }
    transform_mat = case_dir / "T1_to_SRI240GenericAffine.mat"
    transform_prefix = case_dir / "T1_to_SRI24"

    if not overwrite and all(p.exists() for p in out_files.values()):
        log.info("[%s] cached", case_id); return {"case_id": case_id, **{k: str(v) for k, v in out_files.items()}}

    env = os.environ.copy()
    if ants_bin is not None:
        env["PATH"] = f"{ants_bin}{os.pathsep}{env.get('PATH','')}"
        # antsRegistrationSyNQuick.sh checks $ANTSPATH for its helper binaries.
        env["ANTSPATH"] = str(ants_bin).rstrip("/") + "/"

    def _which(name: str) -> str:
        if ants_bin is not None:
            cand = ants_bin / name
            if cand.exists(): return str(cand)
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f"{name} not found on PATH (set --ants_bin).")
        return path

    syn_quick    = _which("antsRegistrationSyNQuick.sh")
    apply_trans  = _which("antsApplyTransforms")

    # ── (1) Register T1 → SRI24 T1 (rigid + affine) ──────────────────────
    # -t a    : translation + rigid + affine (no SyN, fast, atlas-grade)
    # -n N    : threads
    # -p f    : single-precision (faster)
    log.info("[%s] running antsRegistrationSyNQuick (rigid+affine)...", case_id)
    cmd = [
        "bash", syn_quick,
        "-d", "3",
        "-f", str(atlas),
        "-m", str(info["T1"]),
        "-o", str(transform_prefix),
        "-t", "a",
        "-n", "4",
        "-p", "f",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        log.error("[%s] antsRegistrationSyNQuick failed: %s", case_id, res.stderr[-2000:])
        return None

    # antsRegistrationSyNQuick writes <prefix>Warped.nii.gz (the moved T1).
    syn_warped = Path(str(transform_prefix) + "Warped.nii.gz")
    if syn_warped.exists():
        shutil.move(str(syn_warped), str(out_files["T1"]))
    if not transform_mat.exists():
        log.error("[%s] expected transform %s not produced", case_id, transform_mat)
        return None

    # ── (2) Apply transform to T1CE / T2 / FLAIR (linear) ────────────────
    for mod in ("T1CE", "T2", "FLAIR"):
        cmd = [
            apply_trans,
            "-d", "3",
            "-i", str(info[mod]),
            "-r", str(atlas),
            "-o", str(out_files[mod]),
            "-t", str(transform_mat),
            "-n", "Linear",
        ]
        try:
            _run(cmd, f"{case_id}:{mod}")
        except RuntimeError:
            return None

    # ── (3) Apply transform to label (nearest neighbour — preserve ints) ─
    cmd = [
        apply_trans,
        "-d", "3",
        "-i", str(info["label"]),
        "-r", str(atlas),
        "-o", str(out_files["label"]),
        "-t", str(transform_mat),
        "-n", "NearestNeighbor",
        "-u", "int",
    ]
    try:
        _run(cmd, f"{case_id}:label")
    except RuntimeError:
        return None

    # Clean up SyN-Quick's intermediate "InverseWarped" file (we didn't ask
    # for SyN so it's just the inverse-affine sanity image; saves disk).
    inv = Path(str(transform_prefix) + "InverseWarped.nii.gz")
    if inv.exists():
        try: inv.unlink()
        except OSError: pass

    log.info("[%s] OK", case_id)
    return {"case_id": case_id, **{k: str(v) for k, v in out_files.items()}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register LUMIERE sessions to SRI24 with ANTs.")
    p.add_argument("--lumiere_root", type=Path, required=True)
    p.add_argument("--output_dir",   type=Path, required=True,
                   help="Parent dir; registered NIfTIs go to <output_dir>/registered/<case_id>/.")
    p.add_argument("--atlas", type=Path, default=None,
                   help="SRI24 T1 template (default: bundled MNI152_in_SRI24_T1_1mm_brain.nii.gz).")
    p.add_argument("--ants_bin", type=Path, default=None,
                   help="Directory containing antsRegistrationSyNQuick.sh / antsApplyTransforms.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = args.lumiere_root.expanduser().resolve()
    out  = (args.output_dir.expanduser().resolve() / "registered")
    out.mkdir(parents=True, exist_ok=True)

    if args.atlas is None:
        atlas = Path(__file__).resolve().parent.parent / _ATLAS_RELPATH
    else:
        atlas = args.atlas.expanduser().resolve()
    if not atlas.is_file():
        sys.exit(f"Atlas not found: {atlas}")
    log.info("Atlas (fixed): %s", atlas)

    log.info("Discovering sessions under %s ...", root)
    sessions = discover_sessions(root)
    if args.limit:
        sessions = sessions[: args.limit]
    log.info("Registering %d sessions (workers=%d) ...", len(sessions), args.workers)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_register_one, info, atlas, out, args.ants_bin, args.overwrite)
                       for info in sessions]
            for fut in as_completed(futures):
                fut.result()
    else:
        for info in sessions:
            _register_one(info, atlas, out, args.ants_bin, args.overwrite)

    log.info("Done.  Now run scripts/ingest_lumiere.py with --registered_dir %s", out)


if __name__ == "__main__":
    main()
