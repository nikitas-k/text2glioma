"""Compute paired normalised cross-correlation (NCC) from generated NIfTIs.

Walks a directory of ``sample_cond_native_<case:04d>*.nii.gz`` files produced
by the CFG-sweep pipeline and computes NCC per modality against the paired
ground-truth image from the datalist, without re-running the model. Writes a
long-form ``ncc_vs_cfg.csv`` compatible with ``build_table{1,2}.py``.

NCC is the Pearson correlation of flattened voxel intensities:

    NCC(X, Y) = <X - mean(X), Y - mean(Y)> / (||X - mean(X)|| * ||Y - mean(Y)||)

Two variants are reported:
    * ``ncc_whole``   — over the brain-content mask (image > 1e-6).
    * ``ncc_in_mask`` — restricted to the union of tumour classes in the
                       provided segmentation (label > 0). NaN when no label.

Motivation: Eidex et al. (2024) report NCC (not SSIM) for their T1C
translation model; NCC lets us stage a like-for-like whole-brain comparison
without re-running sampling. See paper/2409.01622v1.pdf Table 1.

Usage
-----

Per-model internal cohort::

    python scripts/compute_ncc_from_niftis.py \\
        --samples_root  /g/data/vp06/$USER/text2glioma_train/runs/stage1_kl1e6_freebits_lc6/data/cfg_sweep_text_only \\
        --datalist      datalist_N1510.json \\
        --split         validation \\
        --out           /g/data/vp06/$USER/text2glioma_train/runs/stage1_kl1e6_freebits_lc6/data/cfg_sweep_text_only/ncc_vs_cfg.csv

LUMIERE external cohort::

    python scripts/compute_ncc_from_niftis.py \\
        --samples_root  /g/data/vp06/$USER/text2glioma_train/runs/lumiere_eval/samples/stage1_kl1e6_freebits_lc6 \\
        --datalist      datalist_lumiere.json \\
        --split         validation \\
        --model_tag     stage1_kl1e6_freebits_lc6 \\
        --out           /g/data/vp06/$USER/text2glioma_train/runs/lumiere_eval/ncc_vs_cfg_lumiere.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd


MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]
CFG_DIR_RE = re.compile(r"^cfg_(?P<w>\d+)p(?P<ff>\d+)$")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def ncc(a: np.ndarray, b: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    """Pearson correlation of flattened arrays, optionally masked.

    Returns ``float('nan')`` when the denominator is zero (constant array
    or empty mask).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"NCC shape mismatch: {a.shape} vs {b.shape}")
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
        a = a[m]
        b = b[m]
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_multi_modality_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI whose last axis is the modality channel (X, Y, Z, C)
    and return ``(C, X, Y, Z)``. If the file is 3-D, return ``(1, X, Y, Z)``."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    if arr.ndim == 4:
        return np.moveaxis(arr, -1, 0)
    if arr.ndim == 3:
        return arr[None]
    raise ValueError(f"unexpected NIfTI shape {arr.shape} at {path}")


def _load_label_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.is_file():
        return None
    lbl = nib.load(str(path)).get_fdata()
    return (lbl > 0)


def _cfg_from_dirname(name: str) -> Optional[float]:
    m = CFG_DIR_RE.match(name)
    if not m:
        return None
    w = int(m.group("w"))
    ff = m.group("ff")
    # "cfg_4p50" -> 4.50; "cfg_7p0" -> 7.0
    return float(f"{w}.{ff}")


def _find_sample(case_dir: Path, case_idx: int, prompt_slug: str) -> Optional[Path]:
    """Look up the native-space sample for a case_idx.

    Tries, in order:
        sample_cond_native_<case>_<slug>.nii.gz
        sample_cond_native_<case>.nii.gz
        sample_cond_<case>_<slug>.nii.gz
        sample_cond_<case>.nii.gz
    """
    stem = f"{case_idx:04d}"
    candidates = [
        case_dir / f"sample_cond_native_{stem}_{prompt_slug}.nii.gz",
        case_dir / f"sample_cond_native_{stem}.nii.gz",
        case_dir / f"sample_cond_{stem}_{prompt_slug}.nii.gz",
        case_dir / f"sample_cond_{stem}.nii.gz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def compute_ncc_for_case(
    sample_path: Path,
    image_path: Path,
    label_path: Optional[Path],
) -> dict[str, dict[str, float]]:
    """Return ``{modality: {'ncc_whole': v, 'ncc_in_mask': v}}``."""
    sample = _load_multi_modality_nifti(sample_path)   # (C, X, Y, Z)
    image  = _load_multi_modality_nifti(image_path)
    if sample.shape != image.shape:
        raise ValueError(
            f"shape mismatch: sample {sample.shape} vs image {image.shape} "
            f"({sample_path.name} vs {image_path.name})"
        )
    tumour_mask = _load_label_mask(label_path)          # (X, Y, Z) or None

    results: dict[str, dict[str, float]] = {}
    for c in range(sample.shape[0]):
        name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}"
        s = sample[c]
        im = image[c]
        # Whole-brain: use the real image's brain-content mask so we don't
        # correlate over the zero-padded skull-stripped background.
        brain_mask = (im > 1e-6)
        results[name] = {
            "ncc_whole":   ncc(s, im, mask=brain_mask),
            "ncc_in_mask": (ncc(s, im, mask=tumour_mask)
                            if tumour_mask is not None else float("nan")),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples_root", type=Path, required=True,
                    help="Directory containing cfg_*/sample_cond_native_*.nii.gz. "
                         "Also supports flat layout without cfg_* subdirs "
                         "(single-CFG runs); use --single_cfg in that case.")
    ap.add_argument("--datalist", type=Path, required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data_root", type=Path, default=None,
                    help="Optional path prefix if datalist entries are relative.")
    ap.add_argument("--prompt_slug", default="real",
                    help="Prompt slug embedded in the sample filename "
                         "(default: 'real').")
    ap.add_argument("--cfg_values", default=None,
                    help="Comma-separated CFG values to include (default: "
                         "auto-discover from cfg_* subdirectories).")
    ap.add_argument("--single_cfg", type=float, default=None,
                    help="If set, treat samples_root as a flat directory of "
                         "samples at this single CFG value (skip cfg_* walk).")
    ap.add_argument("--model_tag", default=None,
                    help="Optional model identifier stored in the 'model' "
                         "column of the output CSV. Useful when aggregating "
                         "multiple models into one CSV (e.g. LUMIERE).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: limit to the first N cases.")
    args = ap.parse_args()

    # Datalist ----------------------------------------------------------------
    with open(args.datalist) as f:
        datalist = json.load(f)
    if args.split not in datalist:
        raise KeyError(f"split {args.split!r} not in {args.datalist}")
    data_items = datalist[args.split]
    if args.limit is not None:
        data_items = data_items[:args.limit]

    def _resolve(p: str | None) -> Optional[Path]:
        if p is None:
            return None
        pp = Path(p)
        if args.data_root and not pp.is_absolute():
            pp = args.data_root / pp
        return pp

    # CFG discovery -----------------------------------------------------------
    if args.single_cfg is not None:
        cfg_dirs: list[tuple[float, Path]] = [(float(args.single_cfg), args.samples_root)]
    else:
        cfg_dirs = []
        for child in sorted(args.samples_root.iterdir()):
            if not child.is_dir():
                continue
            cfg = _cfg_from_dirname(child.name)
            if cfg is None:
                continue
            cfg_dirs.append((cfg, child))
        if args.cfg_values:
            wanted = {float(x) for x in args.cfg_values.split(",")}
            cfg_dirs = [(c, d) for (c, d) in cfg_dirs if c in wanted]

    if not cfg_dirs:
        print(f"[error] no CFG directories under {args.samples_root}", file=sys.stderr)
        sys.exit(1)

    print(f"CFG dirs: {[(c, d.name) for c, d in cfg_dirs]}")
    print(f"N cases:  {len(data_items)}")

    # Sweep -------------------------------------------------------------------
    rows: list[dict] = []
    for cfg, case_dir in cfg_dirs:
        n_found = n_missing = n_shape = n_ok = 0
        for case_idx, item in enumerate(data_items):
            sample_path = _find_sample(case_dir, case_idx, args.prompt_slug)
            if sample_path is None:
                n_missing += 1
                continue
            n_found += 1
            image_path = _resolve(item.get("image"))
            label_path = _resolve(item.get("label"))
            if image_path is None or not image_path.is_file():
                print(f"[skip] case {case_idx}: image not found ({item.get('image')})",
                      file=sys.stderr)
                continue
            try:
                per_mod = compute_ncc_for_case(sample_path, image_path, label_path)
            except ValueError as e:
                print(f"[skip] case {case_idx} @ cfg={cfg}: {e}", file=sys.stderr)
                n_shape += 1
                continue
            n_ok += 1
            for mod, vals in per_mod.items():
                row = {
                    "case_idx":   case_idx,
                    "subject_id": item.get("subject_id", f"idx_{case_idx}"),
                    "modality":   mod,
                    "cfg":        cfg,
                    "ncc_whole":  vals["ncc_whole"],
                    "ncc_in_mask": vals["ncc_in_mask"],
                }
                if args.model_tag is not None:
                    row["model"] = args.model_tag
                rows.append(row)
        print(f"  cfg={cfg:>4.1f}  found={n_found:>4}  missing={n_missing:>4}  "
              f"shape_err={n_shape:>3}  ok={n_ok:>4}")

    if not rows:
        print("[error] no rows written — check --samples_root / --prompt_slug / --datalist",
              file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows, "
          f"{df.case_idx.nunique()} cases, "
          f"{df.cfg.nunique()} CFG values, "
          f"{df.modality.nunique()} modalities)")


if __name__ == "__main__":
    main()
