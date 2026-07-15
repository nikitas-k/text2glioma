"""Compute paired normalised cross-correlation (NCC) from generated NIfTIs.

Walks a directory of ``sample_cond_native_<case:04d>*.nii.gz`` files
produced by the CFG-sweep pipeline and pairs each one with the co-located
``sample_original_processed_<case:04d>*.nii.gz`` (real image at the same
preprocessed spatial size). Computes per-modality NCC without re-running
the model, then writes a long-form ``ncc_vs_cfg.csv`` compatible with
``build_table{1,2}.py``.

NCC is the Pearson correlation of flattened voxel intensities:

    NCC(X, Y) = <X - mean(X), Y - mean(Y)> / (||X - mean(X)|| * ||Y - mean(Y)||)

Two variants per (case, cfg, modality) are reported:

    * ``ncc_brain``   — computed over voxels where the paired real image
                        is non-zero (brain-content mask). This matches the
                        convention in Eidex et al. 2024 (arXiv 2409.01622)
                        for whole-brain NCC and is the value to compare
                        against their reported 0.908.
    * ``ncc_in_mask`` — restricted to the tumour segmentation. Requires
                        MONAI + the datalist to bring the raw label into
                        the preprocessed spatial size; skipped (NaN) if
                        MONAI is unavailable or ``--datalist`` is omitted.

Design rationale
----------------

The offline sampler saves three NIfTI pairs per case
(see ``scripts/offline_sample_stage2_compare.py`` lines 745-756):

    sample_cond_<case>.nii.gz                \\  (native scanner space,
    sample_uncond_<case>.nii.gz              /   e.g. 240x240x155)
    sample_original_processed_<case>.nii.gz  \\  preprocessed space
    sample_cond_native_<case>.nii.gz         >  (160x224x160, LPS,
    sample_uncond_native_<case>.nii.gz       /   MSD->T2G channel order)

The *_native* files are the ones the SSIM/LPIPS pipeline uses; pairing
them with *_original_processed* gives a shape-consistent target so no
resampling is required at NCC time.

Usage
-----
::

    python scripts/compute_ncc_from_niftis.py \\
        --samples_root  /g/data/vp06/$USER/text2glioma_train/runs/stage1_kl1e6_freebits_lc6/data/cfg_sweep_text_only \\
        --out           /g/data/vp06/$USER/text2glioma_train/runs/stage1_kl1e6_freebits_lc6/data/cfg_sweep_text_only/ncc_vs_cfg.csv

Add ``--datalist datalist_N1510.json`` to also produce the in-mask NCC
column (requires ``monai``).
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
MSD_TO_T2G = [1, 2, 3, 0]  # matches offline_sample_stage2_compare.py:22


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
# NIfTI helpers
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


def _cfg_from_dirname(name: str) -> Optional[float]:
    m = CFG_DIR_RE.match(name)
    if not m:
        return None
    w = int(m.group("w"))
    ff = m.group("ff")
    return float(f"{w}.{ff}")


def _find_pair(case_dir: Path, case_idx: int,
               prompt_slug: str) -> Optional[tuple[Path, Path]]:
    """Locate ``sample_cond_native_XXXX*.nii.gz`` and the matching
    ``sample_original_processed_XXXX*.nii.gz``. Returns ``None`` if
    either is missing.

    Tries the plain (no-slug) filename first, since the CFG-sweep pipeline
    writes ``sample_cond_native_XXXX.nii.gz`` without a slug. If
    ``prompt_slug`` is non-empty and different from ``''``, the slugged
    variant is tried as a fallback (for runs invoked with
    ``--custom_prompt`` / ``--output_suffix``).
    """
    stem = f"{case_idx:04d}"
    suffixes = [""]
    if prompt_slug:
        suffixes.append(f"_{prompt_slug}")
    for suffix in suffixes:
        pred = case_dir / f"sample_cond_native_{stem}{suffix}.nii.gz"
        real = case_dir / f"sample_original_processed_{stem}{suffix}.nii.gz"
        if pred.is_file() and real.is_file():
            return pred, real
        # Some pipelines only write the "_original_processed" file once per
        # case (independent of the prompt slug); try that fallback too.
        if pred.is_file():
            real_flat = case_dir / f"sample_original_processed_{stem}.nii.gz"
            if real_flat.is_file():
                return pred, real_flat
    return None


# ---------------------------------------------------------------------------
# Optional: preprocessed label loader (for in-mask NCC)
# ---------------------------------------------------------------------------

def _try_import_monai():
    try:
        import torch  # noqa: F401
        from monai import transforms as T
        return T
    except Exception as e:  # pragma: no cover
        print(f"[warn] MONAI not importable ({e}); in-mask NCC will be NaN.",
              file=sys.stderr)
        return None


def _preprocess_label(item: dict, T,
                      channel_reorder: bool,
                      spatial_size: tuple[int, int, int]) -> Optional[np.ndarray]:
    """Apply the same preprocessing as
    ``scripts/offline_sample_stage2_compare.py:_build_val_transform`` to the
    raw image+label pair, and return the label at ``spatial_size`` as a
    boolean tumour mask ``(X, Y, Z)``.

    Uses the raw image only to seed ``CropForegroundd``; it is otherwise
    discarded.
    """
    import torch
    if not item.get("label"):
        return None
    keys = ["image", "label"]
    xforms = [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=["label"], dtype=torch.float32),
        T.Orientationd(keys=keys, axcodes="LPS"),
        T.CropForegroundd(keys=keys, source_key="image"),
        T.SpatialPadd(keys=keys, spatial_size=spatial_size, mode="constant"),
        T.CenterSpatialCropd(keys=keys, roi_size=spatial_size),
        T.ToTensord(keys=keys),
    ])
    batch = T.Compose(xforms)(item)
    label_np = batch["label"].detach().cpu().numpy()
    if label_np.ndim == 4:
        label_np = label_np[0]  # drop channel dim
    return (label_np > 0)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def compute_ncc_for_pair(
    pred_path: Path,
    real_path: Path,
    tumour_mask: Optional[np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """Return ``{modality: {'ncc_brain': v, 'ncc_in_mask': v}}``."""
    pred = _load_multi_modality_nifti(pred_path)   # (C, X, Y, Z)
    real = _load_multi_modality_nifti(real_path)
    if pred.shape != real.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs real {real.shape} "
            f"({pred_path.name} vs {real_path.name})"
        )
    if tumour_mask is not None and tumour_mask.shape != pred.shape[1:]:
        raise ValueError(
            f"tumour_mask shape {tumour_mask.shape} does not match volume "
            f"spatial shape {pred.shape[1:]}"
        )

    out: dict[str, dict[str, float]] = {}
    for c in range(pred.shape[0]):
        name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}"
        p = pred[c]
        r = real[c]
        # Brain-content mask derived from the real image: excludes the
        # zero-padded skull-stripped background so the correlation is
        # not dominated by matching zeros.
        brain_mask = (r > 1e-6)
        out[name] = {
            "ncc_brain":   ncc(p, r, mask=brain_mask),
            "ncc_in_mask": (ncc(p, r, mask=tumour_mask)
                            if tumour_mask is not None else float("nan")),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples_root", type=Path, required=True,
                    help="Directory containing cfg_*/sample_cond_native_*.nii.gz "
                         "and paired sample_original_processed_*.nii.gz. Also "
                         "supports flat layout without cfg_* subdirs (see "
                         "--single_cfg).")
    ap.add_argument("--datalist", type=Path, default=None,
                    help="Optional datalist for in-mask NCC (requires MONAI). "
                         "When omitted, only ncc_brain is computed and "
                         "ncc_in_mask is NaN.")
    ap.add_argument("--split", default="validation",
                    help="Datalist split key (only used with --datalist).")
    ap.add_argument("--prompt_slug", default="",
                    help="Prompt slug embedded in the sample filename. "
                         "Default is empty (matches plain "
                         "'sample_cond_native_XXXX.nii.gz' as written by the "
                         "CFG-sweep pipeline). Set to e.g. 'real' or "
                         "'a_ood_vocab' to match runs launched with "
                         "--custom_prompt / --output_suffix.")
    ap.add_argument("--cfg_values", default=None,
                    help="Comma-separated CFG values to include (default: "
                         "auto-discover from cfg_* subdirectories).")
    ap.add_argument("--single_cfg", type=float, default=None,
                    help="If set, treat samples_root as a flat directory of "
                         "samples at this single CFG value (skip cfg_* walk).")
    ap.add_argument("--model_tag", default=None,
                    help="Optional model identifier stored in the 'model' "
                         "column of the output CSV.")
    ap.add_argument("--no_channel_reorder", action="store_true", default=False,
                    help="Skip MSD->T2G channel reorder when preprocessing the "
                         "label (default: apply, matching offline sampler).")
    ap.add_argument("--spatial_size", type=int, nargs=3, default=(160, 224, 160),
                    help="Preprocessed spatial size for label preprocessing.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: limit to the first N cases.")
    args = ap.parse_args()

    # Optional MONAI import + datalist load for in-mask NCC ---------------
    T = None
    data_items = None
    if args.datalist is not None:
        T = _try_import_monai()
        if T is not None:
            with open(args.datalist) as f:
                dl = json.load(f)
            if args.split not in dl:
                raise KeyError(f"split {args.split!r} not in {args.datalist}")
            data_items = dl[args.split]
            if args.limit is not None:
                data_items = data_items[:args.limit]
    else:
        print("[info] --datalist not supplied; ncc_in_mask will be NaN.",
              file=sys.stderr)

    # CFG discovery -------------------------------------------------------
    if args.single_cfg is not None:
        cfg_dirs: list[tuple[float, Path]] = [(float(args.single_cfg),
                                               args.samples_root)]
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
        print(f"[error] no CFG directories under {args.samples_root}",
              file=sys.stderr)
        sys.exit(1)

    print(f"CFG dirs: {[(c, d.name) for c, d in cfg_dirs]}")
    if data_items is not None:
        print(f"N cases:  {len(data_items)}  (for in-mask NCC)")

    # Cache preprocessed tumour masks so re-visiting a case at another CFG
    # doesn't re-run MONAI transforms.
    tumour_cache: dict[int, Optional[np.ndarray]] = {}

    def _get_tumour_mask(case_idx: int) -> Optional[np.ndarray]:
        if T is None or data_items is None:
            return None
        if case_idx in tumour_cache:
            return tumour_cache[case_idx]
        if case_idx >= len(data_items):
            tumour_cache[case_idx] = None
            return None
        try:
            m = _preprocess_label(
                dict(data_items[case_idx]), T,
                channel_reorder=not args.no_channel_reorder,
                spatial_size=tuple(args.spatial_size),
            )
        except Exception as e:
            print(f"[warn] preprocessing label for case {case_idx} failed: {e}",
                  file=sys.stderr)
            m = None
        tumour_cache[case_idx] = m
        return m

    # Sweep ---------------------------------------------------------------
    rows: list[dict] = []
    for cfg, case_dir in cfg_dirs:
        n_found = n_shape = n_ok = 0
        miss_streak = 0
        max_case = args.limit if args.limit is not None else 3000
        for case_idx in range(max_case):
            pair = _find_pair(case_dir, case_idx, args.prompt_slug)
            if pair is None:
                miss_streak += 1
                # Bail out early after many consecutive misses if we've
                # already found some samples (typical dataset is a few
                # hundred cases).
                if miss_streak > 50 and n_found > 0:
                    break
                continue
            miss_streak = 0
            n_found += 1
            pred_path, real_path = pair
            tumour_mask = _get_tumour_mask(case_idx)
            try:
                per_mod = compute_ncc_for_pair(pred_path, real_path,
                                               tumour_mask)
            except ValueError as e:
                print(f"[skip] case {case_idx} @ cfg={cfg}: {e}",
                      file=sys.stderr)
                n_shape += 1
                continue
            n_ok += 1
            subj = (data_items[case_idx].get("subject_id", f"idx_{case_idx}")
                    if data_items is not None and case_idx < len(data_items)
                    else f"idx_{case_idx}")
            for mod, vals in per_mod.items():
                row = {
                    "case_idx":    case_idx,
                    "subject_id":  subj,
                    "modality":    mod,
                    "cfg":         cfg,
                    "ncc_brain":   vals["ncc_brain"],
                    "ncc_in_mask": vals["ncc_in_mask"],
                }
                if args.model_tag is not None:
                    row["model"] = args.model_tag
                rows.append(row)
        print(f"  cfg={cfg:>4.1f}  found={n_found:>4}  "
              f"shape_err={n_shape:>3}  ok={n_ok:>4}")

    if not rows:
        print("[error] no rows written \u2014 check --samples_root / --prompt_slug",
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
