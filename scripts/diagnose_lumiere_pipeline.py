"""Diagnose input-pipeline consistency between training data and LUMIERE.

Prints raw NIfTI properties and post-transform tensor statistics for one
training-split subject and one LUMIERE subject, side-by-side. Loud
red flags to look for:

* **Raw shape differs by orders of magnitude** on the non-channel axes
  (e.g. training data at (240, 240, 155) vs LUMIERE at (192, 240, 240)).
  The final tensor is always ``(4, 160, 224, 160)`` because both go
  through the same ``SpatialPadd -> CenterSpatialCropd`` combo, but the
  anatomical content of that centre crop depends on the source volume
  size. If the sizes differ, the models saw different anatomy in
  training than they see at test time.
* **Voxel spacing differs** between the two (from the affine diagonal).
  Same target shape at different mm/vox means different fields of view.
* **Modality order mismatch**: raw NIfTI 4th-axis channels are supposed
  to be ``T1, T1CE, T2, FLAIR`` in that order for both datasets. If one
  writes them in a different order, all four channels of the classifier
  see swapped modalities. Symptom: per-channel post-transform means
  should be in a similar rank order (typically T1 < T2 < T1CE < FLAIR
  or similar; big rank changes are suspicious).
* **Skull-strip mismatch**: if the training data is skull-stripped and
  LUMIERE is not (or vice versa), the percentile-clipped intensities
  will be pulled by scalp/skull hyperintensities. Symptom: post-
  transform intensity mean far from 0.5 on any channel.

Usage
-----
::

    python scripts/diagnose_lumiere_pipeline.py \\
        --training_datalist datalist_N494_idh_only.json \\
        --lumiere_datalist  /path/to/datalist_lumiere.json \\
        --n_samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import transforms as T


_TARGET_SPATIAL = (160, 224, 160)


def _build_transforms() -> T.Compose:
    """Copy of `train_molecular_classifier._build_transforms` common list."""
    return T.Compose([
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.SpatialPadd(keys=["image"], spatial_size=_TARGET_SPATIAL, mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=_TARGET_SPATIAL),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0,
            channel_wise=True, clip=True,
        ),
        T.ToTensord(keys=["image"]),
    ])


def _describe_raw(path: Path) -> dict:
    img = nib.load(str(path))
    arr = img.get_fdata()
    affine = np.asarray(img.affine)
    voxel_size = tuple(float(v) for v in np.abs(np.diag(affine))[:3])
    return {
        "path":        str(path),
        "raw_shape":   tuple(int(s) for s in arr.shape),
        "raw_dtype":   str(arr.dtype),
        "axcodes":     "".join(nib.orientations.aff2axcodes(affine)),
        "voxel_size":  voxel_size,
        "nan_count":   int(np.isnan(arr).sum()),
        "min":         float(np.nanmin(arr)),
        "max":         float(np.nanmax(arr)),
        "affine":      affine.tolist(),
    }


def _describe_transformed(item: dict, xforms: T.Compose) -> dict:
    """Run the val transforms and report per-channel intensity stats."""
    out = xforms({"image": str(item["image"])})
    x = out["image"]
    if hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    # (C, H, W, D)
    C = arr.shape[0]
    per_ch = []
    for c in range(C):
        ch = arr[c]
        nonzero = ch[ch > 1e-6]
        per_ch.append({
            "min":     float(ch.min()),
            "max":     float(ch.max()),
            "mean":    float(ch.mean()),
            "std":     float(ch.std()),
            "nonzero_mean": float(nonzero.mean()) if nonzero.size else float("nan"),
            "nonzero_frac": float((ch > 1e-6).mean()),
        })
    return {
        "shape":  tuple(int(s) for s in arr.shape),
        "per_channel": per_ch,
    }


def _load_items(datalist_path: Path, split: str) -> list[dict]:
    with open(datalist_path) as fh:
        dl = json.load(fh)
    entries = dl.get(split, []) or dl.get("validation", []) or dl.get("training", [])
    return list(entries)


def _print_side_by_side(train_diag: dict, lumiere_diag: dict) -> None:
    """Aligned key: value pairs across the two datasets."""
    print(f"{'':<20} {'training':<38}  {'lumiere':<38}")
    print("-" * 100)

    def _row(key: str, tr, lu, formatter=str):
        tr_s = formatter(tr) if tr is not None else "n/a"
        lu_s = formatter(lu) if lu is not None else "n/a"
        marker = "  *" if tr_s != lu_s else ""
        print(f"{key:<20} {tr_s:<38}  {lu_s:<38}{marker}")

    for key in ["raw_shape", "axcodes", "voxel_size", "raw_dtype",
                "nan_count"]:
        _row(key,
              train_diag["raw"].get(key) if train_diag["raw"] else None,
              lumiere_diag["raw"].get(key) if lumiere_diag["raw"] else None)

    _row("raw min/max",
          train_diag["raw"].get("min") if train_diag["raw"] else None,
          lumiere_diag["raw"].get("min") if lumiere_diag["raw"] else None,
          formatter=lambda v: f"{v:.3f}")

    _row("post shape",
          train_diag["post"]["shape"] if train_diag["post"] else None,
          lumiere_diag["post"]["shape"] if lumiere_diag["post"] else None)

    print()
    print("Per-channel post-transform intensity (should be ~[0, 1] with mean ~0.3-0.6):")
    print(f"  {'ch':<3}  {'training  mean/nz_mean/nz_frac':<40}  "
          f"{'lumiere  mean/nz_mean/nz_frac':<40}")
    if train_diag["post"] and lumiere_diag["post"]:
        for c in range(4):
            t = train_diag["post"]["per_channel"][c]
            l = lumiere_diag["post"]["per_channel"][c]
            t_s = f"{t['mean']:.3f} / {t['nonzero_mean']:.3f} / {t['nonzero_frac']:.3f}"
            l_s = f"{l['mean']:.3f} / {l['nonzero_mean']:.3f} / {l['nonzero_frac']:.3f}"
            print(f"  {c:<3}  {t_s:<40}  {l_s:<40}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--training_datalist", type=Path, required=True)
    ap.add_argument("--training_split", type=str, default="validation",
                    help="Which split to pick a training subject from "
                         "(default: validation to avoid training-time "
                         "randomness).")
    ap.add_argument("--lumiere_datalist", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=1,
                    help="How many pairs to inspect. First N from each cohort.")
    args = ap.parse_args()

    train_items = _load_items(args.training_datalist, args.training_split)
    lumiere_items = _load_items(args.lumiere_datalist, "validation")
    print(f"training/{args.training_split}: {len(train_items)} entries",
          file=sys.stderr)
    print(f"lumiere:                        {len(lumiere_items)} entries",
          file=sys.stderr)
    if not train_items or not lumiere_items:
        raise SystemExit("no entries in one or both datalists")

    xforms = _build_transforms()

    for i in range(min(args.n_samples, len(train_items), len(lumiere_items))):
        print()
        print("=" * 100)
        print(f"Sample {i}")
        print("=" * 100)
        t_it = train_items[i]
        l_it = lumiere_items[i]
        t_diag = {"raw": None, "post": None}
        l_diag = {"raw": None, "post": None}
        try:
            t_diag["raw"] = _describe_raw(Path(t_it["image"]))
        except Exception as e:
            print(f"[training] failed to load raw NIfTI: {type(e).__name__}: {e}")
        try:
            l_diag["raw"] = _describe_raw(Path(l_it["image"]))
        except Exception as e:
            print(f"[lumiere ] failed to load raw NIfTI: {type(e).__name__}: {e}")
        try:
            t_diag["post"] = _describe_transformed(t_it, xforms)
        except Exception as e:
            print(f"[training] failed to run transforms: {type(e).__name__}: {e}")
        try:
            l_diag["post"] = _describe_transformed(l_it, xforms)
        except Exception as e:
            print(f"[lumiere ] failed to run transforms: {type(e).__name__}: {e}")
        _print_side_by_side(t_diag, l_diag)

    print()
    print("=" * 100)
    print("Look for '*' markers on the right — those are fields that differ.")
    print("Common failure modes:")
    print("  * raw_shape differs: source volumes are not on the same grid; "
          "SpatialPad/CenterCrop selects different anatomy in each cohort.")
    print("  * voxel_size differs: same tensor shape at different mm/vox means "
          "different fields of view.")
    print("  * per-channel post-transform means differ substantially: modality "
          "order swapped, or one cohort not skull-stripped.")


if __name__ == "__main__":
    main()
