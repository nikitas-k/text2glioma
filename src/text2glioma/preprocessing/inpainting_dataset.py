"""Inpainting dataset wrapper for BraTS-GLI 2025 longitudinal pairs.

Yields one training sample per pair from
``datalist_brats_gli_2025_pairs_split.json``.

Design notes
------------
- **No text.** BraTS-GLI 2025 has no paired radiology reports; conditioning is
  purely categorical (trajectory + treatment_status_a + treatment_status_b).
- **No combined N1510 cohort.** The N1510 source set overlaps with BraTS
  contributors, so training only on the BraTS pairs avoids hidden duplication.
- **Spatial correspondence.** image_a, image_b, label_a, label_b are
  intra-patient co-registered by BraTS-GLI. The transform pipeline applies
  every spatial op (Orientation, CropForeground, SpatialPad, CenterSpatialCrop,
  RandFlip, RandAffine) to all four keys with shared parameters so the
  correspondence is preserved through training.
- **Mask comes from the (A, B) union.** ``sample_pair_inpainting_mask`` returns
  a dilated (M_A \u222a M_B) \u2229 brain ROI \u2014 the region inside which the future
  tumour must live. The model's job is to predict ``image_b`` inside the
  mask, given ``masked_image_a`` (image_a with the ROI zeroed) plus the
  trajectory / treatment_status labels.

Sample schema
-------------
After ``build_pair_transforms`` + ``PairInpaintingMaskd`` runs, each sample dict
contains the following tensors on the canonical (D, H, W) = (160, 224, 160) grid:

  image_a        : (4, D, H, W) float in [0, 1]  \u2014 visible baseline scan
  image_b        : (4, D, H, W) float in [0, 1]  \u2014 ground-truth follow-up
  label_a        : (1, D, H, W) float            \u2014 tumour seg at visit A
  label_b        : (1, D, H, W) float            \u2014 tumour seg at visit B
  mask           : (1, D, H, W) float in {0, 1}  \u2014 ROI to inpaint
  masked_image_a : (4, D, H, W) float            \u2014 image_a * (1 - mask)

Plus passthrough scalar keys (added in ``prepare_pair_records``):

  subject_id     : str
  timepoint_a/b  : str
  trajectory     : int  (0=response, 1=stable, 2=progression)
  treatment_a    : int  (0=pre, 1=post)
  treatment_b    : int  (0=pre, 1=post)
  stratum        : str  e.g. 'post->post/stable'  \u2014 used by weighted sampler
"""
from __future__ import annotations

import collections
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from monai import transforms as T
from monai.config import KeysCollection
from monai.transforms import MapTransform

from text2glioma.preprocessing.inpainting_masks import (
    TRAJECTORY_CLASSES,
    apply_inpainting_mask,
    sample_pair_inpainting_mask,
)


TRAJECTORY_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TRAJECTORY_CLASSES)}
TREATMENT_TO_IDX: dict[str, int] = {"pre": 0, "post": 1}

SPATIAL_KEYS = ("image_a", "image_b", "label_a", "label_b")
IMAGE_KEYS = ("image_a", "image_b")
LABEL_KEYS = ("label_a", "label_b")

CANONICAL_SHAPE = (160, 224, 160)


# ---------------------------------------------------------------------------
# Record preparation
# ---------------------------------------------------------------------------

def prepare_pair_records(pairs: Sequence[Mapping]) -> list[dict]:
    """Convert raw datalist pair dicts into MONAI-ready records.

    - Categorical labels are encoded to int indices.
    - A ``stratum`` string ``'<dir>/<traj>'`` is added for weighted sampling.
    - Unknown / unmapped trajectories are skipped (with a print warning) rather
      than silently demoted, so we fail loudly on schema drift.
    """
    out: list[dict] = []
    skipped = 0
    for p in pairs:
        traj = p["trajectory"]
        if traj not in TRAJECTORY_TO_IDX:
            skipped += 1
            continue
        ta, tb = p["treatment_status_a"], p["treatment_status_b"]
        record = {
            "image_a":     p["image_a"],   # list of 4 modality paths
            "image_b":     p["image_b"],
            "label_a":     p["label_a"],
            "label_b":     p["label_b"],
            "subject_id":  p["subject_id"],
            "timepoint_a": p["timepoint_a"],
            "timepoint_b": p["timepoint_b"],
            "trajectory":  TRAJECTORY_TO_IDX[traj],
            "treatment_a": TREATMENT_TO_IDX[ta],
            "treatment_b": TREATMENT_TO_IDX[tb],
            "stratum":     f"{ta}->{tb}/{traj}",
        }
        out.append(record)
    if skipped:
        print(f"[prepare_pair_records] skipped {skipped} pairs with unknown trajectory")
    return out


# ---------------------------------------------------------------------------
# Mask + masked-image injection transform
# ---------------------------------------------------------------------------

class PairInpaintingMaskd(MapTransform):
    """Inject ``mask`` and ``masked_image_a`` into the sample dict.

    Must run AFTER spatial normalisation (Orientationd / CropForegroundd /
    SpatialPadd / CenterSpatialCropd) and AFTER intensity normalisation, so
    that ``masked_image_a`` carries the same intensity scale as ``image_a``.
    """

    def __init__(
        self,
        image_key: str = "image_a",
        label_a_key: str = "label_a",
        label_b_key: str = "label_b",
        mask_key: str = "mask",
        masked_image_key: str = "masked_image_a",
        dilation_mm: float = 18.0,
        voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=(image_key, label_a_key, label_b_key),
                         allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.label_a_key = label_a_key
        self.label_b_key = label_b_key
        self.mask_key = mask_key
        self.masked_image_key = masked_image_key
        self.dilation_mm = dilation_mm
        self.voxel_size_mm = voxel_size_mm
        # Use a per-instance Generator so seeds are reproducible at sample level
        # only when the user explicitly provides one. Otherwise inherit Python /
        # numpy global state, which under DataLoader workers is itself seeded
        # from worker_init_fn.
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def __call__(self, data: Mapping) -> dict:
        d = dict(data)
        image_a = d[self.image_key]
        if not isinstance(image_a, torch.Tensor):
            image_a = torch.as_tensor(image_a)
        label_a = d[self.label_a_key]
        if not isinstance(label_a, torch.Tensor):
            label_a = torch.as_tensor(label_a)
        label_b = d[self.label_b_key]
        if not isinstance(label_b, torch.Tensor):
            label_b = torch.as_tensor(label_b)

        mask = sample_pair_inpainting_mask(
            image_a=image_a,
            label_a=label_a,
            label_b=label_b,
            rng=self._rng,
            dilation_mm=self.dilation_mm,
            voxel_size_mm=self.voxel_size_mm,
        )
        masked = apply_inpainting_mask(image_a.float(), mask, fill_value=0.0)

        d[self.mask_key] = mask
        d[self.masked_image_key] = masked
        return d


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def _spatial_modes_for(keys: Sequence[str], image_mode: str, label_mode: str) -> list[str]:
    """Per-key interpolation mode list for RandAffined etc.

    Images get ``image_mode`` (e.g. 'bilinear'), labels get ``label_mode``
    (typically 'nearest' to keep segmentation integer-valued).
    """
    out = []
    for k in keys:
        out.append(label_mode if k in LABEL_KEYS else image_mode)
    return out


def build_pair_transforms(
    training: bool,
    dilation_mm: float = 18.0,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: tuple[int, int, int] = CANONICAL_SHAPE,
) -> T.Compose:
    """Build the train- or val-side transform pipeline.

    Loading
      ``image_a`` and ``image_b`` are lists of 4 per-modality .nii.gz paths;
      MONAI's ``LoadImaged`` concatenates list entries into a new leading axis
      so the output is (4, H, W, D) directly \u2014 no EnsureChannelFirstd needed.
      Labels are single .nii.gz files; EnsureChannelFirstd adds the (1, ...)
      channel axis.
    """
    keys_all = list(SPATIAL_KEYS)
    keys_image = list(IMAGE_KEYS)
    keys_label = list(LABEL_KEYS)

    xforms: list = [
        T.LoadImaged(keys=keys_all, image_only=True),
        T.EnsureChannelFirstd(keys=keys_label, channel_dim="no_channel"),
        T.EnsureTyped(keys=keys_all, dtype=torch.float32),
        T.Orientationd(keys=keys_all, axcodes="LPS"),
        T.CropForegroundd(keys=keys_all, source_key="image_a"),
        T.SpatialPadd(keys=keys_all, spatial_size=spatial_size, mode="constant"),
        T.CenterSpatialCropd(keys=keys_all, roi_size=spatial_size),
        T.ScaleIntensityRangePercentilesd(
            keys=keys_image, lower=0, upper=99.5, b_min=0.0, b_max=1.0,
            channel_wise=True,
        ),
    ]

    if training:
        xforms.extend([
            T.RandFlipd(keys=keys_all, prob=0.5, spatial_axis=0),
            T.RandAffined(
                keys=keys_all,
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.05, 0.05, 0.05),
                mode=_spatial_modes_for(keys_all, "bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ])

    xforms.append(PairInpaintingMaskd(
        dilation_mm=dilation_mm,
        voxel_size_mm=voxel_size_mm,
    ))
    return T.Compose(xforms)


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------

def compute_balanced_weights(
    records: Sequence[Mapping],
    mode: str = "joint",
    floor: float = 1.0,
) -> torch.Tensor:
    """Return a per-sample weight tensor suitable for WeightedRandomSampler.

    Parameters
    ----------
    records  : output of ``prepare_pair_records`` (must contain ``stratum``).
    mode     : 'joint'      -> stratum = (treatment_direction, trajectory)
               'trajectory' -> stratum = trajectory only
               'direction'  -> stratum = treatment direction only
               'uniform'    -> all weights = 1.0
    floor    : minimum inverse-count denominator; clamps weights of rare
               strata (count==1) from blowing up.

    A stratum's weight is ``1 / max(count, floor)``, so the expected per-batch
    count of each stratum is proportional to ``1`` rather than to its natural
    frequency. With the BraTS-GLI 2025 split this raises the per-batch share of
    'pre_post' from ~3.8% to ~25% (4 strata under 'direction'; finer-grained
    under 'joint').
    """
    if mode == "uniform":
        return torch.ones(len(records), dtype=torch.double)

    if mode == "joint":
        key = lambda r: r["stratum"]
    elif mode == "trajectory":
        key = lambda r: r["stratum"].split("/", 1)[1]
    elif mode == "direction":
        key = lambda r: r["stratum"].split("/", 1)[0]
    else:
        raise ValueError(f"Unknown balance mode {mode!r}")

    counts = collections.Counter(key(r) for r in records)
    weights = torch.tensor(
        [1.0 / max(counts[key(r)], floor) for r in records],
        dtype=torch.double,
    )
    return weights


def stratum_summary(records: Sequence[Mapping]) -> dict[str, int]:
    """Convenience: per-stratum record counts (useful for logging)."""
    return dict(collections.Counter(r["stratum"] for r in records))
