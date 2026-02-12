"""Mask fidelity: round-trip segmentation Dice / HD95 and mask ablation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

# BraTS evaluation regions
REGION_NAMES = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Label → region mapping  (MSD BraTS convention: 1=edema, 2=nCET, 3=enh)
# ---------------------------------------------------------------------------


def labels_to_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert integer segmentation to binary region masks.

    Returns
    -------
    dict with keys WT (whole tumour), TC (tumour core), ET (enhancing).
    Each value is a binary np.ndarray of the same spatial shape.
    """
    wt = (seg > 0).astype(np.uint8)          # 1+2+3
    tc = ((seg == 2) | (seg == 3)).astype(np.uint8)  # nCET + enh
    et = (seg == 3).astype(np.uint8)          # enhancing only
    return {"WT": wt, "TC": tc, "ET": et}


# ---------------------------------------------------------------------------
# Segmentation model loader
# ---------------------------------------------------------------------------


def load_brats_segmenter(
    device: torch.device = torch.device("cpu"),
    bundle_name: str = "brats_mri_segmentation",
) -> torch.nn.Module:
    """Load a pretrained BraTS segmentation model from MONAI model zoo.

    Falls back to a basic SegResNet if the bundle is unavailable.
    """
    try:
        from monai.bundle import download, load

        bundle_dir = Path.home() / ".cache" / "monai_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        download(name=bundle_name, bundle_dir=str(bundle_dir), source="github")
        model = load(
            name=bundle_name,
            bundle_dir=str(bundle_dir),
            source="github",
            load_ts_module=False,
        )
        model.eval().to(device)
        return model
    except Exception as e:
        logger.warning("Could not load bundle %s: %s. Using fallback SegResNet.", bundle_name, e)
        from monai.networks.nets import SegResNet

        model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,  # bg + 3 tumour regions
            init_filters=32,
        )
        model.eval().to(device)
        return model


# ---------------------------------------------------------------------------
# Round-trip segmentation
# ---------------------------------------------------------------------------


@torch.no_grad()
def segment_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    spatial_size: Tuple[int, int, int] = (160, 224, 160),
) -> np.ndarray:
    """Run BraTS segmenter on a single 4-channel volume.

    Parameters
    ----------
    model : segmentation model (4-ch in, 4-class out)
    volume : [C, D, H, W] float32 normalised to ~[0, 1]
    device : torch device
    spatial_size : model input size (if resize needed)

    Returns
    -------
    seg : [D, H, W] int32 label map
    """
    x = torch.from_numpy(volume).unsqueeze(0).float().to(device)  # [1,C,D,H,W]
    orig_shape = x.shape[2:]

    if x.shape[2:] != spatial_size:
        x = F.interpolate(x, size=spatial_size, mode="trilinear", align_corners=False)

    logits = model(x)  # [1, n_cls, D, H, W]
    pred = logits.argmax(dim=1).squeeze(0)  # [D, H, W]

    if pred.shape != orig_shape:
        pred = F.interpolate(
            pred.float().unsqueeze(0).unsqueeze(0),
            size=orig_shape,
            mode="nearest",
        ).squeeze().long()

    return pred.cpu().numpy().astype(np.int32)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for two binary masks."""
    intersection = float(np.sum(pred & gt))
    volume = float(np.sum(pred) + np.sum(gt))
    if volume == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / volume


def compute_hausdorff95(pred: np.ndarray, gt: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance (mm)."""
    from scipy.ndimage import distance_transform_edt

    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float("nan")

    # distance from pred surface to gt
    border_pred = pred.astype(bool) ^ _erode(pred.astype(bool))
    border_gt = gt.astype(bool) ^ _erode(gt.astype(bool))

    dt_gt = distance_transform_edt(~gt.astype(bool), sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred.astype(bool), sampling=voxel_spacing)

    d_pred_to_gt = dt_gt[border_pred]
    d_gt_to_pred = dt_pred[border_gt]

    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return float("nan")

    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1).astype(mask.dtype)


# ---------------------------------------------------------------------------
# Full round-trip evaluation
# ---------------------------------------------------------------------------


def roundtrip_evaluation(
    synth_volumes: List[np.ndarray],
    gt_masks: List[np.ndarray],
    segmenter: torch.nn.Module,
    device: torch.device,
) -> Dict[str, dict]:
    """Run round-trip segmentation and compute Dice / HD95 per region.

    Parameters
    ----------
    synth_volumes : list of [C, D, H, W] synthetic images
    gt_masks : list of [D, H, W] conditioning masks (ground truth)
    segmenter : pretrained BraTS segmenter
    device : torch device

    Returns
    -------
    dict : {"WT": {"dice_mean": …, "hd95_mean": …}, "TC": …, "ET": …}
    """
    region_scores: Dict[str, dict] = {r: {"dice": [], "hd95": []} for r in REGION_NAMES}

    for vol, gt_mask in tqdm(
        zip(synth_volumes, gt_masks), total=len(synth_volumes), desc="Round-trip segmentation"
    ):
        pred_mask = segment_volume(segmenter, vol, device)
        pred_regions = labels_to_regions(pred_mask)
        gt_regions = labels_to_regions(gt_mask)

        for region in REGION_NAMES:
            d = compute_dice(pred_regions[region], gt_regions[region])
            h = compute_hausdorff95(pred_regions[region], gt_regions[region])
            region_scores[region]["dice"].append(d)
            region_scores[region]["hd95"].append(h)

    results = {}
    for region in REGION_NAMES:
        dices = region_scores[region]["dice"]
        hd95s = [h for h in region_scores[region]["hd95"] if not np.isnan(h)]
        results[region] = {
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "hd95_mean": float(np.mean(hd95s)) if hd95s else float("nan"),
            "hd95_std": float(np.std(hd95s)) if hd95s else float("nan"),
            "n": len(dices),
        }
        logger.info(
            "%s  Dice = %.3f ± %.3f  |  HD95 = %.1f ± %.1f",
            region, results[region]["dice_mean"], results[region]["dice_std"],
            results[region]["hd95_mean"], results[region]["hd95_std"],
        )
    return results


# ---------------------------------------------------------------------------
# Mask ablation: text-only vs dual-conditioned
# ---------------------------------------------------------------------------


def mask_ablation(
    synth_dual: List[np.ndarray],
    synth_textonly: List[np.ndarray],
    gt_masks: List[np.ndarray],
    segmenter: torch.nn.Module,
    device: torch.device,
) -> Dict[str, dict]:
    """Compare round-trip Dice for dual-conditioned vs text-only samples.

    Returns
    -------
    dict with "dual" and "text_only" sub-dicts, each containing per-region Dice.
    """
    logger.info("Evaluating dual-conditioned samples …")
    dual_result = roundtrip_evaluation(synth_dual, gt_masks, segmenter, device)
    logger.info("Evaluating text-only samples …")
    textonly_result = roundtrip_evaluation(synth_textonly, gt_masks, segmenter, device)
    return {"dual": dual_result, "text_only": textonly_result}


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_mask_fidelity(
    synth_dir: str,
    mask_dir: str,
    output_json: str,
    synth_textonly_dir: Optional[str] = None,
    device: str = "cpu",
    max_n: Optional[int] = None,
) -> dict:
    """End-to-end mask fidelity evaluation."""
    from text2glioma.validation.image_quality import load_nifti_volumes

    dev = torch.device(device)
    synth_vols = load_nifti_volumes(synth_dir, max_n)
    gt_masks = [
        nib.load(str(p)).get_fdata().astype(np.float32)
        for p in sorted(Path(mask_dir).glob("*.nii*"))[:max_n]
    ]

    segmenter = load_brats_segmenter(dev)

    results: dict = {}
    results["roundtrip"] = roundtrip_evaluation(synth_vols, gt_masks, segmenter, dev)

    if synth_textonly_dir:
        textonly_vols = load_nifti_volumes(synth_textonly_dir, max_n)
        results["ablation"] = mask_ablation(synth_vols, textonly_vols, gt_masks, segmenter, dev)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Mask fidelity results saved to %s", out_path)
    return results
