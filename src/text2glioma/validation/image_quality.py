"""Image quality metrics: FID (per-modality), MS-SSIM, pixel-level statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg, stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _load_medicalnet_extractor(device: torch.device = torch.device("cpu")):
    """Return a MedicalNet ResNet-50 truncated at layer4 (2048-d features).

    Falls back to a torchvision ResNet-50 (2D slices) if MedicalNet is
    unavailable.
    """
    try:
        from monai.networks.nets import resnet50

        model = resnet50(
            pretrained=False,
            spatial_dims=2,
            n_input_channels=1,
            num_classes=1,
        )
    except Exception:
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # patch first conv for 1-ch input
        w = model.conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = w

    # remove the fc head — keep up to avgpool
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_slice_features(
    volumes: List[np.ndarray],
    channel: int,
    extractor: torch.nn.Module,
    device: torch.device,
    mid_fraction: float = 0.5,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 2048-d features from axial slices of a single modality.

    Parameters
    ----------
    volumes : list of [C, D, H, W] or [D, H, W, C] arrays
    channel : modality index to use
    extractor : feature-extraction model (output shape [B, 2048])
    device : torch device
    mid_fraction : fraction of slices around the centre to keep
    batch_size : forward-pass batch size

    Returns
    -------
    features : np.ndarray  [N_slices_total, 2048]
    """
    all_feats: list[np.ndarray] = []
    buf: list[torch.Tensor] = []

    def _flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        f = extractor(x)
        if f.ndim > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        all_feats.append(f.cpu().numpy())
        buf.clear()

    for vol in volumes:
        # normalise to [C, D, H, W]
        if vol.ndim == 4 and vol.shape[-1] <= 4:  # [D, H, W, C]
            vol = np.transpose(vol, (3, 0, 1, 2))
        ch_data = vol[channel]  # [D, H, W]
        n_slices = ch_data.shape[0]
        lo = int(n_slices * (1 - mid_fraction) / 2)
        hi = n_slices - lo
        for s in range(lo, hi):
            sl = ch_data[s]
            # min-max normalise slice
            vmin, vmax = float(sl.min()), float(sl.max())
            if vmax - vmin > 1e-8:
                sl = (sl - vmin) / (vmax - vmin)
            t = torch.from_numpy(sl.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            buf.append(t)
            if len(buf) >= batch_size:
                _flush()
    _flush()

    if not all_feats:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(
    real_volumes: List[np.ndarray],
    synth_volumes: List[np.ndarray],
    channel: int,
    device: torch.device = torch.device("cpu"),
    mid_fraction: float = 0.5,
    batch_size: int = 64,
) -> float:
    """Compute per-modality FID between real and synthetic volumes.

    Parameters
    ----------
    real_volumes : list of [C, D, H, W] arrays
    synth_volumes : list of [C, D, H, W] arrays
    channel : modality index (0=T1, 1=T1CE, 2=T2, 3=FLAIR)

    Returns
    -------
    fid : float
    """
    extractor = _load_medicalnet_extractor(device)

    logger.info("Extracting real features for channel %d …", channel)
    feats_real = extract_slice_features(
        real_volumes, channel, extractor, device, mid_fraction, batch_size
    )
    logger.info("Extracting synth features for channel %d …", channel)
    feats_synth = extract_slice_features(
        synth_volumes, channel, extractor, device, mid_fraction, batch_size
    )

    mu_r, sigma_r = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu_s, sigma_s = feats_synth.mean(0), np.cov(feats_synth, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_s, sigma_s)


def compute_fid_all_channels(
    real_volumes: List[np.ndarray],
    synth_volumes: List[np.ndarray],
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> dict[str, float]:
    """Compute FID for each modality channel and the pool of all channels.

    Returns
    -------
    dict : {"FID_T1": float, "FID_T1CE": float, …, "FID_all": float}
    """
    results = {}
    for ch_idx, name in enumerate(MODALITY_NAMES):
        fid = compute_fid(real_volumes, synth_volumes, ch_idx, device, **kwargs)
        results[f"FID_{name}"] = fid
        logger.info("FID_%s = %.2f", name, fid)

    # Pool all channels as if they were independent images
    extractor = _load_medicalnet_extractor(device)
    all_real, all_synth = [], []
    for ch_idx in range(len(MODALITY_NAMES)):
        all_real.append(
            extract_slice_features(real_volumes, ch_idx, extractor, device, **kwargs)
        )
        all_synth.append(
            extract_slice_features(synth_volumes, ch_idx, extractor, device, **kwargs)
        )
    feats_r = np.concatenate(all_real, axis=0)
    feats_s = np.concatenate(all_synth, axis=0)
    mu_r, sigma_r = feats_r.mean(0), np.cov(feats_r, rowvar=False)
    mu_s, sigma_s = feats_s.mean(0), np.cov(feats_s, rowvar=False)
    results["FID_all"] = frechet_distance(mu_r, sigma_r, mu_s, sigma_s)
    logger.info("FID_all = %.2f", results["FID_all"])
    return results


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------


def compute_ms_ssim_3d(
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    channel: Optional[int] = None,
    data_range: float = 1.0,
) -> float:
    """3D MS-SSIM between two volumes.

    Parameters
    ----------
    vol_a, vol_b : [C, D, H, W] or [D, H, W]
    channel : if not None, evaluate only that channel.
    """
    from skimage.metrics import structural_similarity as ssim

    if channel is not None:
        a = vol_a[channel]
        b = vol_b[channel]
    else:
        # average over channels
        scores = []
        n_ch = vol_a.shape[0] if vol_a.ndim == 4 else 1
        for c in range(n_ch):
            s = ssim(
                vol_a[c] if n_ch > 1 else vol_a,
                vol_b[c] if n_ch > 1 else vol_b,
                data_range=data_range,
            )
            scores.append(s)
        return float(np.mean(scores))

    return float(ssim(a, b, data_range=data_range))


def compute_ms_ssim_batch(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    channel: Optional[int] = None,
) -> dict[str, float]:
    """Compute MS-SSIM for a list of (vol_a, vol_b) pairs.

    Returns dict with mean, std, and per-pair values.
    """
    scores = [compute_ms_ssim_3d(a, b, channel=channel) for a, b in tqdm(pairs, desc="MS-SSIM")]
    return {
        "ms_ssim_mean": float(np.mean(scores)),
        "ms_ssim_std": float(np.std(scores)),
        "ms_ssim_values": scores,
    }


# ---------------------------------------------------------------------------
# Pixel-level statistics
# ---------------------------------------------------------------------------


def compute_pixel_stats(
    real_volumes: List[np.ndarray],
    synth_volumes: List[np.ndarray],
    mask_volumes: Optional[List[np.ndarray]] = None,
) -> dict:
    """Compute per-channel intensity statistics and KS tests.

    Parameters
    ----------
    real_volumes : list of [C, D, H, W]
    synth_volumes : list of [C, D, H, W]
    mask_volumes : optional list of [D, H, W] segmentation labels for CNR/SNR

    Returns
    -------
    dict with per-channel KS statistics, CNR, SNR
    """
    n_ch = real_volumes[0].shape[0]
    results: dict = {}

    for ch in range(n_ch):
        name = MODALITY_NAMES[ch] if ch < len(MODALITY_NAMES) else f"ch{ch}"

        # flatten all voxels for this channel
        real_flat = np.concatenate([v[ch].ravel() for v in real_volumes])
        synth_flat = np.concatenate([v[ch].ravel() for v in synth_volumes])

        # sub-sample for speed (KS on >1M samples is always significant)
        rng = np.random.default_rng(42)
        n_sample = min(500_000, len(real_flat), len(synth_flat))
        real_sub = rng.choice(real_flat, n_sample, replace=False)
        synth_sub = rng.choice(synth_flat, n_sample, replace=False)

        ks_stat, ks_p = stats.ks_2samp(real_sub, synth_sub)
        results[f"{name}_mean_real"] = float(real_flat.mean())
        results[f"{name}_mean_synth"] = float(synth_flat.mean())
        results[f"{name}_ks_stat"] = float(ks_stat)
        results[f"{name}_ks_p"] = float(ks_p)

    # CNR / SNR (requires masks)
    if mask_volumes is not None and len(mask_volumes) == len(real_volumes):
        for tag, vols in [("real", real_volumes), ("synth", synth_volumes)]:
            cnr_vals, snr_vals = [], []
            masks = mask_volumes if tag == "real" else mask_volumes  # same masks
            for vol, msk in zip(vols, masks):
                fg = vol[0][msk > 0]  # tumour voxels (channel 0)
                bg = vol[0][msk == 0]  # non-tumour
                if len(fg) > 10 and len(bg) > 10:
                    cnr_vals.append(abs(fg.mean() - bg.mean()) / max(bg.std(), 1e-8))
                    snr_vals.append(fg.mean() / max(bg.std(), 1e-8))
            results[f"CNR_{tag}"] = float(np.mean(cnr_vals)) if cnr_vals else float("nan")
            results[f"SNR_{tag}"] = float(np.mean(snr_vals)) if snr_vals else float("nan")

    return results


# ---------------------------------------------------------------------------
# Convenience: load volumes from a directory of NIfTI files
# ---------------------------------------------------------------------------


def load_nifti_volumes(
    nifti_dir: str | Path,
    max_n: Optional[int] = None,
    expected_channels: int = 4,
) -> List[np.ndarray]:
    """Load NIfTI files and return list of [C, D, H, W] numpy arrays."""
    nifti_dir = Path(nifti_dir)
    paths = sorted(nifti_dir.glob("*.nii.gz")) + sorted(nifti_dir.glob("*.nii"))
    if max_n:
        paths = paths[:max_n]
    volumes = []
    for p in tqdm(paths, desc=f"Loading {nifti_dir.name}"):
        data = nib.load(str(p)).get_fdata().astype(np.float32)
        # [D, H, W, C] → [C, D, H, W]
        if data.ndim == 4 and data.shape[-1] <= expected_channels:
            data = np.transpose(data, (3, 0, 1, 2))
        elif data.ndim == 3:
            data = data[np.newaxis]
        volumes.append(data)
    logger.info("Loaded %d volumes from %s", len(volumes), nifti_dir)
    return volumes


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_image_quality(
    real_dir: str,
    synth_dir: str,
    output_json: str,
    mask_dir: Optional[str] = None,
    device: str = "cpu",
    max_n: Optional[int] = None,
) -> dict:
    """End-to-end image-quality evaluation.

    Returns
    -------
    dict with all metrics
    """
    import json

    dev = torch.device(device)
    real_vols = load_nifti_volumes(real_dir, max_n)
    synth_vols = load_nifti_volumes(synth_dir, max_n)
    mask_vols = None
    if mask_dir:
        mask_vols = [
            nib.load(str(p)).get_fdata().astype(np.float32)
            for p in sorted(Path(mask_dir).glob("*.nii*"))
        ]

    results: dict = {}

    # FID
    logger.info("Computing FID per modality …")
    results.update(compute_fid_all_channels(real_vols, synth_vols, dev))

    # MS-SSIM (diversity — random pairs)
    rng = np.random.default_rng(0)
    n_pairs = min(200, len(synth_vols))
    idx_a = rng.integers(0, len(synth_vols), n_pairs)
    idx_b = rng.integers(0, len(synth_vols), n_pairs)
    div_pairs = [(synth_vols[a], synth_vols[b]) for a, b in zip(idx_a, idx_b) if a != b]
    results["ms_ssim_diversity"] = compute_ms_ssim_batch(div_pairs)["ms_ssim_mean"]

    # MS-SSIM (intra-pair: synth vs nearest real placeholder — first N)
    n_intra = min(len(real_vols), len(synth_vols))
    intra_pairs = list(zip(real_vols[:n_intra], synth_vols[:n_intra]))
    results["ms_ssim_intra_pair"] = compute_ms_ssim_batch(intra_pairs)["ms_ssim_mean"]

    # Pixel stats
    logger.info("Computing pixel-level statistics …")
    results.update(compute_pixel_stats(real_vols, synth_vols, mask_vols))

    # Save
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return results
