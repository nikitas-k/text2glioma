"""DDIM-with-CFG sampler + SSIM helpers for the inpainting LDM.

Reusable building blocks used by the inference CLI in ``run_inference.py``
and (optionally) by future qualitative visualisation scripts.

The sampler does **not** assume DDP — it takes the unwrapped ``InpaintingModel``
(or any module exposing ``unet`` and ``cond_encoder``) and runs synchronously
on a single device. Multi-GPU inference is a non-goal here: the test set is
small (95 pairs) and DDIM with 50 steps × 3 conditioning variants finishes in
minutes on one H100.

Why we re-implement sampling rather than calling a MONAI helper
---------------------------------------------------------------
MONAI Generative ships a ``LatentDiffusionInferer`` but it doesn't expose the
classifier-free-guidance path cleanly: CFG requires either two unet forward
calls per step (the readable way) or a duplicated batch trick (the fast way).
Both fit in ~30 lines of explicit code and the readable version is what we
want here for auditability of the conditioning maths.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .conditioning import downsample_binary_mask_to_latent


# ---------------------------------------------------------------------------
# DDIM-with-CFG sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_inpainting(
    *,
    inpainting_model: nn.Module,                       # raw InpaintingModel (no DDP)
    stage1: nn.Module,                                 # frozen Stage1Wrapper
    scheduler,                                         # DDIMScheduler or DDPMScheduler
    masked_image_a: torch.Tensor,                      # (B, C_img, D, H, W)
    mask: torch.Tensor,                                # (B, 1, D, H, W) in {0, 1}
    trajectory: torch.Tensor,                          # (B,) long, in [0, N_TRAJ)
    treatment_a: torch.Tensor,                         # (B,) long, in [0, N_TREAT)
    treatment_b: torch.Tensor,                         # (B,) long, in [0, N_TREAT)
    scale_factor: float,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    use_uncond: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``image_b`` for one batch.

    Modes:
      - ``use_uncond=False, guidance_scale > 0`` → CFG: blend conditional and
        all-NULL passes per step (standard SD / Imagen recipe).
      - ``use_uncond=False, guidance_scale == 0`` → single conditional pass
        (no CFG).
      - ``use_uncond=True`` → single pass with NULL conditioning on all three
        tokens (Task C, unconditional baseline). ``guidance_scale`` ignored.

    Returns
    -------
    pred_image_b : (B, C_img, D, H, W) in the same intensity scale as
        ``masked_image_a`` (Stage-1 decoder output, not re-thresholded).
    """
    device = masked_image_a.device
    raw = inpainting_model
    unet = raw.unet
    cond_encoder = raw.cond_encoder

    # --- Encode the masked prior into latent space -----------------------
    z_masked_a = stage1(masked_image_a) * scale_factor               # (B, Cz, D', H', W')
    z_mask = downsample_binary_mask_to_latent(mask, tuple(z_masked_a.shape[2:]))

    B, Cz, *latent_spatial = z_masked_a.shape

    # --- Context vectors -------------------------------------------------
    if use_uncond:
        context_cond = cond_encoder.get_uncond_context(B, device=device)
        context_uncond = None
        do_cfg = False
    else:
        context_cond = cond_encoder(
            trajectory=trajectory,
            treatment_a=treatment_a,
            treatment_b=treatment_b,
        )
        if guidance_scale and guidance_scale > 0.0:
            context_uncond = cond_encoder.get_uncond_context(B, device=device)
            do_cfg = True
        else:
            context_uncond = None
            do_cfg = False

    # --- Init noise + scheduler ------------------------------------------
    scheduler.set_timesteps(num_inference_steps, device=device)
    z = torch.randn(
        (B, Cz, *latent_spatial),
        device=device, dtype=z_masked_a.dtype, generator=generator,
    )
    # Some schedulers expose init_noise_sigma; DDIM keeps it at 1.0.
    z = z * float(getattr(scheduler, "init_noise_sigma", 1.0))

    # --- Denoising loop --------------------------------------------------
    for t in scheduler.timesteps:
        t_batch = t.expand(B).to(device).long() if t.ndim == 0 else t.to(device).long()
        model_input = torch.cat([z, z_masked_a, z_mask], dim=1)
        pred_cond = unet(x=model_input, timesteps=t_batch, context=context_cond)
        if do_cfg:
            pred_uncond = unet(x=model_input, timesteps=t_batch, context=context_uncond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred = pred_cond
        # scheduler.step expects (model_output, timestep, sample); returns (prev, pred_x0)
        z = scheduler.step(pred, t, z)[0]

    # --- Decode ---------------------------------------------------------
    z = z / scale_factor
    # Stage1Wrapper.decode(z) returns the reconstruction
    pred_image_b = stage1.decode(z) if hasattr(stage1, "decode") else stage1.model.decode(z)
    return pred_image_b


# ---------------------------------------------------------------------------
# SSIM helpers
# ---------------------------------------------------------------------------

def _ssim_3d(vol_a: np.ndarray, vol_b: np.ndarray, data_range: float) -> float:
    """3D SSIM via skimage (single-channel). vol_a, vol_b : (D, H, W)."""
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(vol_a, vol_b, data_range=data_range))


def compute_ssim_per_modality(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
) -> dict:
    """Per-modality 3D SSIM. Returns a flat dict.

    Parameters
    ----------
    pred, target : (C, D, H, W) torch or numpy
    mask : optional (1, D, H, W) or (D, H, W) binary mask. If given, returns
        also an ROI-SSIM computed over the tight bounding box of the mask.
    data_range : intensity range. If None, inferred from target.max() - target.min()
        per modality. (Inpainting outputs aren't normalised to a fixed [0, 1].)

    Returns
    -------
    {
      "ssim_global_mean":   float,           # mean across modalities, full volume
      "ssim_global_perch":  list[float],     # per-modality SSIM, full volume
      "ssim_roi_mean":      float or None,   # mean across modalities inside mask bbox
      "ssim_roi_perch":     list[float] or None,
      "bbox":               list[int] or None,
    }
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().float().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().float().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
        if mask.ndim == 4:
            mask = mask[0]                              # (D, H, W)

    n_ch = pred.shape[0]
    global_scores: list[float] = []
    roi_scores: list[float] = []
    bbox: Optional[list[int]] = None

    if mask is not None and mask.sum() > 0:
        idx = np.where(mask > 0.5)
        dlo, dhi = int(idx[0].min()), int(idx[0].max()) + 1
        hlo, hhi = int(idx[1].min()), int(idx[1].max()) + 1
        wlo, whi = int(idx[2].min()), int(idx[2].max()) + 1
        bbox = [dlo, dhi, hlo, hhi, wlo, whi]

    for c in range(n_ch):
        p, g = pred[c], target[c]
        dr = float(data_range) if data_range is not None else float(g.max() - g.min())
        if dr <= 0:
            dr = 1.0
        global_scores.append(_ssim_3d(p, g, data_range=dr))
        if bbox is not None:
            p_roi = p[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            g_roi = g[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            # skimage's default win_size is 7; refuse to compute if any roi
            # dim is < win_size (the ROI is then too small for SSIM to be
            # meaningful — fall back to a global value as a placeholder).
            if min(p_roi.shape) >= 7:
                roi_scores.append(_ssim_3d(p_roi, g_roi, data_range=dr))
            else:
                roi_scores.append(float("nan"))

    if roi_scores:
        # Suppress numpy's "Mean of empty slice" warning when every roi is NaN
        # (legitimate case: mask present but always smaller than skimage's
        # 7-vox SSIM window). nanmean already returns nan in that case.
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                roi_mean = float(np.nanmean(roi_scores))
        roi_perch = [float(s) for s in roi_scores]
    else:
        roi_mean = None
        roi_perch = None

    return {
        "ssim_global_mean":  float(np.mean(global_scores)),
        "ssim_global_perch": [float(s) for s in global_scores],
        "ssim_roi_mean":     roi_mean,
        "ssim_roi_perch":    roi_perch,
        "bbox":              bbox,
    }
