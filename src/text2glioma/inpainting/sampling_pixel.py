"""DDIM-with-CFG sampler for the **pixel-space** inpainting DDM.

Sibling of ``sampling.py``. Identical denoising loop and CFG maths; the only
differences are:

  - **No Stage-1 encode/decode.** The masked image and mask are used at
    native resolution; the sampler returns the denoised image directly.
  - **No ``scale_factor``.** Image intensities live in roughly [0, 1] from
    the dataset transforms.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def sample_pixel_inpainting(
    *,
    inpainting_model: nn.Module,                       # raw InpaintingModel
    scheduler,                                         # DDIMScheduler / DDPMScheduler
    masked_image_a: torch.Tensor,                      # (B, C, D, H, W)
    mask: torch.Tensor,                                # (B, 1, D, H, W) in {0, 1}
    trajectory: torch.Tensor,                          # (B,) long
    treatment_a: torch.Tensor,                         # (B,) long
    treatment_b: torch.Tensor,                         # (B,) long
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    use_uncond: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``image_b`` for one batch in pixel space.

    Modes mirror ``sample_inpainting`` (Task A / B / C). Returns a tensor
    of shape ``(B, C, D, H, W)`` in the same intensity scale as ``masked_image_a``.
    """
    device = masked_image_a.device
    unet = inpainting_model.unet
    cond_encoder = inpainting_model.cond_encoder

    B, C, *spatial = masked_image_a.shape

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

    scheduler.set_timesteps(num_inference_steps, device=device)
    x = torch.randn(
        (B, C, *spatial),
        device=device, dtype=masked_image_a.dtype, generator=generator,
    )
    x = x * float(getattr(scheduler, "init_noise_sigma", 1.0))

    for t in scheduler.timesteps:
        t_batch = t.expand(B).to(device).long() if t.ndim == 0 else t.to(device).long()
        model_input = torch.cat([x, masked_image_a, mask], dim=1)
        pred_cond = unet(x=model_input, timesteps=t_batch, context=context_cond)
        if do_cfg:
            pred_uncond = unet(x=model_input, timesteps=t_batch, context=context_uncond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred = pred_cond
        x = scheduler.step(pred, t, x)[0]

    return x
