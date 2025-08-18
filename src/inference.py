"""Utilities for image generation with classifier-free guidance.

This module provides a :func:`generate_images` function which synthesises
images conditioned on a text prompt.  Sampling hyper-parameters are read from
an external YAML configuration file allowing users to tweak values such as the
classifier-free guidance scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from utils.text import prepare_conditioning


def load_config(path: str | Path) -> dict[str, Any]:
    """Return the dictionary stored in YAML file at ``path``."""

    with open(path, "r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def generate_images(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    text_encoder: Any,
    prompt: str,
    config_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Generate an image using classifier-free guidance.

    Parameters
    ----------
    model:
        Denoising UNet operating in latent space.
    stage1:
        Autoencoder model providing a ``decode`` method.
    scheduler:
        Diffusion scheduler controlling the sampling loop.
    text_encoder:
        Hugging Face ``PreTrainedModel`` producing text embeddings.
    prompt:
        Textual description guiding the generation.
    config_path:
        Path to a YAML configuration file.  At minimum it must contain a
        ``guidance_scale`` entry.  Additional optional keys are ``num_steps``,
        ``depth``, ``height``, ``width`` and ``scale_factor``.  Further
        ``healthy_prompt`` may specify a baseline prompt to compare against,
        ``difference_threshold`` sets the cutoff for the difference map and
        ``save_difference`` enables storing that map to ``difference.pt``.
    device:
        Device on which to run the computation.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        Generated image.  If a ``healthy_prompt`` is provided the second
        element of the tuple contains the difference map.
    """

    cfg = load_config(config_path)
    guidance_scale = float(cfg.get("guidance_scale", 7.5))
    num_steps = int(cfg.get("num_steps", 50))
    depth = int(cfg.get("depth", 80))
    height = int(cfg.get("height", 112))
    width = int(cfg.get("width", 128))
    scale_factor = float(cfg.get("scale_factor", 1.0))

    healthy_prompt = cfg.get("healthy_prompt")
    difference_threshold = float(cfg.get("difference_threshold", 0.0))
    save_difference = bool(cfg.get("save_difference", False))

    def _sample(prompt_text: str) -> torch.Tensor:
        cond, uncond = prepare_conditioning([prompt_text], text_encoder, device)
        context = torch.cat([uncond, cond])

        scheduler.set_timesteps(num_steps, device=device)
        latents = torch.randn(
            1,
            model.in_channels,
            depth // 8,
            height // 8,
            width // 8,
            device=device,
        )

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents, latents])
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = model(latent_model_input, t, context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / scale_factor
        with torch.no_grad():
            return stage1.decode(latents).clamp(0, 1)

    images = _sample(prompt)

    if healthy_prompt:
        healthy = _sample(healthy_prompt)
        difference = (images - healthy).abs()
        if difference_threshold:
            difference = (difference > difference_threshold).float()
        if save_difference:
            torch.save(difference, "difference.pt")
        return images, difference

    return images
