"""Training loops for the inpainting LDM.

Forks ``text2glioma.training.training_functions.train_ldm`` with the following
changes:

  - **No text encoder.** Categorical-only conditioning via
    ``CategoricalConditioningEncoder``.
  - **Dual latent input.** UNet sees
    ``[noisy_z_b ; z_masked_image_a ; z_mask]`` (latent_ch + latent_ch + 1
    channels).
  - **Mask-weighted MSE.** Inside-mask voxels weighted ``1 + mask_weight``,
    outside-mask weighted ``1``. Default ``mask_weight = 4.0`` ⇒ inside ROI
    sees 5× the gradient.
  - **No SSIM during eval (yet).** Eval reports loss only, plus an optional
    DDIM-sampled image grid every ``sample_interval`` val rounds.

Keeps the production-grade improvements from ``train_ldm``:
  - EMA (exponential moving average of UNet + cond encoder weights)
  - Min-SNR-γ loss weighting (Hang et al. 2023, γ = 5)
  - Linear warmup → cosine decay LR schedule
  - bf16 autocast
  - Rank-0-gated checkpoint writes (avoids Lustre FS contention; see
    ``/memories/repo`` note "DDP checkpoint hangs")
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from text2glioma.utils import get_lr, print_gpu_memory_report

from .conditioning import (
    CategoricalConditioningEncoder,
    downsample_binary_mask_to_latent,
)


# ---------------------------------------------------------------------------
# Combined module: cond_encoder + UNet, DDP-wrapped together.
# ---------------------------------------------------------------------------

class InpaintingModel(nn.Module):
    """Holds the conditioning encoder + the diffusion UNet.

    Wrapping both in a single ``nn.Module`` lets DDP own them in one pass; no
    ``find_unused_parameters=True`` needed because every forward exercises
    both submodules. The dataset->latent encoding step (Stage-1 VAE) remains
    outside the wrapper because it's frozen.
    """

    def __init__(self, unet: nn.Module, cond_encoder: CategoricalConditioningEncoder) -> None:
        super().__init__()
        self.unet = unet
        self.cond_encoder = cond_encoder

    def forward(
        self,
        noisy_z_b: torch.Tensor,
        z_masked_a: torch.Tensor,
        z_mask: torch.Tensor,
        timesteps: torch.Tensor,
        trajectory: torch.Tensor,
        treatment_a: torch.Tensor,
        treatment_b: torch.Tensor,
        p_traj: float = 0.0,
        p_treat: float = 0.0,
    ) -> torch.Tensor:
        context = self.cond_encoder.forward_with_dropout(
            trajectory=trajectory,
            treatment_a=treatment_a,
            treatment_b=treatment_b,
            p_traj=p_traj,
            p_treat=p_treat,
        )
        model_input = torch.cat([noisy_z_b, z_masked_a, z_mask], dim=1)
        return self.unet(x=model_input, timesteps=timesteps, context=context)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _mask_weighted_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    z_mask: torch.Tensor,
    mask_weight: float,
) -> torch.Tensor:
    """Per-sample mask-weighted MSE.

    pred, target : (B, C, D', H', W') in latent space.
    z_mask       : (B, 1, D', H', W') soft mask in [0, 1] at latent resolution.
    mask_weight  : extra weight applied to in-mask voxels (in addition to 1.0).

    Returns ``[B]`` of per-sample mean-squared error.
    """
    weight = 1.0 + mask_weight * z_mask                       # (B, 1, ...)
    sq = (pred.float() - target.float()).pow(2)               # (B, C, ...)
    weighted = sq * weight                                    # (B, C, ...) — broadcast over C
    # Reduce over (C, D', H', W'): numerator and denominator both summed over
    # the same spatial extent so the per-sample MSE is comparable across batches.
    num = weighted.sum(dim=(1, 2, 3, 4))                      # (B,)
    den = weight.expand_as(sq).sum(dim=(1, 2, 3, 4))          # (B,)  pred channels
    return num / den.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch_inpainting(
    model: nn.Module,                  # DDP-wrapped InpaintingModel
    stage1: nn.Module,                 # frozen Stage1Wrapper
    scheduler: Any,                    # DDIM/DDPMScheduler
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Any,
    scale_factor: float,
    p_traj: float,
    p_treat: float,
    mask_weight: float,
    max_grad_norm: float = 1.0,
    lr_scheduler: Any = None,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.9999,
    min_snr_weights: Optional[torch.Tensor] = None,
) -> None:
    model.train()
    raw_model = model.module if hasattr(model, "module") else model

    is_main = (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_rank() == 0
    )
    pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_main)

    for step, batch in pbar:
        image_b = batch["image_b"].to(device, non_blocking=True)
        masked_a = batch["masked_image_a"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        trajectory = batch["trajectory"].to(device, non_blocking=True).long()
        treatment_a = batch["treatment_a"].to(device, non_blocking=True).long()
        treatment_b = batch["treatment_b"].to(device, non_blocking=True).long()

        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (image_b.shape[0],), device=device
        ).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                z_b = stage1(image_b) * scale_factor
                z_masked_a = stage1(masked_a) * scale_factor
                z_mask = downsample_binary_mask_to_latent(mask, tuple(z_b.shape[2:]))

            noise = torch.randn_like(z_b)
            noisy_z_b = scheduler.add_noise(original_samples=z_b, noise=noise, timesteps=timesteps)

            pred = model(
                noisy_z_b=noisy_z_b,
                z_masked_a=z_masked_a,
                z_mask=z_mask,
                timesteps=timesteps,
                trajectory=trajectory,
                treatment_a=treatment_a,
                treatment_b=treatment_b,
                p_traj=p_traj,
                p_treat=p_treat,
            )

            if scheduler.prediction_type == "v_prediction":
                target = scheduler.get_velocity(z_b, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unsupported prediction_type {scheduler.prediction_type}")

            mse_per_sample = _mask_weighted_mse_per_sample(
                pred=pred, target=target, z_mask=z_mask, mask_weight=mask_weight,
            )  # (B,)

            if min_snr_weights is not None:
                snr_w = min_snr_weights[timesteps]
                loss = (mse_per_sample * snr_w).mean()
            else:
                loss = mse_per_sample.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # EMA update across the full combined module (UNet + cond encoder).
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, src_p in zip(ema_model.parameters(), raw_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(src_p.data, alpha=1.0 - ema_decay)

        if writer is not None:
            global_step = epoch * len(loader) + step
            writer.add_scalar("loss", loss.item(), global_step)
            writer.add_scalar("lr", get_lr(optimizer), global_step)

        pbar.set_postfix({
            "epoch": epoch,
            "loss": f"{loss.item():.5f}",
            "lr": f"{get_lr(optimizer):.6f}",
        })


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_inpainting(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: Any,
    scale_factor: float,
    mask_weight: float,
) -> float:
    """Loss-only eval. Per-sample mask-weighted MSE averaged across the val set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        image_b = batch["image_b"].to(device, non_blocking=True)
        masked_a = batch["masked_image_a"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        trajectory = batch["trajectory"].to(device, non_blocking=True).long()
        treatment_a = batch["treatment_a"].to(device, non_blocking=True).long()
        treatment_b = batch["treatment_b"].to(device, non_blocking=True).long()

        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (image_b.shape[0],), device=device
        ).long()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z_b = stage1(image_b) * scale_factor
            z_masked_a = stage1(masked_a) * scale_factor
            z_mask = downsample_binary_mask_to_latent(mask, tuple(z_b.shape[2:]))

            noise = torch.randn_like(z_b)
            noisy_z_b = scheduler.add_noise(original_samples=z_b, noise=noise, timesteps=timesteps)

            pred = model(
                noisy_z_b=noisy_z_b,
                z_masked_a=z_masked_a,
                z_mask=z_mask,
                timesteps=timesteps,
                trajectory=trajectory,
                treatment_a=treatment_a,
                treatment_b=treatment_b,
                p_traj=0.0, p_treat=0.0,
            )

            if scheduler.prediction_type == "v_prediction":
                target = scheduler.get_velocity(z_b, noise, timesteps)
            else:
                target = noise

            mse_per_sample = _mask_weighted_mse_per_sample(
                pred=pred, target=target, z_mask=z_mask, mask_weight=mask_weight,
            )

        bs = image_b.shape[0]
        total_loss += float(mse_per_sample.mean().item()) * bs
        n_samples += bs

    val_loss = total_loss / max(n_samples, 1)
    if writer is not None:
        writer.add_scalar("loss", val_loss, step)
    return val_loss


# ---------------------------------------------------------------------------
# Outer training loop
# ---------------------------------------------------------------------------

def train_inpainting(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int,
    start_epoch: int = 0,
    val_interval: int = 5,
    p_traj: float = 0.2,
    p_treat: float = 0.2,
    mask_weight: float = 4.0,
    run_dir: Path = Path("./runs"),
    writer_train: Any = None,
    writer_val: Any = None,
    scale_factor: float = 1.0,
    warmup_epochs: int = 10,
    ema_decay: float = 0.9999,
    ema_state_dict: Optional[dict] = None,
    snr_gamma: float = 5.0,
) -> float:
    raw_model = model.module if hasattr(model, "module") else model
    is_main = (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_rank() == 0
    )

    best_loss = float("inf")
    run_dir = Path(run_dir)

    # ── EMA ───────────────────────────────────────────────────────────
    ema_model = deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)
        if is_main:
            print("[rank-0] [INFO] Loaded EMA state from checkpoint.")

    # ── Min-SNR weights ───────────────────────────────────────────────
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    snr = alphas_cumprod / (1.0 - alphas_cumprod)
    min_snr_weights = torch.clamp(snr, max=snr_gamma) / snr

    # ── LR schedule: linear warmup → cosine decay (1% min floor) ─────
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    min_lr_ratio = 0.01

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(start_epoch * steps_per_epoch):
        lr_scheduler.step()

    # ── Initial val ──────────────────────────────────────────────────
    val_loss = eval_inpainting(
        model=model, stage1=stage1, scheduler=scheduler,
        loader=val_loader, device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val, scale_factor=scale_factor,
        mask_weight=mask_weight,
    )
    if is_main:
        print(f"[rank-0] [INFO] epoch {start_epoch} val loss: {val_loss:.4f}")

    # ── Main loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_epoch_inpainting(
            model=model, stage1=stage1, scheduler=scheduler,
            loader=train_loader, optimizer=optimizer, device=device, epoch=epoch,
            writer=writer_train, scale_factor=scale_factor,
            p_traj=p_traj, p_treat=p_treat, mask_weight=mask_weight,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model, ema_decay=ema_decay,
            min_snr_weights=min_snr_weights,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_inpainting(
                model=model, stage1=stage1, scheduler=scheduler,
                loader=val_loader, device=device,
                step=len(train_loader) * (epoch + 1),
                writer=writer_val, scale_factor=scale_factor,
                mask_weight=mask_weight,
            )
            if is_main:
                print(f"[rank-0] [INFO] epoch {epoch + 1} val loss: {val_loss:.4f}")
                print_gpu_memory_report()

            # Rank-0-only checkpoint write to avoid Lustre contention
            # (see /memories/repo on DDP checkpoint hangs).
            if is_main:
                checkpoint = {
                    "epoch":        epoch + 1,
                    "model":        raw_model.state_dict(),
                    "ema":          ema_model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_loss":    best_loss,
                    "scale_factor": scale_factor,
                }
                torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

                if val_loss <= best_loss:
                    best_loss = val_loss
                    torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))
                    torch.save(ema_model.state_dict(), str(run_dir / "best_model_ema.pth"))

    if is_main:
        print("[rank-0] [INFO] Training finished. Saving final model …")
        torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))
        torch.save(ema_model.state_dict(), str(run_dir / "final_model_ema.pth"))

    return val_loss
