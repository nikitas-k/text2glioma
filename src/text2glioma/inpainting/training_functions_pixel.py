"""Training loops for the **pixel-space** inpainting DDM.

Sibling of ``training_functions.py`` (the latent-space LDM trainer). The
diffusion loop is structurally identical; the only differences are:

  - **No Stage-1 VAE.** The UNet sees raw images, not latents.
  - **Mask at native resolution.** No downsample to a latent grid; the
    mask-weighted MSE applies directly to the per-voxel error tensor.
  - **No ``scale_factor``.** Image intensities are already in the
    ``[0, 1]``-ish range produced by ``ScaleIntensityRangePercentilesd``
    in ``build_pair_transforms``; no further rescaling is needed before
    the noise schedule.

Everything else is shared with the LDM trainer:
  - ``InpaintingModel`` wrapper (UNet + ``CategoricalConditioningEncoder``).
  - bf16 autocast, EMA, min-SNR-γ loss weighting, linear warmup → cosine LR.
  - Rank-0-gated checkpoint writes.
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from text2glioma.utils import get_lr, print_gpu_memory_report


# ---------------------------------------------------------------------------
# Loss — pixel-space mask-weighted MSE
# ---------------------------------------------------------------------------

def _mask_weighted_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float,
) -> torch.Tensor:
    """Per-sample mask-weighted MSE in image space.

    pred, target : (B, C, D, H, W).
    mask         : (B, 1, D, H, W) soft mask in [0, 1] at native resolution.
    mask_weight  : extra weight applied to in-mask voxels (additive to 1.0).
    """
    weight = 1.0 + mask_weight * mask                         # (B, 1, ...)
    sq = (pred.float() - target.float()).pow(2)               # (B, C, ...)
    weighted = sq * weight
    num = weighted.sum(dim=(1, 2, 3, 4))
    den = weight.expand_as(sq).sum(dim=(1, 2, 3, 4))
    return num / den.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch_pixel_inpainting(
    model: nn.Module,                  # DDP-wrapped InpaintingModel
    scheduler: Any,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Any,
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
            0, scheduler.num_train_timesteps, (image_b.shape[0],), device=device,
        ).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn_like(image_b)
            noisy_b = scheduler.add_noise(original_samples=image_b, noise=noise, timesteps=timesteps)

            # InpaintingModel concatenates (noisy, cond_image, cond_mask) along
            # channel dim and forwards through the UNet with cross-attn cond.
            # Variable names in the wrapper say ``noisy_z_b`` etc. but the
            # signature is image-space-agnostic.
            pred = model(
                noisy_z_b=noisy_b,
                z_masked_a=masked_a,
                z_mask=mask,
                timesteps=timesteps,
                trajectory=trajectory,
                treatment_a=treatment_a,
                treatment_b=treatment_b,
                p_traj=p_traj,
                p_treat=p_treat,
            )

            if scheduler.prediction_type == "v_prediction":
                target = scheduler.get_velocity(image_b, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unsupported prediction_type {scheduler.prediction_type}")

            mse_per_sample = _mask_weighted_mse_per_sample(
                pred=pred, target=target, mask=mask, mask_weight=mask_weight,
            )
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
def eval_pixel_inpainting(
    model: nn.Module,
    scheduler: Any,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: Any,
    mask_weight: float,
) -> float:
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
            0, scheduler.num_train_timesteps, (image_b.shape[0],), device=device,
        ).long()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn_like(image_b)
            noisy_b = scheduler.add_noise(original_samples=image_b, noise=noise, timesteps=timesteps)

            pred = model(
                noisy_z_b=noisy_b,
                z_masked_a=masked_a,
                z_mask=mask,
                timesteps=timesteps,
                trajectory=trajectory,
                treatment_a=treatment_a,
                treatment_b=treatment_b,
                p_traj=0.0, p_treat=0.0,
            )

            if scheduler.prediction_type == "v_prediction":
                target = scheduler.get_velocity(image_b, noise, timesteps)
            else:
                target = noise

            mse_per_sample = _mask_weighted_mse_per_sample(
                pred=pred, target=target, mask=mask, mask_weight=mask_weight,
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

def train_pixel_inpainting(
    model: nn.Module,
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

    # EMA shadow of the full module
    ema_model = deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)
        if is_main:
            print("[rank-0] [INFO] Loaded EMA state from checkpoint.")

    # Min-SNR-γ weights
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    snr = alphas_cumprod / (1.0 - alphas_cumprod)
    min_snr_weights = torch.clamp(snr, max=snr_gamma) / snr

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

    val_loss = eval_pixel_inpainting(
        model=model, scheduler=scheduler, loader=val_loader, device=device,
        step=len(train_loader) * start_epoch, writer=writer_val,
        mask_weight=mask_weight,
    )
    if is_main:
        print(f"[rank-0] [INFO] epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_epoch_pixel_inpainting(
            model=model, scheduler=scheduler, loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch,
            writer=writer_train, p_traj=p_traj, p_treat=p_treat,
            mask_weight=mask_weight, lr_scheduler=lr_scheduler,
            ema_model=ema_model, ema_decay=ema_decay,
            min_snr_weights=min_snr_weights,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_pixel_inpainting(
                model=model, scheduler=scheduler, loader=val_loader, device=device,
                step=len(train_loader) * (epoch + 1), writer=writer_val,
                mask_weight=mask_weight,
            )
            if is_main:
                print(f"[rank-0] [INFO] epoch {epoch + 1} val loss: {val_loss:.4f}")
                print_gpu_memory_report()

                checkpoint = {
                    "epoch":     epoch + 1,
                    "model":     raw_model.state_dict(),
                    "ema":       ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
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
