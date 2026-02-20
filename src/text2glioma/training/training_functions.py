from pathlib import Path
from typing import Any
from collections import OrderedDict
from copy import deepcopy
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from generative.losses import PatchAdversarialLoss

from text2glioma.utils import print_gpu_memory_report, get_lr, log_reconstructions, log_ldm_sample_unconditioned, prepare_mask_conditioning, get_text_encoder_hidden_states

@torch.no_grad()
def encode_text(tokenizer, text_encoder, texts, pad_to_max=True, device='cpu'):
    """Encode a list of texts into text embeddings using the provided tokenizer and text encoder."""
    tokens = tokenizer(
        text=texts,
        max_length=tokenizer.model_max_length if pad_to_max else None,
        padding="max_length" if pad_to_max else True,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    out = text_encoder(**tokens)
    return get_text_encoder_hidden_states(out).to(device)

def get_uncond(tokenizer, text_encoder, batch_size, device):
    return encode_text(tokenizer, text_encoder, [""] * batch_size, device=device)

def prepare_conditioning(tokenizer, text_encoder, texts, batch_size, dropout_p=0.2, uncond_cache=None, device='cpu'):
    B = len(texts)
    cond = encode_text(tokenizer, text_encoder, texts, device=device)
    uncond = uncond_cache if (uncond_cache is not None and uncond_cache.size(0) == B) \
        else get_uncond(tokenizer, text_encoder, batch_size, device=device)
    # text dropout for classifier-free guidance
    drop = (torch.rand(B) < dropout_p).float().to(device).view(B, 1, 1)
    context = cond * (1 - drop) + uncond * drop
    return context, uncond

def train_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    train_loader,
    val_loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
    device: torch.device,
    n_epochs: int,
    start_epoch: int = 0,
    best_loss: float = float("inf"),
    val_interval: int = 1,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str  = "./runs",
    kl_weight: float = 1e-6,
    perceptual_weight: float = 2e-3,
    adversarial_weight: float = 1e-3,
):
    raw_model = model.module if hasattr(model, "module") else model

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_loss = eval_autoencoder(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        kl_weight=kl_weight,
        adversarial_weight=adversarial_weight,
        perceptual_weight=perceptual_weight,
    )

    for epoch in range(start_epoch, n_epochs):
        train_epoch_autoencoder(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            kl_weight=kl_weight,
            adversarial_weight=adversarial_weight,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_autoencoder(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                kl_weight=kl_weight,
                adversarial_weight=adversarial_weight,
                perceptual_weight=perceptual_weight,
            )
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"[rank-0] [INFO] Training finished!")
    print(f"[rank-0] [INFO] Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss

def train_epoch_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
) -> None:
    model.train()
    discriminator.train()

    # Underlying module (bypasses DDP wrapper for the generator's
    # adversarial-loss forward — avoids DDP gradient-sync hooks and
    # the in-place buffer-version errors introduced in PyTorch ≥ 2.6).
    disc_module = getattr(discriminator, "module", discriminator)

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)

        # Shared forward pass — kept in the graph so the generator
        # backward can reach the autoencoder parameters.
        with torch.amp.autocast("cuda"):
            reconstruction, z_mu, z_sigma = model(x=images)

        # -------- DISCRIMINATOR --------
        # Concatenate fake + real into one batch for a SINGLE
        # discriminator forward.  This avoids the PyTorch ≥ 2.6
        # in-place version-mismatch error: BatchNorm updates
        # running_mean/running_var in-place during forward, so two
        # separate forwards through the same BN layer create two
        # saved-variable versions and the second backward fails.
        # One forward = one BN update = no conflict.
        if adversarial_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                disc_input = torch.cat(
                    [reconstruction.contiguous().detach(),
                     images.contiguous().detach()],
                    dim=0,
                )
                logits_all = discriminator(disc_input.float())[-1]
                logits_fake, logits_real = torch.chunk(logits_all, 2, dim=0)

                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                d_loss = (adversarial_weight * discriminator_loss).mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        # -------- GENERATOR --------
        # Freeze discriminator — no need to track its params/buffers
        # during the generator backward pass.  Use unwrapped module
        # to bypass DDP's in-place buffer broadcast.
        for p in disc_module.parameters():
            p.requires_grad_(False)

        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # MedicalNet perceptual loss expects single-channel → average over channels
            n_ch = images.shape[1]
            p_loss = sum(
                perceptual_loss(
                    reconstruction[:, c:c+1].float(),
                    images[:, c:c+1].float()
                )
                for c in range(n_ch)
            ) / n_ch

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adversarial_weight > 0:
                logits_fake_g = disc_module(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake_g, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adversarial_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # Unfreeze discriminator for the next iteration.
        for p in disc_module.parameters():
            p.requires_grad_(True)

        losses["d_loss"] = discriminator_loss

        if writer is not None:
            writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
            writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
            for k, v in losses.items():
                writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )

@torch.no_grad()
def eval_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    n_samples = 0
    for x in loader:
        images = x["image"].to(device)

        with torch.amp.autocast("cuda"):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # MedicalNet perceptual loss expects single-channel → average over channels
            n_ch = images.shape[1]
            p_loss = sum(
                perceptual_loss(
                    reconstruction[:, c:c+1].float(),
                    images[:, c:c+1].float()
                )
                for c in range(n_ch)
            ) / n_ch
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adversarial_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adversarial_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adversarial_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

        n_samples += images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= max(n_samples, 1)

    if writer is not None:
        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]

# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    tokenizer: Any,
    text_encoder: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    n_epochs: int,
    text_field: str = "impression",
    start_epoch: int = 0,
    val_interval: int = 1,
    dropout_p: float = 0.2,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str = "./runs",
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    mask_dropout_p: float = 0.2,
    latent_channels: int = 3,
    warmup_epochs: int = 10,
    ema_decay: float = 0.9999,
    ema_state_dict: dict = None,
    snr_gamma: float = 5.0,
) -> float:
    raw_model = model.module if hasattr(model, "module") else model

    best_loss = float("inf")

    # ── EMA (exponential moving average) ─────────────────────────────
    ema_model = deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)
        if _is_main:
            print("[rank-0] [INFO] Loaded EMA state from checkpoint.")

    # ── Pre-compute min-SNR weights (Hang et al., 2023) ────────────
    # SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
    # For v-prediction the per-timestep weight is:
    #   w(t) = min(SNR(t), gamma) / SNR(t)  where gamma = snr_gamma (typically 5)
    # This down-weights high-noise timesteps that produce noisy gradients.
    alphas_cumprod = scheduler.alphas_cumprod.to(device)     # [T]
    snr = alphas_cumprod / (1.0 - alphas_cumprod)            # [T]
    # For v-prediction: weight = min(SNR, gamma) / SNR
    # For epsilon-prediction: weight = min(SNR, gamma) / SNR
    # (both reduce to clamping the effective weight at high-noise steps)
    min_snr_weights = torch.clamp(snr, max=snr_gamma) / snr  # [T]
    # ── LR schedule: linear warmup → cosine decay with min_lr floor ──
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = 0.01  # LR never drops below 1% of base_lr

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Fast-forward scheduler if resuming
    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            lr_scheduler.step()

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_field=text_field,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
        scale_factor=scale_factor,
        num_mask_classes=num_mask_classes,
        latent_channels=latent_channels,
    )

    # Determine rank for gated printing
    _is_main = writer_train is not None  # only rank 0 has a writer
    if _is_main:
        print(f"[rank-0] [INFO] epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_epoch_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_field=text_field,
            dropout_p=dropout_p,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
            num_mask_classes=num_mask_classes,
            mask_dropout_p=mask_dropout_p,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            min_snr_weights=min_snr_weights,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                text_field=text_field,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (val_interval * 2) == 0 else False,
                scale_factor=scale_factor,
                num_mask_classes=num_mask_classes,
                latent_channels=latent_channels,
            )

            if _is_main:
                print(f"[rank-0] [INFO] epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": raw_model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))
                torch.save(ema_model.state_dict(), str(run_dir / "best_model_ema.pth"))

    if _is_main:
        print(f"[rank-0] [INFO] Training finished!")
        print(f"[rank-0] [INFO] Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))
    torch.save(ema_model.state_dict(), str(run_dir / "final_model_ema.pth"))

    return val_loss


def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    tokenizer: Any,
    text_encoder: Any,
    text_field: str,
    dropout_p: float,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: torch.amp.GradScaler,
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    mask_dropout_p: float = 0.2,
    max_grad_norm: float = 1.0,
    lr_scheduler: Any = None,
    ema_model: Any = None,
    ema_decay: float = 0.9999,
    min_snr_weights: torch.Tensor = None,
) -> None:
    model.train()
    raw_model = model.module if hasattr(model, "module") else model

    # Only show progress bar on rank 0 to avoid interleaved output
    is_main = not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    rank = 0 if not (torch.distributed.is_available() and torch.distributed.is_initialized()) else torch.distributed.get_rank()
    pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_main)
    for step, x in pbar:
        images = x["image"].to(device)
        labels = x["label"].to(device)
        reports = x[text_field]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                e = stage1(images) * scale_factor
                latent_spatial = e.shape[2:]  # (D', H', W')

                # Prepare mask conditioning: one-hot → downsample → dropout
                mask_cond = prepare_mask_conditioning(
                    labels, latent_spatial,
                    num_classes=num_mask_classes,
                    dropout_p=mask_dropout_p,
                ).to(device)

            # Prepare text conditioning (with independent text dropout)
            cond, _ = prepare_conditioning(tokenizer, text_encoder, reports, images.size(0), dropout_p=dropout_p, device=device)

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            # Concatenate noisy latent with mask conditioning: [B, latent_ch + num_classes, D', H', W']
            model_input = torch.cat([noisy_e, mask_cond], dim=1)

            noise_pred = model(x=model_input, timesteps=timesteps, context=cond)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

            # Per-sample MSE (reduce over all dims except batch)
            mse = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            mse = mse.mean(dim=list(range(1, mse.ndim)))  # [B]

            # Min-SNR-γ weighting (Hang et al., 2023) — down-weights
            # high-noise timesteps that produce noisy, unhelpful gradients.
            if min_snr_weights is not None:
                snr_w = min_snr_weights[timesteps]  # [B]
                loss = (mse * snr_w).mean()
            else:
                loss = mse.mean()

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), raw_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1.0 - ema_decay)

        if writer is not None:
            writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

            for k, v in losses.items():
                writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    tokenizer: Any,
    text_encoder: Any,
    text_field: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    latent_channels: int = 3,
) -> float:
    model.eval()
    total_losses = OrderedDict()
    n_samples = 0

    for x in loader:
        images = x["image"].to(device)
        labels = x["label"].to(device)
        reports = x[text_field]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with torch.amp.autocast("cuda"):
            e = stage1(images) * scale_factor
            latent_spatial = e.shape[2:]

            # Prepare mask conditioning (no dropout during eval)
            mask_cond = prepare_mask_conditioning(
                labels, latent_spatial,
                num_classes=num_mask_classes,
                dropout_p=0.0,
            ).to(device)

            cond, _ = prepare_conditioning(tokenizer, text_encoder, reports, images.size(0), dropout_p=0.0, device=device)

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            # Concatenate noisy latent with mask conditioning
            model_input = torch.cat([noisy_e, mask_cond], dim=1)

            noise_pred = model(x=model_input, timesteps=timesteps, context=cond)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        n_samples += images.shape[0]
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    # Normalise by samples this rank actually processed (not full dataset
    # size, which over-counts under DDP sharded loaders).
    for k in total_losses.keys():
        total_losses[k] /= max(n_samples, 1)

    if writer is not None:
        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)

    if sample:
        log_ldm_sample_unconditioned(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            latent_channels=latent_channels,
            num_mask_classes=num_mask_classes,
        )

    return total_losses["loss"]
