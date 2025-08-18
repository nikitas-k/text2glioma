"""Training utilities for latent diffusion models."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

# The following imports assume the MONAI Generative components are available
# in the runtime environment.
from generative.inferers import DiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_AUTOENCODER_CONFIG = CONFIG_DIR / "autoencoder.json"
DEFAULT_DIFFUSION_CONFIG = CONFIG_DIR / "diffusion_model.json"
DEFAULT_SCHEDULER_CONFIG = CONFIG_DIR / "scheduler.json"
DEFAULT_OPTIMIZER_CONFIG = CONFIG_DIR / "optimizer.json"


def _load_config(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


class Stage1Wrapper(nn.Module):
    """Wrapper for the stage 1 autoencoder so it can be used in DataParallel."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z


def _prepare_batch(batch, tokenizer, device: torch.device):
    """Prepare conditioning and image tensors from a dataloader batch."""
    context, _ = prepare_conditioning(
        tokenizer, batch["text"], device, dropout_p=0.2, uncond_cache=None
    )
    images = batch["image"].to(device)
    return context, images


def _compute_loss(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    context: torch.Tensor,
    images: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """Shared forward pass and loss computation for diffusion training."""
    with torch.no_grad():
        latents = stage1(images) * scale_factor
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, (latents.size(0),), device=latents.device
    ).long()
    noisy = scheduler.add_noise(latents, noise, timesteps)
    noise_pred = model(x=noisy, timesteps=timesteps, context=context)
    if scheduler.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    loss = F.mse_loss(noise_pred.float(), target.float())
    return loss


def _train_epoch(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    scale_factor: float,
) -> None:
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, batch in pbar:
        context, images = _prepare_batch(batch, tokenizer, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            loss = _compute_loss(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                context=context,
                images=images,
                scale_factor=scale_factor,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix({"epoch": epoch, "loss": f"{loss.item():.5f}"})


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    inferer: nn.Module,
    loader: torch.utils.data.DataLoader,
    tokenizer,
    device: torch.device,
    step: int = 0,
    sample: bool = False,
    scale_factor: float = 1.0,
) -> float:
    """Evaluate the diffusion model on a validation dataloader."""
    del inferer  # unused but kept for API parity
    model.eval()
    total_loss = 0.0
    for _, batch in enumerate(loader):
        context, images = _prepare_batch(batch, tokenizer, device)
        with autocast(enabled=True):
            loss = _compute_loss(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                context=context,
                images=images,
                scale_factor=scale_factor,
            )
        total_loss += loss.item() * images.shape[0]
    total_loss /= len(loader.dataset)
    return total_loss


def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    inferer: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    run_dir: Path,
    scale_factor: float,
) -> float:
    """Full latent diffusion training loop."""
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        inferer=inferer,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        step=len(train_loader) * start_epoch,
        sample=False,
        scale_factor=1.0,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    run_dir_path = Path(run_dir)
    for epoch in range(start_epoch, n_epochs):
        _train_epoch(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            scale_factor=scale_factor,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                inferer=inferer,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                step=len(train_loader) * epoch,
                sample=(epoch + 1) % (eval_freq * 2) == 0,
                scale_factor=scale_factor,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            with open(run_dir_path / "ldm_val_T2_losses.txt", "a+") as f:
                f.write(f"\n{val_loss}")
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, run_dir_path / "diffusion_T2_checkpoint.pth")
            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(
                    raw_model.state_dict(), run_dir_path / "diffusion_T2_best_model.pth"
                )

    print("Training finished!")
    print("Saving final model...")
    torch.save(raw_model.state_dict(), run_dir_path / "diffusion_T2_final_model.pth")
    return val_loss


def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    tokenizer,
    n_epochs: int = 1000,
    eval_freq: int = 1,
    run_dir: Path | str = "./models",
    scale_factor: float = 1.0,
    device: Optional[torch.device] = None,
    autoencoder_config: Path | str = DEFAULT_AUTOENCODER_CONFIG,
    diffusion_config: Path | str = DEFAULT_DIFFUSION_CONFIG,
    scheduler_config: Path | str = DEFAULT_SCHEDULER_CONFIG,
    optimizer_config: Path | str = DEFAULT_OPTIMIZER_CONFIG,
) -> float:
    """Entry point to train a latent diffusion model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1 autoencoder setup
    autoencoder_cfg = _load_config(autoencoder_config)
    autoencoder = AutoencoderKL(**autoencoder_cfg)
    if torch.cuda.device_count() > 1:
        autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.to(device)
    autoencoder.eval()
    stage1 = Stage1Wrapper(model=autoencoder.module if hasattr(autoencoder, "module") else autoencoder)
    if torch.cuda.device_count() > 1:
        stage1 = torch.nn.DataParallel(stage1)
    stage1.to(device)
    stage1.eval()

    # Diffusion model setup
    diffusion_cfg = _load_config(diffusion_config)
    diffusion_model = DiffusionModelUNet(**diffusion_cfg)
    if torch.cuda.device_count() > 1:
        diffusion_model = torch.nn.DataParallel(diffusion_model)
    diffusion_model.to(device)

    optimizer_cfg = _load_config(optimizer_config)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), **optimizer_cfg)
    scheduler_cfg = _load_config(scheduler_config)
    scheduler = DDIMScheduler(**scheduler_cfg)
    inferer = DiffusionInferer(scheduler)

    val_loss = train_ldm(
        model=diffusion_model,
        stage1=stage1,
        scheduler=scheduler,
        inferer=inferer,
        start_epoch=0,
        best_loss=float("inf"),
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        device=device,
        run_dir=Path(run_dir),
        scale_factor=scale_factor,
    )
    return val_loss
