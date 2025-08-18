"""Training utilities for Latent Diffusion Models.

This module exposes a :func:`run_training` entrypoint used to train a
latent diffusion model (LDM).  The original implementation of the
training, evaluation and epoch loops lived in the project notebook.  The
functions have been consolidated here and refactored to remove code
repetition and to operate on batched tensors where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.nets import UNet

from .prompt import generate_prompt
try:  # prepare_conditioning comes from the ldm library
    from ldm.util import prepare_conditioning
except Exception:  # pragma: no cover - placeholder for external dependency
    prepare_conditioning = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _prepare_context(tokenizer: Any, text: list[str], device: torch.device) -> torch.Tensor:
    """Tokenise ``text`` and move the result to ``device``.

    The helper wraps ``prepare_conditioning`` so that training and
    evaluation share the same implementation.
    """

    if prepare_conditioning is None:
        raise RuntimeError("prepare_conditioning is unavailable")
    context, _ = prepare_conditioning(tokenizer, text, device, dropout_p=0.2, uncond_cache=None)
    return context


def _extract_prompts(batch: Mapping[str, Any]) -> list[str]:
    """Return text prompts for ``batch``.

    When the dataloader already provides a ``"text"`` entry the function simply
    returns it.  Otherwise it looks for ``label_meta_dict`` populated by MONAI's
    :class:`~monai.transforms.LoadImaged` and derives prompts from the stored
    file names via :func:`~src.prompt.generate_prompt`.
    """

    if "text" in batch:
        return batch["text"]
    meta = batch.get("label_meta_dict")
    if isinstance(meta, Mapping) and "filename_or_obj" in meta:
        filenames = meta["filename_or_obj"]
        if isinstance(filenames, (list, tuple)):
            return [generate_prompt(f) for f in filenames]
        return [generate_prompt(filenames)]
    raise KeyError("text prompts missing from batch")

def _ldm_forward(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    images: torch.Tensor,
    context: torch.Tensor,
    *,
    scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``images`` and predict the diffusion noise.

    Returns a tuple ``(noise_pred, target)`` where ``noise_pred`` is the
    model output and ``target`` the reference noise used for computing the
    loss.
    """

    with torch.no_grad():
        latents = stage1(images) * scale_factor

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, (latents.size(0),), device=latents.device, dtype=torch.long
    )
    noisy = scheduler.add_noise(latents, noise, timesteps)
    noise_pred = model(x=noisy, timesteps=timesteps, context=context)

    if scheduler.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    return noise_pred, target


def _backward_step(loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler: GradScaler) -> None:
    """Scale ``loss`` and update ``optimizer`` using ``scaler``."""

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    *,
    scale_factor: float,
) -> None:
    """Train ``model`` for a single epoch."""

    model.train()
    pbar = enumerate(loader)
    for step, batch in pbar:
        texts = _extract_prompts(batch)
        context = _prepare_context(tokenizer, texts, device)
        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            noise_pred, target = _ldm_forward(
                model, stage1, scheduler, images, context, scale_factor=scale_factor
            )
            loss = F.mse_loss(noise_pred.float(), target.float()).mean()
        _backward_step(loss, optimizer, scaler)


def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: torch.device,
    *,
    scale_factor: float,
) -> float:
    """Evaluate ``model`` and return the mean loss."""

    model.eval()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        texts = _extract_prompts(batch)
        context = _prepare_context(tokenizer, texts, device)
        with torch.no_grad():
            with autocast():
                noise_pred, target = _ldm_forward(
                    model, stage1, scheduler, images, context, scale_factor=scale_factor
                )
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="sum")
        total_loss += loss.item()
        n_samples += images.size(0)

    return total_loss / max(1, n_samples)


def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    *,
    start_epoch: int,
    best_loss: float,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    scale_factor: float,
    val_loss_file: Path,
    checkpoint_path: Path,
    best_model_path: Path,
    final_model_path: Path,
) -> float:
    """Run the main training loop.

    The function saves checkpoints, best and final model weights as well as a
    text file containing validation losses.  The paths for these artefacts are
    provided via the keyword arguments.  Returns the validation loss of the
    last evaluation."""

    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        scale_factor=1.0,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
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
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                scale_factor=scale_factor,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            val_loss_file.open("a+").write(f"\n{val_loss}")

            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(checkpoint_path))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(best_model_path))

    print("Training finished! Saving final model...")
    torch.save(raw_model.state_dict(), str(final_model_path))
    return val_loss


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model: nn.Module
    stage1: nn.Module
    scheduler: nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    tokenizer: Any
    n_epochs: int
    eval_freq: int
    device: torch.device
    run_dir: Path
    scale_factor: float
    start_epoch: int = 0
    best_loss: float = float("inf")
    val_loss_file: Path | None = None
    checkpoint_path: Path | None = None
    best_model_path: Path | None = None
    final_model_path: Path | None = None

    def __post_init__(self) -> None:
        """Populate file paths and ensure ``run_dir`` exists."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.val_loss_file is None:
            self.val_loss_file = self.run_dir / "ldm_val_T2_losses.txt"
        if self.checkpoint_path is None:
            self.checkpoint_path = self.run_dir / "diffusion_T2_checkpoint.pth"
        if self.best_model_path is None:
            self.best_model_path = self.run_dir / "diffusion_T2_best_model.pth"
        if self.final_model_path is None:
            self.final_model_path = self.run_dir / "diffusion_T2_final_model.pth"


def run_training(cfg: TrainConfig) -> float:
    """Convenience wrapper around :func:`train_ldm`.

    ``cfg`` bundles together the models, data loaders, optimiser and file
    paths used during training.  The function forwards these fields to
    :func:`train_ldm` and returns the last validation loss.
    """

    return train_ldm(
        model=cfg.model,
        stage1=cfg.stage1,
        scheduler=cfg.scheduler,
        train_loader=cfg.train_loader,
        val_loader=cfg.val_loader,
        optimizer=cfg.optimizer,
        tokenizer=cfg.tokenizer,
        start_epoch=cfg.start_epoch,
        best_loss=cfg.best_loss,
        n_epochs=cfg.n_epochs,
        eval_freq=cfg.eval_freq,
        device=cfg.device,
        scale_factor=cfg.scale_factor,
        val_loss_file=cfg.val_loss_file,
        checkpoint_path=cfg.checkpoint_path,
        best_model_path=cfg.best_model_path,
        final_model_path=cfg.final_model_path,
    )


# ---------------------------------------------------------------------------
# Simple segmentation utilities
# ---------------------------------------------------------------------------


class SimpleUNet(nn.Module):
    """A minimal 3D UNet based on MONAI's implementation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.unet(x)


def train_segmentation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train ``model`` for a single epoch using Dice loss.

    The ``loader`` is expected to return batches containing ``diff_mask`` and
    ``label`` tensors. ``diff_mask`` provides the network input while ``label``
    contains the ground truth segmentation. The mean training loss is
    returned.
    """

    model.train()
    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    epoch_loss = 0.0
    for batch in loader:
        diff = batch["diff_mask"].to(device)
        target = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(diff)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / max(len(loader), 1)


@torch.no_grad()
def eval_segmentation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate ``model`` on ``loader`` returning the average Dice loss."""

    model.eval()
    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    total_loss = 0.0
    for batch in loader:
        diff = batch["diff_mask"].to(device)
        target = batch["label"].to(device)
        output = model(diff)
        loss = criterion(output, target)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)
