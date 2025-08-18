"""Training and inference helpers for MGMT and IDH status classification.

This module provides simple utilities to train binary classifiers predicting
MGMT promoter methylation and IDH mutation status from imaging data.  The
functions are intentionally lightweight and accept generic PyTorch models
operating on batches of ``(N, C, ...)`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

__all__ = [
    "ClassifierTrainingConfig",
    "train_classifier",
    "evaluate_classifier",
    "predict",
]


@dataclass
class ClassifierTrainingConfig:
    """Configuration options for :func:`train_classifier`.

    Attributes
    ----------
    n_epochs:
        Number of epochs to run the training loop.
    device:
        Device on which to execute the model (e.g. ``torch.device('cuda')``).
    log_interval:
        Frequency (in steps) at which the training loss is printed.
    """

    n_epochs: int = 10
    device: torch.device = torch.device("cpu")
    log_interval: int = 10


def train_classifier(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    config: ClassifierTrainingConfig,
) -> None:
    """Train ``model`` using ``loader`` and ``optimizer``.

    The function expects ``loader`` to yield dictionaries containing ``image``
    and ``label`` entries where ``label`` is a tensor of class indices.  Mixed
    precision is employed when a CUDA device is available.
    """

    scaler = GradScaler(enabled=config.device.type == "cuda")
    model.to(config.device)

    step = 0
    for epoch in range(config.n_epochs):
        model.train()
        for batch in loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % config.log_interval == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")
            step += 1


def evaluate_classifier(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """Return the mean accuracy of ``model`` over ``loader``."""

    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def predict(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return classification probabilities for ``images``.

    The returned tensor has shape ``(N, C)`` where ``C`` is the number of
    classes produced by ``model``.  Softmax is applied along ``dim=1``.
    """

    model.to(device).eval()
    with torch.no_grad():
        logits = model(images.to(device))
        probs = logits.softmax(dim=1)
    return probs.cpu()
