"""Example CLI for lesion localisation and segmentation training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .functions import localise_pathology


class SegDataset(Dataset):
    """Dataset loading diseased images and pre-computed masks."""

    def __init__(self, image_dir: Path, mask_dir: Path) -> None:
        self.image_files = sorted(image_dir.glob("*.npy"))
        self.mask_files = sorted(mask_dir.glob("*.npy"))
        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Mismatched number of images and masks")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_files)

    def __getitem__(self, index: int):
        image = np.load(self.image_files[index])
        mask = np.load(self.mask_files[index])
        image_t = torch.from_numpy(image).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_t, mask_t


class TinySeg(nn.Module):
    """Minimal segmentation network for demonstration."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return self.layers(x)


def collect_masks(healthy_dir: Path, diseased_dir: Path, mask_dir: Path, threshold: float) -> None:
    """Collect and save masks for all image pairs."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    healthy_files = sorted(healthy_dir.glob("*.npy"))
    diseased_files = sorted(diseased_dir.glob("*.npy"))
    for h, d in zip(healthy_files, diseased_files):
        mask = localise_pathology(np.load(h), np.load(d), threshold)
        out = mask_dir / d.name
        np.save(out, mask.astype(np.uint8))


def train_segmentation(image_dir: Path, mask_dir: Path, epochs: int) -> None:
    dataset = SegDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySeg().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect masks and train a segmentation network")
    parser.add_argument("healthy_dir", type=Path, help="Directory with healthy images (.npy)")
    parser.add_argument("diseased_dir", type=Path, help="Directory with diseased images (.npy)")
    parser.add_argument("mask_dir", type=Path, help="Output directory for generated masks")
    parser.add_argument("--threshold", type=float, default=0.1, help="Difference threshold")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    args = parser.parse_args()

    collect_masks(args.healthy_dir, args.diseased_dir, args.mask_dir, args.threshold)
    train_segmentation(args.diseased_dir, args.mask_dir, args.epochs)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
