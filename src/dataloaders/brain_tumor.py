from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BrainTumorDataset(Dataset):
    """Dataset for brain tumor NIfTI volumes and associated text reports.

    This dataset expects a directory structure of::

        root_dir/
            images/   # NIfTI files (.nii or .nii.gz)
            reports/  # text reports with identical file stems

    Parameters
    ----------
    root_dir:
        Directory containing ``images`` and ``reports`` subfolders.
    transform:
        Optional callable applied to the image tensor for normalization and
        augmentation.
    tokenizer:
        Callable that converts a string report into a token representation.
    cache_dir:
        If provided, tokenized reports are cached on disk to avoid repeated
        preprocessing.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable[[str], Any]] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        images_dir = self.root_dir / "images"
        reports_dir = self.root_dir / "reports"
        if not images_dir.exists() or not reports_dir.exists():
            raise FileNotFoundError(
                f"Expected 'images' and 'reports' directories in {self.root_dir}"
            )

        self.image_paths = sorted(images_dir.glob("*.nii*"))
        self.report_paths = [reports_dir / (p.stem + ".txt") for p in self.image_paths]

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_or_tokenize(self, report_path: Path):
        cache_path = self.cache_dir / f"{report_path.stem}.pt" if self.cache_dir else None
        if cache_path and cache_path.exists():
            return torch.load(cache_path)

        text = report_path.read_text(encoding="utf-8")
        tokens = self.tokenizer(text) if self.tokenizer else text

        if cache_path:
            torch.save(tokens, cache_path)
        return tokens

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        report_path = self.report_paths[idx]

        img = nib.load(str(img_path), mmap=True)
        volume = np.asanyarray(img.dataobj, dtype=np.float32)
        tensor = torch.from_numpy(volume)

        if self.transform:
            tensor = self.transform(tensor)

        tokens = self._load_or_tokenize(report_path)
        return {"image": tensor, "report": tokens}


def create_dataloader(
    root_dir: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Callable[[str], Any]] = None,
    cache_dir: Optional[str | Path] = None,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` for ``BrainTumorDataset``."""

    dataset = BrainTumorDataset(
        root_dir=root_dir,
        transform=transform,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
