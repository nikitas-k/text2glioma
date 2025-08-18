"""Dataset and dataloader utilities for brain tumour NIfTI volumes.

This module exposes a :class:`BrainTumourDataset` for loading images and
labels using nibabel's memory mapped loader and applying MONAI transforms.
It also provides a :func:`create_dataloaders` factory returning PyTorch
``DataLoader`` instances for training and validation splits.

The transform pipeline integrates the custom ``ConvertToMultiChannel...``
transform and ``ConvertTextd`` defined below. Additional MONAI transforms can
be injected via the :class:`LoaderConfig` to tailor preprocessing or
augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from monai.transforms import (
    Compose,
    EnsureTyped,
    MapTransform,
    RandFlipd,
)
from monai.transforms.transform import Transform
from monai.config import KeysCollection, DtypeLike
from monai.utils.enums import TransformBackends


def _load_nifti_memmap(filename: str | Path) -> np.ndarray:
    """Load a NIfTI file using nibabel with memory mapping enabled.

    Parameters
    ----------
    filename: str or :class:`~pathlib.Path`
        Path to the NIfTI file to load.

    Returns
    -------
    np.ndarray
        Loaded image array.
    """
    img = nib.load(str(filename), mmap=True)
    # ``img.dataobj`` is a proxy object that leverages memmap when possible.
    data = np.asanyarray(img.dataobj)
    return data


class ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024(Transform):
    """Convert single-channel labels to BraTS 2024 multi-channel format.

    The mapping creates four channels representing tumour core (TC), whole
    tumour (WT), enhancing tumour (ET) and resection cavity (RC).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [
            (img == 1) | (img == 3),
            (img == 1) | (img == 2) | (img == 3),
            img == 3,
            img == 4,
        ]
        if isinstance(img, torch.Tensor):
            return torch.stack(result, dim=0)
        return np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d(MapTransform):
    """Dictionary-based wrapper for
    :class:`ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024`."""

    backend = ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024()

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ConvertTextd(MapTransform):
    """Simple transform ensuring text key is propagated in a dict."""

    def __init__(self, keys: KeysCollection = ("image",), text_key: str = "text") -> None:
        super().__init__(keys)
        self.text_key = text_key

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        d = dict(data)
        d[self.text_key] = data[self.text_key]
        return d


class AddChannelD(MapTransform):
    """Add a channel dimension to specified keys if missing."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            if arr.ndim == 3:
                d[key] = np.expand_dims(arr, 0)
        return d


class BrainTumourDataset(Dataset):
    """Dataset for loading brain tumour volumes and labels.

    Each item is expected to be a dict containing ``image`` and ``label`` paths
    along with a ``text`` description. Images are loaded using nibabel with
    memory mapping and then passed through the provided transform pipeline.
    """

    def __init__(self, data: Sequence[Dict[str, Any]], transform: Transform | None = None) -> None:
        self.data = list(data)
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.data[index])
        item["image"] = _load_nifti_memmap(item["image"])
        if "label" in item:
            item["label"] = _load_nifti_memmap(item["label"])
        if self.transform is not None:
            item = self.transform(item)
        return item


@dataclass
class LoaderConfig:
    """Configuration options for :func:`create_dataloaders`.

    ``train_transforms`` and ``val_transforms`` allow injection of additional
    MONAI transforms that will be prepended to the standard pipeline for the
    training and validation datasets respectively.
    """

    train_files: Sequence[Dict[str, Any]]
    val_files: Sequence[Dict[str, Any]]
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    train_transforms: Sequence[Transform] | None = None
    val_transforms: Sequence[Transform] | None = None


def _build_transforms(extra_transforms: Sequence[Transform] | None = None) -> Transform:
    """Create the transform pipeline for dataset items.

    Parameters
    ----------
    extra_transforms: sequence of :class:`~monai.transforms.Transform`, optional
        Additional transforms to prepend to the standard pipeline. This can be
        used to inject augmentation transforms from the configuration.
    """

    transforms: List[Transform] = []
    if extra_transforms is not None:
        transforms.extend(extra_transforms)

    transforms.extend(
        [
            AddChannelD(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d(keys="label"),
            ConvertTextd(keys="image", text_key="text"),
            EnsureTyped(keys=["image", "label"], dtype=np.float32),
        ]
    )
    return Compose(transforms)


def create_dataloaders(config: LoaderConfig) -> Dict[str, DataLoader]:
    """Factory creating training and validation dataloaders.

    Parameters
    ----------
    config: :class:`LoaderConfig`
        Configuration describing datasets and loader behaviour.

    Returns
    -------
    dict
        Dictionary with ``"train"`` and ``"val"`` ``DataLoader`` instances.
    """

    train_ds = BrainTumourDataset(
        config.train_files, transform=_build_transforms(config.train_transforms)
    )
    val_ds = BrainTumourDataset(
        config.val_files, transform=_build_transforms(config.val_transforms)
    )

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    if config.num_workers > 0:
        loader_args["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    return {"train": train_loader, "val": val_loader}


__all__ = [
    "BrainTumourDataset",
    "ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024",
    "ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d",
    "ConvertTextd",
    "LoaderConfig",
    "create_dataloaders",
]
