"""Dataloader utilities integrating MONAI's BraTS dataset.

The helper functions in this module wrap the official
``monai.apps.BratsDataset`` and automatically derive textual prompts from the
segmentation labels. These prompts are attached to each sample under the
``"text"`` key so latent diffusion model training can consume them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
)
from monai.transforms.transform import Transform

from src.prompt import generate_prompt
from .brain_tumour_dataset import (
    ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d,
)


class PromptFromLabeld(MapTransform):
    """Generate text prompts from label file paths.

    The transform looks up the ``filename_or_obj`` entry produced by
    :class:`LoadImaged` for ``label`` and uses :func:`~src.prompt.generate_prompt`
    to derive a textual description.  The resulting string is stored under the
    ``text`` key so that downstream training functions can consume it.
    """

    def __init__(self, keys: KeysCollection = ("label",), text_key: str = "text") -> None:
        super().__init__(keys)
        self.text_key = text_key

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        d = dict(data)
        for key in self.key_iterator(d):
            meta = d.get(f"{key}_meta_dict", {})
            filename = meta.get("filename_or_obj")
            if filename is not None:
                d[self.text_key] = generate_prompt(filename)
        return d


@dataclass
class BratsLoaderConfig:
    """Configuration for :func:`create_brats_dataloaders`."""

    root: str | Path
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    train_transforms: Sequence[Transform] | None = None
    val_transforms: Sequence[Transform] | None = None
    train_section: str = "training"
    val_section: str = "validation"


def _import_brats_dataset() -> Any:
    """Return MONAI's :class:`BratsDataset` class or raise an error."""

    candidates = [
        "monai.apps.brats.mri_dataset",
        "monai.apps",
    ]
    for module in candidates:
        try:  # pragma: no cover - best effort import
            pkg = __import__(module, fromlist=["BratsDataset"])
            return getattr(pkg, "BratsDataset")
        except Exception:
            continue
    raise RuntimeError("BratsDataset is unavailable. Please install MONAI with BraTS extras.")


def _build_transforms(extra: Sequence[Transform] | None = None) -> Transform:
    """Compose preprocessing transforms for BraTS items."""

    transforms: list[Transform] = []
    if extra is not None:
        transforms.extend(extra)
    transforms.extend(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d(
                keys="label", allow_missing_keys=True
            ),
            PromptFromLabeld(keys="label", text_key="text"),
            EnsureTyped(keys=["image", "label"], dtype=np.float32),
        ]
    )
    return Compose(transforms)


def _create_dataset(root: str | Path, section: str, transform: Transform) -> Any:
    """Instantiate a :class:`BratsDataset` with ``transform`` applied."""

    BratsDataset = _import_brats_dataset()
    try:  # pragma: no cover - constructor signature may vary
        return BratsDataset(root_dir=root, section=section, transform=transform, download=False)
    except TypeError:
        return BratsDataset(root_dir=root, section=section, download=False, transform=transform)


def create_brats_dataloaders(config: BratsLoaderConfig) -> Dict[str, DataLoader]:
    """Factory returning training and validation ``DataLoader`` instances."""

    train_ds = _create_dataset(
        root=config.root,
        section=config.train_section,
        transform=_build_transforms(config.train_transforms),
    )
    val_ds = _create_dataset(
        root=config.root,
        section=config.val_section,
        transform=_build_transforms(config.val_transforms),
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


__all__ = ["BratsLoaderConfig", "create_brats_dataloaders", "PromptFromLabeld"]
