import yaml
from pathlib import Path

import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.data.dataset import PersistentDataset
from monai import transforms as T

import gdown

class Stage1Wrapper(nn.Module):
    """Wraps the stage 1 model to bypass DataParallel issues."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)

        return z
    
def stage1_ify(stage1):
    """Wraps the stage 1 model if it is not already wrapped."""
    if not isinstance(stage1, Stage1Wrapper):
        stage1 = Stage1Wrapper(stage1)
    return stage1

def load_config(config_path):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dataloaders(
        cache_dir,
        train_dataset, 
        val_dataset, 
        batch_size, 
        num_workers,
        pin_memory=False,
        shuffle=False,
        model_type="AutoencoderKL",
    ):
    if model_type not in ["AutoencoderKL", "DiffusionModelUNet"]:
        raise ValueError(f"Model type {model_type} not supported for dataloaders.")
    
    if model_type == "AutoencoderKL":
        train_transform = T.Compose(
            [
                T.LoadImaged(keys=["image"]),
                T.EnsureChannelFirstd(keys=["image"]),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.Orientationd(keys=["image"], axcodes="LPS"),
                T.CropForegroundd(keys=["image"], source_key="image"),
                T.Resized(keys=["image"], spatial_size=(160, 192, 96), mode="trilinear"),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1
                ),
                T.RandFlipd(
                    keys=["image"], prob=0.5, spatial_axis=0),
                T.RandAffined(
                    keys=["image"],
                    prob=0.1,
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 192, 96],
                    mode="trilinear",
                ),
                T.RandShiftIntensityd(
                    keys=["image"], offsets=0.05, prob=0.1
                ),
                T.RandAdjustContrastd(
                    keys=["image"], prob=0.1, gamma=(0.97, 1.03)
                ),
                T.ToTensord(keys=["image"]),
            ]
        )
    else:  # DiffusionModelUNet
        train_transform = T.Compose(
            [
                T.LoadImaged(keys=["image"]),
                T.EnsureChannelFirstd(keys=["image"]),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.Orientationd(keys=["image"], axcodes="LPS"),
                T.CropForegroundd(keys=["image"], source_key="image"),
                T.Resized(keys=["image"], spatial_size=(160, 192, 96), mode="trilinear"),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1
                ),
                T.RandAffined(
                    keys=["image"],
                    prob=0.1,
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 192, 96],
                    mode="trilinear",
                ),
                T.RandShiftIntensityd(
                    keys=["image"], offsets=0.05, prob=0.1
                ),
                T.RandAdjustContrastd(
                    keys=["image"], prob=0.1, gamma=(0.97, 1.03)
                ),
                T.ToTensord(keys=["image"]),
            ]
        )
    train_dataset = PersistentDataset(
        data=train_dataset,
        transform=train_transform,
        cache_dir=cache_dir / "train",
        num_workers=num_workers,
    )

    val_dataset = PersistentDataset(
        data=val_dataset,
        transform=T.Compose(
            [
                T.LoadImaged(keys=["image"]),
                T.EnsureChannelFirstd(keys=["image"]),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.Orientationd(keys=["image"], axcodes="LPS"),
                T.CropForegroundd(keys=["image"], source_key="image"),
                T.Resized(keys=["image"], spatial_size=(160, 192, 96), mode="trilinear"),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1
                ),
                T.ToTensord(keys=["image"]),
            ]
        ),
        cache_dir=cache_dir / "val",
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader

def get_model(model_type, config, pretrained=False):
    if model_type == "AutoencoderKL":
        from generative.networks.nets import AutoEncoderKL
        model = AutoEncoderKL(**config["model"])
        if pretrained:
            print("Using pretrained weights from Pinaya et al. for the autoencoder.")
            state_dict = gdown.download(
                id="1r0K8gkH2v2xw3tqT3m6d8l5X9f5JcX4j",  # Example ID, replace with actual
                quiet=False,
            )
        
            model.load_state_dict(state_dict)

    elif model_type == "DiffusionModelUNet":
        from generative.networks.nets import DiffusionModelUNet
        model = DiffusionModelUNet(**config["model"])
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    return model