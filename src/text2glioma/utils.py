import yaml
from pathlib import Path
import psutil
import datetime
from typing import Tuple, Any

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.data.dataset import PersistentDataset
from monai import transforms as T
from pynvml_utils import nvidia_smi
from torch.utils.tensorboard import SummaryWriter

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
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
    cache_dir: str = None,
):
    img_npy_0 = np.clip(a=img[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_0 = np.clip(a=recons[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    img_npy_1 = np.clip(a=img[0, 0, :, :, 30].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_1 = np.clip(a=recons[0, 0, :, :, 30].cpu().numpy(), a_min=0, a_max=1)
    img_npy_2 = np.clip(a=img[0, 0, :, :, 50].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_2 = np.clip(a=recons[0, 0, :, :, 50].cpu().numpy(), a_min=0, a_max=1)
    img_npy_3 = np.clip(a=img[0, 0, :, :, 40].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_3 = np.clip(a=recons[0, 0, :, :, 40].cpu().numpy(), a_min=0, a_max=1)

    img_row_0 = np.concatenate(
        (
            img_npy_0,
            recons_npy_0,
            img_npy_1,
            recons_npy_1,
        ),
        axis=1,
    )

    img_row_1 = np.concatenate(
        (
            img_npy_2,
            recons_npy_2,
            img_npy_3,
            recons_npy_3,
        ),
        axis=1,
    )

    img = np.concatenate(
        (
            img_row_0,
            img_row_1,
        ),
        axis=0,
    )

    fig = plt.figure(dpi=300)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    if cache_dir is not None:
        plt.savefig(str(Path(cache_dir, f'sample_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.png')), dpi=300)
    return fig
    
def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
    cache_dir: str = None,
) -> None:
    fig = get_figure(
        image,
        reconstruction,
        cache_dir=cache_dir,
    )
    writer.add_figure(title, fig, step)
    
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
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        model_type="AutoencoderKL",
        initialize=False,
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
                T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
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
                    spatial_size=[160, 224, 160],
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
                T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1
                ),
                T.RandAffined(
                    keys=["image"],
                    prob=0.1,
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
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
    train_data = PersistentDataset(
        data=train_dataset,
        transform=train_transform,
        cache_dir=cache_dir / "train",
    )

    val_data = PersistentDataset(
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
    )
    if initialize:
        train_data.set_data(train_dataset)  # Reset to ensure data is correct
        val_data.set_data(val_dataset)  # Reset to ensure data is correct

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader

def get_experiment_dataloaders(
        datalist,
        cache_dir,
        batch_size, 
        num_workers,
        pin_memory=False,
        shuffle=False,
        model_type="densenet121",
        ratio=0.1,
    ):
    if model_type not in ["densenet121", "resnet50", "efficientnet_b0"]:
        raise ValueError(f"Model type {model_type} not supported for dataloaders.")
    
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")
    if ratio > 0.0:
        print(f"Using {ratio*100}% synthetic data for training.")
        datalist["train"] = datalist["real_train"] + datalist["synthetic_train"]
    else:
        datalist["train"] = datalist["real_train"]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_transform = T.Compose(
        [
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.EnsureTyped(keys=["image"], dtype=torch.float32),
            T.Orientationd(keys=["image"], axcodes="LPS"),
            T.CropForegroundd(keys=["image"], source_key="image"),
            T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
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
    val_transform = T.Compose(
        [
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.EnsureTyped(keys=["image"], dtype=torch.float32),
            T.Orientationd(keys=["image"], axcodes="LPS"),
            T.CropForegroundd(keys=["image"], source_key="image"),
            T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
            T.ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1
            ),
            T.ToTensord(keys=["image"]),
        ]
    )

    train_ds = PersistentDataset(
        data=datalist["train"],
        transform=train_transform,
        cache_dir=cache_dir / "train",
        num_workers=num_workers,
    )
    val_ds = PersistentDataset(
        data=datalist["val"],
        transform=val_transform,
        cache_dir=cache_dir / "val",
        num_workers=num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader

def get_model(model_type, config, pretrained=False):
    if model_type == "AutoencoderKL":
        from generative.networks.nets import AutoencoderKL
        model = AutoencoderKL(**config["model"]["params"])
        if pretrained:
            print("Using pretrained weights from Pinaya et al. for the autoencoder.")
            state_dict = gdown.download(
                id="1r0K8gkH2v2xw3tqT3m6d8l5X9f5JcX4j",  # Example ID, replace with actual
                quiet=False,
            )
        
            model.load_state_dict(state_dict)

    elif model_type == "DiffusionModelUNet":
        from generative.networks.nets import DiffusionModelUNet
        model = DiffusionModelUNet(**config["model"]["params"])
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    return model

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

def print_resource_usage(epoch: int = None):
    if epoch:
        print(f"Epoch: {epoch}")
    print("--- Resource Usage ---")
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent:0.2f}% | RAM Usage: {ram.percent:0.2f}% ({ram.used // (1024**2):0.3f}MB/{ram.total // (1024**2):0.3f}MB)")
    if torch.cuda.is_available():
        cuda_device_list = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            vram_used = torch.cuda.memory_allocated(i) // (1024**2)
            vram_total = torch.cuda.get_device_properties(i).total_memory // (1024**2)
            print(f"GPU {i}: {gpu_name} | VRAM Usage: {vram_used:0.2f}MB/{vram_total:0.2f}MB")
            cuda_device_list.append({
                'device': gpu_name,
                "vram_used": vram_used,
                "vram_total": vram_total,
            })

@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    prompt_embeds = text_encoder(prompt_embeds.squeeze(1).to(device))
    prompt_embeds = prompt_embeds[0]

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)

    x_hat = stage1.decode(latent / scale_factor)
    img_0 = np.clip(a=x_hat[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    fig = plt.figure(dpi=300)
    plt.imshow(img_0, cmap="gray")
    plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)