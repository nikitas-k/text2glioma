import os
import yaml
from pathlib import Path
import psutil
from datetime import datetime as _datetime
from typing import Tuple, Any, Optional
from itertools import islice

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from monai.data import DataLoader
from monai.data.dataset import PersistentDataset
from monai import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

try:
    from pynvml_utils import nvidia_smi
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

import gdown
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTextModel, CLIPTextModelWithProjection

try:
    from transformers import AutoModelForMaskedLM
    _HAS_AUTO_MLM = True
except ImportError:
    _HAS_AUTO_MLM = False

class Stage1Wrapper(nn.Module):
    """Wraps the stage 1 model to bypass DataParallel issues."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _base_in_channels(self) -> int:
        return int(getattr(self.model, "in_channels", 1))

    def _base_out_channels(self) -> int:
        return int(getattr(self.model, "out_channels", 1))

    def _base_latent_channels(self) -> int:
        if hasattr(self.model, "latent_channels"):
            return int(self.model.latent_channels)
        if hasattr(self.model, "quant_conv_mu") and hasattr(self.model.quant_conv_mu, "out_channels"):
            return int(self.model.quant_conv_mu.out_channels)
        raise ValueError("Unable to infer stage-1 latent channels from model.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_in = self._base_in_channels()

        # Standard path: model input channels match tensor channels.
        if x.shape[1] == base_in:
            z_mu, z_sigma = self.model.encode(x)
            return self.model.sampling(z_mu, z_sigma)

        # Pinaya path: 1-channel VAE used on multi-channel data.
        # Encode each channel independently and concatenate latents.
        if base_in == 1 and x.shape[1] > 1:
            z_list = []
            for c in range(x.shape[1]):
                x_ch = x[:, c : c + 1]
                z_mu, z_sigma = self.model.encode(x_ch)
                z_list.append(self.model.sampling(z_mu, z_sigma))
            return torch.cat(z_list, dim=1)

        raise ValueError(
            f"Stage-1 input channel mismatch: model expects {base_in}, got {x.shape[1]}"
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to image space (supports channel-wise Pinaya path)."""
        base_latent = self._base_latent_channels()
        base_out = self._base_out_channels()

        # Standard path.
        if z.shape[1] == base_latent:
            return self.model.decode(z)

        # Pinaya path: split concatenated latents into per-channel chunks,
        # decode each chunk independently, then concatenate image channels.
        if base_out == 1 and z.shape[1] % base_latent == 0:
            recons = []
            n_chunks = z.shape[1] // base_latent
            for i in range(n_chunks):
                z_chunk = z[:, i * base_latent : (i + 1) * base_latent]
                recons.append(self.model.decode(z_chunk))
            return torch.cat(recons, dim=1)

        raise ValueError(
            f"Stage-1 latent channel mismatch: base latent={base_latent}, got {z.shape[1]}"
        )

# ── Mask conditioning utilities ──────────────────────────────────────────────

def masks_to_onehot(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Convert integer label tensor to one-hot encoding.
    
    Args:
        labels: [B, 1, D, H, W] integer label tensor (0=bg, 1=nCET, 2=edema, 3=enhancing).
        num_classes: Number of classes including background.
    
    Returns:
        One-hot tensor [B, num_classes, D, H, W] float32.
    """
    B, C, D, H, W = labels.shape
    labels_long = labels[:, 0].long()  # [B, D, H, W]
    labels_long = labels_long.clamp(0, num_classes - 1)
    onehot = F_torch.one_hot(labels_long, num_classes)  # [B, D, H, W, num_classes]
    onehot = onehot.permute(0, 4, 1, 2, 3).float()    # [B, num_classes, D, H, W]
    return onehot

def downsample_mask_to_latent(
    mask_onehot: torch.Tensor,
    latent_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Downsample one-hot mask to match the VAE latent spatial dimensions.
    
    Args:
        mask_onehot: [B, num_classes, D, H, W] one-hot mask at full resolution.
        latent_shape: (D', H', W') target spatial dimensions of the latent.
    
    Returns:
        Downsampled mask [B, num_classes, D', H', W'] via trilinear interpolation.
    """
    return F_torch.interpolate(
        mask_onehot, size=latent_shape, mode="trilinear", align_corners=False
    )

def prepare_mask_conditioning(
    labels: torch.Tensor,
    latent_shape: Tuple[int, int, int],
    num_classes: int = 4,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Full pipeline: labels → one-hot → downsample → optional dropout.
    
    Args:
        labels: [B, 1, D, H, W] integer segmentation labels.
        latent_shape: (D', H', W') spatial dims of VAE latent.
        num_classes: Number of label classes (including background).
        dropout_p: Probability of dropping mask conditioning (replacing with zeros)
                   for classifier-free guidance training.
    
    Returns:
        Mask conditioning tensor [B, num_classes, D', H', W'].
    """
    onehot = masks_to_onehot(labels, num_classes=num_classes)
    mask_cond = downsample_mask_to_latent(onehot, latent_shape)
    
    if dropout_p > 0.0 and mask_cond.requires_grad is False:
        B = mask_cond.shape[0]
        drop = (torch.rand(B, device=mask_cond.device) < dropout_p).float()
        drop = drop.view(B, 1, 1, 1, 1)  # broadcast over C, D, H, W
        mask_cond = mask_cond * (1.0 - drop)
    
    return mask_cond

def batchify(data, batch_size):
    """Yield successive n-sized chunks from data."""
    it = iter(data)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
# Channel names for multi-modal visualization
MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]

def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
    cache_dir: str = None,
):
    """Create reconstruction comparison figure.
    
    Supports multi-channel images: shows each modality as a separate row,
    with original and reconstruction side-by-side at different slice depths.
    For single-channel, produces the legacy 2×4 layout.
    """
    n_channels = img.shape[1]
    slice_depths = [60, 30, 50, 40]
    
    rows = []
    for ch in range(n_channels):
        cols = []
        for d in slice_depths:
            d_safe = min(d, img.shape[-1] - 1)
            orig = np.clip(img[0, ch, :, :, d_safe].float().cpu().numpy(), 0, 1)
            rec = np.clip(recons[0, ch, :, :, d_safe].float().cpu().numpy(), 0, 1)
            cols.append(np.concatenate((orig, rec), axis=1))  # side-by-side
        rows.append(np.concatenate(cols, axis=1))
    
    grid = np.concatenate(rows, axis=0)
    
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(grid, cmap="gray")
    ax.axis("off")
    # Add modality labels on the left
    if n_channels > 1:
        h_per_ch = grid.shape[0] / n_channels
        for ch in range(n_channels):
            name = MODALITY_NAMES[ch] if ch < len(MODALITY_NAMES) else f"ch{ch}"
            ax.text(2, int(h_per_ch * (ch + 0.5)), name,
                    fontsize=6, color="yellow", va="center")
    if cache_dir is not None:
        plt.savefig(str(Path(cache_dir, f'sample_{_datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.png')), dpi=300)
    return fig
    
def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
    cache_dir: str = None,
) -> None:
    if writer is None:
        return
    fig = get_figure(
        image,
        reconstruction,
        cache_dir=cache_dir,
    )
    writer.add_figure(title, fig, step)
    
def stage1_ify(stage1 : Any) -> Any:
    """Wraps the stage 1 model if it is not already wrapped."""
    if not isinstance(stage1, Stage1Wrapper):
        stage1 = Stage1Wrapper(stage1)
    return stage1

@torch.no_grad()
def compute_scale_factor(
    stage1: nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Compute 1 / std(latents) over a subset of the training set.

    This normalises the VAE latent space to approximately unit variance so
    the diffusion noise schedule works correctly.  Analogous to the
    ``scale_factor = 0.18215`` constant in Stable Diffusion.

    Args:
        stage1: Frozen Stage-1 VAE (or Stage1Wrapper).
        loader: Training DataLoader.
        device: Device to run on.
        max_batches: Number of batches to estimate from (default 50).

    Returns:
        Scalar scale factor ``1 / std(z)``.
    """
    stage1.eval()
    all_z = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch["image"].to(device)
        z = stage1(images)  # [B, C, D', H', W']
        all_z.append(z.flatten())
    all_z = torch.cat(all_z)
    std = all_z.std().item()
    scale_factor = 1.0 / std
    print(f"[scale_factor] latent std = {std:.4f}  →  scale_factor = {scale_factor:.4f}")
    return scale_factor


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
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
    if model_type not in ["AutoencoderKL", "DiffusionModelUNet"]:
        raise ValueError(f"Model type {model_type} not supported for dataloaders.")
    
    if model_type == "AutoencoderKL":
        train_transform = T.Compose(
            [
                T.LoadImaged(keys=["image"]),
                T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.Orientationd(keys=["image"], axcodes="LPS"),
                T.CropForegroundd(keys=["image"], source_key="image"),
                T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
                T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
                    channel_wise=True,
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
                    keys=["image"], offsets=0.05, prob=0.1,
                    channel_wise=True,
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
                T.LoadImaged(keys=["image", "label"]),
                T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
                T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.EnsureTyped(keys=["label"], dtype=torch.float32),
                T.Orientationd(keys=["image", "label"], axcodes="LPS"),
                T.CropForegroundd(keys=["image", "label"], source_key="image"),
                T.SpatialPadd(keys=["image", "label"], spatial_size=(160, 224, 160), mode="constant"),
                T.CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 224, 160)),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
                    channel_wise=True,
                ),
                T.RandAffined(
                    keys=["image", "label"],
                    prob=0.1,
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    mode=["trilinear", "nearest"],
                ),
                T.RandShiftIntensityd(
                    keys=["image"], offsets=0.05, prob=0.1,
                    channel_wise=True,
                ),
                T.RandAdjustContrastd(
                    keys=["image"], prob=0.1, gamma=(0.97, 1.03)
                ),
                T.ToTensord(keys=["image", "label"]),
            ]
        )
    train_data = PersistentDataset(
        data=train_dataset,
        transform=train_transform,
        cache_dir=cache_dir / "train",
    )

    if model_type == "DiffusionModelUNet":
        val_transform = T.Compose(
            [
                T.LoadImaged(keys=["image", "label"]),
                T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
                T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.EnsureTyped(keys=["label"], dtype=torch.float32),
                T.Orientationd(keys=["image", "label"], axcodes="LPS"),
                T.CropForegroundd(keys=["image", "label"], source_key="image"),
                T.SpatialPadd(keys=["image", "label"], spatial_size=(160, 224, 160), mode="constant"),
                T.CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 224, 160)),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
                    channel_wise=True,
                ),
                T.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        val_transform = T.Compose(
            [
                T.LoadImaged(keys=["image"]),
                T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
                T.EnsureTyped(keys=["image"], dtype=torch.float32),
                T.Orientationd(keys=["image"], axcodes="LPS"),
                T.CropForegroundd(keys=["image"], source_key="image"),
                T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
                T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
                T.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
                    channel_wise=True,
                ),
                T.ToTensord(keys=["image"]),
            ]
        )

    val_data = PersistentDataset(
        data=val_dataset,
        transform=val_transform,
        cache_dir=cache_dir / "val",
    )
    if initialize:
        train_data.set_data(train_dataset)  # Reset to ensure data is correct
        val_data.set_data(val_dataset)  # Reset to ensure data is correct

    train_sampler = None
    val_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=shuffle)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
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
            T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
            T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
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
    val_transform = T.Compose(
        [
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.EnsureTyped(keys=["image"], dtype=torch.float32),
            T.Orientationd(keys=["image"], axcodes="LPS"),
            T.CropForegroundd(keys=["image"], source_key="image"),
            T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
            T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
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

def _patch_attention_proj(model: nn.Module) -> None:
    """Fix MONAI Generative bug: AttentionBlock defines proj_attn but never
    calls it in forward().  This monkey-patches the forward method of every
    AttentionBlock to apply the output projection after multi-head attention,
    which is standard transformer behaviour and necessary for DDP (otherwise
    the unused parameters cause a reduction error).

    See: generative/networks/nets/autoencoderkl.py  AttentionBlock.__init__
    """
    from generative.networks.nets.autoencoderkl import AttentionBlock

    for module in model.modules():
        if isinstance(module, AttentionBlock) and hasattr(module, "proj_attn"):
            _bind_patched_forward(module)


def _bind_patched_forward(block: nn.Module) -> None:
    """Bind a patched forward to *block* that includes ``proj_attn``."""
    import types

    _orig = block.forward  # keep a reference

    def _forward(self, x: torch.Tensor) -> torch.Tensor:      # noqa: D401
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = self.proj_attn(x)            # ← FIX: output projection
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual

    block.forward = types.MethodType(_forward, block)


def apply_spectral_norm(discriminator: nn.Module) -> nn.Module:
    """Wrap every Conv layer in *discriminator* with spectral normalisation.

    Spectral normalisation (Miyato et al., 2018) constrains the Lipschitz
    constant of each layer to 1, which stabilises PatchGAN training and
    prevents the discriminator from creating sharp patch-boundary gradients
    that imprint grid artifacts onto the generator.

    The module is modified **in-place** and also returned for convenience.
    """
    count = 0
    for name, mod in list(discriminator.named_modules()):
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            # spectral_norm modifies the module in-place (reparameterises
            # weight → weight_orig + weight_v/u), so no parent navigation
            # or setattr is needed.
            try:
                nn.utils.spectral_norm(mod)
                count += 1
            except Exception:
                pass  # already wrapped or unsupported — skip silently
    if count:
        print(f"[spectral_norm] Wrapped {count} conv layers in discriminator")
    return discriminator


def _extract_state_dict_from_checkpoint(checkpoint_obj: Any) -> dict:
    """Extract a raw state_dict from common checkpoint containers."""
    if isinstance(checkpoint_obj, dict):
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
        for key in ("state_dict", "model", "autoencoder", "vae"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict) and value:
                return value
    raise ValueError("Unsupported checkpoint format: unable to extract state_dict.")


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> Optional[dict]:
    """Return a prefix-stripped copy if all keys start with prefix, else None."""
    keys = list(state_dict.keys())
    if not keys:
        return None
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return None


def _normalise_autoencoder_state_dict(state_dict: dict) -> dict:
    """Normalise common wrapper prefixes for AutoencoderKL checkpoints.

    Handles formats such as:
      - module.vae.*
      - vae.*
      - module.*
    """
    for prefix in ("module.vae.", "vae.", "module."):
        stripped = _strip_prefix_if_present(state_dict, prefix)
        if stripped is not None:
            return stripped
    return state_dict


def get_model(model_type, config, pretrained=False, from_file=None):
    if model_type == "AutoencoderKL":
        from generative.networks.nets import AutoencoderKL
        model = AutoencoderKL(**config["model"]["params"])
        _patch_attention_proj(model)
        if pretrained:
            print("Using pretrained weights from Pinaya et al. for the autoencoder.")
            state_dict = gdown.download(
                id="1r0K8gkH2v2xw3tqT3m6d8l5X9f5JcX4j",  # Example ID, replace with actual
                quiet=False,
            )
        
            model.load_state_dict(state_dict)
        elif from_file is not None:
            print(f"Loading autoencoder weights from {from_file}.")
            ckpt = torch.load(from_file, map_location="cpu")
            state_dict = _extract_state_dict_from_checkpoint(ckpt)
            state_dict = _normalise_autoencoder_state_dict(state_dict)
            model.load_state_dict(state_dict)

    elif model_type == "DiffusionModelUNet":
        from generative.networks.nets import DiffusionModelUNet
        model = DiffusionModelUNet(**config["model"]["params"])
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    return model

def _get_rank() -> int:
    """Return the current distributed rank, or 0 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def print_gpu_memory_report(rank: Optional[int] = None):
    """Print GPU memory utilisation.  Only prints on rank 0 (or when rank is explicitly 0)."""
    if rank is None:
        rank = _get_rank()
    if rank != 0:
        return
    if torch.cuda.is_available() and _HAS_PYNVML:
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print(f"[rank-{rank}] [INFO] Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"[rank-{rank}] [INFO] gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")
    elif torch.cuda.is_available():
        print(f"[rank-{rank}] [INFO] Memory report")
        for i in range(torch.cuda.device_count()):
            vram_used = torch.cuda.memory_allocated(i) // (1024**2)
            vram_total = torch.cuda.get_device_properties(i).total_mem // (1024**2)
            print(f"[rank-{rank}] [INFO] gpu:{i} mem(%) {int(vram_used * 100.0 / max(vram_total, 1))}")

def print_resource_usage(epoch: int = None):
    rank = _get_rank()
    if rank != 0:
        return
    if epoch:
        print(f"[rank-0] [INFO] Epoch: {epoch}")
    print("[rank-0] [INFO] --- Resource Usage ---")
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    print(f"[rank-0] [INFO] CPU Usage: {cpu_percent:0.2f}% | RAM Usage: {ram.percent:0.2f}% ({ram.used // (1024**2):0.3f}MB/{ram.total // (1024**2):0.3f}MB)")
    if torch.cuda.is_available():
        cuda_device_list = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            vram_used = torch.cuda.memory_allocated(i) // (1024**2)
            vram_total = torch.cuda.get_device_properties(i).total_memory // (1024**2)
            print(f"[rank-0] [INFO] GPU {i}: {gpu_name} | VRAM Usage: {vram_used:0.2f}MB/{vram_total:0.2f}MB")
            cuda_device_list.append({
                'device': gpu_name,
                "vram_used": vram_used,
                "vram_total": vram_total,
            })

def get_text_encoder_hidden_states(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, "last_hidden_state") and encoder_output.last_hidden_state is not None:
        return encoder_output.last_hidden_state
    if hasattr(encoder_output, "text_model_output") and encoder_output.text_model_output is not None:
        return encoder_output.text_model_output.last_hidden_state
    if hasattr(encoder_output, "hidden_states") and encoder_output.hidden_states is not None:
        return encoder_output.hidden_states[-1]
    if hasattr(encoder_output, "text_embeds") and encoder_output.text_embeds is not None:
        return encoder_output.text_embeds[:, None, :]
    if isinstance(encoder_output, (tuple, list)) and encoder_output and torch.is_tensor(encoder_output[0]):
        return encoder_output[0]
    raise ValueError("Unsupported text encoder output; unable to extract hidden states.")

def _resolve_hf_local(name_or_path: str, cache_dir: str = None, subfolder: str = "") -> str:
    """Resolve a HuggingFace hub identifier to its local cache snapshot path.

    If *name_or_path* is already a local directory it is returned as-is.
    Otherwise we look up the cached snapshot via ``huggingface_hub``.
    Returning a real directory path prevents ``transformers`` from making
    network requests inside ``has_file()`` (a bug in some versions).
    """
    path = os.path.join(name_or_path, subfolder) if subfolder else name_or_path
    if os.path.isdir(path):
        return name_or_path

    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            name_or_path,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return local
    except Exception:
        # Cache miss or huggingface_hub not installed — fall back to the
        # original name and let transformers handle it.
        return name_or_path


def load_text_encoder_and_tokenizer(conditioning_config: dict, cache_dir: str = None, local_files_only: bool = True):
    tokenizer_name = conditioning_config.get("tokenizer")
    text_encoder_name = conditioning_config.get("text_encoder")
    if not tokenizer_name or not text_encoder_name:
        raise ValueError("Tokenizer and text encoder must be specified in the configuration file.")

    # Force offline mode at the env-var level as well — some transformers
    # versions still make network requests inside has_file() even when
    # local_files_only=True.
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer_subfolder = conditioning_config.get("tokenizer_subfolder", "tokenizer")
    text_encoder_subfolder = conditioning_config.get("text_encoder_subfolder", "text_encoder")
    text_encoder_class = conditioning_config.get("text_encoder_class", "CLIPTextModel")

    def normalize_subfolder(subfolder_value: Any) -> str:
        if not subfolder_value:
            return ""
        return str(subfolder_value)

    tok_sub = normalize_subfolder(tokenizer_subfolder)
    enc_sub = normalize_subfolder(text_encoder_subfolder)

    # Resolve hub names to local cache snapshot directories so that
    # from_pretrained never attempts a network call.
    if local_files_only:
        tokenizer_name = _resolve_hf_local(tokenizer_name, cache_dir, tok_sub)
        text_encoder_name = _resolve_hf_local(text_encoder_name, cache_dir, enc_sub)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        subfolder=tok_sub,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    encoder_classes = {
        "CLIPTextModel": CLIPTextModel,
        "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
        "CLIPModel": CLIPModel,
        "AutoModel": AutoModel,
    }
    if _HAS_AUTO_MLM:
        encoder_classes["AutoModelForMaskedLM"] = AutoModelForMaskedLM
    encoder_cls = encoder_classes.get(text_encoder_class)
    if encoder_cls is None:
        raise ValueError(
            f"Unsupported text encoder class: {text_encoder_class}. "
            f"Supported: {list(encoder_classes.keys())}"
        )

    text_encoder = encoder_cls.from_pretrained(
        text_encoder_name,
        subfolder=enc_sub,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    return tokenizer, text_encoder


def _sample_latent_once(
    model: nn.Module,
    scheduler: nn.Module,
    start_latent: torch.Tensor,
    mask_cond: torch.Tensor,
    prompt_embeds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one diffusion sampling trajectory from a provided initial latent."""
    latent = start_latent.clone()
    model_input = torch.cat([latent, mask_cond], dim=1)

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(
            x=model_input,
            timesteps=torch.asarray((t,)).to(device),
            context=prompt_embeds,
        )
        latent, _ = scheduler.step(noise_pred, t, latent)
        model_input = torch.cat([latent, mask_cond], dim=1)

    return latent


def _build_multislice_grid(x_hat: torch.Tensor, depth_indices: list[int]) -> np.ndarray:
    """Create a grid with rows=depth slices and columns=modalities."""
    n_ch = x_hat.shape[1]
    rows = []
    for d in depth_indices:
        d_safe = max(0, min(int(d), x_hat.shape[-1] - 1))
        cols = []
        for c in range(n_ch):
            cols.append(np.clip(x_hat[0, c, :, :, d_safe].float().cpu().numpy(), 0, 1))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)

@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    tokenizer,
    text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
    latent_channels: int = 3,
    num_mask_classes: int = 4,
    conditioning_label: torch.Tensor = None,
    prompt_text: str = None,
) -> None:
    # Only rank 0 has a writer; skip the expensive sampling on other ranks.
    if writer is None:
        return

    latent_spatial = spatial_shape[1:]  # strip channel dim → (D', H', W')
    latent = torch.randn((1, latent_channels) + latent_spatial, device=device)

    # Optional conditional mask from validation batch; fallback to zero mask.
    if conditioning_label is not None:
        label = conditioning_label.to(device)
        if label.ndim == 4:
            label = label.unsqueeze(0)
        if label.shape[0] > 1:
            label = label[:1]
        mask_cond = prepare_mask_conditioning(
            labels=label,
            latent_shape=latent_spatial,
            num_classes=num_mask_classes,
            dropout_p=0.0,
        ).to(device)
    else:
        mask_cond = torch.zeros((1, num_mask_classes) + latent_spatial, device=device)

    mask_uncond = torch.zeros_like(mask_cond)

    # Build embeddings for conditioned and unconditional sampling.
    cond_text = prompt_text if prompt_text is not None else ""

    def _encode_text(text: str) -> torch.Tensor:
        tokens = tokenizer(
            [text],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return get_text_encoder_hidden_states(text_encoder(**tokens))

    cond_embeds = _encode_text(cond_text)
    uncond_embeds = _encode_text("")

    # CRITICAL: configure the scheduler for inference (sets internal
    # timestep spacing based on num_inference_steps). Without this,
    # scheduler.timesteps and scheduler.step() are not properly initialised.
    num_inference_steps = min(200, scheduler.num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    # Use the same starting latent to compare conditioned vs unconditioned outputs.
    latent_cond = _sample_latent_once(model, scheduler, latent, mask_cond, cond_embeds, device)
    latent_uncond = _sample_latent_once(model, scheduler, latent, mask_uncond, uncond_embeds, device)

    x_hat_cond = stage1.decode(latent_cond / scale_factor)
    x_hat_uncond = stage1.decode(latent_uncond / scale_factor)

    n_ch = x_hat_cond.shape[1]
    depth = x_hat_cond.shape[-1]
    depth_indices = [depth // 4, depth // 2, (3 * depth) // 4]

    grid_cond = _build_multislice_grid(x_hat_cond, depth_indices)
    grid_uncond = _build_multislice_grid(x_hat_uncond, depth_indices)

    has_mask = conditioning_label is not None
    n_rows = 3 if has_mask else 2
    fig, axes = plt.subplots(n_rows, 1, dpi=300, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    row_idx = 0
    if has_mask:
        label_vis = conditioning_label
        if label_vis.ndim == 5:
            label_vis = label_vis[0]
        if label_vis.ndim == 4:
            label_vis = label_vis[0]
        mask_depth = max(0, min(depth_indices[1], label_vis.shape[-1] - 1))
        label_slice = label_vis[:, :, mask_depth].detach().float().cpu().numpy()
        label_slice = np.clip(label_slice, 0, num_mask_classes - 1)
        axes[row_idx].imshow(label_slice, cmap="tab20", vmin=0, vmax=max(num_mask_classes - 1, 1))
        axes[row_idx].set_title("Input Mask (middle depth)")
        axes[row_idx].axis("off")
        row_idx += 1

    axes[row_idx].imshow(grid_cond, cmap="gray")
    axes[row_idx].set_title("Conditioned Sample (3 depths)")
    axes[row_idx].axis("off")
    row_idx += 1

    axes[row_idx].imshow(grid_uncond, cmap="gray")
    axes[row_idx].set_title("Unconditioned Sample (3 depths)")
    axes[row_idx].axis("off")

    for ax in axes[1:] if has_mask else axes:
        if n_ch > 1:
            w_per_ch = grid_cond.shape[1] / n_ch
            for c in range(n_ch):
                name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}"
                ax.text(int(w_per_ch * c) + 2, 10, name, fontsize=6, color="yellow")

    if prompt_text is not None:
        fig.suptitle(f"Prompt: {prompt_text[:200]}", fontsize=8)
        writer.add_text("SAMPLE_PROMPT", prompt_text, step)

    fig.tight_layout()
    writer.add_figure("SAMPLE", fig, step)
    plt.close(fig)
