#!/usr/bin/env python
"""Fine-tune the Pinaya 2022 pretrained VAE on multi-channel BraTS data.

Two strategies are supported:

  Strategy A — ``--strategy decoder_only``
    Keep the Pinaya architecture (1-channel in/out, num_channels=[64,128,128,128]).
    Freeze the entire encoder; fine-tune only the decoder.
    Each modality channel is processed independently through the VAE.
    The discriminator and perceptual loss operate on single-channel volumes.

  Strategy B — ``--strategy full_4ch``
    Adapt the Pinaya architecture to 4-channel in/out by inflating
    ``encoder.conv_in`` (1→4) and ``decoder.conv_out`` (1→4) with replicated
    + scaled Pinaya weights.  Fine-tune ALL parameters with a small LR.
    The discriminator and perceptual loss work on the full 4-channel output.

Both strategies use the existing ``train_autoencoder`` / ``eval_autoencoder``
machinery from the text2glioma training pipeline, so TensorBoard logs include
the same metrics (L1, SSIM, MS-SSIM per-channel, perceptual, KL, etc.).

Usage on Gadi::

    torchrun --standalone --nproc_per_node=4 scripts/launch_pinaya_finetune.py \\
        --config configs/stage1_pinaya_finetune.yaml \\
        --run_dir /g/data/vp06/$USER/text2glioma_train/runs \\
        --strategy decoder_only \\
        --datalist datalist_N1510.json --no_channel_reorder \\
        --num_epochs 50 --batch_size 4 --val_interval 5

Local MPS testing::

    python scripts/launch_pinaya_finetune.py \\
        --config configs/stage1_pinaya_finetune.yaml \\
        --run_dir ./runs --strategy decoder_only \\
        --data_dir ./data --batch_size 1 --num_epochs 10 \\
        --dist_backend gloo
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.training.training_functions import (
    train_autoencoder,
)
from text2glioma.utils import (
    _patch_attention_proj,
    apply_spectral_norm,
    load_config,
)

warnings.filterwarnings("ignore")

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]

# ── Pinaya pretrained weights ──────────────────────────────────────────────
PINAYA_WEIGHTS_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1CZHwxHJWybOsDavipD0EorDPOo_mzNeX"
)


# ---------------------------------------------------------------------------
# Weight download & channel inflation
# ---------------------------------------------------------------------------

def download_pinaya_weights(cache_dir: Path) -> Path:
    """Download autoencoder.pth from Google Drive via gdown."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "pinaya_autoencoder.pth"
    if out_path.exists():
        print(f"  Cached weights: {out_path}")
        return out_path
    import gdown

    print(f"  Downloading Pinaya VAE weights → {out_path}")
    gdown.download(PINAYA_WEIGHTS_URL, str(out_path), quiet=False)
    if not out_path.exists():
        raise RuntimeError("Download failed.")
    return out_path


def inflate_conv_weight(w: torch.Tensor, target_channels: int, dim: int) -> torch.Tensor:
    """Replicate a 1-channel conv weight to target_channels and scale.

    For encoder.conv_in (dim=1, input channels):
        (64, 1, k, k, k) → (64, 4, k, k, k)  by repeating + /4
    For decoder.conv_out (dim=0, output channels):
        (1, 64, k, k, k) → (4, 64, k, k, k)  by repeating + /4
    """
    reps = [1] * w.ndim
    reps[dim] = target_channels
    return w.repeat(*reps) / target_channels


def load_pinaya_for_strategy(
    strategy: str,
    config: dict,
    weights_path: Path,
) -> AutoencoderKL:
    """Build AutoencoderKL and load Pinaya weights, with strategy-specific adaptation.

    Returns a model ready for fine-tuning (not yet on device).
    """
    params = config["model"]["params"]
    model = AutoencoderKL(**params)
    _patch_attention_proj(model)

    pinaya_sd = torch.load(str(weights_path), map_location="cpu")

    if strategy == "decoder_only":
        # Architecture must match Pinaya exactly (1ch in/out).
        # Strict load, then freeze encoder.
        model.load_state_dict(pinaya_sd, strict=True)
        # Freeze encoder + quant_conv (encoder side)
        for name, param in model.named_parameters():
            if name.startswith("encoder.") or name.startswith("quant_conv_mu.") or name.startswith("quant_conv_log_sigma."):
                param.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in model.parameters())
        print(f"  [decoder_only] Frozen {n_frozen}/{n_total} parameter tensors "
              f"({sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6:.1f}M params)")
        print(f"  [decoder_only] Trainable: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")

    elif strategy == "full_4ch":
        # Architecture has in_channels=4, out_channels=4 but otherwise
        # matches Pinaya's structure.  We inflate the first/last convs.
        new_sd = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in pinaya_sd.items():
            if k not in new_sd:
                skipped += 1
                continue
            target = new_sd[k]
            if v.shape == target.shape:
                new_sd[k] = v
                loaded += 1
            elif k == "encoder.conv_in.weight":
                # (64, 1, 3, 3, 3) → (64, 4, 3, 3, 3)
                new_sd[k] = inflate_conv_weight(v, target.shape[1], dim=1)
                loaded += 1
                print(f"  [inflate] {k}: {tuple(v.shape)} → {tuple(target.shape)}")
            elif k == "decoder.conv_out.weight":
                # (1, C, 3, 3, 3) → (4, C, 3, 3, 3)
                new_sd[k] = inflate_conv_weight(v, target.shape[0], dim=0)
                loaded += 1
                print(f"  [inflate] {k}: {tuple(v.shape)} → {tuple(target.shape)}")
            elif k == "decoder.conv_out.bias":
                # (1,) → (4,)
                new_sd[k] = v.repeat(target.shape[0])
                loaded += 1
                print(f"  [inflate] {k}: {tuple(v.shape)} → {tuple(target.shape)}")
            else:
                # Shape mismatch in deeper layers → skip (random init)
                skipped += 1
                print(f"  [skip] {k}: pinaya {tuple(v.shape)} ≠ ours {tuple(target.shape)}")
        model.load_state_dict(new_sd)
        print(f"  [full_4ch] Loaded {loaded} tensors, skipped {skipped}")
        print(f"  [full_4ch] All {sum(p.numel() for p in model.parameters())/1e6:.1f}M params trainable")

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return model


# ---------------------------------------------------------------------------
# 1-channel VAE wrapper for decoder_only strategy
# ---------------------------------------------------------------------------

class PerChannelVAEWrapper(nn.Module):
    """Wraps a 1-channel AutoencoderKL so it can handle 4-channel batches.

    forward(x) processes each channel independently through the VAE and
    returns concatenated 4-channel output, matching the signature expected
    by train_autoencoder: ``reconstruction, z_mu, z_sigma = model(x=images)``.

    The z_mu and z_sigma are concatenated along the channel dim across all
    4 passes.
    """

    def __init__(self, vae: AutoencoderKL, n_channels: int = 4):
        super().__init__()
        self.vae = vae
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert C == self.n_channels
        recons, z_mus, z_sigmas = [], [], []
        for c in range(C):
            x_ch = x[:, c:c+1]  # (B, 1, D, H, W)
            r, zm, zs = self.vae(x=x_ch)
            recons.append(r)
            z_mus.append(zm)
            z_sigmas.append(zs)
        return (
            torch.cat(recons, dim=1),    # (B, 4, D, H, W)
            torch.cat(z_mus, dim=1),     # (B, 4*3, d, h, w)
            torch.cat(z_sigmas, dim=1),  # (B, 4*3, d, h, w)
        )

    def encode(self, x: torch.Tensor):
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor):
        return self.vae.decode(z)


# ---------------------------------------------------------------------------
# Transforms (reused from train_stage1_ddp)
# ---------------------------------------------------------------------------

def get_train_transform(channel_reorder: bool = True) -> T.Compose:
    xforms = [
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        T.RandAffined(
            keys=["image"], prob=0.1,
            translate_range=(1, 1, 1), scale_range=(-0.02, 0.02),
            spatial_size=[160, 224, 160], mode="trilinear",
        ),
        T.RandShiftIntensityd(
            keys=["image"], offsets=0.05, prob=0.1, channel_wise=True,
        ),
        T.RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.97, 1.03)),
        T.ToTensord(keys=["image"]),
    ])
    return T.Compose(xforms)


def get_val_transform(channel_reorder: bool = True) -> T.Compose:
    xforms = [
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.SpatialPadd(keys=["image"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.ToTensord(keys=["image"]),
    ])
    return T.Compose(xforms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Pinaya VAE on BraTS")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--strategy", type=str, required=True,
                   choices=["decoder_only", "full_4ch"],
                   help="decoder_only: freeze encoder, 1ch-per-pass. "
                        "full_4ch: inflate to 4ch, train all.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--datalist", type=str, default=None)
    p.add_argument("--no_channel_reorder", action="store_true", default=False)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dist_backend", type=str, default="nccl",
                   choices=["nccl", "gloo"])
    p.add_argument("--find_unused_parameters", action="store_true", default=False)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--pinaya_weights", type=str, default=None,
                   help="Path to pinaya_autoencoder.pth (auto-downloads if absent)")
    p.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers (identical to train_stage1_ddp)
# ---------------------------------------------------------------------------

def setup_distributed(backend: str):
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return rank, world_size, local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def print0(msg, rank):
    if is_main(rank):
        print(msg)


def _apply_overrides(config, overrides):
    import ast
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if value.lower() in ("true", "yes"):
                value = True
            elif value.lower() in ("false", "no"):
                value = False
        parts = key.split(".")
        d = config
        for p_part in parts[:-1]:
            d = d[p_part]
        leaf = parts[-1]
        old = d.get(leaf, "<NEW>")
        print(f"[override] {key}: {old!r} -> {value!r}")
        d[leaf] = value
    return config


def _save_config_snapshot(config, args, run_dir):
    import yaml
    snapshot = copy.deepcopy(config)
    snapshot["_cli"] = {k: v for k, v in vars(args).items()
                        if v is not None and k != "config"}
    snapshot["_meta"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(Path(args.config).resolve()),
        "strategy": args.strategy,
    }
    out_path = run_dir / "config_snapshot.yaml"
    with open(out_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    print(f"[rank-0] Config snapshot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_determinism(args.seed)

    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    distributed = world_size > 1
    print0(f"Strategy: {args.strategy}  |  World size: {world_size}", rank)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = load_config(args.config)
    if args.set:
        config = _apply_overrides(config, args.set)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    channel_reorder = not args.no_channel_reorder

    if args.datalist:
        print0(f"Loading datalist from {args.datalist}", rank)
        with open(args.datalist) as f:
            datalist = json.load(f)
        train_data = datalist["training"]
        val_data = datalist["validation"]
        print0(f"  {len(train_data)} training, {len(val_data)} validation", rank)
        train_ds = Dataset(data=train_data, transform=get_train_transform(channel_reorder))
        val_ds = Dataset(data=val_data, transform=get_val_transform(channel_reorder))
    else:
        if is_main(rank):
            Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        task_dir = Path(args.data_dir) / "Task01_BrainTumour"
        need_download = not task_dir.is_dir()
        download = need_download and is_main(rank)
        if distributed:
            dist.barrier()
        train_ds = DecathlonDataset(
            root_dir=args.data_dir, task="Task01_BrainTumour",
            section="training", download=download, seed=args.seed,
            val_frac=args.val_frac,
            transform=get_train_transform(channel_reorder),
            num_workers=args.num_workers,
        )
        if distributed:
            dist.barrier()
        val_ds = DecathlonDataset(
            root_dir=args.data_dir, task="Task01_BrainTumour",
            section="validation", download=False, seed=args.seed,
            val_frac=args.val_frac,
            transform=get_val_transform(channel_reorder),
            num_workers=args.num_workers,
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
    )
    print0(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}", rank)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Download Pinaya weights
    # ------------------------------------------------------------------
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.run_dir) / "cache"
    if args.pinaya_weights:
        weights_path = Path(args.pinaya_weights)
    else:
        if is_main(rank):
            weights_path = download_pinaya_weights(cache_dir)
        if distributed:
            dist.barrier()
        weights_path = cache_dir / "pinaya_autoencoder.pth"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print0("Loading Pinaya VAE …", rank)
    model = load_pinaya_for_strategy(args.strategy, config, weights_path)

    if args.strategy == "decoder_only":
        # Wrap in PerChannelVAEWrapper so 4-ch images are handled correctly
        model = PerChannelVAEWrapper(model, n_channels=4)

    model = model.to(device)

    # ------------------------------------------------------------------
    # Discriminator + perceptual loss
    # ------------------------------------------------------------------
    disc_params = config["discriminator"]["params"]
    discriminator = PatchDiscriminator(**disc_params)
    if config["discriminator"].get("spectral_norm", False):
        apply_spectral_norm(discriminator)
    discriminator = discriminator.to(device)

    perceptual_loss = PerceptualLoss(
        **config["perceptual_network"]["params"], cache_dir=cache_dir,
    ).to(device)

    # ------------------------------------------------------------------
    # DDP
    # ------------------------------------------------------------------
    if distributed:
        # decoder_only: the encoder params are frozen, so we need
        # find_unused_parameters=True (or they'd error in DDP).
        fup = args.find_unused_parameters or (args.strategy == "decoder_only")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=fup)
        discriminator = DDP(discriminator, device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=args.find_unused_parameters)

    # ------------------------------------------------------------------
    # Optimisers — only trainable parameters
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = optim.AdamW(trainable_params, lr=config["model"]["lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["lr"])
    print0(f"Optimiser G: {len(trainable_params)} param groups, "
           f"lr={config['model']['lr']}", rank)

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    default_name = f"pinaya_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = args.run_name or default_name
    run_dir = Path(args.run_dir) / run_name / "autoencoder_stage1"
    output_dir = run_dir / "output"
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    if is_main(rank):
        for d in [output_dir, model_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        _save_config_snapshot(config, args, run_dir)
    if distributed:
        dist.barrier()

    writer_train = SummaryWriter(log_dir / "train") if is_main(rank) else None
    writer_val = SummaryWriter(log_dir / "val") if is_main(rank) else None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = float("inf")
    ckpt_path = run_dir / "checkpoint.pth"
    if args.resume and ckpt_path.exists():
        print0(f"Resuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("best_loss", float("inf"))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print0(f"Fine-tuning for {args.num_epochs} epochs  "
           f"(strategy={args.strategy}) …", rank)

    train_autoencoder(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        n_epochs=args.num_epochs,
        start_epoch=start_epoch,
        best_loss=best_loss,
        val_interval=args.val_interval,
        model_dir=model_dir,
        writer_train=writer_train,
        writer_val=writer_val,
        run_dir=run_dir,
        kl_weight=config["model"]["kl_weight"],
        perceptual_weight=config["model"]["perceptual_weight"],
        adversarial_weight=config["model"]["adv_weight"],
        autoencoder_warm_up_n_epochs=config["model"].get("autoencoder_warm_up_n_epochs", 0),
        d_skip_threshold=config["model"].get("d_skip_threshold", 0.0),
        r1_gamma=config["model"].get("r1_gamma", 0.0),
        kl_warmup_epochs=config["model"].get("kl_warmup_epochs", 0),
        kl_max=config["model"].get("kl_max", 0.0),
        adaptive_adv_weight=config["model"].get("adaptive_adv_weight", False),
        wavelet_loss_weight=config["model"].get("wavelet_loss_weight", 0.0),
        wavelet_detail_weight=config["model"].get("wavelet_detail_weight", 2.0),
        wavelet_name=config["model"].get("wavelet_name", "haar"),
    )

    print0("Fine-tuning complete.", rank)
    cleanup()


if __name__ == "__main__":
    main()
