"""DDP-aware VAE training using MONAI DecathlonDataset (BraTS).

Launch with ``torchrun``::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage1_ddp \
        --config configs/stage1.yaml --run_dir /runs/ --num_epochs 300

On a PBS cluster::

    qsub scripts/torchrun_hpc.sh \
        -v TRAIN_ARGS="--nproc_per_node 4 \
            -m text2glioma.training.train_stage1_ddp \
            --config configs/stage1.yaml --run_dir /runs/"
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.training.training_functions import train_autoencoder
from text2glioma.utils import get_model, load_config

warnings.filterwarnings("ignore")

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-1 VAE with DDP on BraTS DecathlonDataset",
    )
    p.add_argument("--config", type=str, required=True, help="Stage-1 YAML config.")
    p.add_argument("--run_dir", type=str, required=True, help="Root output directory.")
    p.add_argument("--data_dir", type=str, default="./data",
                    help="Root for DecathlonDataset download / cache.")
    p.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of training set reserved for validation.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False,
                    help="Resume from checkpoint.pth in run_dir.")
    p.add_argument("--pretrained", action="store_true", default=False,
                    help="Load pretrained VAE weights (Pinaya et al.).")
    p.add_argument("--dist_backend", type=str, default="nccl",
                    choices=["nccl", "gloo"],
                    help="Distributed backend (nccl for GPU, gloo for CPU/fallback).")
    p.add_argument("--find_unused_parameters", action="store_true", default=False,
                    help="Pass find_unused_parameters=True to DDP (slower but needed for some models).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed(backend: str) -> tuple[int, int, int]:
    """Initialise the process group and return (rank, world_size, local_rank).

    Works automatically when launched via ``torchrun``, which sets the
    environment variables ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, and
    ``MASTER_ADDR``/``MASTER_PORT``.  If those variables are missing the
    function falls back to single-process mode.
    """
    if "RANK" not in os.environ:
        # Single-process mode
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


def is_main(rank: int) -> bool:
    return rank == 0


def print0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main(rank):
        print(msg)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform() -> T.Compose:
    """Training transforms for 4-ch BraTS images."""
    return T.Compose([
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        # Reorder channels: MSD (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
        T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
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


def get_val_transform() -> T.Compose:
    """Validation transforms (deterministic)."""
    return T.Compose([
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.Resized(keys=["image"], spatial_size=(160, 224, 160), mode="trilinear"),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.ToTensord(keys=["image"]),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_determinism(args.seed)

    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    distributed = world_size > 1

    print0(f"World size: {world_size}  |  Backend: {args.dist_backend}", rank)
    if is_main(rank):
        print_config()

    # ------------------------------------------------------------------
    # Dataset: download on login node first if compute nodes lack internet
    #   python -c "from monai.apps import DecathlonDataset; \
    #     DecathlonDataset('/path/to/data', 'Task01_BrainTumour', download=True)"
    # ------------------------------------------------------------------
    if is_main(rank):
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    task_dir = Path(args.data_dir) / "Task01_BrainTumour"
    need_download = not task_dir.is_dir()
    download = need_download and is_main(rank)
    if distributed:
        dist.barrier()  # other ranks wait for download

    train_ds = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        section="training",
        download=download,
        seed=args.seed,
        val_frac=args.val_frac,
        transform=get_train_transform(),
        num_workers=args.num_workers,
    )
    if distributed:
        dist.barrier()

    val_ds = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        section="validation",
        download=False,
        seed=args.seed,
        val_frac=args.val_frac,
        transform=get_val_transform(),
        num_workers=args.num_workers,
    )

    # Samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    print0(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples", rank)

    # ------------------------------------------------------------------
    # Model, discriminator, perceptual loss
    # ------------------------------------------------------------------
    config = load_config(args.config)
    model_type = config["model"]["name"]
    model = get_model(model_type, config, args.pretrained)

    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    cache_dir = Path(args.run_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    perceptual_loss = PerceptualLoss(
        **config["perceptual_network"]["params"], cache_dir=cache_dir,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    discriminator = discriminator.to(device)
    perceptual_loss = perceptual_loss.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=args.find_unused_parameters)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank,
                            find_unused_parameters=args.find_unused_parameters)

    # ------------------------------------------------------------------
    # Optimisers & scalers
    # ------------------------------------------------------------------
    optimizer_g = optim.AdamW(model.parameters(), lr=config["model"]["lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["lr"])
    scaler_g = torch.amp.GradScaler()
    scaler_d = torch.amp.GradScaler()

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    run_dir = Path(args.run_dir) / "text2glioma" / "autoencoder_stage1"
    output_dir = run_dir / "output"
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    if is_main(rank):
        for d in [output_dir, model_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)
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
    else:
        print0("Starting fresh training.", rank)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print0("Starting training …", rank)
    val_loss = train_autoencoder(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
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
    )
    print0(f"Training finished.  Best val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
