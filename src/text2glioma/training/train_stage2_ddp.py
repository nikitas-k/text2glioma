"""DDP-aware LDM (stage 2) training using MONAI DecathlonDataset or custom datalist.

Requires a pre-trained Stage-1 VAE checkpoint (``--stage1_uri``).

With DecathlonDataset::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage2_ddp \
        --config configs/ldm.yaml --stage1_config configs/stage1.yaml \
        --stage1_uri /path/to/best_model.pth --run_dir /runs/

With a custom datalist::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage2_ddp \
        --config configs/ldm.yaml --stage1_config configs/stage1.yaml \
        --stage1_uri /path/to/best_model.pth --run_dir /runs/ \
        --datalist datalist_task03.json --no_channel_reorder
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.training.training_functions import train_ldm
from text2glioma.utils import (
    get_model,
    load_config,
    load_text_encoder_and_tokenizer,
    stage1_ify,
)

warnings.filterwarnings("ignore")

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-2 LDM with DDP on BraTS or a custom datalist",
    )
    p.add_argument("--config", type=str, required=True,
                    help="LDM (stage 2) YAML config (e.g. configs/ldm.yaml).")
    p.add_argument("--stage1_config", type=str, required=True,
                    help="Stage-1 VAE YAML config (e.g. configs/stage1.yaml).")
    p.add_argument("--stage1_uri", type=str, required=True,
                    help="Path to pretrained Stage-1 VAE checkpoint (.pth).")
    p.add_argument("--run_dir", type=str, required=True,
                    help="Root output directory.")
    p.add_argument("--data_dir", type=str, default="./data",
                    help="Root for DecathlonDataset download / cache.")
    p.add_argument("--datalist", type=str, default=None,
                    help="Path to a JSON datalist (overrides --data_dir / DecathlonDataset). "
                         "JSON must have 'training' and 'validation' keys, each a list of "
                         "dicts with 'image' and 'label' paths.")
    p.add_argument("--no_channel_reorder", action="store_true", default=False,
                    help="Skip MSD→pipeline channel reorder (use for non-MSD data).")
    p.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of training set reserved for validation.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=250)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False,
                    help="Resume from checkpoint.pth in run_dir.")
    p.add_argument("--dist_backend", type=str, default="nccl",
                    choices=["nccl", "gloo"],
                    help="Distributed backend (nccl for GPU, gloo for CPU/fallback).")
    p.add_argument("--find_unused_parameters", action="store_true", default=False,
                    help="Pass find_unused_parameters=True to DDP.")
    p.add_argument("--scale_factor", type=float, default=1.0,
                    help="Latent scale factor (default 1.0).")
    p.add_argument("--train_spec", type=str, default="impression",
                    choices=["impression", "findings"],
                    help="Text field used for conditioning.")
    p.add_argument("--mask_dropout_p", type=float, default=None,
                    help="Override mask dropout probability (default: from config).")
    p.add_argument("--text_dropout_p", type=float, default=None,
                    help="Override text dropout probability (default: from config).")
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Cache directory for HuggingFace models / tokenizers.")
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
# Transforms  (image + label for mask conditioning)
# ---------------------------------------------------------------------------

def get_train_transform(channel_reorder: bool = True) -> T.Compose:
    """Training transforms for 4-ch BraTS images + segmentation labels."""
    xforms = [
        T.LoadImaged(keys=["image", "label"]),
        # Image: (H,W,D,4) → (4,H,W,D)
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        # Label: (H,W,D,1) → (1,H,W,D) or already channel-first
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=["label"], dtype=torch.float32),
        # Spatial
        T.Orientationd(keys=["image", "label"], axcodes="LPS"),
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        T.Resized(
            keys=["image"], spatial_size=(160, 224, 160), mode="trilinear",
        ),
        T.Resized(
            keys=["label"], spatial_size=(160, 224, 160), mode="nearest",
        ),
        # Intensity (image only)
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        # Augmentation
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        T.RandAffined(
            keys=["image", "label"], prob=0.1,
            translate_range=(1, 1, 1), scale_range=(-0.02, 0.02),
            spatial_size=[160, 224, 160],
            mode=["trilinear", "nearest"],
        ),
        T.RandShiftIntensityd(
            keys=["image"], offsets=0.05, prob=0.1, channel_wise=True,
        ),
        T.RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.97, 1.03)),
        T.ToTensord(keys=["image", "label"]),
    ])
    return T.Compose([x for x in xforms if x is not None])


def get_val_transform(channel_reorder: bool = True) -> T.Compose:
    """Validation transforms (deterministic)."""
    xforms = [
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=["label"], dtype=torch.float32),
        T.Orientationd(keys=["image", "label"], axcodes="LPS"),
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        T.Resized(
            keys=["image"], spatial_size=(160, 224, 160), mode="trilinear",
        ),
        T.Resized(
            keys=["label"], spatial_size=(160, 224, 160), mode="nearest",
        ),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.ToTensord(keys=["image", "label"]),
    ])
    return T.Compose([x for x in xforms if x is not None])


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
    # Config
    # ------------------------------------------------------------------
    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    channel_reorder = not args.no_channel_reorder

    if args.datalist:
        # ── Custom JSON datalist ─────────────────────────────────────
        print0(f"Loading datalist from {args.datalist}", rank)
        with open(args.datalist) as f:
            datalist = json.load(f)
        train_data = datalist["training"]
        val_data = datalist["validation"]
        print0(f"  {len(train_data)} training, {len(val_data)} validation entries", rank)

        train_ds = Dataset(data=train_data, transform=get_train_transform(channel_reorder))
        val_ds = Dataset(data=val_data, transform=get_val_transform(channel_reorder))
    else:
        # ── DecathlonDataset (BraTS) ─────────────────────────────────
        if is_main(rank):
            Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        task_dir = Path(args.data_dir) / "Task01_BrainTumour"
        need_download = not task_dir.is_dir()
        download = need_download and is_main(rank)
        if distributed:
            dist.barrier()

        train_ds = DecathlonDataset(
            root_dir=args.data_dir,
            task="Task01_BrainTumour",
            section="training",
            download=download,
            seed=args.seed,
            val_frac=args.val_frac,
            transform=get_train_transform(channel_reorder),
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
            transform=get_val_transform(channel_reorder),
            num_workers=args.num_workers,
        )

    # Samplers
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed else None
    )

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
    # Device
    # ------------------------------------------------------------------
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Stage-1 VAE (frozen)
    # ------------------------------------------------------------------
    print0("Loading frozen Stage-1 VAE …", rank)
    stage1 = stage1_ify(
        get_model(
            model_type="AutoencoderKL",
            config=stage1_config,
            from_file=args.stage1_uri,
        )
    )
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False
    stage1 = stage1.to(device)

    # ------------------------------------------------------------------
    # LDM (diffusion model)
    # ------------------------------------------------------------------
    print0("Initialising LDM …", rank)
    model_type = config["model"].get("name", "DiffusionModelUNet")
    ldm = get_model(model_type, config)
    ldm = ldm.to(device)

    if distributed:
        ldm = DDP(
            ldm, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    # ------------------------------------------------------------------
    # Noise scheduler
    # ------------------------------------------------------------------
    scheduler_name = config["scheduler"].get("name", "DDIMScheduler")
    scheduler_params = config["scheduler"].get("params", {})
    if scheduler_name == "DDPMScheduler":
        scheduler = DDPMScheduler(**scheduler_params)
    elif scheduler_name == "DDIMScheduler":
        scheduler = DDIMScheduler(**scheduler_params)
    else:
        raise ValueError(f"Unsupported noise scheduler: {scheduler_name}")

    # ------------------------------------------------------------------
    # Text encoder + tokenizer (frozen)
    # ------------------------------------------------------------------
    print0("Loading text encoder …", rank)
    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"],
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    # Override model_max_length if config specifies it (avoids wasteful
    # 512-token padding for BERT-family encoders)
    cfg_max_len = config["conditioning"].get("max_length")
    if cfg_max_len is not None:
        tokenizer.model_max_length = cfg_max_len
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Optimiser & scaler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(ldm.parameters(), lr=config["model"].get("base_lr", 1e-4))
    scaler = torch.amp.GradScaler()

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    run_dir = Path(args.run_dir) / "text2glioma" / "ldm_stage2"
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
    ckpt_path = run_dir / "checkpoint.pth"

    if args.resume and ckpt_path.exists():
        print0(f"Resuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ldm_state = ckpt.get("diffusion", ckpt.get("ldm_state_dict"))
        if ldm_state is None:
            raise KeyError("Checkpoint missing 'diffusion' or 'ldm_state_dict' key.")
        ldm.load_state_dict(ldm_state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print0(f"Resumed at epoch {start_epoch}", rank)
    else:
        print0("Starting fresh training.", rank)

    # ------------------------------------------------------------------
    # Dropout overrides
    # ------------------------------------------------------------------
    text_dropout = (
        args.text_dropout_p
        if args.text_dropout_p is not None
        else config["conditioning"].get("dropout_p", 0.2)
    )
    mask_dropout = (
        args.mask_dropout_p
        if args.mask_dropout_p is not None
        else config.get("mask", {}).get("dropout_p", 0.2)
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print0("Starting training …", rank)
    val_loss = train_ldm(
        model=ldm,
        stage1=stage1,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        n_epochs=args.num_epochs,
        start_epoch=start_epoch,
        text_field=args.train_spec,
        val_interval=args.val_interval,
        dropout_p=text_dropout,
        model_dir=model_dir,
        writer_train=writer_train,
        writer_val=writer_val,
        run_dir=run_dir,
        scale_factor=args.scale_factor,
        num_mask_classes=config.get("mask", {}).get("num_classes", 4),
        mask_dropout_p=mask_dropout,
        latent_channels=config.get("model", {}).get("latent_channels", 3),
    )

    print0(f"Training finished.  Final val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
