"""DDP-aware VAE training using MONAI DecathlonDataset or a custom JSON datalist.

With DecathlonDataset (BraTS)::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage1_ddp \
        --config configs/stage1.yaml --run_dir /runs/ --num_epochs 300

With a custom datalist and a named run::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage1_ddp \\
        --config configs/stage1.yaml --run_dir /runs/ \\
        --run_name lr_d_sweep_2.5e-5 \\
        --datalist datalist_task03.json --no_channel_reorder

If ``--run_name`` is omitted a timestamp is used (e.g. ``2026-02-25_143022``).
Each run saves a ``config_snapshot.yaml`` capturing the full config + CLI args.
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
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.training.training_functions import train_autoencoder
from text2glioma.utils import apply_spectral_norm, get_model, load_config

warnings.filterwarnings("ignore")

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-1 VAE with DDP on BraTS or a custom datalist",
    )
    p.add_argument("--config", type=str, required=True, help="Stage-1 YAML config.")
    p.add_argument("--run_dir", type=str, required=True, help="Root output directory.")
    p.add_argument("--run_name", type=str, default=None,
                    help="Run name (used as sub-directory under run_dir). "
                         "Defaults to a timestamp, e.g. 2026-02-25_143022.")
    p.add_argument("--data_dir", type=str, default="./data",
                    help="Root for DecathlonDataset download / cache.")
    p.add_argument("--datalist", type=str, default=None,
                    help="Path to a JSON datalist (overrides --data_dir / DecathlonDataset). "
                         "JSON must have 'training' and 'validation' keys, each a list of "
                         "dicts with 'image' (and optionally 'label') paths.")
    p.add_argument("--no_channel_reorder", action="store_true", default=False,
                    help="Skip MSD→pipeline channel reorder (use for non-MSD data "
                         "where channels are already in the desired order).")
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
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Directory for perceptual-loss network cache. "
                         "Defaults to <run_dir>/cache if not specified.")
    p.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                    help="Override config values using dot notation, e.g. "
                         "--set model.lr=1e-4 discriminator.lr=2.5e-5 "
                         "model.adv_weight=0.025")
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


def _save_config_snapshot(
    config: dict,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """Dump the merged YAML config + CLI overrides into *run_dir* for reproducibility.

    Creates ``run_dir/config_snapshot.yaml`` containing:
    - The full YAML config (with any CLI-driven overrides applied)
    - A ``_cli`` block recording every CLI argument
    - A ``_meta`` block with timestamp, config source path, etc.
    """
    import yaml

    snapshot = copy.deepcopy(config)

    # Apply CLI overrides that shadow YAML values
    cli_overrides = {}
    if args.batch_size != 2:
        cli_overrides["batch_size"] = args.batch_size
    if args.num_epochs != 300:
        cli_overrides["num_epochs"] = args.num_epochs
    if args.seed != 42:
        cli_overrides["seed"] = args.seed

    snapshot["_cli"] = {k: v for k, v in vars(args).items()
                        if v is not None and k != "config"}
    snapshot["_meta"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(Path(args.config).resolve()),
        "run_name": run_dir.parent.name,
    }

    out_path = run_dir / "config_snapshot.yaml"
    with open(out_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    print(f"[rank-0] Config snapshot saved to {out_path}")


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``key=value`` overrides using dot notation.

    Examples::

        --set model.lr=1e-4  discriminator.lr=2.5e-5  model.adv_weight=0.025
              model.params.num_channels='[64,128,256]'
    """
    import ast

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        # Auto-cast value
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Handle YAML-style booleans (true/false) that Python's
            # ast.literal_eval doesn't recognise.
            if value.lower() in ("true", "yes"):
                value = True
            elif value.lower() in ("false", "no"):
                value = False
            # else: keep as string

        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            if p not in d:
                raise KeyError(f"Config key {key!r}: sub-key {p!r} not found. "
                               f"Available: {list(d.keys())}")
            d = d[p]
        leaf = parts[-1]
        if leaf not in d:
            print(f"[override] {key}: NEW key (not in config) -> {value!r}")
        else:
            print(f"[override] {key}: {d[leaf]!r} -> {value!r}")
        d[leaf] = value
    return config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform(channel_reorder: bool = True) -> T.Compose:
    """Training transforms for 4-ch BraTS images."""
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
    return T.Compose([x for x in xforms if x is not None])


def get_val_transform(channel_reorder: bool = True) -> T.Compose:
    """Validation transforms (deterministic)."""
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
    if args.set:
        config = _apply_overrides(config, args.set)
    model_type = config["model"]["name"]
    model = get_model(model_type, config, args.pretrained)

    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    if config["discriminator"].get("spectral_norm", False):
        apply_spectral_norm(discriminator)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.run_dir) / "cache"
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
    # Optimisers
    # ------------------------------------------------------------------
    optimizer_g = optim.AdamW(model.parameters(), lr=config["model"]["lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["lr"])

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.run_dir) / run_name / "autoencoder_stage1"
    output_dir = run_dir / "output"
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    if is_main(rank):
        for d in [output_dir, model_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # Save config snapshot with CLI overrides for reproducibility
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
    )
    print0(f"Training finished.  Best val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
