"""DDP launcher for the **pixel-space** inpainting DDM.

Sibling of ``train_inpainting_ddp.py``. The only structural differences are:

  - No ``--stage1_config`` / ``--stage1_uri`` / scale_factor logic.
  - UNet ``in_channels`` / ``out_channels`` are fixed by the dataset's
    image channel count (4 BraTS modalities), no probe needed.

Example::

    torchrun --standalone --nproc_per_node=4 \\
        -m text2glioma.inpainting.train_pixel_inpainting_ddp \\
        --config configs/inpainting_pixel.yaml \\
        --run_dir /runs/inpainting_pixel/ \\
        --datalist datalist_brats_gli_2025_pairs_split.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.preprocessing.inpainting_dataset import (
    build_pair_transforms,
    compute_balanced_weights,
    prepare_pair_records,
    stratum_summary,
)
from text2glioma.utils import get_model, load_config

from .conditioning import CategoricalConditioningEncoder
from .training_functions import InpaintingModel
from .training_functions_pixel import train_pixel_inpainting

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the pixel-space inpainting DDM with DDP",
    )
    p.add_argument("--config", type=str, required=True,
                   help="Pixel-DDM YAML config (e.g. configs/inpainting_pixel.yaml).")
    p.add_argument("--datalist", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-rank batch size. Defaults to 1 because pixel-space "
                        "3D activations are large; raise only if you've measured spare memory.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dist_backend", type=str, default="nccl",
                   choices=["nccl", "gloo"])
    p.add_argument("--mask_weight", type=float, default=None)
    p.add_argument("--p_traj", type=float, default=None)
    p.add_argument("--p_treat", type=float, default=None)
    p.add_argument("--balance_mode", type=str, default="uniform",
                   choices=["uniform", "trajectory", "direction", "joint"],
                   help="Stratum-balancing mode (single-process only; under DDP "
                        "this is ignored and DistributedSampler is used — bake "
                        "any oversampling into the datalist instead).")
    p.add_argument("--dilation_mm", type=float, default=18.0)
    return p.parse_args()


def setup_distributed(backend: str) -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend, init_method="env://",
        timeout=timedelta(minutes=30),
    )
    dist.barrier()
    return rank, world_size, local_rank


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def print0(msg: str, rank: int) -> None:
    if is_main(rank):
        print(msg)


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    distributed = world_size > 1
    print0(f"World size: {world_size}  |  Backend: {args.dist_backend}", rank)
    if is_main(rank):
        print_config()

    config = load_config(args.config)

    # ── Dataset ──────────────────────────────────────────────────────
    print0(f"Loading datalist from {args.datalist}", rank)
    with open(args.datalist) as f:
        datalist = json.load(f)
    train_records = prepare_pair_records(datalist["training"])
    val_records = prepare_pair_records(datalist["validation"])
    print0(f"  train: {len(train_records)} pairs  |  val: {len(val_records)} pairs", rank)
    if is_main(rank):
        print("  training stratum distribution:")
        for s, n in sorted(stratum_summary(train_records).items(), key=lambda x: -x[1]):
            print(f"    {s:<35s} {n:>4d}")

    spatial_size = tuple(config.get("data", {}).get("spatial_size", (160, 224, 160)))
    dilation_mm = float(args.dilation_mm)

    train_xforms = build_pair_transforms(training=True,  dilation_mm=dilation_mm, spatial_size=spatial_size)
    val_xforms   = build_pair_transforms(training=False, dilation_mm=dilation_mm, spatial_size=spatial_size)

    train_ds = Dataset(data=train_records, transform=train_xforms)
    val_ds = Dataset(data=val_records, transform=val_xforms)

    if distributed:
        if args.balance_mode != "uniform":
            print0(
                f"[WARN] balance_mode={args.balance_mode!r} is ignored under DDP; "
                "DistributedSampler with shuffle=True is used instead.",
                rank,
            )
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_shuffle = False
    else:
        if args.balance_mode == "uniform":
            train_sampler = None
            train_shuffle = True
        else:
            weights = compute_balanced_weights(train_records, mode=args.balance_mode)
            train_sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(train_records), replacement=True,
            )
            train_shuffle = False
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=train_shuffle, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── UNet + cond encoder ──────────────────────────────────────────
    cfg_params = config.setdefault("model", {}).setdefault("params", {})
    image_ch = int(config["model"].get("image_channels", 4))
    expected_in = 2 * image_ch + 1
    cfg_in = int(cfg_params.get("in_channels", expected_in))
    cfg_out = int(cfg_params.get("out_channels", image_ch))
    if cfg_in != expected_in or cfg_out != image_ch:
        print0(
            f"[INFO] Auto-aligning UNet channels: in {cfg_in}->{expected_in}, "
            f"out {cfg_out}->{image_ch} (image_ch={image_ch})",
            rank,
        )
    cfg_params["in_channels"] = expected_in
    cfg_params["out_channels"] = image_ch

    print0("Initialising UNet + cond encoder …", rank)
    unet = get_model(config["model"].get("name", "DiffusionModelUNet"), config)

    embed_dim = int(cfg_params.get("cross_attention_dim", 256))
    cond_encoder = CategoricalConditioningEncoder(embed_dim=embed_dim)

    # Zero-init the conv_in weights for the masked-image + mask channels:
    # model starts as a vanilla DDM in noisy_b and learns to use the cond
    # channels gradually. Same trick as the LDM.
    with torch.no_grad():
        first_conv = unet.conv_in.conv
        first_conv.weight[:, image_ch:].zero_()
        print0(
            f"  Zero-initialised conv_in cond channels [{image_ch}:{first_conv.weight.shape[1]}]",
            rank,
        )

    model = InpaintingModel(unet=unet, cond_encoder=cond_encoder).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── Scheduler ────────────────────────────────────────────────────
    sch_name = config["scheduler"].get("name", "DDIMScheduler")
    sch_params = config["scheduler"].get("params", {})
    if sch_name == "DDPMScheduler":
        scheduler = DDPMScheduler(**sch_params)
    elif sch_name == "DDIMScheduler":
        scheduler = DDIMScheduler(**sch_params)
    else:
        raise ValueError(f"Unsupported scheduler {sch_name!r}")

    # ── Optimizer ────────────────────────────────────────────────────
    base_lr = float(config["model"].get("base_lr", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    # ── Output dirs ──────────────────────────────────────────────────
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_dir = run_dir / "logs"
    if is_main(rank):
        for d in [run_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    writer_train = SummaryWriter(log_dir / "train") if is_main(rank) else None
    writer_val = SummaryWriter(log_dir / "val") if is_main(rank) else None

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    ckpt_path = run_dir / "checkpoint.pth"
    ema_state_dict = None
    if args.resume and ckpt_path.exists():
        print0(f"Resuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        ema_state_dict = ckpt.get("ema")

    # ── Dropout / loss overrides ─────────────────────────────────────
    cfg_cfg = config.get("cfg", {})
    p_traj = args.p_traj if args.p_traj is not None else float(cfg_cfg.get("p_traj", 0.2))
    p_treat = args.p_treat if args.p_treat is not None else float(cfg_cfg.get("p_treat", 0.2))
    mask_weight = (
        args.mask_weight
        if args.mask_weight is not None
        else float(config.get("loss", {}).get("mask_weight", 4.0))
    )
    print0(
        f"CFG dropout: p_traj={p_traj:.2f}  p_treat={p_treat:.2f}  |  mask_weight={mask_weight:.2f}",
        rank,
    )

    # ── Train ────────────────────────────────────────────────────────
    print0("Starting training …", rank)
    val_loss = train_pixel_inpainting(
        model=model, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        n_epochs=args.num_epochs, start_epoch=start_epoch,
        val_interval=args.val_interval,
        p_traj=p_traj, p_treat=p_treat, mask_weight=mask_weight,
        run_dir=run_dir,
        writer_train=writer_train, writer_val=writer_val,
        ema_state_dict=ema_state_dict,
    )

    print0(f"Training finished. Final val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
