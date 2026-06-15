"""DDP launcher for the inpainting LDM.

Trains on BraTS-GLI 2025 longitudinal pairs (see
``datalist_brats_gli_2025_pairs_split.json``). Categorical-only conditioning;
no text encoder.

Example::

    torchrun --standalone --nproc_per_node=4 \\
        -m text2glioma.inpainting.train_inpainting_ddp \\
        --config configs/inpainting.yaml \\
        --stage1_config configs/stage1.yaml \\
        --stage1_uri /path/to/stage1/final_model.pth \\
        --run_dir /runs/ \\
        --datalist datalist_brats_gli_2025_pairs_split.json

Channel reconciliation
----------------------
At startup we probe Stage-1 with one batch to detect the real latent channel
count (kl=1e-6 VAE → 6). We then rewrite the UNet config to:
    in_channels  = 2 * latent_ch + 1    # noisy_z_b ⊕ z_masked_a ⊕ z_mask
    out_channels = latent_ch            # v / ε in latent space

Stage-1 checkpoint compatibility
--------------------------------
``--stage1_uri`` accepts both ``final_model.pth`` (raw state_dict) and
``checkpoint.pth`` (wrapped {"state_dict", "discriminator", ...}). The DDP
``module.`` prefix is stripped by ``_normalise_autoencoder_state_dict``.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Optional

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
from text2glioma.utils import (
    compute_scale_factor,
    get_model,
    load_config,
    stage1_ify,
)

from .conditioning import CategoricalConditioningEncoder
from .training_functions import InpaintingModel, train_inpainting

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the BraTS-GLI longitudinal inpainting LDM with DDP",
    )
    p.add_argument("--config", type=str, required=True,
                   help="Inpainting LDM YAML config (e.g. configs/inpainting.yaml).")
    p.add_argument("--stage1_config", type=str, required=True,
                   help="Stage-1 VAE YAML config (e.g. configs/stage1.yaml).")
    p.add_argument("--stage1_uri", type=str, required=True,
                   help="Path to pretrained Stage-1 VAE checkpoint (.pth). "
                        "Accepts final_model.pth or checkpoint.pth.")
    p.add_argument("--datalist", type=str, required=True,
                   help="Path to a JSON datalist with 'training' and 'validation' "
                        "keys, each a list of pair-dicts (see scripts/split_brats_gli_pairs.py).")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Output directory for checkpoints, logs, and best_model.pth.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from checkpoint.pth in run_dir.")
    p.add_argument("--dist_backend", type=str, default="nccl",
                   choices=["nccl", "gloo"])
    p.add_argument("--scale_factor", type=float, default=None,
                   help="Latent scale factor. If omitted, auto-computed as "
                        "1/std(latents) over training data.")
    p.add_argument("--mask_weight", type=float, default=None,
                   help="Extra in-mask loss weight (override config.mask_weight).")
    p.add_argument("--p_traj", type=float, default=None,
                   help="CFG dropout on trajectory (override config.cfg.p_traj).")
    p.add_argument("--p_treat", type=float, default=None,
                   help="CFG dropout on treatment_a + treatment_b "
                        "(override config.cfg.p_treat).")
    p.add_argument("--balance_mode", type=str, default="direction",
                   choices=["uniform", "trajectory", "direction", "joint"],
                   help="Weighted-sampler stratum key. 'direction' boosts "
                        "the rare pre->post bucket from 3.8%% to ~33%%. See "
                        "compute_balanced_weights().")
    p.add_argument("--dilation_mm", type=float, default=18.0,
                   help="Inpainting mask = dilation_mm-dilated (M_A ∪ M_B) ∩ brain.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP setup helpers (mirror train_stage2_ddp; 30-min NCCL timeout for Gadi)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    distributed = world_size > 1
    print0(f"World size: {world_size}  |  Backend: {args.dist_backend}", rank)
    if is_main(rank):
        print_config()

    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
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

    # ── Samplers ─────────────────────────────────────────────────────
    # Weighted sampling for stratum balance, then DistributedSampler on top
    # is incompatible: WeightedRandomSampler doesn't shard. So under DDP we
    # use DistributedSampler with shuffle=True and rely on the implicit
    # natural-frequency mix unless balance_mode='uniform'. For single-process
    # runs (debug / single-GPU) we honour the weighted sampler.
    if distributed:
        if args.balance_mode != "uniform":
            print0(
                f"[WARN] balance_mode={args.balance_mode!r} is ignored under DDP; "
                "DistributedSampler with shuffle=True is used instead. To balance "
                "under DDP, oversample the rare strata in the datalist itself.",
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

    # ------------------------------------------------------------------
    # Stage-1 VAE (frozen)
    # ------------------------------------------------------------------
    print0("Loading frozen Stage-1 VAE …", rank)
    stage1 = stage1_ify(
        get_model(model_type="AutoencoderKL", config=stage1_config, from_file=args.stage1_uri)
    )
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    stage1 = stage1.to(device)

    # ── Probe for latent channels ────────────────────────────────────
    stage1_latent_ch: Optional[int] = None
    try:
        probe_batch = next(iter(train_loader))
        probe_img = probe_batch["image_a"][:1].to(device)
        with torch.no_grad():
            probe_z = stage1(probe_img)
        stage1_latent_ch = int(probe_z.shape[1])
    except Exception as exc:
        print0(f"[WARN] Stage-1 probe failed: {exc!r}", rank)
    if stage1_latent_ch is None and hasattr(stage1, "model") and hasattr(stage1.model, "latent_channels"):
        stage1_latent_ch = int(stage1.model.latent_channels)
    if stage1_latent_ch is None:
        stage1_latent_ch = int(config["model"].get("latent_channels", 4))
        print0(f"[WARN] Defaulting to config latent_channels={stage1_latent_ch}", rank)

    expected_in_channels = 2 * stage1_latent_ch + 1
    cfg_params = config.setdefault("model", {}).setdefault("params", {})
    cfg_in = int(cfg_params.get("in_channels", expected_in_channels))
    cfg_out = int(cfg_params.get("out_channels", stage1_latent_ch))
    if cfg_in != expected_in_channels or cfg_out != stage1_latent_ch:
        print0(
            f"[INFO] Auto-aligning UNet channels: in {cfg_in}->{expected_in_channels}, "
            f"out {cfg_out}->{stage1_latent_ch} (latent_ch={stage1_latent_ch})",
            rank,
        )
    config["model"]["latent_channels"] = stage1_latent_ch
    cfg_params["in_channels"] = expected_in_channels
    cfg_params["out_channels"] = stage1_latent_ch

    # ------------------------------------------------------------------
    # UNet + conditioning encoder
    # ------------------------------------------------------------------
    print0("Initialising UNet + cond encoder …", rank)
    unet = get_model(config["model"].get("name", "DiffusionModelUNet"), config)

    embed_dim = int(cfg_params.get("cross_attention_dim", 256))
    cond_encoder = CategoricalConditioningEncoder(embed_dim=embed_dim)

    # Zero-init the conv_in weights for the masked-image-latent + mask channels.
    # The model starts as a standard latent diffusion and *learns* to use the
    # extra conditioners gradually. (Same trick as train_stage2_ddp.)
    with torch.no_grad():
        first_conv = unet.conv_in.conv
        first_conv.weight[:, stage1_latent_ch:].zero_()
        print0(
            f"  Zero-initialised conv_in cond channels [{stage1_latent_ch}:{first_conv.weight.shape[1]}]",
            rank,
        )

    model = InpaintingModel(unet=unet, cond_encoder=cond_encoder).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    sch_name = config["scheduler"].get("name", "DDIMScheduler")
    sch_params = config["scheduler"].get("params", {})
    if sch_name == "DDPMScheduler":
        scheduler = DDPMScheduler(**sch_params)
    elif sch_name == "DDIMScheduler":
        scheduler = DDIMScheduler(**sch_params)
    else:
        raise ValueError(f"Unsupported scheduler {sch_name!r}")

    # ------------------------------------------------------------------
    # Optimizer + scale factor
    # ------------------------------------------------------------------
    base_lr = float(config["model"].get("base_lr", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    if args.scale_factor is not None:
        scale_factor = args.scale_factor
        print0(f"Using user-specified scale_factor = {scale_factor:.4f}", rank)
    else:
        print0("Auto-computing latent scale_factor from training data …", rank)
        # compute_scale_factor expects loader batches with key 'image'; our
        # batches use 'image_a' / 'image_b'. Build a tiny ad-hoc loop instead.
        with torch.no_grad():
            zs = []
            for i, batch in enumerate(train_loader):
                if i >= 50:
                    break
                imgs = batch["image_b"].to(device)  # latent stats of the target
                zs.append(stage1(imgs).flatten())
            std = torch.cat(zs).std().item()
            scale_factor = 1.0 / std
        print0(f"  latent std = {std:.4f}  ->  scale_factor = {scale_factor:.4f}", rank)

    # ------------------------------------------------------------------
    # Output dirs (rank-0 mkdir; barrier)
    # ------------------------------------------------------------------
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_dir = run_dir / "logs"
    if is_main(rank):
        for d in [run_dir, log_dir]:
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
    ema_state_dict = None
    if args.resume and ckpt_path.exists():
        print0(f"Resuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        ema_state_dict = ckpt.get("ema")
        if ckpt.get("scale_factor") is not None and args.scale_factor is None:
            scale_factor = float(ckpt["scale_factor"])
            print0(f"  Recovered scale_factor from checkpoint = {scale_factor:.4f}", rank)

    # ------------------------------------------------------------------
    # Dropout overrides
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print0("Starting training …", rank)
    val_loss = train_inpainting(
        model=model, stage1=stage1, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        n_epochs=args.num_epochs, start_epoch=start_epoch,
        val_interval=args.val_interval,
        p_traj=p_traj, p_treat=p_treat, mask_weight=mask_weight,
        run_dir=run_dir,
        writer_train=writer_train, writer_val=writer_val,
        scale_factor=scale_factor,
        ema_state_dict=ema_state_dict,
    )

    print0(f"Training finished. Final val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
