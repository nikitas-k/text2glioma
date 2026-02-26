""" Training script for autoencoder (stage 1) with KL regularization. """
import argparse
import copy
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from text2glioma.training.training_functions import train_autoencoder
from text2glioma.utils import load_config, get_dataloaders, get_model
from torch.utils.tensorboard import SummaryWriter
import gdown
import json

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str, help="Path to the data JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and logs.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (used as sub-directory under run_dir). "
                             "Defaults to a timestamp, e.g. 2026-02-25_143022.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for data loading.")
    parser.add_argument("--no_shuffle", action="store_true", default=False, help="Disable shuffling of the training data.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (in epochs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Maximum number of training epochs.")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained weights from Pinaya et al. for the autoencoder.")
    parser.add_argument("--use_parallel", action="store_true", default=False, help="Use DataParallel for multi-GPU training.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Enable DistributedDataParallel training.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--initialize", action="store_true", default=False, help="Initialize (reset) the data for PersistentDataset.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for perceptual-loss network cache. "
                             "Defaults to <output_dir>/cache if not specified.")
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                        help="Override config values using dot notation, e.g. "
                             "--set model.lr=1e-4 discriminator.lr=2.5e-5")

    return parser.parse_args()


def _save_config_snapshot(
    config: dict,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """Dump the merged YAML config + CLI overrides into *run_dir* for reproducibility."""
    import yaml

    snapshot = copy.deepcopy(config)
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
    print(f"Config snapshot saved to {out_path}")


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``key=value`` overrides using dot notation."""
    import ast

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

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


def init_distributed(args):
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Distributed mode requested but environment variables RANK/WORLD_SIZE not set. Falling back to single process.")
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return False

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank)
    dist.barrier()
    return True

def main():
    args = parse_args()

    set_determinism(args.seed)
    print_config()
    distributed = init_distributed(args)
    is_main_process = args.rank == 0

    datalist_json = args.data
    with open(datalist_json, "r") as f:
        datalist = json.load(f)
    train_dataset = datalist["training"]
    val_dataset = datalist["validation"]

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.run_dir) / run_name / "autoencoder_stage1"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.set:
        config = _apply_overrides(config, args.set)

    # Save config snapshot for reproducibility
    if is_main_process:
        _save_config_snapshot(config, args, run_dir)

    model_dir = output_dir / "models"
    if is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    if is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)

    resume = args.resume

    if resume is True:
        if (run_dir.exists() and (run_dir / "checkpoint.pth").exists()):
            print(f"Resuming from checkpoint in {run_dir}")
            checkpoint = torch.load(run_dir / "checkpoint.pth", map_location="cpu")
        else:
            print(f"No checkpoint found in {run_dir}."
                  "Running training for the first time...")
    else:
        print("Running training for the first time...")
        checkpoint = None

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"Config: {str(config)}")

    writer_train = SummaryWriter(log_dir / "train") if is_main_process else None
    writer_val = SummaryWriter(log_dir / "val") if is_main_process else None
    if is_main_process:
        print("Getting data...")
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)  # cache needs to exist on all ranks

    model_type = config["model"]["name"]
    train_loader, val_loader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=not args.no_shuffle,
        model_type=model_type,
        initialize=args.initialize,
        distributed=distributed,
        rank=args.rank,
        world_size=args.world_size if distributed else 1,
    )

    if is_main_process:
        print("Initializing model...")        
    model = get_model(model_type, config, args.pretrained)

    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"], cache_dir=cache_dir)        

    if distributed:
        if args.use_parallel and is_main_process:
            print("DistributedDataParallel enabled; ignoring DataParallel flag.")
    elif torch.cuda.device_count() > 1 and args.use_parallel:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}" if distributed else args.device)
    else:
        device = torch.device("cpu")
    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer_g = optim.AdamW(model.parameters(), lr=config["model"]["lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["lr"])

    scaler_g = torch.amp.GradScaler()
    scaler_d = torch.amp.GradScaler()

    # get checkpoint to resume
    best_loss = float("inf")
    start_epoch = 0
    if resume and checkpoint is not None:
        print("Using checkpoint to resume training...")
        checkpoint = torch.load(run_dir / "checkpoint.pth", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print("No checkpoint found. Starting fresh training.")

    print("Starting training...")
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
        autoencoder_warm_up_n_epochs=config["model"].get("autoencoder_warm_up_n_epochs", 0),
        d_skip_threshold=config["model"].get("d_skip_threshold", 0.0),
    )
    if is_main_process:
        print(f"Training completed. Best validation loss: {val_loss:.4f}")
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    # get things going...
    main()
