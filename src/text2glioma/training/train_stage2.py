""" Training script for LDM (stage 2) with frozen autoencoder. """
import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from text2glioma.training.training_functions import train_ldm
from text2glioma.utils import (
    get_dataloaders,
    get_model,
    load_config,
    load_text_encoder_and_tokenizer,
    stage1_ify,
)  # workaround for DataParallel
from torch.utils.tensorboard import SummaryWriter
import json

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str, help="Path to the data JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--stage1_config", type=str, required=True, help="Path to the stage 1 autoencoder config file.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--stage1_uri", type=str, required=True, help="URI for the pretrained stage 1 autoencoder.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading.")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the training data.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (in epochs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--n_epochs", type=int, default=250, help="Maximum number of training epochs.")
    parser.add_argument("--use_parallel", action="store_true", help="Use DataParallel for multi-GPU training.")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel training.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend to use.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models and tokenizers.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor for input images.")
    parser.add_argument("--train_spec", type=str, default="impression", metavar=["impression", "findings"], help="Which version of training to run.")
    parser.add_argument("--mask_dropout_p", type=float, default=None, help="Override mask dropout probability (default: from config).")
    parser.add_argument("--text_dropout_p", type=float, default=None, help="Override text dropout probability (default: from config).")

    return parser.parse_args()

def init_distributed(args):
    if not args.distributed:
        if dist.is_available() and dist.is_initialized():
            args.rank = dist.get_rank()
            args.world_size = dist.get_world_size()
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
        return False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))     

    torch.cuda.set_device(args.local_rank)
    device_id = torch.device("cuda", args.local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        device_id=device_id,
    )
    dist.barrier(device_ids=[args.local_rank])
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    return True

def main():
    args = parse_args()
    set_determinism(args.seed)
    print_config()
    distributed = args.distributed

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        node = os.uname()[1]
        world_size = torch.cuda.device_count()

        # create model and move it to GPU with id rank
        device_id = rank % world_size
        is_main_process = rank == 0

    if args.train_spec not in ["impression", "findings"]:
        raise ValueError(f"Unrecognized training option: {args.train_spec}"
                          "Expected: impression, findings")
    else:
        print(f"Training {args.train_spec}...")

    datalist_json = args.data
    with open(datalist_json, "r") as f:
        datalist = json.load(f)
    train_dataset = datalist["training"]
    val_dataset = datalist["validation"]

    run_dir = Path(args.run_dir) / "text2glioma" / "ldm_stage2"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "models"
    if is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    if is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)

    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        print(f"Resuming from checkpoint in {run_dir}")
        checkpoint = torch.load(run_dir / "checkpoint.pth", map_location="cpu")
        resume = True
    else:
        print(f"No checkpoint found in {run_dir}. Starting fresh training.")
        checkpoint = None
        resume = False

    print("Initializing models...")
    stage1 = stage1_ify(
        get_model(
            model_type="AutoencoderKL", config=stage1_config, from_file=args.stage1_uri
            )
        )
    stage1.eval()

    for param in stage1.parameters():
        param.requires_grad = False

    model_type = config["model"].get("name", "DiffusionModelUNet")
    ldm = get_model(model_type, config)

    print("Preparing data loaders...")
    train_loader, val_loader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=not args.no_shuffle,
        model_type=model_type,
        initialize=False,
        distributed=distributed,
        rank=rank if distributed else 0,
        world_size=world_size if distributed else 1,
    )
    
    noise_scheduler_type = config["scheduler"].get("name", "DDPMScheduler")
    if noise_scheduler_type == "DDPMScheduler":
        scheduler = DDPMScheduler(**config["scheduler"].get("params", {}))
    elif noise_scheduler_type == "DDIMScheduler":
        scheduler = DDIMScheduler(**config["scheduler"].get("params", {}))
    else:
        raise ValueError(f"Unsupported noise scheduler type: {noise_scheduler_type}")

    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"],
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    if distributed:
        if args.use_parallel and is_main_process:
            print("DistributedDataParallel enabled; ignoring DataParallel flag.")
    elif args.use_parallel and torch.cuda.device_count() > 1:
        ldm = torch.nn.DataParallel(ldm)
        #tokenizer = torch.nn.DataParallel(tokenizer) if tokenizer else None
        text_encoder = torch.nn.DataParallel(text_encoder) if text_encoder else None

    if resume and checkpoint is not None:
        ldm_state_dict = checkpoint.get("ldm_state_dict", checkpoint.get("diffusion"))
        if ldm_state_dict is None:
            raise KeyError("Checkpoint missing 'ldm_state_dict' or 'diffusion' keys.")
        ldm.load_state_dict(ldm_state_dict)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0

    if is_main_process:
        print(f"Starting training from epoch {start_epoch}")
        print(f"Run directory: {str(run_dir)}")
        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print(f"Config: {str(config)}")

    writer_train = SummaryWriter(log_dir / "train") if is_main_process else None
    writer_val = SummaryWriter(log_dir / "val") if is_main_process else None

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device}" if distributed else args.device)
        if distributed:
            torch.cuda.set_device(device)
    else:
        if distributed and args.dist_backend == "nccl":
            raise RuntimeError("Distributed training with NCCL requires CUDA, but no GPU was detected.")
        device = torch.device("cpu")

    if distributed:
        ldm = DDP(ldm, device_ids=[device_id], find_unused_parameters=False)
        text_encoder = DDP(text_encoder, device_ids=[device_id])
        stage1 = DDP(stage1, device_ids=[device_id])

    text_encoder = text_encoder.to(device)
    ldm = ldm.to(device)
    stage1 = stage1.to(device)

    if distributed:
        if device.type == "cuda":
            ldm = torch.nn.parallel.DistributedDataParallel(
                ldm,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )
        else:
            ldm = torch.nn.parallel.DistributedDataParallel(ldm, find_unused_parameters=False)
    optimizer = optim.AdamW(ldm.parameters(), lr=config["model"].get("base_lr", 1e-4))
    scaler = torch.cuda.amp.GradScaler()
    
    if is_main_process:
        print("Starting training...")
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
        model_dir=model_dir,
        writer_train=writer_train,
        writer_val=writer_val,
        n_epochs=args.n_epochs,
        start_epoch=start_epoch,
        text_field=args.train_spec,
        val_interval=args.val_interval,
        dropout_p=args.text_dropout_p if args.text_dropout_p is not None else config["conditioning"].get("dropout_p", 0.2),
        run_dir=run_dir,
        scale_factor=args.scale_factor,
        num_mask_classes=config.get("mask", {}).get("num_classes", 4),
        mask_dropout_p=args.mask_dropout_p if args.mask_dropout_p is not None else config.get("mask", {}).get("dropout_p", 0.2),
        latent_channels=config.get("model", {}).get("latent_channels", 3),
    )

    print(f"Training completed, final validation loss: {val_loss:0.5f}")
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
