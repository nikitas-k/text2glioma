""" Training script for LDM (stage 2) with frozen autoencoder. """
import argparse
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from training_functions import train_ldm
from text2glioma.utils import load_config, get_dataloaders, get_model, stage1_ify #workaround for DataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPTextModel
import json

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str, help="Path to the data JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--stage1_uri", type=str, required=True, help="URI for the pretrained stage 1 autoencoder.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading.")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the training data.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (in epochs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--use_parallel", action="store_true", help="Use DataParallel for multi-GPU training.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models and tokenizers.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor for input images.")

    return parser.parse_args()

def main():
    args = parse_args()
    set_determinism(args.seed)
    print_config()

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

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
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
    stage1 = get_model("AutoencoderKL", args.stage1_uri, device=args.device)
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False

    model_type = config["model"].get("name", "DiffusionModelUNet")
    ldm = get_model(model_type, config, device=args.device)

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
        initialize=False
    )
    
    noise_scheduler_type = config["scheduler"].get("name", "DDPMScheduler")
    if noise_scheduler_type == "DDPMScheduler":
        scheduler = DDPMScheduler(**config["model"]["scheduler"].get("params", {}))
    elif noise_scheduler_type == "DDIMScheduler":
        scheduler = DDIMScheduler(**config["model"]["scheduler"].get("params", {}))
    else:
        raise ValueError(f"Unsupported noise scheduler type: {noise_scheduler_type}")

    tokenizer = config["conditioning"].get("tokenizer", None)
    text_encoder = config["conditioning"].get("text_encoder", None)
    if tokenizer and txt_encoder:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, subfolder="tokenizer", cache_dir=args.cache_dir)
        text_encoder = CLIPTextModel.from_pretrained(text_encoder, subfolder="text_encoder", cache_dir=args.cache_dir)

    if args.use_parallel and torch.cuda.device_count() > 1:
        ldm = torch.nn.DataParallel(ldm)
        tokenizer = torch.nn.DataParallel(tokenizer) if tokenizer else None
        txt_encoder = torch.nn.DataParallel(txt_encoder) if txt_encoder else None
        stage1 = torch.nn.DataParallel(stage1_ify(stage1))

    if resume and checkpoint is not None:
        ldm.load_state_dict(checkpoint["ldm_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0

    print(f"Starting training from epoch {start_epoch}")
    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"Config: {str(config)}")

    writer_train = SummaryWriter(log_dir / "train")
    writer_val = SummaryWriter(log_dir / "val")

    optimizer = optim.AdamW(ldm.parameters(), lr=config.get("learning_rate", 1e-4))
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ldm = ldm.to(device)
    stage1 = stage1.to(device)
    
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
        val_interval=args.val_interval,
        run_dir=run_dir,
        scale_factor=args.scale_factor,
    )

    print(f"Training completed, final validation loss: {val_loss:0.5f}")

if __name__ == "__main__":
    main()