""" Training script for autoencoder (stage 1) with KL regularization. """
import argparse
import warnings
from pathlib import Path

import torch
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for data loading.")
    parser.add_argument("--no_shuffle", action="store_true", default=True, help="Disable shuffling of the training data.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (in epochs).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained weights from Pinaya et al. for the autoencoder.")
    parser.add_argument("--use_parallel", action="store_true", default=False, help="Use DataParallel for multi-GPU training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--initialize", action="store_true", default=False, help="Initialize (reset) the data for PersistentDataset.")

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

    run_dir = Path(args.run_dir) / "text2glioma" / "autoencoder_stage1"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

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

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"Config: {str(config)}")

    writer_train = SummaryWriter(log_dir / "train")
    writer_val = SummaryWriter(log_dir / "val")
    print("Getting data...")
    
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    )

    print("Initializing model...")        
    model = get_model(model_type, config, args.pretrained)

    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])        
    
    if torch.cuda.device_count() > 1 and args.use_parallel:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = optim.AdamW(model.parameters(), lr=config["model"]["lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["lr"])

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    # get checkpoint to resume
    best_loss = float("inf")
    start_epoch = 0
    if resume and checkpoint is not None:
        print("Using checkpoint to resume training...")
        checkpoint = torch.load(run_dir / "checkpoint.pth")
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
        n_epochs=args.n_epochs,
        start_epoch=start_epoch,
        best_loss=best_loss,
        val_interval=args.val_interval,
        model_dir=model_dir,
        writer_train=writer_train,
        writer_val=writer_val,
        run_dir=run_dir,
        kl_weight=config["model"]["kl_weight"],
        recon_weight=config["model"]["recon_weight"],
        perceptual_weight=config["model"]["perceptual_weight"],
        adversarial_weight=config["model"]["adversarial_weight"],
    )
    print(f"Training completed. Best validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    # get things going...
    main()