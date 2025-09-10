from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPTextModel
from generative.losses import PatchAdversarialLoss

from text2glioma.utils import print_resource_usage

@torch.no_grad()
def encode_text(tokenizer, text_encoder, texts, device, pad_to_max=True):
    tokens = tokenizer(
        text=texts,
        max_length=tokenizer.model_max_length if pad_to_max else None,
        padding="max_length" if pad_to_max else True,
        truncation=True,
        return_tensors="pt",
    )
    out = text_encoder(**tokens)
    return out.last_hidden_state.to(device)

def get_uncond(tokenizer, text_encoder, batch_size, device):
    return encode_text(tokenizer, text_encoder, [""] * batch_size, device)

def prepare_conditioning(tokenizer, text_encoder, texts, batch_size, device, dropout_p=0.2, uncond_cache=None):
    B = len(texts)
    cond = encode_text(tokenizer, text_encoder, texts, device)
    uncond = uncond_cache if (uncond_cache is not None and uncond_cache.size(0) == B) \
        else get_uncond(tokenizer, text_encoder, batch_size, device)
    # text dropout for classifier-free guidance
    drop = (torch.rand(B) < dropout_p).to(device).float().view(B, 1, 1)
    context = cond * (1 - drop) + uncond * drop
    return context, uncond

def train_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    train_loader,
    val_loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler_g: torch.cuda.amp.GradScaler,
    scaler_d: torch.cuda.amp.GradScaler,
    device: str,
    n_epochs: int,
    start_epoch: int = 0,
    best_loss: float = float("inf"),
    val_interval: int = 1,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str  = "./runs",
    kl_weight: float = 1e-6,
    perceptual_weight: float = 2e-3,
    adversarial_weight: float = 1e-3,
    resource_monitor: bool = True,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)

            # -----------------
            # Train Generator
            # -----------------
            with torch.cuda.amp.autocast(enabled=True):
                reconstruction, z_mu, z_sigma = model(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
            
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss.mean()) + (adversarial_weight * generator_loss)

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adversarial_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            
        avg_recon_loss = epoch_loss / (step + 1)
        avg_g_loss = gen_epoch_loss / (step + 1)
        avg_d_loss = disc_epoch_loss / (step + 1)
        if resource_monitor:
            print_resource_usage(epoch)

        if writer_train:
            writer_train.add_scalar("Loss/Generator", avg_g_loss, epoch)
            writer_train.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
            writer_train.add_scalar("Loss/Reconstruction", avg_recon_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{n_epochs}] "
            f"Generator Loss: {avg_g_loss:.4f} "
            f"Discriminator Loss: {avg_d_loss:.4f} "
            f"Reconstruction Loss: {avg_recon_loss:.4f} "
        )

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            avg_val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)
                    optimizer_g.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=True):
                        reconstruction, z_mu, z_sigma = model(images)
                        recons_loss = F.l1_loss(reconstruction.float(), images.float())

                    avg_val_loss += recons_loss.item()

            avg_val_loss /= val_step
            if writer_val:
                writer_val.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            if resource_monitor:
                print_resource_usage(epoch)

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), model_dir / "best_autoencoder.pth")
                torch.save(discriminator.state_dict(), model_dir / "best_discriminator.pth")
                print(f"Best model saved with validation loss: {best_loss:.4f}")
                  
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "best_loss": best_loss,
                },
                run_dir / "checkpoint.pth",
            )
    return best_loss

def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    tokenizer: Any,
    text_encoder: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    n_epochs: int,
    text_field: str = "impression",
    start_epoch: int = 0,
    val_interval: int = 1,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str = "./runs",
    scale_factor: float = 1.0,
    resource_monitor: bool = True,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            texts = batch[text_field]

            # Prepare conditioning
            cond, _ = prepare_conditioning(tokenizer, text_encoder, texts, images.size(0), device)

            # Encode images to latent space
            with torch.no_grad():
                latents = stage1(images) * scale_factor

            # Sample noise and timestep
            noise = torch.randn_like(latents).to(device)
            bsz = latents.size(0)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                noise_pred = model(noisy_latents, timesteps, context=cond)

            if scheduler.prediction_type == "epsilon":
                loss = nn.functional.mse_loss(noise_pred, noise)
            elif scheduler.prediction_type == "v_prediction":
                v = scheduler.get_velocity(latents, noise, timesteps)
                loss = nn.functional.mse_loss(noise_pred, v)
            else:
                raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        if writer_train:
            writer_train.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{n_epochs}] Training Loss: {avg_loss:.4f}")
        scheduler.step()

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    texts = batch[text_field]

                    # Prepare conditioning
                    cond, _ = prepare_conditioning(tokenizer, text_encoder, texts, images.size(0), device)

                    # Encode images to latent space
                    latents = stage1(images).detach() * scale_factor

                    # Sample noise and timestep
                    noise = torch.randn_like(latents).to(device)
                    bsz = latents.size(0)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()

                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                    with torch.cuda.amp.autocast():
                        noise_pred = model(noisy_latents, timesteps, context=cond)

                    if scheduler.prediction_type == "epsilon":
                        loss = nn.functional.mse_loss(noise_pred, noise)
                    elif scheduler.prediction_type == "v_prediction":
                        v = scheduler.get_velocity(latents, noise, timesteps)
                        loss = nn.functional.mse_loss(noise_pred, v)
                    else:
                        raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")

                    val_loss += loss.item() * images.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)

            if writer_val:
                writer_val.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save best model
            torch.save(model.state_dict(), model_dir / "best_ldm.pth")
            print(f"Model saved at epoch {epoch+1}")
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "ldm_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                run_dir / "checkpoint.pth",
            )
    return avg_val_loss

