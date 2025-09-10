from pathlib import Path
from typing import Any
from collections import OrderedDict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPTextModel
from generative.losses import PatchAdversarialLoss

from text2glioma.utils import print_gpu_memory_report, get_lr, log_reconstructions

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
    device: torch.device,
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
    raw_model = model.module if hasattr(model, "module") else model

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_loss = eval_autoencoder(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        kl_weight=kl_weight,
        adversarial_weight=adversarial_weight,
        perceptual_weight=perceptual_weight,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_autoencoder(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            kl_weight=kl_weight,
            adversarial_weight=adversarial_weight,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_autoencoder(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                kl_weight=kl_weight,
                adversarial_weight=adversarial_weight,
                perceptual_weight=perceptual_weight,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss

def train_epoch_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adversarial_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adversarial_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        if adversarial_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                d_loss = adversarial_weight * discriminator_loss
                d_loss = d_loss.mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        losses["d_loss"] = discriminator_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )

@torch.no_grad()
def eval_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    for x in loader:
        images = x["image"].to(device)

        with torch.cuda.amp.autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adversarial_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adversarial_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adversarial_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]


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

