import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np

from text2glioma.utils import masks_to_onehot, downsample_mask_to_latent, get_text_encoder_hidden_states


@torch.no_grad()
def encode_text(tokenizer, text_encoder, texts, pad_to_max=True, device='cpu'):
    """Encode a list of texts into text embeddings using the provided tokenizer and text encoder."""
    tokens = tokenizer(
        text=texts,
        max_length=tokenizer.model_max_length if pad_to_max else None,
        padding="max_length" if pad_to_max else True,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    out = text_encoder(**tokens)
    return get_text_encoder_hidden_states(out).to(device)

def get_uncond(tokenizer, text_encoder, batch_size, device):
    return encode_text(tokenizer, text_encoder, [""] * batch_size, device=device)

def prepare_conditioning(tokenizer, text_encoder, texts, batch_size, dropout_p=0.2, uncond_cache=None, device='cpu'):
    B = len(texts)
    cond = encode_text(tokenizer, text_encoder, texts, device=device)
    uncond = uncond_cache if (uncond_cache is not None and uncond_cache.size(0) == B) \
        else get_uncond(tokenizer, text_encoder, batch_size, device=device)
    # text dropout for classifier-free guidance
    drop = (torch.rand(B) < dropout_p).float().to(device).view(B, 1, 1)
    context = cond * (1 - drop) + uncond * drop
    return context, uncond

def prepare_mask_for_inference(
    labels: torch.Tensor = None,
    latent_spatial: tuple = None,
    num_classes: int = 4,
    batch_size: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare mask conditioning tensor for inference.
    
    Args:
        labels: Optional [B, 1, D, H, W] integer segmentation labels.
                If None, returns zero tensor (mask-unconditional).
        latent_spatial: (D', H', W') target spatial dims.
        num_classes: Number of classes including background.
        batch_size: Batch size (used when labels is None).
        device: Target device.
    
    Returns:
        Mask conditioning tensor [B, num_classes, D', H', W'].
    """
    if labels is not None:
        onehot = masks_to_onehot(labels, num_classes=num_classes)
        mask_cond = downsample_mask_to_latent(onehot, latent_spatial)
        return mask_cond.to(device)
    else:
        return torch.zeros(batch_size, num_classes, *latent_spatial, device=device)


@torch.no_grad()
def cfg_sample(model, x, t, text_cond, text_uncond, mask_cond, mask_uncond,
               guidance_scale_text=7.5, guidance_scale_mask=3.0):
    """
    Perform dual classifier-free guidance sampling step over text and mask.
    
    Supports three-way CFG: fully unconditional, mask-only, and full conditioning.
    The guidance formula is:
        eps = eps_uncond 
              + scale_text * (eps_text_only - eps_uncond)
              + scale_mask * (eps_full - eps_text_only)
    
    For simplicity, we use a two-way formulation when guidance_scale_mask <= 0:
        eps = eps_uncond + scale_text * (eps_cond - eps_uncond)
    
    Args:
        model: The diffusion model.
        x: Current latent tensor [B, latent_ch, D, H, W].
        t: Current timestep tensor [B].
        text_cond/text_uncond: Text conditioning [B, T, D] / unconditional.
        mask_cond/mask_uncond: Mask conditioning [B, num_classes, D', H', W'] / zeros.
        guidance_scale_text: CFG scale for text.
        guidance_scale_mask: CFG scale for mask (0 = ignore mask guidance).
    Returns:
        The predicted noise tensor after applying dual classifier-free guidance.
    """
    if guidance_scale_mask > 0:
        # Three-way CFG: uncond, text-only (no mask), fully conditioned
        # 1) Fully unconditional
        x_in_uncond = torch.cat([x, mask_uncond], dim=1)
        eps_uncond = model(x=x_in_uncond, timesteps=t, context=text_uncond)
        
        # 2) Text-only (mask dropped = zeros)
        x_in_text = torch.cat([x, mask_uncond], dim=1)
        eps_text = model(x=x_in_text, timesteps=t, context=text_cond)
        
        # 3) Full conditioning (text + mask)
        x_in_full = torch.cat([x, mask_cond], dim=1)
        eps_full = model(x=x_in_full, timesteps=t, context=text_cond)
        
        # Compose
        eps = (eps_uncond 
               + guidance_scale_text * (eps_text - eps_uncond) 
               + guidance_scale_mask * (eps_full - eps_text))
    else:
        # Standard two-way CFG on text only (mask always provided)
        x_cond_in = torch.cat([x, mask_cond], dim=1)
        x_uncond_in = torch.cat([x, mask_cond], dim=1)  # mask present for both
        
        x_in = torch.cat([x_uncond_in, x_cond_in], dim=0)
        t_in = torch.cat([t, t], dim=0)
        ctx_in = torch.cat([text_uncond, text_cond], dim=0)
        
        model_output = model(x=x_in, timesteps=t_in, context=ctx_in)
        eps_uncond, eps_cond = model_output.chunk(2)
        eps = eps_uncond + guidance_scale_text * (eps_cond - eps_uncond)
    
    return eps

@torch.no_grad()
def cfg_img2img(model, scheduler, z_init, text_cond, text_uncond, 
                mask_cond, mask_uncond,
                steps=60, start_strength=0.85, 
                guidance_scale_text=5.0, guidance_scale_mask=3.0):
    """
    img2img generation with dual CFG over text and mask.
    
    z_init: VAE latent of source image (clean); add noise at a high t and denoise down.
    start_strength in [0,1]: 0 -> no edit; 1 -> pure noise (full rewrite).
    """
    device = z_init.device
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps

    start_idx = int(min(max(start_strength, 0.0), 1.0) * (len(timesteps) - 1))
    t_start = timesteps[start_idx]

    noise = torch.randn_like(z_init)
    x = scheduler.add_noise(z_init, noise, t_start)

    for t in timesteps[start_idx:]:
        t_b = torch.full((x.size(0),), int(t.item()), dtype=torch.long, device=device)
        
        eps_hat = cfg_sample(
            model, x, t_b, text_cond, text_uncond, 
            mask_cond, mask_uncond,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_mask=guidance_scale_mask,
        )
        x = scheduler.step(eps_hat, int(t.item()), x)[0]

    return x


class GenericSampler():
    def __init__(self, stage1, model, scheduler, tokenizer, 
                 text_encoder, device='cpu', cache_dir=None, 
                 return_latents=False, img2img=False,
                 num_mask_classes=4, latent_channels=3,
                 scale_factor=0.866):
        """
        stage1: the autoencoder model to decode latents to image space.
        model: the diffusion model.
        scheduler: the noise scheduler (e.g., DDIMScheduler).
        device: device to run the sampling on.
        cache_dir: directory for caching models if needed.
        return_latents: if True, return latents along with images.
        num_mask_classes: number of mask classes (including background).
        latent_channels: number of VAE latent channels.
        scale_factor: latent whitening factor used at stage-2 training time;
            latents are divided by this before AE decode. Default 0.866
            matches the released MaxFeat/RadBERT LDM.
        """
        super().__init__()
        self.stage1 = stage1
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.cache_dir = cache_dir
        self.return_latents = return_latents
        self.img2img = img2img
        self.num_mask_classes = num_mask_classes
        self.latent_channels = latent_channels
        self.scale_factor = float(scale_factor)

    @torch.no_grad()
    def sample(self, steps, batch_size, latent_shape, 
               texts, masks=None,
               guidance_scale_text=7.5, guidance_scale_mask=3.0,
               eta=0.0, verbose=False, 
               rescale_intensity=False,
               decode_amp_dtype=None,
               offload_diffusion_during_decode=False):
        """
        Generate samples using the diffusion model with dual classifier-free guidance.
        
        Args:
            steps: Number of diffusion steps.
            batch_size: Number of samples to generate.
            latent_shape: Shape of the latent space (C, D, H, W).
            texts: List of text prompts for conditioning.
            masks: Optional [B, 1, D_full, H_full, W_full] integer segmentation labels.
                   If None, generates mask-unconditionally (text-only mode).
            guidance_scale_text: CFG scale for text conditioning.
            guidance_scale_mask: CFG scale for mask conditioning (0 = text-only mode).
            eta: DDIM eta parameter for stochasticity.
            verbose: Whether to print progress information.
            rescale_intensity: Whether to rescale intensity to [0, 1].
        
        Returns:
            Generated samples in image space (and latent representations if return_latents=True).
        """
        C, W, H, D = latent_shape
        latent_spatial = (W, H, D)
        latents = torch.randn(batch_size, C, W, H, D).to(self.device)
        
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare text conditioning
        cond, uncond = prepare_conditioning(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            texts=texts,
            batch_size=batch_size,
            dropout_p=0.0,  # no dropout during sampling
            uncond_cache=None,
            device=self.device
        )
        
        # Prepare mask conditioning
        mask_cond = prepare_mask_for_inference(
            labels=masks,
            latent_spatial=latent_spatial,
            num_classes=self.num_mask_classes,
            batch_size=batch_size,
            device=self.device,
        )
        mask_uncond = torch.zeros_like(mask_cond)
        
        if verbose:
            has_mask = masks is not None
            print(f"Sampling {batch_size} samples | latent {latent_shape} | "
                  f"{steps} steps | mask={'yes' if has_mask else 'no'} | "
                  f"cfg_text={guidance_scale_text} cfg_mask={guidance_scale_mask}")
        
        for i, t in enumerate(timesteps):
            if verbose and i % max(1, steps // 10) == 0:
                print(f"Step {i+1}/{steps}, Timestep {t.item()}")
            
            t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            
            # Dual CFG step
            noise_pred = cfg_sample(
                self.model, latents, t_tensor,
                text_cond=cond, text_uncond=uncond,
                mask_cond=mask_cond, mask_uncond=mask_uncond,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_mask=guidance_scale_mask,
            )
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)[0]
        
        if not torch.isfinite(latents).all():
            n = (~torch.isfinite(latents)).float().mean().item()
            raise RuntimeError(
                f"Diffusion produced non-finite latents ({n:.1%} bad); "
                "typical cause is CFG too high, wrong scale_factor, or a "
                "config/checkpoint mismatch."
            )
        
        # Free VRAM on 32 GB cards: UNet + text encoder are unused for the AE decode.
        if offload_diffusion_during_decode:
            model_dev = next(self.model.parameters()).device
            txt_dev = next(self.text_encoder.parameters()).device
            self.model.to("cpu")
            self.text_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Undo stage-2 latent whitening before decode (mirrors the training-time scaling).
        latents = latents / self.scale_factor
        
        # Decode the latents to image space using stage1 model
        if decode_amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=decode_amp_dtype):
                images = self.stage1.model.decode(latents)
            images = images.float()
        else:
            images = self.stage1.model.decode(latents)
        
        if not torch.isfinite(images).all():
            n = (~torch.isfinite(images)).float().mean().item()
            raise RuntimeError(
                f"AE decode produced non-finite images ({n:.1%} bad); "
                "decode_amp_dtype=fp16 is a known offender for 3D AutoencoderKL "
                "(GroupNorm variance can overflow); rerun in fp32."
            )
        
        if offload_diffusion_during_decode:
            self.model.to(model_dev)
            self.text_encoder.to(txt_dev)
        if rescale_intensity:
            images = (images - images.min()) / (images.max() - images.min())
        images = images.clamp(0, 1)
        
        if self.return_latents:
            return images, latents
        return images