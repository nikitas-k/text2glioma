# Architecture

text2glioma uses a two-stage latent diffusion architecture for 3D
multi-sequence brain MRI synthesis, with dual conditioning via text (CLIP
cross-attention) and segmentation masks (channel concatenation).

## Stage 1 — Variational Autoencoder (VAE)

The VAE is a 3D `AutoencoderKL` from MONAI Generative that compresses
4-channel MRI volumes into a compact latent space:

| Property | Value |
|----------|-------|
| Input | `[B, 4, 160, 224, 160]` — T1, T1CE, T2, FLAIR |
| Latent | `[B, 3, 10, 14, 10]` — 16× spatial downsampling |
| Architecture | 4 levels `[32, 64, 128, 128]`, no attention, 2 res blocks |
| Output | `[B, 4, 160, 224, 160]` — reconstructed multi-channel MRI |
| Loss | L1 + KL (1e-8) + per-channel MedicalNet perceptual (2e-3) + PatchGAN adversarial (5e-3) |

### Per-channel perceptual loss

The MedicalNet ResNet-50 perceptual loss expects single-channel input.
For 4-channel images, the loss is computed independently per modality and
averaged:

```
p_loss = (1/4) × Σ_{c=0}^{3} PerceptualLoss(recon[:, c:c+1], image[:, c:c+1])
```

### Discriminator

A 3D `PatchDiscriminator` with 3 layers and 96 channels operates on
the full 4-channel output.

## Stage 2 — Latent Diffusion Model (LDM)

The LDM is a 3D `DiffusionModelUNet` that learns to denoise in the frozen
VAE latent space.

### Input construction

At each denoising step, the model receives a **7-channel** input:

```
model_input = cat([noisy_latent, mask_conditioning], dim=1)
              ────────────────  ───────────────────
                  3 channels        4 channels
```

- **Noisy latent** `[B, 3, 10, 14, 10]`: noise-corrupted VAE encoding.
- **Mask conditioning** `[B, 4, 10, 14, 10]`: one-hot encoded segmentation
  labels (0=bg, 1=nCET, 2=edema, 3=ET), trilinearly downsampled from
  `160 × 224 × 160` to the latent spatial dims.

### Text conditioning

Text is encoded using the CLIP text encoder from Stable Diffusion 2.1
(`cross_attention_dim = 1024`, `max_length = 77` tokens) and injected
via cross-attention at UNet levels 2 and 3.

### UNet architecture

| Property | Value |
|----------|-------|
| Channels | `[256, 512, 768]` |
| Attention | Levels 2 and 3 |
| Head channels | `[0, 512, 768]` |
| Transformer layers | 1 per block |
| Output | `[B, 3, 10, 14, 10]` — predicted noise / velocity |

### Noise scheduler

DDIM scheduler with:

- `schedule = "scaled_linear_beta"`
- `num_train_timesteps = 1000`
- `prediction_type = "v_prediction"`

### Dual classifier-free guidance

During training, text and mask are independently dropped (each with
probability 0.2) by replacing them with empty-string embeddings or zero
tensors.  This creates four training modes, enabling flexible CFG at
inference:

**Three-way CFG formula:**

```
ε = ε_uncond
    + s_text × (ε_text_only − ε_uncond)
    + s_mask × (ε_full − ε_text_only)
```

Where:
- `ε_uncond`: both text and mask unconditional
- `ε_text_only`: text conditioned, mask = zeros
- `ε_full`: both text and mask conditioned
- `s_text`, `s_mask`: independent guidance scales

This allows 4 inference modes:

| Text | Mask | Use case |
|------|------|----------|
| ✓ | ✓ | Spatially-constrained generation from a description |
| ✓ | ✗ | Free-form generation from text alone |
| ✗ | ✓ | Generate MRI fitting a given segmentation mask |
| ✗ | ✗ | Unconditional sampling |

## Data pipeline

### Input format

- **Images**: 4D NIfTI `[D, H, W, 4]`, with volumes in order
  T1 / T1CE / T2 / FLAIR.  `EnsureChannelFirstd(channel_dim=3)` moves
  this to `[4, D, H, W]`.
- **Labels**: 3D integer NIfTI `[D, H, W]` with classes 0–3.
  `EnsureChannelFirstd(channel_dim="no_channel")` adds a unit channel
  `[1, D, H, W]`.

### Transforms

All transforms resize to `160 × 224 × 160`.  Key per-channel operations:

- `ScaleIntensityRangePercentilesd(channel_wise=True)`: each MRI sequence
  normalised to [0, 1] using its own 0th–99.5th percentile range.
- `RandShiftIntensityd(channel_wise=True)`: random intensity shifts applied
  independently per channel.

### Output format

Generated images are saved as 4D NIfTI `[D, H, W, C]` (NIfTI convention:
channels in last dimension).  Per-channel rescaling to [0, 255] is applied
before saving.
