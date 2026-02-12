# Configuration reference

text2glioma uses YAML configuration files stored in `configs/`.  This
page documents every setting.

## `configs/stage1.yaml` — VAE (AutoencoderKL)

```yaml
model:
  name: "AutoencoderKL"
  lr: 0.00005                     # generator learning rate
  perceptual_weight: 0.002        # weight for MedicalNet perceptual loss
  adv_weight: 0.005               # PatchGAN adversarial loss weight
  kl_weight: 0.00000001           # KL divergence weight
  params:
    spatial_dims: 3
    in_channels: 4                # T1, T1CE, T2, FLAIR
    out_channels: 4
    num_channels: [32, 64, 128, 128]
    latent_channels: 3
    num_res_blocks: 2
    attention_levels: [False, False, False, False]
    with_encoder_nonlocal_attn: False
    with_decoder_nonlocal_attn: False

discriminator:
  lr: 0.0001                      # discriminator learning rate
  params:
    spatial_dims: 3
    num_channels: 96
    num_layers_d: 3
    in_channels: 4

perceptual_network:
  params:
    spatial_dims: 3
    network_type: "medicalnet_resnet50_23datasets"
    is_fake_3d: False
```

### Notes

- `in_channels` / `out_channels` must match the number of MRI sequences
  in your 4D NIfTI files.
- `perceptual_weight` controls the MedicalNet loss.  This is computed
  **per channel** and averaged.
- Set `adv_weight: 0` to disable the discriminator during early epochs.

---

## `configs/ldm.yaml` — Latent Diffusion Model

```yaml
model:
  name: "DiffusionModelUNet"
  base_lr: 0.000025
  latent_channels: 3
  params:
    spatial_dims: 3
    in_channels: 7                # 3 latent + 4 one-hot mask
    out_channels: 3
    num_res_blocks: 2
    num_channels: [256, 512, 768]
    attention_levels: [False, True, True]
    with_conditioning: True
    cross_attention_dim: 1024     # CLIP SD 2.1 embedding dim
    num_head_channels: [0, 512, 768]
    upcast_attention: True
    use_flash_attention: False
    transformer_num_layers: 1
    norm_num_groups: 32
    norm_eps: 1e-6

scheduler:
  name: "DDIMScheduler"           # or "DDPMScheduler"
  params:
    schedule: "scaled_linear_beta"
    num_train_timesteps: 1000
    beta_start: 0.0015
    beta_end: 0.0205
    prediction_type: "v_prediction"

conditioning:
  tokenizer: "stabilityai/stable-diffusion-2-1-base"
  text_encoder: "stabilityai/stable-diffusion-2-1-base"
  max_length: 77
  projection_dim: 1024
  dropout_p: 0.2                  # text dropout for CFG

mask:
  num_classes: 4                  # bg + nCET + edema + ET
  dropout_p: 0.2                  # independent mask dropout for CFG
  spatial_size: [160, 224, 160]
```

### Key relationships

- `in_channels` = `latent_channels` + `mask.num_classes`
- `out_channels` = `latent_channels`
- `cross_attention_dim` must match the CLIP model dimensionality

---

## `configs/inference.yaml` — Sampling

```yaml
guidance_scale_text: 7.5          # CFG scale for text
guidance_scale_mask: 3.0          # CFG scale for mask (0 = text-only)
num_steps: 50                     # DDIM denoising steps
scale_factor: 0.18215            # latent scaling factor

num_mask_classes: 4
mask_field: "label"               # key in JSON for mask paths

healthy_prompt: "a healthy brain MRI"
difference_threshold: 0.0
save_difference: false
```

---

## `configs/cnn.yaml` — Downstream classification

```yaml
model:
  name: densenet121
  params:
    num_classes: 2
    in_channels: 1
    spatial_dims: 3
    input_shape: [1, 160, 192, 96]

lr: 0.001
weight_decay: 0.0001
batch_size: 2
n_epochs: 1000
val_interval: 10
early_stopping: 50
experiment_name: mgmt             # mgmt | 1p19q | idh | grade
exp_type: real                    # real | synthetic | real_synthetic
```

---

## CLI arguments

### `train_stage1`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data` | str | required | Path to datalist JSON |
| `--config` | str | required | Path to stage1.yaml |
| `--run_dir` | str | required | Output directory |
| `--num_epochs` | int | 500 | Total epochs |
| `--batch_size` | int | 4 | Batch size |
| `--device` | str | cuda | Device |
| `--resume` | flag | — | Resume from checkpoint |
| `--pretrained` | flag | — | Load pretrained weights |
| `--use_parallel` | flag | — | DataParallel |
| `--initialize` | flag | — | Reset PersistentDataset cache |

### `train_stage2`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data` | str | required | Path to datalist JSON |
| `--config` | str | required | Path to ldm.yaml |
| `--stage1_config` | str | required | Path to stage1.yaml |
| `--stage1_uri` | str | required | Path to trained VAE weights |
| `--run_dir` | str | required | Output directory |
| `--n_epochs` | int | 250 | Total epochs |
| `--batch_size` | int | 4 | Batch size |
| `--train_spec` | str | impression | Text field: `impression` or `findings` |
| `--scale_factor` | float | 1.0 | Latent scaling factor |
| `--mask_dropout_p` | float | config | Override mask dropout |
| `--text_dropout_p` | float | config | Override text dropout |

### `sample`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source_json` | str | required | JSON with text prompts (+ optional mask paths) |
| `output_dir` | str | required | Output directory |
| `--config` | str | required | LDM config |
| `--stage1_config` | str | required | VAE config |
| `--stage1_uri` | str | required | VAE weights |
| `--model_ckpt` | str | required | LDM weights |
| `--n_samples` | int | 100 | Number of samples |
| `--guidance_scale_text` | float | 7.5 | Text CFG scale |
| `--guidance_scale_mask` | float | 3.0 | Mask CFG scale |
| `--ddim_steps` | int | 50 | Denoising steps |
| `--ddim_eta` | float | 0.0 | DDIM stochasticity |
