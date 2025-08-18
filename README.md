# text2glioma

## Inference configuration

Sampling options are stored in `configs/inference.yaml`:

- `guidance_scale`: strength of classifier-free guidance.
- `num_steps`: number of denoising steps.
- `depth`, `height`, `width`: spatial dimensions of the generated volume.
- `scale_factor`: latent scaling factor used by the autoencoder.
- `healthy_prompt`: optional prompt describing a healthy reference image.
- `difference_threshold`: minimum absolute difference to keep in the difference map.
- `save_difference`: when `true`, writes the difference map to `difference.pt`.

