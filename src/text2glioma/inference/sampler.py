import argparse
import pathlib
import json
from tqdm import tqdm

import nibabel as nib
import numpy as np
import torch
from monai.config import print_config
from text2glioma.utils import print_gpu_memory_report, get_model, load_config, stage1_ify, batchify, load_text_encoder_and_tokenizer
from text2glioma.inference.inference_functions import GenericSampler
from text2glioma.inference.saver import NiftiSaver

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion model.")
    parser.add_argument("source_json", type=str, help="JSON file containing text prompts and optional mask paths.")
    parser.add_argument("output_dir", type=str, help="Directory to save converted NIfTI files.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--stage1_config", type=str, required=True, help="Path to the stage 1 autoencoder config file.")
    parser.add_argument("--stage1_uri", type=str, required=True, help="URI for the pretrained stage 1 autoencoder.")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--text_field", type=str, default="impression", metavar=["impression", "findings"], help="Which text field to use for conditioning (default: impression).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sampling.")
    parser.add_argument("--n_samples", type=int, default=100, help="Total number of samples to generate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading.")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the data.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models and tokenizers.")
    parser.add_argument("--guidance_scale_text", type=float, default=7.5, help="CFG scale for text conditioning.")
    parser.add_argument("--guidance_scale_mask", type=float, default=3.0, help="CFG scale for mask conditioning (0 = text-only).")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps for sampling.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta parameter.")
    parser.add_argument("--rescale_intensity", action="store_true", help="Whether to rescale intensity to [0, 1].")
    parser.add_argument("--use_parallel", action="store_true", help="Use DataParallel for multi-GPU sampling.")
    parser.add_argument("--img2img", action="store_true", help="Whether to perform img2img sampling (default: False).")
    parser.add_argument("--return_latents", action="store_true", help="Whether to return latents along with images (default: False).")
    parser.add_argument("--verbose", action="store_true", help="Whether to print outputs sometimes (default: False).")
    parser.add_argument("--mask_field", type=str, default="label", help="Key in JSON for mask/label paths (default: label).")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[160, 224, 160], help="Full-resolution spatial size D H W for loading masks.")
    parser.add_argument("--scale_factor", type=float, default=0.866, help="Latent whitening factor; latents / scale_factor before AE decode. Default 0.866 matches the released MaxFeat/RadBERT LDM.")

    return parser.parse_args()

def main():
    args = parse_args()
    print_config()
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    if args.verbose:
        print(f"Config: {config}")

    with open(args.source_json, "r") as f:
        datalist = json.load(f)

    if datalist is None:
        raise ValueError("A source JSON file must be provided for inference.")

    n_samples = args.n_samples
    cache_dir = args.cache_dir if args.cache_dir else None
    batch_size = args.batch_size
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    guidance_scale_text = args.guidance_scale_text
    guidance_scale_mask = args.guidance_scale_mask
    rescale_intensity = args.rescale_intensity
    latent_shape = config.get("sampling", {}).get("latent_shape", (28, 32, 28))
    latent_channels = config.get("model", {}).get("latent_channels", 3)
    latent_shape = (latent_channels, ) + tuple(latent_shape)
    num_mask_classes = config.get("mask", {}).get("num_classes", 4)
    text_field = args.text_field
    mask_field = args.mask_field
    spatial_size = tuple(args.spatial_size)
    if text_field not in ["impression", "findings"]:
        raise ValueError(f"Unrecognized text field: {text_field}. Expected: impression, findings")
    print(f"Using text field: {text_field}")

    verbose = args.verbose

    run_dir = pathlib.Path(args.output_dir) / "text2glioma" / "inference"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage1_config = load_config(args.stage1_config)
    stage1 = stage1_ify(
        get_model(
            model_type="AutoencoderKL", config=stage1_config, from_file=args.stage1_uri
        )
    )
    stage1.eval()
    print(f"Loaded stage 1 autoencoder from {args.stage1_uri}")

    model = get_model(config["model"].get("name", "DiffusionModelUNet"), config)
    model_ckpt = args.model_ckpt
    if not pathlib.Path(model_ckpt).exists():
        raise ValueError(f"Model checkpoint {model_ckpt} does not exist.")
    checkpoint = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Loaded model from {model_ckpt}")

    if verbose:
        print(f"Using tokenizer: {config['conditioning'].get('tokenizer')}")
        print(f"Using text encoder: {config['conditioning'].get('text_encoder')}")

    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"],
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    # Override model_max_length if config specifies it
    cfg_max_len = config["conditioning"].get("max_length")
    if cfg_max_len is not None:
        tokenizer.model_max_length = cfg_max_len

    from generative.networks.schedulers import DDIMScheduler
    scheduler = DDIMScheduler(**config["scheduler"].get("params", {}))

    if args.use_parallel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        stage1 = torch.nn.DataParallel(stage1)
        model = torch.nn.DataParallel(model)
        text_encoder = torch.nn.DataParallel(text_encoder)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stage1.to(device)
    model.to(device)
    text_encoder.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print_gpu_memory_report()

    print(f"Using device: {device}")

    sampler = GenericSampler(
        stage1=stage1,
        model=model,
        scheduler=scheduler,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        device=device, 
        cache_dir=cache_dir,
        img2img=args.img2img,
        return_latents=args.return_latents,
        num_mask_classes=num_mask_classes,
        latent_channels=latent_channels,
        scale_factor=args.scale_factor,
    )
    saver = NiftiSaver(output_dir)

    # Flatten datalist if it's a dict with splits
    if isinstance(datalist, dict):
        samples_list = datalist.get("test", datalist.get("validation", datalist.get("training", [])))
    elif isinstance(datalist, list):
        samples_list = datalist
    else:
        raise ValueError("Unexpected datalist format.")

    batches = list(batchify(samples_list, batch_size))
    progress_bar = tqdm(batches, total=len(batches), desc="Sampling")
    
    total_samples = 0
    with torch.no_grad():
        for batch in progress_bar:
            texts = [item[text_field] for item in batch]
            B = len(texts)
            
            # Load masks if available
            masks = None
            mask_paths = [item.get(mask_field) for item in batch]
            if any(p is not None for p in mask_paths):
                mask_list = []
                for mp in mask_paths:
                    if mp is not None and pathlib.Path(mp).exists():
                        label_nii = nib.load(str(mp))
                        label_data = np.asarray(label_nii.dataobj, dtype=np.float32)
                        # Resize to expected spatial size via nearest interpolation
                        label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                        label_tensor = torch.nn.functional.interpolate(
                            label_tensor, size=spatial_size, mode="nearest"
                        )
                        mask_list.append(label_tensor)
                    else:
                        # No mask for this sample → zeros (will be handled as unconditional)
                        mask_list.append(torch.zeros(1, 1, *spatial_size))
                masks = torch.cat(mask_list, dim=0).to(device)  # [B, 1, D, H, W]
            
            if verbose:
                print(f"Generating {B} samples...")
            
            samples = sampler.sample(
                steps=ddim_steps,
                batch_size=B,
                latent_shape=latent_shape,
                texts=texts,
                masks=masks,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_mask=guidance_scale_mask,
                eta=ddim_eta,
                verbose=verbose,
                rescale_intensity=rescale_intensity,
            )
            if args.return_latents:
                samples, latents = samples
            
            if verbose:
                print(f"Sampled tensor shape: {samples.shape}, dtype: {samples.dtype}, "
                      f"min: {samples.min().item():.4f}, max: {samples.max().item():.4f}")
            samples = samples.clamp(0, 1)

            for i in range(B):
                sample_np = samples[i].cpu().numpy()  # [C,D,H,W] or [D,H,W]
                file_name = f"sample_{total_samples + i + 1:04d}.nii.gz"
                saver.save(sample_np, file_name, meta=None, is_label=False)
                if verbose:
                    n_ch = sample_np.shape[0] if sample_np.ndim == 4 else 1
                    print(f"Saved sample {total_samples + i + 1} ({n_ch} channels)")
            total_samples += B
            if total_samples >= n_samples:
                break
    
    print(f"Generated {total_samples} samples → {output_dir}")
                