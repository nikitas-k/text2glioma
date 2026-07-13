import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms as T
from tqdm import tqdm

from text2glioma.utils import (
    MODALITY_NAMES,
    get_model,
    get_text_encoder_hidden_states,
    load_config,
    load_text_encoder_and_tokenizer,
    prepare_mask_conditioning,
    stage1_ify,
    WhiteningStage1Wrapper,
)


MSD_TO_T2G = [1, 2, 3, 0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline Stage-2 sampling (conditioned vs unconditioned).")
    p.add_argument("--datalist", type=str, required=True, help="Path to datalist JSON.")
    p.add_argument("--config", type=str, required=True, help="Path to Stage-2 config YAML.")
    p.add_argument("--stage1_config", type=str, required=True, help="Path to Stage-1 config YAML.")
    p.add_argument("--stage1_uri", type=str, required=True, help="Path to Stage-1 checkpoint.")
    p.add_argument("--model_ckpt", type=str, required=True, help="Path to Stage-2 checkpoint (best_model.pth).")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save PNG outputs.")
    p.add_argument("--split", type=str, default="validation", choices=["training", "validation", "test"])
    p.add_argument("--start_index", type=int, default=0, help="Start index in the chosen split.")
    p.add_argument("--num_cases", type=int, default=4, help="Number of cases to sample.")
    p.add_argument("--text_field", type=str, default="impression", choices=["impression", "findings"])
    p.add_argument("--steps", type=int, default=200, help="Number of sampling steps.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--scale_factor", type=float, default=None, help="Override scale factor.")
    p.add_argument("--latent_whitening_path", type=str, default=None,
                   help="Optional whitening .pt (from scripts/fit_latent_whitening.py). "
                        "When set, stage-1 is wrapped so its output is whitened and "
                        "decode() inverts the whitening. scale_factor is forced to 1.0.")
    p.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale for conditioned sampling.")
    p.add_argument(
        "--cfg_mode",
        type=str,
        default="text_only",
        choices=["text_only", "joint"],
        help=(
            "CFG variant: 'text_only' keeps the mask in both branches and only "
            "guides on text (recommended for mask-conditioned models); 'joint' "
            "drops both text and mask in the uncond branch."
        ),
    )
    p.add_argument("--no_channel_reorder", action="store_true", default=False)
    p.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help=(
            "If set, override each case's text field with this string before "
            "sampling. Use together with --output_suffix to avoid overwriting "
            "the standard cfg-sweep output filenames."
        ),
    )
    p.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help=(
            "Optional slug appended to output filenames "
            "(e.g. sample_cond_native_{case_idx}_{suffix}.nii.gz). "
            "Recommended whenever --custom_prompt is used."
        ),
    )
    p.add_argument(
        "--drop_mask",
        action="store_true",
        default=False,
        help=(
            "If set, zero-out the mask conditioning channels even when the "
            "datalist provides a label. Use to probe text-only generation."
        ),
    )
    p.add_argument("--latent_smooth_sigma", type=float, default=0.0,
                   help="Isotropic Gaussian blur sigma (in latent voxels) applied "
                        "to the sampled latent. 0 disables. Diagnoses whether the "
                        "stage-1 latent's high-frequency content is what breaks the "
                        "LDM's output (see §3.5 diagnostic).")
    p.add_argument("--latent_smooth_mode", type=str, default="end",
                   choices=["end", "each_step"],
                   help="'end' blurs the final latent once before decoding. "
                        "'each_step' blurs inside the diffusion loop after every "
                        "scheduler.step() so the trajectory stays in a smoother "
                        "latent subspace throughout denoising.")
    return p.parse_args()


def _build_val_transform(channel_reorder: bool, has_label: bool) -> T.Compose:
    keys = ["image"] + (["label"] if has_label else [])
    xforms = [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))

    if has_label:
        xforms.extend(
            [
                T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
                T.EnsureTyped(keys=["label"], dtype=torch.float32),
            ]
        )

    spatial_keys = keys
    xforms.extend(
        [
            T.Orientationd(keys=spatial_keys, axcodes="LPS"),
            T.CropForegroundd(keys=spatial_keys, source_key="image"),
            T.SpatialPadd(keys=spatial_keys, spatial_size=(160, 224, 160), mode="constant"),
            T.CenterSpatialCropd(keys=spatial_keys, roi_size=(160, 224, 160)),
            T.ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1, channel_wise=True
            ),
            T.ToTensord(keys=keys),
        ]
    )
    return T.Compose(xforms)


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "diffusion" in checkpoint_obj:
            return checkpoint_obj["diffusion"]
        if "ldm_state_dict" in checkpoint_obj:
            return checkpoint_obj["ldm_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format for Stage-2 model.")


def _extract_stage1_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return checkpoint_obj["state_dict"]
        if "autoencoder" in checkpoint_obj and isinstance(checkpoint_obj["autoencoder"], dict):
            return checkpoint_obj["autoencoder"]
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return checkpoint_obj["model"]
    raise ValueError("Unsupported checkpoint format for Stage-1 model.")


def _infer_stage1_latent_channels(checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_stage1_state_dict(checkpoint)
    for key in (
        "quant_conv_mu.conv.weight",
        "module.quant_conv_mu.conv.weight",
        "model.quant_conv_mu.conv.weight",
    ):
        weight = state_dict.get(key)
        if torch.is_tensor(weight):
            return int(weight.shape[0])
    return None


def _gaussian_kernel_1d(sigma: float, dtype: torch.dtype, device: torch.device,
                        truncate: float = 4.0) -> torch.Tensor:
    """Return a normalised 1-D Gaussian kernel of appropriate length."""
    r = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k.to(dtype)


def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic 3-D Gaussian blur applied per channel to an NCDHW tensor.

    Separable: three 1-D convolutions along D, H, W with reflection padding.
    Returns a tensor with the same shape / dtype / device as `x`. When
    ``sigma <= 0`` returns ``x`` unchanged.
    """
    if sigma <= 0:
        return x
    if x.ndim != 5:
        raise ValueError(f"expected NCDHW, got shape {tuple(x.shape)}")
    k = _gaussian_kernel_1d(sigma, x.dtype, x.device)
    r = k.numel() // 2
    C = x.shape[1]
    for spatial_dim in (2, 3, 4):
        kshape = [1, 1, 1, 1, 1]
        kshape[spatial_dim] = k.numel()
        kernel = k.reshape(kshape).expand(C, 1, *kshape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]  # padding for (W_l, W_r, H_l, H_r, D_l, D_r)
        pad_index = 2 * (4 - spatial_dim)
        pad[pad_index] = r
        pad[pad_index + 1] = r
        x = F.pad(x, pad, mode="replicate")
        x = F.conv3d(x, kernel, groups=C)
    return x


def _sample_latent(
    model,
    scheduler,
    latent0,
    mask_cond,
    prompt_embeds,
    device,
    uncond_embeds: torch.Tensor = None,
    uncond_mask_cond: torch.Tensor = None,
    guidance_scale: float = 1.0,
    cfg_mode: str = "text_only",
    smooth_sigma: float = 0.0,
    smooth_each_step: bool = False,
):
    """Run DDIM sampling with optional classifier-free guidance.

    CFG variants:
      - ``text_only`` (default): mask is concatenated to the latent in BOTH the
        conditional and unconditional branches; only the text context is
        swapped. This matches ``cfg_sample`` in ``inference_functions.py`` and
        is the theoretically sound choice for this mask-conditioned U-Net,
        whose mask channels are part of the input convolution rather than
        cross-attention.
      - ``joint``: drop both text AND mask in the uncond branch (zeros mask).
        Use only if you specifically want to extrapolate jointly along both
        modalities; this typically degrades SSIM because the uncond branch
        sees out-of-distribution latent input statistics.

    Falls back to a single conditional forward pass when ``guidance_scale ==
    1.0`` or the uncond inputs are not provided.
    """
    latent = latent0.clone()
    do_cfg = (
        uncond_embeds is not None
        and uncond_mask_cond is not None
        and guidance_scale != 1.0
    )
    if do_cfg:
        # Mask routing depends on cfg_mode.
        mask_for_uncond = mask_cond if cfg_mode == "text_only" else uncond_mask_cond

    for t in scheduler.timesteps:
        ts = torch.asarray((t,)).to(device)
        if do_cfg:
            cond_input = torch.cat([latent, mask_cond], dim=1)
            uncond_input = torch.cat([latent, mask_for_uncond], dim=1)
            # Batch both branches into a single forward pass.
            x_in = torch.cat([uncond_input, cond_input], dim=0)
            ts_in = torch.cat([ts, ts], dim=0)
            ctx_in = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            model_output = model(x=x_in, timesteps=ts_in, context=ctx_in)
            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            model_input = torch.cat([latent, mask_cond], dim=1)
            noise_pred = model(x=model_input, timesteps=ts, context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)
        if smooth_each_step and smooth_sigma > 0:
            latent = _gaussian_blur_3d(latent, smooth_sigma)
    if smooth_sigma > 0 and not smooth_each_step:
        latent = _gaussian_blur_3d(latent, smooth_sigma)
    return latent


def _make_grid(x_hat: torch.Tensor, depth_indices: list[int]) -> np.ndarray:
    n_ch = x_hat.shape[1]
    rows = []
    for d in depth_indices:
        d_safe = max(0, min(int(d), x_hat.shape[-1] - 1))
        cols = []
        for c in range(n_ch):
            cols.append(np.clip(x_hat[0, c, :, :, d_safe].float().cpu().numpy(), 0, 1))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def _encode_text(tokenizer, text_encoder, text: str, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return get_text_encoder_hidden_states(text_encoder(**tokens))


def _resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if mps_available else "cpu")
    return torch.device(requested)


def _ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    """Compute mean SSIM for a single-channel 3D or 2D tensor (NCDHW/NCHW)."""
    ndim = pred.ndim - 2
    if ndim == 3:
        kernel = torch.ones(
            1,
            1,
            window_size,
            window_size,
            window_size,
            device=pred.device,
            dtype=pred.dtype,
        )
        conv_fn = F.conv3d
    else:
        kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype)
        conv_fn = F.conv2d
    kernel = kernel / kernel.numel()
    pad = window_size // 2

    mu_p = conv_fn(pred, kernel, padding=pad)
    mu_t = conv_fn(target, kernel, padding=pad)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = conv_fn(pred * pred, kernel, padding=pad) - mu_pp
    sigma_tt = conv_fn(target * target, kernel, padding=pad) - mu_tt
    sigma_pt = conv_fn(pred * target, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_pp + mu_tt + c1) * (sigma_pp + sigma_tt + c2))
    return ssim_map.mean()


def _compute_per_channel_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    modality_names: list[str],
) -> tuple[dict[str, float], float]:
    """Compute per-modality SSIM and return (per_channel, mean)."""
    n_ch = int(pred.shape[1])
    values: dict[str, float] = {}
    for c in range(n_ch):
        name = modality_names[c] if c < len(modality_names) else f"ch{c}"
        ssim_val = _ssim_3d(pred[:, c : c + 1], target[:, c : c + 1])
        values[name] = float(ssim_val.item())
    mean_ssim = float(sum(values.values()) / max(len(values), 1))
    return values, mean_ssim


def _resize_to_common_space(x: torch.Tensor, spatial_size: tuple[int, int, int]) -> torch.Tensor:
    """Resize NCDHW tensor to a shared spatial space for metric computation."""
    if tuple(int(v) for v in x.shape[2:]) == tuple(int(v) for v in spatial_size):
        return x
    return F.interpolate(x.float(), size=spatial_size, mode="trilinear", align_corners=False)


def _save_sample_nifti(
    sample: torch.Tensor,
    reference_image_path: str,
    out_path: Path,
) -> None:
    """Save generated sample as NIfTI with reference affine/header and spatial shape."""
    ref_img = nib.load(reference_image_path)
    ref_shape = ref_img.shape

    if len(ref_shape) == 4:
        target_spatial = tuple(int(v) for v in ref_shape[:3])
        target_channels = int(ref_shape[3])
    elif len(ref_shape) == 3:
        target_spatial = tuple(int(v) for v in ref_shape)
        target_channels = int(sample.shape[1])
    else:
        raise ValueError(f"Unsupported reference image shape: {ref_shape}")

    resized = F.interpolate(sample.float(), size=target_spatial, mode="trilinear", align_corners=False)
    arr = resized[0].detach().cpu().numpy().astype(np.float32)  # (C, X, Y, Z)

    if target_channels == 1:
        out_arr = arr[0]
    else:
        if arr.shape[0] != target_channels:
            min_ch = min(arr.shape[0], target_channels)
            fixed = np.zeros((target_channels,) + arr.shape[1:], dtype=np.float32)
            fixed[:min_ch] = arr[:min_ch]
            arr = fixed
        out_arr = np.moveaxis(arr, 0, -1)  # (X, Y, Z, C)

    out_img = nib.Nifti1Image(out_arr, affine=ref_img.affine, header=ref_img.header.copy())
    nib.save(out_img, str(out_path))


def _get_tensor_affine(tensor: torch.Tensor) -> np.ndarray:
    """Best-effort affine extraction from MONAI MetaTensor; fallback to identity."""
    affine = None

    tensor_affine = getattr(tensor, "affine", None)
    if tensor_affine is not None:
        affine = np.asarray(tensor_affine, dtype=np.float64)

    if affine is None:
        meta = getattr(tensor, "meta", None)
        if isinstance(meta, dict) and meta.get("affine") is not None:
            affine = np.asarray(meta["affine"], dtype=np.float64)

    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    return affine


def _save_tensor_nifti_native(
    sample: torch.Tensor,
    out_path: Path,
    affine: np.ndarray,
) -> None:
    """Save a sample tensor (N,C,X,Y,Z) directly without spatial interpolation."""
    arr = sample[0].detach().cpu().numpy().astype(np.float32)
    if arr.shape[0] == 1:
        out_arr = arr[0]
    else:
        out_arr = np.moveaxis(arr, 0, -1)
    out_img = nib.Nifti1Image(out_arr, affine=affine)
    nib.save(out_img, str(out_path))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Hard determinism: without these, cuDNN heuristics and non-deterministic
    # reductions (e.g. flash attention atomics) can make the uncond trajectory
    # differ across subprocesses launched with different --cfg_scale values,
    # even though the uncond forward itself takes no CFG-dependent input.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.datalist, "r") as f:
        datalist = json.load(f)
    if args.split not in datalist:
        raise KeyError(f"Split '{args.split}' not found in datalist.")

    data_split = datalist[args.split]
    if len(data_split) == 0:
        raise ValueError(f"Split '{args.split}' is empty.")

    end_index = min(args.start_index + args.num_cases, len(data_split))
    indices = list(range(args.start_index, end_index))
    if not indices:
        raise ValueError("No cases selected. Check --start_index and --num_cases.")

    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)
    common_ssim_size = tuple(int(v) for v in config.get("mask", {}).get("spatial_size", [160, 224, 160]))

    device = _resolve_device(args.device)

    stage1_params = stage1_config.setdefault("model", {}).setdefault("params", {})
    inferred_stage1_latent_ch = _infer_stage1_latent_channels(args.stage1_uri)
    if inferred_stage1_latent_ch is not None:
        stage1_params["latent_channels"] = inferred_stage1_latent_ch

    if device.type != "cuda":
        # AutoencoderKL flash attention path requires CUDA; disable for MPS/CPU local runs.
        stage1_params["use_flash_attention"] = False

    stage1 = stage1_ify(get_model("AutoencoderKL", stage1_config, from_file=args.stage1_uri))
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad = False

    # Wrap stage-1 with whitening if requested (must be applied *before* any
    # channel-count probing so downstream code sees the whitened channel count).
    if args.latent_whitening_path is not None:
        whit_path = Path(args.latent_whitening_path).expanduser().resolve()
        if not whit_path.is_file():
            raise FileNotFoundError(f"--latent_whitening_path not found: {whit_path}")
        print(f"Loading latent whitening from {whit_path}")
        whit = torch.load(str(whit_path), map_location="cpu")
        stage1 = WhiteningStage1Wrapper(
            stage1,
            mu=whit["mu"].to(device),
            W=whit["W"].to(device),
            W_inv=whit["W_inv"].to(device),
        ).to(device).eval()
        for p in stage1.parameters():
            p.requires_grad = False

    # Probe actual latent channel count through Stage1Wrapper forward.
    # This is robust when a 1-ch Pinaya VAE is wrapped to process 4-ch inputs
    # channel-wise (effective latent channels become 4 * base_latent_channels).
    stage1_latent_ch = None
    try:
        probe_item = dict(data_split[indices[0]])
        probe_has_label = "label" in probe_item and probe_item["label"]
        probe_transform = _build_val_transform(
            channel_reorder=not args.no_channel_reorder,
            has_label=probe_has_label,
        )
        probe_batch = probe_transform(probe_item)
        probe_image = probe_batch["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probe_z = stage1(probe_image)
        stage1_latent_ch = int(probe_z.shape[1])
    except Exception:
        stage1_latent_ch = None

    if stage1_latent_ch is None and hasattr(stage1, "model") and hasattr(stage1.model, "latent_channels"):
        stage1_latent_ch = int(stage1.model.latent_channels)
    elif stage1_latent_ch is None and hasattr(stage1, "model") and hasattr(stage1.model, "quant_conv_mu"):
        stage1_latent_ch = int(stage1.model.quant_conv_mu.out_channels)

    if stage1_latent_ch is None:
        stage1_latent_ch = int(config.get("model", {}).get("latent_channels", 3))

    num_mask_classes = int(config.get("mask", {}).get("num_classes", 4))
    model_cfg = config.setdefault("model", {})
    params = model_cfg.setdefault("params", {})
    model_cfg["latent_channels"] = stage1_latent_ch
    params["in_channels"] = stage1_latent_ch + num_mask_classes
    params["out_channels"] = stage1_latent_ch

    model = get_model(model_cfg.get("name", "DiffusionModelUNet"), config)
    checkpoint = torch.load(args.model_ckpt, map_location="cpu")
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()

    scheduler_name = config.get("scheduler", {}).get("name", "DDIMScheduler")
    scheduler_params = config.get("scheduler", {}).get("params", {})
    if scheduler_name == "DDIMScheduler":
        from generative.networks.schedulers import DDIMScheduler

        scheduler = DDIMScheduler(**scheduler_params)
    elif scheduler_name == "DDPMScheduler":
        from generative.networks.schedulers import DDPMScheduler

        scheduler = DDPMScheduler(**scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"], cache_dir=args.cache_dir, local_files_only=True
    )
    cfg_max_len = config["conditioning"].get("max_length")
    if cfg_max_len is not None:
        tokenizer.model_max_length = cfg_max_len
    text_encoder = text_encoder.to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    for case_i in tqdm(indices, desc="Offline sampling"):
        item = dict(data_split[case_i])
        has_label = "label" in item and item["label"]
        transform = _build_val_transform(channel_reorder=not args.no_channel_reorder, has_label=has_label)
        batch = transform(item)

        _suffix = f"_{args.output_suffix}" if args.output_suffix else ""

        image = batch["image"].unsqueeze(0).to(device)
        label = batch["label"].unsqueeze(0).to(device) if has_label else None

        prompt = item.get(args.text_field)
        if prompt is None:
            fallback = "findings" if args.text_field == "impression" else "impression"
            prompt = item.get(fallback, "")
        prompt = str(prompt)
        if args.custom_prompt is not None:
            prompt = str(args.custom_prompt)

        with torch.no_grad():
            z = stage1(image)
            if args.latent_whitening_path is not None:
                # Whitening enforces unit-variance channels; keep sf = 1.0.
                scale_factor = 1.0
            elif args.scale_factor is None:
                scale_factor = 1.0 / max(z.std().item(), 1e-8)
            else:
                scale_factor = float(args.scale_factor)

            latent_spatial = z.shape[2:]
            latent0 = torch.randn((1, stage1_latent_ch) + latent_spatial, device=device)

            if label is not None:
                mask_cond = prepare_mask_conditioning(
                    labels=label,
                    latent_shape=latent_spatial,
                    num_classes=num_mask_classes,
                    dropout_p=0.0,
                ).to(device)
            else:
                mask_cond = torch.zeros((1, num_mask_classes) + latent_spatial, device=device)
            if args.drop_mask:
                mask_cond = torch.zeros((1, num_mask_classes) + latent_spatial, device=device)
            mask_uncond = torch.zeros_like(mask_cond)

            cond_embeds = _encode_text(tokenizer, text_encoder, prompt, device)
            uncond_embeds = _encode_text(tokenizer, text_encoder, "", device)

            scheduler.set_timesteps(min(args.steps, scheduler.num_train_timesteps))
            _smooth_each_step = args.latent_smooth_mode == "each_step"
            # Compute the unconditional trajectory FIRST so its output cannot
            # depend on --cfg_scale. Non-deterministic CUDA kernels (flash
            # attention, cuDNN algorithm selection) and allocator/workspace
            # reuse mean that running latent_cond first perturbs the model's
            # numerical behaviour on the subsequent uncond forward, even after
            # scheduler.set_timesteps and torch.manual_seed resets.
            torch.manual_seed(args.seed)
            latent_uncond = _sample_latent(
                model, scheduler, latent0, mask_uncond, uncond_embeds, device,
                smooth_sigma=float(args.latent_smooth_sigma),
                smooth_each_step=_smooth_each_step,
            )
            scheduler.set_timesteps(min(args.steps, scheduler.num_train_timesteps))
            torch.manual_seed(args.seed)
            latent_cond = _sample_latent(
                model,
                scheduler,
                latent0,
                mask_cond,
                cond_embeds,
                device,
                uncond_embeds=uncond_embeds,
                uncond_mask_cond=mask_uncond,
                guidance_scale=float(args.cfg_scale),
                cfg_mode=str(args.cfg_mode),
                smooth_sigma=float(args.latent_smooth_sigma),
                smooth_each_step=_smooth_each_step,
            )

            x_cond = stage1.decode(latent_cond / scale_factor).float().clamp(0.0, 1.0)
            x_uncond = stage1.decode(latent_uncond / scale_factor).float().clamp(0.0, 1.0)

            image_for_ssim = _resize_to_common_space(image, common_ssim_size)
            x_cond_for_ssim = _resize_to_common_space(x_cond, common_ssim_size)
            x_uncond_for_ssim = _resize_to_common_space(x_uncond, common_ssim_size)

            ssim_cond_per_ch, ssim_cond_mean = _compute_per_channel_ssim(x_cond_for_ssim, image_for_ssim, MODALITY_NAMES)
            ssim_uncond_per_ch, ssim_uncond_mean = _compute_per_channel_ssim(x_uncond_for_ssim, image_for_ssim, MODALITY_NAMES)

        subj = item.get("subject_id", f"idx_{case_i}")
        cond_ssim_text = ", ".join(f"{k}={v:.4f}" for k, v in ssim_cond_per_ch.items())
        uncond_ssim_text = ", ".join(f"{k}={v:.4f}" for k, v in ssim_uncond_per_ch.items())
        print(f"[{subj}] SSIM conditioned mean={ssim_cond_mean:.4f} | {cond_ssim_text}")
        print(f"[{subj}] SSIM unconditioned mean={ssim_uncond_mean:.4f} | {uncond_ssim_text}")

        depth = x_cond.shape[-1]
        depth_indices = [depth // 4, depth // 2, (3 * depth) // 4]
        grid_cond = _make_grid(x_cond, depth_indices)
        grid_uncond = _make_grid(x_uncond, depth_indices)

        n_ch = x_cond.shape[1]
        grid_mask = None
        if label is not None:
            _label_np = label[0, 0].detach().cpu().numpy()
            _mask_rows = []
            for _d in depth_indices:
                _d_safe = max(0, min(int(_d), _label_np.shape[-1] - 1))
                _sl = _label_np[:, :, _d_safe]
                _mask_rows.append(np.tile(_sl, (1, n_ch)))
            _grid_mask_raw = np.concatenate(_mask_rows, axis=0)
            grid_mask = np.ma.masked_where(_grid_mask_raw == 0, _grid_mask_raw)

        n_rows = 3 if label is not None else 2
        fig, axes = plt.subplots(n_rows, 1, dpi=300, figsize=(10, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]

        row = 0
        if label is not None:
            label_vis = label[0, 0].detach().cpu().numpy()
            mask_depth = max(0, min(depth_indices[1], label_vis.shape[-1] - 1))
            axes[row].imshow(label_vis[:, :, mask_depth], cmap="tab20", vmin=0, vmax=max(num_mask_classes - 1, 1))
            axes[row].set_title("Input Mask (middle depth)")
            axes[row].axis("off")
            row += 1

        axes[row].imshow(grid_cond, cmap="gray")
        if grid_mask is not None:
            axes[row].imshow(
                grid_mask,
                cmap="tab20",
                vmin=0,
                vmax=max(num_mask_classes - 1, 1),
                alpha=0.2,
                interpolation="nearest",
            )
        axes[row].set_title("Conditioned Sample (3 depths, mask overlay α=0.2)")
        axes[row].axis("off")
        row += 1

        axes[row].imshow(grid_uncond, cmap="gray")
        axes[row].set_title("Unconditioned Sample (3 depths)")
        axes[row].axis("off")

        for ax in axes[1:] if label is not None else axes:
            n_ch = x_cond.shape[1]
            w_per_ch = grid_cond.shape[1] / n_ch
            for c in range(n_ch):
                name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}"
                ax.text(int(w_per_ch * c) + 2, 10, name, fontsize=6, color="yellow")

        _header = f"idx={case_i} | scale_factor={scale_factor:.4f} | cfg_scale={args.cfg_scale:.2f} | cfg_mode={args.cfg_mode} | prompt="
        _ssim_summary = f"SSIM mean: cond={ssim_cond_mean:.4f}, uncond={ssim_uncond_mean:.4f}"
        _wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=120))
        _n_prompt_lines = len(textwrap.wrap(prompt, width=120)) + 2
        fig.suptitle(
            _ssim_summary + "\n" + _header + "\n" + _wrapped_prompt,
            fontsize=7,
            ha="left",
            x=0.01,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0, 1, max(0.0, 1 - 0.018 * _n_prompt_lines)])

        out_png = output_dir / f"sample_compare_{case_i:04d}{_suffix}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        out_nifti_cond = output_dir / f"sample_cond_{case_i:04d}{_suffix}.nii.gz"
        out_nifti_uncond = output_dir / f"sample_uncond_{case_i:04d}{_suffix}.nii.gz"
        _save_sample_nifti(x_cond, item["image"], out_nifti_cond)
        _save_sample_nifti(x_uncond, item["image"], out_nifti_uncond)

        proc_affine = _get_tensor_affine(batch["image"])
        out_nifti_orig_proc = output_dir / f"sample_original_processed_{case_i:04d}{_suffix}.nii.gz"
        out_nifti_cond_native = output_dir / f"sample_cond_native_{case_i:04d}{_suffix}.nii.gz"
        out_nifti_uncond_native = output_dir / f"sample_uncond_native_{case_i:04d}{_suffix}.nii.gz"
        _save_tensor_nifti_native(image, out_nifti_orig_proc, proc_affine)
        _save_tensor_nifti_native(x_cond, out_nifti_cond_native, proc_affine)
        _save_tensor_nifti_native(x_uncond, out_nifti_uncond_native, proc_affine)

    print(f"Saved {len(indices)} comparison panels to {output_dir}")


if __name__ == "__main__":
    main()
