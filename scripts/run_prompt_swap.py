"""CLI driver for the prompt-swap ablation.

Three subcommands, matching the pipeline stages in
``text2glioma.validation.prompt_swap``:

    build       Select contrasting subject pairs and write pairs.csv.
    generate    Sample native (mask_i, prompt_i) and swap (mask_i, prompt_j)
                4-channel NIfTI volumes for every pair.
    analyse     Given round-trip nnU-Net segmentations of both, compute
                per-attribute shift statistics.

External nnU-Net segmentation is not run in-tree — after ``generate`` finishes,
run your existing nnU-Net predictor over ``<out>/native/`` and ``<out>/swap/``
and drop the predicted labels into ``<out>/seg_native/`` and ``<out>/seg_swap/``
before invoking ``analyse``.

Example
-------

    # 1. Build 100 pairs contrasting on laterality:
    python scripts/run_prompt_swap.py build \\
        --datalist datalist_N1510_val3.json \\
        --atlas_dir src/text2glioma/preprocessing/atlas_masks/sri24 \\
        --target laterality --n_pairs 100 \\
        --out runs/prompt_swap/laterality/

    # 2. Generate images (native + swap) per pair:
    python scripts/run_prompt_swap.py generate \\
        --pairs runs/prompt_swap/laterality/pairs.csv \\
        --config configs/ldm.yaml --stage1_config configs/stage1.yaml \\
        --stage1_uri runs/.../final_model.pth --model_ckpt runs/.../best_model.pth \\
        --out runs/prompt_swap/laterality/

    # 3. [external] nnUNetv2_predict → runs/prompt_swap/laterality/seg_{native,swap}/

    # 4. Run vasari-auto + shift analysis:
    python scripts/run_prompt_swap.py analyse \\
        --pairs runs/prompt_swap/laterality/pairs.csv \\
        --seg_native runs/prompt_swap/laterality/seg_native \\
        --seg_swap runs/prompt_swap/laterality/seg_swap \\
        --atlas_dir src/text2glioma/preprocessing/atlas_masks/sri24 \\
        --out runs/prompt_swap/laterality/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from text2glioma.validation.prompt_swap import (
    build_swap_pairs,
    save_pairs,
    load_pairs,
    analyse_swap_recovery,
    summarise_shift,
    CATEGORICAL_TARGETS,
    ORDINAL_TARGETS,
)


def _flatten_datalist(dl):
    if isinstance(dl, dict):
        return dl.get("validation", []) + dl.get("test", []) + dl.get("training", [])
    return dl


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def cmd_build(args):
    with open(args.datalist) as f:
        data = _flatten_datalist(json.load(f))
    if args.max_subjects:
        data = data[: args.max_subjects]

    pairs = build_swap_pairs(
        datalist=data,
        target=args.target,
        n_pairs=args.n_pairs,
        text_field=args.text_field,
        label_field=args.label_field,
        seed=args.seed,
        min_ordinal_gap=args.min_ordinal_gap,
        atlas_dir=args.atlas_dir,
        use_vasari_auto=args.use_vasari_auto,
    )
    out = Path(args.out)
    save_pairs(pairs, out / "pairs.csv")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _load_generation_stack(args):
    """Reproduce the model / sampler build used in text2glioma.inference.sampler."""
    from generative.networks.schedulers import DDIMScheduler
    from text2glioma.utils import (
        get_model,
        load_config,
        stage1_ify,
        load_text_encoder_and_tokenizer,
    )
    from text2glioma.inference.inference_functions import GenericSampler

    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    stage1 = stage1_ify(get_model("AutoencoderKL", stage1_config, from_file=args.stage1_uri)).eval()

    model = get_model(config["model"].get("name", "DiffusionModelUNet"), config)
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"], cache_dir=args.cache_dir, local_files_only=True
    )
    cfg_max_len = config["conditioning"].get("max_length")
    if cfg_max_len is not None:
        tokenizer.model_max_length = cfg_max_len

    scheduler = DDIMScheduler(**config["scheduler"].get("params", {}))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stage1.to(device); model.to(device); text_encoder.to(device)

    latent_ch = config.get("model", {}).get("latent_channels", 3)
    mask_spatial = tuple(config.get("mask", {}).get("spatial_size", (160, 224, 160)))
    stage1_levels = len(stage1_config["model"]["params"]["num_channels"]) - 1
    downsample_factor = 2 ** stage1_levels
    explicit_latent = config.get("sampling", {}).get("latent_shape")
    if explicit_latent is not None:
        latent_spatial = tuple(explicit_latent)
    else:
        latent_spatial = tuple(s // downsample_factor for s in mask_spatial)
    num_mask_classes = config.get("mask", {}).get("num_classes", 4)
    print(f"[generate] mask spatial={mask_spatial}  AE factor={downsample_factor}  "
          f"latent_shape=({latent_ch},)+{latent_spatial}")

    sampler = GenericSampler(
        stage1=stage1,
        model=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        num_mask_classes=num_mask_classes,
        latent_channels=latent_ch,
        scale_factor=args.scale_factor,
    )
    return sampler, (latent_ch,) + latent_spatial, device


def _load_mask(mask_path, spatial_size, device):
    arr = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    if tuple(t.shape[-3:]) != tuple(spatial_size):
        t = torch.nn.functional.interpolate(t, size=spatial_size, mode="nearest")
    return t.to(device)


def _save_4ch(images: torch.Tensor, out_path: Path, affine=None):
    """Save a [1, 4, D, H, W] tensor as a (D, H, W, 4) NIfTI matching the release format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = images[0].detach().cpu().numpy()  # [4, D, H, W]
    vol = np.moveaxis(vol, 0, -1).astype(np.float32)  # [D, H, W, 4]
    affine = affine if affine is not None else np.eye(4)
    nib.save(nib.Nifti1Image(vol, affine), str(out_path))


def cmd_generate(args):
    pairs = load_pairs(args.pairs)
    sampler, latent_shape, device = _load_generation_stack(args)

    out = Path(args.out)
    (out / "native").mkdir(parents=True, exist_ok=True)
    (out / "swap").mkdir(parents=True, exist_ok=True)

    spatial_size = tuple(args.spatial_size)
    manifest_rows = []

    for pair in pairs:
        mask = _load_mask(pair.mask_i, spatial_size, device)

        for tag, prompt in (("native", pair.prompt_i), ("swap", pair.prompt_j)):
            torch.manual_seed(int(pair.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(pair.seed))
            sampler.scheduler.set_timesteps(args.ddim_steps)

            images = sampler.sample(
                steps=args.ddim_steps,
                batch_size=1,
                latent_shape=latent_shape,
                texts=[prompt],
                masks=mask,
                guidance_scale_text=args.guidance_scale_text,
                guidance_scale_mask=args.guidance_scale_mask,
                eta=0.0,
                verbose=False,
                decode_amp_dtype=torch.float16 if args.decode_fp16 else None,
                offload_diffusion_during_decode=args.offload_during_decode,
            )
            out_path = out / tag / f"{pair.pair_id}.nii.gz"
            _save_4ch(images, out_path)

        manifest_rows.append(
            {
                "pair_id": pair.pair_id,
                "target": pair.target,
                "subj_i": pair.subj_i,
                "subj_j": pair.subj_j,
                "mask_i": pair.mask_i,
                "native_image": str(out / "native" / f"{pair.pair_id}.nii.gz"),
                "swap_image": str(out / "swap" / f"{pair.pair_id}.nii.gz"),
                "seed": int(pair.seed),
                "cfg_text": args.guidance_scale_text,
                "cfg_mask": args.guidance_scale_mask,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "generation_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def _pad_or_crop_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center pad or crop a 3D volume to ``target_shape`` (no resampling)."""
    out = arr
    for axis, (cur, tgt) in enumerate(zip(out.shape, target_shape)):
        if cur == tgt:
            continue
        if cur < tgt:
            pad_total = tgt - cur
            before = pad_total // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, pad_total - before)
            out = np.pad(out, pad, mode="constant", constant_values=0)
        else:
            crop_total = cur - tgt
            before = crop_total // 2
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(before, before + tgt)
            out = out[tuple(slicer)]
    return out


def _atlas_reference(atlas_dir: str):
    ref = nib.load(str(Path(atlas_dir) / "brainstem.nii.gz"))
    return tuple(np.asarray(ref.dataobj).shape), ref.affine


def _align_seg_to_atlas(seg_path: str, atlas_shape, atlas_affine, out_dir: Path) -> str:
    nii = nib.load(seg_path)
    arr = np.asarray(nii.dataobj)
    if tuple(arr.shape) == tuple(atlas_shape):
        return seg_path
    aligned = _pad_or_crop_to(arr, atlas_shape).astype(np.int16)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(seg_path).name
    nib.save(nib.Nifti1Image(aligned, atlas_affine), str(out_path))
    return str(out_path)


def _extract_vasari_from_dir(
    pair_ids, seg_dir, atlas_dir, enhancing_label, nonenhancing_label, oedema_label,
    min_lesion_voxels: int = 10,
):
    """Return a DataFrame indexed by pair_id, one row per successfully-segmented pair.

    Pairs with fewer than ``min_lesion_voxels`` foreground voxels or where
    vasari-auto raises are silently skipped and reported at the end.
    """
    from text2glioma.preprocessing.vasari_auto import get_vasari_features

    atlas_dir_str = atlas_dir.rstrip("/") + "/"
    atlas_shape, atlas_affine = _atlas_reference(atlas_dir)
    seg_dir = Path(seg_dir)
    aligned_dir = seg_dir.parent / f"{seg_dir.name}_atlasspace"
    rows = {}
    skipped_empty: list[str] = []
    skipped_error: list[tuple[str, str]] = []
    for pid in pair_ids:
        candidates = list(seg_dir.glob(f"{pid}*.nii*"))
        if not candidates:
            continue
        raw_arr = np.asarray(nib.load(str(candidates[0])).dataobj)
        if int((raw_arr > 0).sum()) < min_lesion_voxels:
            skipped_empty.append(pid)
            continue
        aligned = _align_seg_to_atlas(
            str(candidates[0]), atlas_shape, atlas_affine, aligned_dir,
        )
        try:
            df = get_vasari_features(
                file=aligned,
                atlases=atlas_dir_str,
                enhancing_label=enhancing_label,
                nonenhancing_label=nonenhancing_label,
                oedema_label=oedema_label,
                verbose=False,
            )
        except (UnboundLocalError, ZeroDivisionError, ValueError, KeyError) as e:
            skipped_error.append((pid, f"{type(e).__name__}: {e}"))
            continue
        rows[pid] = df.iloc[0].to_dict() if isinstance(df, pd.DataFrame) else dict(df)

    if skipped_empty:
        logging.warning(
            "vasari extraction skipped %d empty segmentations in %s: %s",
            len(skipped_empty), seg_dir.name, skipped_empty[:10],
        )
    if skipped_error:
        logging.warning(
            "vasari extraction failed on %d cases in %s: %s",
            len(skipped_error), seg_dir.name, skipped_error[:5],
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def cmd_analyse(args):
    pairs = load_pairs(args.pairs)
    pair_ids = [p.pair_id for p in pairs]

    native_v = _extract_vasari_from_dir(
        pair_ids, args.seg_native, args.atlas_dir,
        args.enhancing_label, args.nonenhancing_label, args.oedema_label,
    )
    swap_v = _extract_vasari_from_dir(
        pair_ids, args.seg_swap, args.atlas_dir,
        args.enhancing_label, args.nonenhancing_label, args.oedema_label,
    )

    recovery = analyse_swap_recovery(pairs, native_v, swap_v, ordinal_tol=args.ordinal_tol)
    summary = summarise_shift(recovery)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recovery.to_csv(out / "recovery_per_pair.csv", index=False)
    summary.to_csv(out / "shift_summary.csv", index=False)

    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Select swap pairs.")
    b.add_argument("--datalist", required=True)
    b.add_argument("--atlas_dir", default=None,
                   help="Only required with --use_vasari_auto.")
    b.add_argument("--use_vasari_auto", action="store_true",
                   help="Re-extract VASARI from the segmentation labels via "
                        "vasari-auto instead of parsing the impression field.")
    b.add_argument(
        "--target", required=True,
        choices=list(CATEGORICAL_TARGETS) + list(ORDINAL_TARGETS),
    )
    b.add_argument("--n_pairs", type=int, default=100)
    b.add_argument("--text_field", default="impression")
    b.add_argument("--label_field", default="label")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--min_ordinal_gap", type=int, default=2)
    b.add_argument("--max_subjects", type=int, default=None,
                   help="Cap the datalist for a quick smoke test.")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_build)

    g = sub.add_parser("generate", help="Generate native + swap NIfTI pairs.")
    g.add_argument("--pairs", required=True)
    g.add_argument("--config", required=True)
    g.add_argument("--stage1_config", required=True)
    g.add_argument("--stage1_uri", required=True)
    g.add_argument("--model_ckpt", required=True)
    g.add_argument("--cache_dir", default=None)
    g.add_argument("--device", default="cuda")
    g.add_argument("--guidance_scale_text", type=float, default=1.0)
    g.add_argument("--guidance_scale_mask", type=float, default=1.0)
    g.add_argument("--ddim_steps", type=int, default=50)
    g.add_argument("--spatial_size", type=int, nargs=3, default=[160, 224, 160])
    g.add_argument("--scale_factor", type=float, default=0.866,
                   help="Latent whitening factor; latents are divided by this "
                        "before AE decode. Default 0.866 matches the released "
                        "MaxFeat/RadBERT LDM.")
    g.add_argument("--decode_fp16", action="store_true",
                   help="Decode with fp16 autocast (V100-safe; halves AE activation VRAM).")
    g.add_argument("--offload_during_decode", action="store_true",
                   help="Move UNet + text encoder to CPU during AE decode to free VRAM.")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("analyse", help="Compute per-attribute shift metrics.")
    a.add_argument("--pairs", required=True)
    a.add_argument("--seg_native", required=True)
    a.add_argument("--seg_swap", required=True)
    a.add_argument("--atlas_dir", required=True)
    a.add_argument("--enhancing_label", type=int, default=3)
    a.add_argument("--nonenhancing_label", type=int, default=2)
    a.add_argument("--oedema_label", type=int, default=1)
    a.add_argument("--ordinal_tol", type=float, default=0.0)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_analyse)
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
