#!/usr/bin/env python
"""Zero-shot inference with the Pinaya 2022 pretrained VAE on BraTS modalities.

Downloads the pretrained AutoencoderKL weights from the MONAI Model Zoo
(brain_image_synthesis_latent_diffusion_model) and runs encode→decode on
existing multi-channel BraTS samples from the local runs folder.

The Pinaya VAE was trained on single-channel T1w UK Biobank data at
160×224×160.  Our BraTS originals are saved as 4-channel NIfTI with
shape (150, 214, 150, 4) after the NiftiSaver border crop, scaled to
[0, 255].  We:
  1. Pad back to 160×224×160 (zero-pad the cropped borders)
  2. Rescale to [0, 1]
  3. Run each channel independently through the Pinaya VAE
  4. Save reconstructions as NIfTI via the project's NiftiSaver
  5. Compute per-channel SSIM, PSNR, L1 and log to CSV

Usage (local, MPS):
    python scripts/pinaya_vae_zero_shot.py \
        --input_dir /Users/nk233/mhf/projects/text2glioma/runs/bf16_adaptive_v2_pw_1.0_r1_1.0_adv_weight_0.1/autoencoder_stage1/samples \
        --epoch 305 --n_samples 5

The script auto-detects MPS / CUDA / CPU.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pinaya VAE architecture (MONAI Model Zoo)
# ---------------------------------------------------------------------------
PINAYA_AUTOENCODER_CONFIG = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=3,
    num_channels=[64, 128, 128, 128],
    num_res_blocks=2,
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)

PINAYA_WEIGHTS_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1CZHwxHJWybOsDavipD0EorDPOo_mzNeX"
)

MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]
# Spatial size the Pinaya VAE expects (same as our training pipeline)
FULL_SPATIAL = (160, 224, 160)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_pinaya_weights(cache_dir: pathlib.Path) -> pathlib.Path:
    """Download autoencoder.pth from Google Drive via gdown."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "pinaya_autoencoder.pth"
    if out_path.exists():
        print(f"  Cached weights found: {out_path}")
        return out_path

    import gdown
    print(f"  Downloading Pinaya VAE weights → {out_path}")
    gdown.download(
        PINAYA_WEIGHTS_URL,
        str(out_path),
        quiet=False,
    )
    if not out_path.exists():
        raise RuntimeError("Download failed – check network / gdown.")
    return out_path


def load_pinaya_vae(weights_path: pathlib.Path, device: torch.device):
    """Instantiate the Pinaya AutoencoderKL and load pretrained weights."""
    from generative.networks.nets import AutoencoderKL

    model = AutoencoderKL(**PINAYA_AUTOENCODER_CONFIG)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"  Pinaya VAE loaded on {device}  "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


# ---------------------------------------------------------------------------
# I/O – load originals from the NiftiSaver format
# ---------------------------------------------------------------------------

def load_original(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 4-channel original NIfTI saved by NiftiSaver.

    Returns
    -------
    volume : (4, D, H, W)  float32 in [0, 1]
    affine : (4, 4)
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)  # (D, H, W, 4)
    # Rescale [0,255] → [0,1]
    data = data / 255.0
    # Channel last → channel first  (4, D, H, W)
    data = np.transpose(data, (3, 0, 1, 2))
    return data, img.affine


def pad_to_full(vol_chw: np.ndarray) -> torch.Tensor:
    """Pad a (C, D', H', W') array back to (C, 160, 224, 160).

    NiftiSaver crops:  data[:, 5:-5, 5:-5, :-10]
    so D'=150, H'=214, W'=150 → need +5 on each D side, +5 each H side,
    +0 front +10 back on W.
    """
    t = torch.from_numpy(vol_chw)  # (C, 150, 214, 150)
    # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
    t = F.pad(t, (0, 10, 5, 5, 5, 5), mode="constant", value=0.0)
    return t  # (C, 160, 224, 160)


def crop_from_full(vol: torch.Tensor) -> torch.Tensor:
    """Re-apply the NiftiSaver border crop: [:, 5:-5, 5:-5, :-10]."""
    return vol[:, 5:-5, 5:-5, :-10]


# ---------------------------------------------------------------------------
# Metrics (numpy, per-channel)
# ---------------------------------------------------------------------------

def _ssim_3d(a: np.ndarray, b: np.ndarray,
             data_range: float = 1.0) -> float:
    """Structural similarity for a pair of 3-D volumes."""
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=data_range)


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """Compute L1, PSNR, SSIM between two (D, H, W) volumes in [0,1]."""
    return {
        "l1": float(np.mean(np.abs(orig - recon))),
        "psnr": _psnr(orig, recon),
        "ssim": _ssim_3d(orig, recon),
    }


# ---------------------------------------------------------------------------
# NIfTI saver (simplified, reuses project affine)
# ---------------------------------------------------------------------------

DEFAULT_AFFINE = np.array([
    [-1., 0., 0., 96.48149872],
    [0., 1., 0., -141.47715759],
    [0., 0., 1., -156.55375671],
    [0., 0., 0., 1.],
])


def save_nifti(data: np.ndarray, path: pathlib.Path,
               affine: np.ndarray = None) -> None:
    """Save a (D, H, W) or (D, H, W, C) float32 volume as NIfTI."""
    if affine is None:
        affine = DEFAULT_AFFINE
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    hdr = img.header
    hdr.set_xyzt_units("mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_original_files(input_dir: pathlib.Path, epoch: int,
                        n_samples: int) -> List[pathlib.Path]:
    """Find ep{epoch}_sample{00..n}_original.nii.gz files."""
    files = []
    for i in range(n_samples):
        name = f"ep{epoch}_sample{i:02d}_original.nii.gz"
        p = input_dir / name
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping.")
            continue
        files.append(p)
    return files


@torch.no_grad()
def run_inference(
    model,
    vol_full: torch.Tensor,       # (4, 160, 224, 160)  float32 [0,1]
    device: torch.device,
) -> torch.Tensor:
    """Run each channel through the Pinaya VAE independently.

    Returns (4, 160, 224, 160) float32 [0, 1] (clamped).
    """
    recons = []
    for ch in range(vol_full.shape[0]):
        x = vol_full[ch:ch+1].unsqueeze(0).to(device)  # (1, 1, D, H, W)
        z_mu, z_sigma = model.encode(x)
        z = model.sampling(z_mu, z_sigma)
        r = model.decode(z)                             # (1, 1, D, H, W)
        r = r.clamp(0.0, 1.0).cpu().squeeze(0)          # (1, D, H, W)
        recons.append(r)
    return torch.cat(recons, dim=0)  # (4, D, H, W)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Pinaya VAE inference on BraTS modalities")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing ep*_sample*_original.nii.gz files")
    parser.add_argument(
        "--epoch", type=int, default=305,
        help="Epoch prefix to select (default: 305)")
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of samples to process (default: 5)")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <input_dir>/../pinaya_zero_shot)")
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Where to cache downloaded weights (default: ~/.cache/text2glioma)")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = input_dir.parent / "pinaya_zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = pathlib.Path(
        args.cache_dir or pathlib.Path.home() / ".cache" / "text2glioma"
    )

    device = pick_device()
    print(f"Device: {device}")

    # ── Download & load Pinaya VAE ──────────────────────────────────
    weights_path = download_pinaya_weights(cache_dir)
    model = load_pinaya_vae(weights_path, device)

    # ── Find input files ────────────────────────────────────────────
    files = find_original_files(input_dir, args.epoch, args.n_samples)
    if not files:
        print("ERROR: No original files found.")
        sys.exit(1)
    print(f"\nProcessing {len(files)} samples from epoch {args.epoch}")
    print(f"Output → {output_dir}\n")

    # ── Run inference ───────────────────────────────────────────────
    csv_path = output_dir / "metrics.csv"
    csv_rows: List[Dict] = []

    for fpath in files:
        sample_name = fpath.stem.replace("_original.nii", "")
        print(f"─── {sample_name} ───")

        # Load & prep
        orig_np, affine = load_original(fpath)          # (4, 150, 214, 150)
        vol_full = pad_to_full(orig_np)                  # (4, 160, 224, 160)

        # Inference
        t0 = time.time()
        recon_full = run_inference(model, vol_full, device)  # (4, 160, 224, 160)
        elapsed = time.time() - t0
        print(f"  Inference: {elapsed:.1f}s")

        # Crop back to NiftiSaver dims for fair comparison
        recon_crop = crop_from_full(recon_full)          # (4, 150, 214, 150)
        orig_crop = orig_np                              # already cropped

        # Save per-channel NIfTI + compute metrics
        row = {"sample": sample_name}
        for ch, mod in enumerate(MODALITY_NAMES):
            # Save reconstruction
            recon_ch = recon_crop[ch].numpy()            # (D, H, W)
            orig_ch = orig_crop[ch]                      # (D, H, W)

            out_name = f"{sample_name}_pinaya_{mod}.nii.gz"
            save_nifti(recon_ch, output_dir / out_name, affine)

            # Metrics
            m = compute_metrics(orig_ch, recon_ch)
            row[f"{mod}_L1"] = f"{m['l1']:.5f}"
            row[f"{mod}_PSNR"] = f"{m['psnr']:.2f}"
            row[f"{mod}_SSIM"] = f"{m['ssim']:.4f}"
            print(f"  {mod:5s}  L1={m['l1']:.5f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

        csv_rows.append(row)

        # Also save the full 4-channel reconstruction as a single NIfTI
        recon_4ch = np.transpose(recon_crop.numpy(), (1, 2, 3, 0))  # (D,H,W,4)
        save_nifti(recon_4ch, output_dir / f"{sample_name}_pinaya_recon.nii.gz", affine)

        # Save original copy for side-by-side viewing
        orig_4ch = np.transpose(orig_crop, (1, 2, 3, 0))  # (D,H,W,4)
        save_nifti(orig_4ch * 255.0, output_dir / f"{sample_name}_original_copy.nii.gz", affine)

    # ── Write CSV ───────────────────────────────────────────────────
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nMetrics saved to {csv_path}")

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY  (mean across samples)")
    print("=" * 60)
    for mod in MODALITY_NAMES:
        l1s = [float(r[f"{mod}_L1"]) for r in csv_rows]
        psnrs = [float(r[f"{mod}_PSNR"]) for r in csv_rows]
        ssims = [float(r[f"{mod}_SSIM"]) for r in csv_rows]
        print(f"  {mod:5s}  L1={np.mean(l1s):.5f}±{np.std(l1s):.5f}  "
              f"PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}  "
              f"SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}")
    print()


if __name__ == "__main__":
    main()
