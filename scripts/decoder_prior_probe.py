#!/usr/bin/env python3
"""Decoder DC-prior probe for Stage-1 VAE.

Diagnoses whether the central T2-contrast "bright patch" artifact observed
in reconstructions is the decoder's intrinsic response to uninformative
latents (i.e. a learned template that emerges when the central latent
voxels are near the training mean).

Procedure
---------
1. Encode K training samples (and K val samples) → collect z_mu.
2. Compute mean latent ``mean_z`` (across all encoded samples).
3. Decode four conditions per sample:
     (a) ``z = 0``                              — pure decoder DC response
     (b) ``z = mean_z``                         — decoder response at training-mean latent
     (c) ``z = z_mu`` with central B³ block zeroed — does erasing center remove artifact?
     (d) ``z = z_mu``                           — baseline reconstruction
4. Save all four NIfTIs + the original input per sample.
5. Compute a center-vs-outer abs-intensity ratio diagnostic across channels
   for each condition; write to ``decoder_prior_probe.json``.

Usage
-----
    python scripts/decoder_prior_probe.py \
        --run_dir /g/data/vp06/$USER/text2glioma_train/runs/stage1_overfit_memsafe \
        --datalist /path/to/mini_datalist.json --no_channel_reorder \
        --n_samples 5 --center_block 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse transform & constants from offline eval script.
from offline_val_metrics import CH_NAMES, get_val_transform


def parse_args():
    p = argparse.ArgumentParser(description="Decoder DC-prior probe")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--datalist", type=str, required=True)
    p.add_argument("--no_channel_reorder", action="store_true", default=False)
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of train+val samples to probe (each split).")
    p.add_argument("--center_block", type=int, default=2,
                   help="Side length of the central latent block to zero out in cond (c).")
    p.add_argument("--center_frac", type=float, default=0.25,
                   help="Fraction of each spatial dim defining the 'center' region for the "
                        "abs-intensity ratio diagnostic (default 0.25 → central 25%).")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def load_model(run_dir: Path, device: torch.device):
    import yaml
    from text2glioma.utils import get_model

    ae_dir = run_dir / "autoencoder_stage1"
    config_path = ae_dir / "config_snapshot.yaml"
    ckpt_path = ae_dir / "checkpoint.pth"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = get_model(config["model"]["name"], config).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    clean = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in sd.items()}
    model.load_state_dict(clean)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded epoch {epoch} from {ckpt_path}")
    return model, config, epoch


def center_outer_ratio(vol: torch.Tensor, frac: float) -> dict:
    """Mean |intensity| in a centered cube vs the surrounding shell, per channel.

    vol: (C, D, H, W). Returns dict per channel and an aggregate.
    """
    c, d, h, w = vol.shape
    # Center crop bounds.
    cd, ch_, cw = int(d * frac), int(h * frac), int(w * frac)
    d0, h0, w0 = (d - cd) // 2, (h - ch_) // 2, (w - cw) // 2
    center = vol[:, d0:d0 + cd, h0:h0 + ch_, w0:w0 + cw].abs()
    full = vol.abs()
    full_sum = full.sum(dim=(1, 2, 3))
    full_count = float(d * h * w)
    center_sum = center.sum(dim=(1, 2, 3))
    center_count = float(cd * ch_ * cw)
    outer_sum = full_sum - center_sum
    outer_count = full_count - center_count
    mean_center = (center_sum / max(center_count, 1.0)).cpu().tolist()
    mean_outer = (outer_sum / max(outer_count, 1.0)).cpu().tolist()
    ratio = [
        (mc / mo) if mo > 1e-12 else float("inf")
        for mc, mo in zip(mean_center, mean_outer)
    ]
    return {
        "per_channel_mean_center": dict(zip(CH_NAMES, mean_center)),
        "per_channel_mean_outer": dict(zip(CH_NAMES, mean_outer)),
        "per_channel_ratio": dict(zip(CH_NAMES, ratio)),
        "mean_ratio": float(sum(ratio) / len(ratio)),
    }


def probe_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    epoch,
    saver,
    out_dir: Path,
    center_block: int,
    center_frac: float,
):
    z_mu_list = []
    inputs = []
    print(f"\n[{split_name}] Encoding samples...")
    with torch.no_grad():
        for x in tqdm(loader, desc=f"encode-{split_name}"):
            images = x["image"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z_mu, z_sigma = model.encode(images)
            z_mu_list.append(z_mu.float().cpu())
            inputs.append(images.float().cpu())

    z_mu_all = torch.cat(z_mu_list, dim=0)  # (N, C, D, H, W)
    inputs_all = torch.cat(inputs, dim=0)
    N, Cz, Dz, Hz, Wz = z_mu_all.shape
    mean_z = z_mu_all.mean(dim=0, keepdim=True)  # (1, C, D, H, W)
    print(f"[{split_name}] z_mu shape: {tuple(z_mu_all.shape)}; mean_z abs mean = {mean_z.abs().mean().item():.4e}")

    # Central block bounds.
    b = max(1, center_block)
    d0 = (Dz - b) // 2
    h0 = (Hz - b) // 2
    w0 = (Wz - b) // 2
    print(f"[{split_name}] Zeroing central block of size {b}^3 at [{d0}:{d0+b}, {h0}:{h0+b}, {w0}:{w0+b}]")

    metrics = []
    for i in range(N):
        z_baseline = z_mu_all[i:i+1].to(device)
        z_zero = torch.zeros_like(z_baseline)
        z_mean = mean_z.to(device).expand_as(z_baseline).clone()
        z_centerzero = z_baseline.clone()
        z_centerzero[:, :, d0:d0+b, h0:h0+b, w0:w0+b] = 0.0

        conds = {
            "a_zero": z_zero,
            "b_mean": z_mean,
            "c_centerzero": z_centerzero,
            "d_baseline": z_baseline,
        }

        sample_metrics = {"sample_idx": i, "split": split_name}
        with torch.no_grad():
            for name, z in conds.items():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    rec = model.decode(z)
                rec_f = rec.float().squeeze(0).cpu()
                saver.save(
                    rec_f,
                    f"ep{epoch}_{split_name}_sample{i:02d}_{name}.nii.gz",
                )
                sample_metrics[name] = center_outer_ratio(rec_f, center_frac)

        # Save original input too.
        saver.save(
            inputs_all[i],
            f"ep{epoch}_{split_name}_sample{i:02d}_input.nii.gz",
        )
        sample_metrics["input"] = center_outer_ratio(inputs_all[i], center_frac)
        metrics.append(sample_metrics)

    return {
        "split": split_name,
        "n_samples": N,
        "latent_shape": list(z_mu_all.shape),
        "mean_z_abs_mean": float(mean_z.abs().mean().item()),
        "center_block": b,
        "per_sample": metrics,
    }


def summarize(split_report: dict) -> dict:
    """Aggregate per-sample mean_ratio per condition."""
    conds = ["a_zero", "b_mean", "c_centerzero", "d_baseline", "input"]
    agg = {}
    for c in conds:
        vals = [s[c]["mean_ratio"] for s in split_report["per_sample"]]
        agg[c] = sum(vals) / len(vals) if vals else float("nan")
    return agg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    model, config, epoch = load_model(run_dir, device)

    channel_reorder = not args.no_channel_reorder
    xform = get_val_transform(channel_reorder)

    with open(args.datalist) as f:
        datalist = json.load(f)
    train_data = datalist["training"][:args.n_samples]
    val_data = datalist["validation"][:args.n_samples]

    train_ds = Dataset(data=train_data, transform=xform)
    val_ds = Dataset(data=val_data, transform=xform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    from text2glioma.inference.saver import NiftiSaver
    out_dir = run_dir / "autoencoder_stage1" / "decoder_prior_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = NiftiSaver(output_dir=str(out_dir), rescale=False)
    print(f"Writing probe outputs to {out_dir}")

    train_report = probe_split(
        model, train_loader, device, "train", epoch, saver, out_dir,
        args.center_block, args.center_frac,
    )
    val_report = probe_split(
        model, val_loader, device, "val", epoch, saver, out_dir,
        args.center_block, args.center_frac,
    )

    report = {
        "run_dir": str(run_dir),
        "epoch": epoch,
        "center_frac": args.center_frac,
        "center_block": args.center_block,
        "train": train_report,
        "val": val_report,
        "summary_center_outer_ratio_mean_over_samples": {
            "train": summarize(train_report),
            "val": summarize(val_report),
        },
    }
    out_json = run_dir / "autoencoder_stage1" / "decoder_prior_probe.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")

    print("\nCenter/outer abs-intensity ratio (mean over samples):")
    for split in ("train", "val"):
        print(f"  [{split}]")
        for cond, val in report["summary_center_outer_ratio_mean_over_samples"][split].items():
            print(f"    {cond:>14s}: {val:.4f}")


if __name__ == "__main__":
    main()
