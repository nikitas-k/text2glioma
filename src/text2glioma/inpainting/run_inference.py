"""Inference + SSIM evaluation for the inpainting LDM.

Runs one of three tasks over the held-out test fold:

  - ``conditional``    (Task A) — sample image_b given the *true* trajectory
                                  + treatments; report SSIM vs real image_b.
  - ``sweep``          (Task B) — sample image_b for *each* of the 3 trajectory
                                  classes (true treatments held fixed); report
                                  SSIM vs real image_b for each. The diagonal
                                  (sampled_traj == true_traj) is Task A; the
                                  off-diagonals probe whether the model actually
                                  uses trajectory conditioning.
  - ``unconditional``  (Task C) — sample with NULL tokens on all three
                                  categorical inputs; baseline against which to
                                  measure the value of conditioning.

Outputs
-------
``<out_dir>/per_sample.csv``           one row per (pair_id, sampled_traj)
``<out_dir>/summary.json``             aggregate SSIM by trajectory / direction
``<out_dir>/sample_<i>_<traj>.nii.gz`` (optional; --save_samples)

Example::

    python -m text2glioma.inpainting.run_inference \\
        --config configs/inpainting.yaml \\
        --stage1_config configs/stage1.yaml \\
        --stage1_uri /runs/stage1/checkpoint.pth \\
        --inpainting_ckpt /runs/inpainting/best_model_ema.pth \\
        --datalist datalist_brats_gli_2025_pairs_split.json \\
        --out_dir /runs/inpainting/eval_test/ \\
        --task sweep \\
        --num_inference_steps 50 \\
        --guidance_scale 3.0

Notes on the checkpoint format
------------------------------
``--inpainting_ckpt`` accepts:
  - a raw ``state_dict`` (e.g. ``best_model.pth`` or ``best_model_ema.pth``),
  - a wrapped dict with a ``model`` or ``state_dict`` key (e.g. ``checkpoint.pth``).
The ``module.`` prefix from DDP is stripped automatically.

By default we recommend ``best_model_ema.pth`` because EMA weights typically
produce more stable samples than the live training weights at the same epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from tqdm import tqdm

from text2glioma.preprocessing.inpainting_dataset import (
    TRAJECTORY_TO_IDX, build_pair_transforms, prepare_pair_records,
)
from text2glioma.utils import get_model, load_config, stage1_ify

from .conditioning import CategoricalConditioningEncoder
from .sampling import compute_ssim_per_modality, sample_inpainting
from .training_functions import InpaintingModel

IDX_TO_TRAJECTORY = {v: k for k, v in TRAJECTORY_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--stage1_config", type=str, required=True)
    p.add_argument("--stage1_uri", type=str, required=True)
    p.add_argument("--inpainting_ckpt", type=str, required=True,
                   help="Path to trained inpainting weights (best_model_ema.pth recommended).")
    p.add_argument("--datalist", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=["conditional", "sweep", "unconditional"])
    p.add_argument("--fold", type=str, default="testing",
                   choices=["training", "validation", "testing"])
    p.add_argument("--batch_size", type=int, default=1,
                   help="Sampling batch size. Defaults to 1 because each batch holds "
                        "a full 3D volume; raise only on >40GB GPUs.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="CFG strength; 0.0 disables CFG. Ignored for --task unconditional.")
    p.add_argument("--scale_factor", type=float, default=None,
                   help="Latent scale factor. If omitted, read from the inpainting checkpoint.")
    p.add_argument("--dilation_mm", type=float, default=18.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cuda:0 / cpu (default: auto).")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N records (smoke testing).")
    p.add_argument("--save_samples", action="store_true",
                   help="Also save each predicted image_b as NIfTI under out_dir/samples/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _strip_ddp_prefix(state_dict: dict) -> dict:
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _load_inpainting_state_dict(ckpt_path: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        sd, container_scale = raw["model"], raw.get("scale_factor")
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd, container_scale = raw["state_dict"], raw.get("scale_factor")
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd, container_scale = raw, None
    else:
        raise ValueError(
            f"Unrecognised inpainting checkpoint structure in {ckpt_path}: "
            f"top-level keys = {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )
    return {"state_dict": _strip_ddp_prefix(sd), "scale_factor": container_scale}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    if args.save_samples:
        samples_dir.mkdir(exist_ok=True)

    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    print(f"Loading datalist from {args.datalist}")
    with open(args.datalist) as f:
        datalist = json.load(f)
    records = prepare_pair_records(datalist[args.fold])
    if args.limit:
        records = records[: args.limit]
    print(f"  {args.fold}: {len(records)} pairs (limit={args.limit})")

    spatial_size = tuple(config.get("data", {}).get("spatial_size", (160, 224, 160)))
    xforms = build_pair_transforms(training=False, dilation_mm=args.dilation_mm,
                                   spatial_size=spatial_size)
    loader = DataLoader(
        Dataset(data=records, transform=xforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )

    # ── Stage-1 (frozen) ─────────────────────────────────────────────
    print("Loading Stage-1 VAE …")
    stage1 = stage1_ify(
        get_model(model_type="AutoencoderKL", config=stage1_config, from_file=args.stage1_uri)
    ).eval().to(device)
    for p in stage1.parameters():
        p.requires_grad = False

    # Probe to align UNet channel counts (same logic as the trainer).
    probe = next(iter(loader))
    with torch.no_grad():
        probe_z = stage1(probe["image_a"][:1].to(device))
    latent_ch = int(probe_z.shape[1])
    cfg_params = config.setdefault("model", {}).setdefault("params", {})
    cfg_params["in_channels"] = 2 * latent_ch + 1
    cfg_params["out_channels"] = latent_ch
    config["model"]["latent_channels"] = latent_ch
    print(f"  latent_ch={latent_ch}  unet in/out = {cfg_params['in_channels']}/{cfg_params['out_channels']}")

    # ── UNet + cond encoder ──────────────────────────────────────────
    print("Building UNet + cond encoder …")
    unet = get_model(config["model"].get("name", "DiffusionModelUNet"), config)
    embed_dim = int(cfg_params.get("cross_attention_dim", 256))
    cond_encoder = CategoricalConditioningEncoder(embed_dim=embed_dim)
    inpainting = InpaintingModel(unet=unet, cond_encoder=cond_encoder)

    # Load weights
    payload = _load_inpainting_state_dict(args.inpainting_ckpt)
    missing, unexpected = inpainting.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"    first missing: {missing[:5]}")
        if unexpected:
            print(f"    first unexpected: {unexpected[:5]}")
    inpainting = inpainting.eval().to(device)
    for p in inpainting.parameters():
        p.requires_grad = False

    # ── Scheduler ────────────────────────────────────────────────────
    sch_name = config["scheduler"].get("name", "DDIMScheduler")
    sch_params = config["scheduler"].get("params", {})
    scheduler = (
        DDPMScheduler(**sch_params) if sch_name == "DDPMScheduler"
        else DDIMScheduler(**sch_params)
    )

    # ── scale_factor: CLI > checkpoint > error ───────────────────────
    if args.scale_factor is not None:
        scale_factor = float(args.scale_factor)
        src = "cli"
    elif payload["scale_factor"] is not None:
        scale_factor = float(payload["scale_factor"])
        src = "checkpoint"
    else:
        raise SystemExit(
            "scale_factor not in checkpoint and not provided via --scale_factor. "
            "Inference without the correct scale_factor will produce garbage."
        )
    print(f"  scale_factor = {scale_factor:.4f}  (from {src})")

    # ── Pick conditioning sweep ─────────────────────────────────────
    # Each entry is (label_for_csv, override_trajectory_fn_or_None, use_uncond)
    if args.task == "conditional":
        variants = [("true", None, False)]
    elif args.task == "sweep":
        variants = [
            (name, idx, False) for name, idx in TRAJECTORY_TO_IDX.items()
        ]
    elif args.task == "unconditional":
        variants = [("uncond", None, True)]
    else:
        raise SystemExit(f"Unknown task {args.task!r}")

    print(f"\nTask: {args.task}  variants: {[v[0] for v in variants]}\n")

    # ── Inference loop ──────────────────────────────────────────────
    rng = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[dict] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval[{args.task}]")):
        masked_a = batch["masked_image_a"].to(device)
        image_b_real = batch["image_b"].to(device)
        mask = batch["mask"].to(device)
        true_traj = batch["trajectory"].to(device).long()
        ta = batch["treatment_a"].to(device).long()
        tb = batch["treatment_b"].to(device).long()
        B = masked_a.shape[0]

        # Per-sample metadata for CSV rows
        meta_per_sample = []
        for i in range(B):
            r = records[batch_idx * args.batch_size + i]
            meta_per_sample.append({
                "pair_id":         r.get("pair_id", f"{r['subject_id']}_{r['timepoint_a']}_{r['timepoint_b']}"),
                "subject_id":      r["subject_id"],
                "true_trajectory": IDX_TO_TRAJECTORY.get(int(true_traj[i]), str(int(true_traj[i]))),
                "treatment_a":     int(ta[i]),
                "treatment_b":     int(tb[i]),
                "stratum":         r.get("stratum", ""),
            })

        for variant_label, traj_override, use_uncond in variants:
            if traj_override is not None:
                traj_in = torch.full_like(true_traj, fill_value=int(traj_override))
            else:
                traj_in = true_traj

            pred = sample_inpainting(
                inpainting_model=inpainting, stage1=stage1, scheduler=scheduler,
                masked_image_a=masked_a, mask=mask,
                trajectory=traj_in, treatment_a=ta, treatment_b=tb,
                scale_factor=scale_factor,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_uncond=use_uncond,
                generator=rng,
            )

            for i in range(B):
                ssim = compute_ssim_per_modality(
                    pred[i], image_b_real[i], mask=mask[i],
                )
                row = dict(meta_per_sample[i])
                row.update({
                    "variant":          variant_label,
                    "ssim_global_mean": ssim["ssim_global_mean"],
                    "ssim_roi_mean":    ssim["ssim_roi_mean"],
                })
                for c, (g, r) in enumerate(zip(
                    ssim["ssim_global_perch"], ssim["ssim_roi_perch"] or []
                )):
                    row[f"ssim_global_ch{c}"] = g
                    if ssim["ssim_roi_perch"]:
                        row[f"ssim_roi_ch{c}"] = r
                rows.append(row)

                if args.save_samples:
                    _save_nifti(
                        pred[i].cpu().numpy(),
                        samples_dir / f"{meta_per_sample[i]['pair_id']}__{variant_label}.nii.gz",
                    )

    # ── Write per-sample CSV ────────────────────────────────────────
    csv_path = out_dir / "per_sample.csv"
    if rows:
        # Stable column order: meta first, then ssim columns sorted
        meta_cols = ["pair_id", "subject_id", "true_trajectory", "treatment_a",
                     "treatment_b", "stratum", "variant"]
        ssim_cols = sorted(k for k in rows[0].keys() if k.startswith("ssim_"))
        fieldnames = meta_cols + ssim_cols
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nWrote {csv_path}  ({len(rows)} rows)")

    # ── Aggregate summary ────────────────────────────────────────────
    summary = _summarise(rows, args.task)
    summary["_provenance"] = {
        "task":                  args.task,
        "fold":                  args.fold,
        "config":                args.config,
        "stage1_uri":            args.stage1_uri,
        "inpainting_ckpt":       args.inpainting_ckpt,
        "datalist":              args.datalist,
        "num_inference_steps":   args.num_inference_steps,
        "guidance_scale":        args.guidance_scale,
        "scale_factor":          scale_factor,
        "scale_factor_source":   src,
        "n_samples":             len(records),
        "n_rows":                len(rows),
        "seed":                  args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")

    # Short stdout summary
    print("\n=== Summary ===")
    print(json.dumps(summary.get("by_variant", {}), indent=2))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _summarise(rows: list[dict], task: str) -> dict:
    """Aggregate per-sample rows into mean/std by variant, by true trajectory,
    and (for sweep task) the variant × true-trajectory cross-tab."""
    by_variant: dict[str, dict] = defaultdict(lambda: {"global": [], "roi": []})
    by_traj: dict[str, dict] = defaultdict(lambda: {"global": [], "roi": []})
    cross: dict[tuple[str, str], dict] = defaultdict(lambda: {"global": [], "roi": []})

    for r in rows:
        v = r["variant"]
        t = r["true_trajectory"]
        g, ro = r["ssim_global_mean"], r["ssim_roi_mean"]
        by_variant[v]["global"].append(g)
        if ro is not None and not (isinstance(ro, float) and np.isnan(ro)):
            by_variant[v]["roi"].append(ro)
        by_traj[t]["global"].append(g)
        if ro is not None and not (isinstance(ro, float) and np.isnan(ro)):
            by_traj[t]["roi"].append(ro)
        cross[(v, t)]["global"].append(g)
        if ro is not None and not (isinstance(ro, float) and np.isnan(ro)):
            cross[(v, t)]["roi"].append(ro)

    def _stats(lst):
        if not lst:
            return None
        a = np.array(lst, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    summary = {
        "by_variant": {
            v: {"global": _stats(d["global"]), "roi": _stats(d["roi"])}
            for v, d in by_variant.items()
        },
        "by_true_trajectory": {
            t: {"global": _stats(d["global"]), "roi": _stats(d["roi"])}
            for t, d in by_traj.items()
        },
    }
    if task == "sweep":
        summary["variant_x_true_trajectory"] = {
            f"{v}|{t}": {"global": _stats(d["global"]), "roi": _stats(d["roi"])}
            for (v, t), d in cross.items()
        }
    return summary


def _save_nifti(arr: np.ndarray, path: Path) -> None:
    import nibabel as nib
    if arr.ndim == 4:
        # (C, D, H, W) -> (D, H, W, C) for NIfTI convention
        arr = np.transpose(arr, (1, 2, 3, 0))
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), str(path))


if __name__ == "__main__":
    main()
