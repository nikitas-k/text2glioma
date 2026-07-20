"""Per-worker generation driver for the Text2Glioma synthetic release.

Consumes the manifest from ``prepare_manifest.py``. Each worker processes
one shard's worth of rows, resumes from partial state (skips samples
whose output already exists), and writes one small sample directory per
row.

The mask geometry has ALREADY been deformed at manifest-prep time and
persisted to ``<out_root>/shard_XXXX/sample_YYYYYYY/_mask_raw.nii.gz``.
This driver only loads that mask, runs the LDM sampler, and writes the
paired image + preprocessed mask + metadata. The raw mask file is
consumed (deleted on success) so only the released files remain.

Output layout per sample::

    <out_root>/<shard_id>/<sample_id>/
        image.nii.gz       (4 modalities stacked on channel axis; shape (160, 224, 160, 4))
        mask.nii.gz        (integer segmentation preprocessed to 160x224x160, paired with image)
        metadata.json      (prompt, seeds, deform params, model version, checkpoint hash)

Shard-level completion marker::

    <out_root>/<shard_id>/_DONE

Usage
-----

Single shard on one GPU::

    python scripts/dataset_release/generate_dataset.py \\
        --manifest data/synth_release_10k/manifest.csv \\
        --shard 0 \\
        --out_root /path/to/output \\
        --stage1_config configs/stage1.yaml \\
        --stage2_config configs/ldm_radbert.yaml \\
        --stage1_ckpt /runs/stage1_kl1e6_freebits_lc6/autoencoder_stage1/checkpoint.pth \\
        --stage2_ckpt /runs/stage1_kl1e6_freebits_lc6/ldm_stage2/best_model.pth

All shards via a PBS job array (see ``launch_gadi_array.pbs``): the
$PBS_ARRAY_INDEX env var picks the shard index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Repo-relative imports
sys.path.insert(0, str(Path(__file__).parent))
# NOTE: deformation is done once at manifest-prep time and persisted to
# <out_root>/shard_XXXX/sample_YYYYYYY/_mask_raw.nii.gz. We do not re-apply
# any deform here to eliminate the possibility of computational drift
# between the mask VASARI-auto saw and the mask conditioning the LDM.

# Package import
from text2glioma.inference.engine import Text2GliomaEngine, GenerationResult  # noqa: E402


MODALITIES = ("T1", "T1CE", "T2", "FLAIR")


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, computed lazily. Used for checkpoint provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _save_sample_4d(image: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    """Save a (4, X, Y, Z) float32 array as a channel-last 4D NIfTI (X, Y, Z, 4)."""
    if image.shape[0] != 4:
        raise ValueError(f"expected (4, X, Y, Z), got {image.shape}")
    arr = np.moveaxis(image, 0, -1).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, affine=affine), str(out_path))


def _save_mask(mask: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    """Save an integer segmentation (X, Y, Z) as int16 NIfTI."""
    nib.save(nib.Nifti1Image(mask.astype(np.int16), affine=affine), str(out_path))


def process_shard(
    engine: Text2GliomaEngine,
    manifest: pd.DataFrame,
    shard: int,
    out_root: Path,
    steps: int,
    stage1_ckpt: Path,
    stage2_ckpt: Path,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> None:
    shard_df = manifest[manifest.shard == shard].reset_index(drop=True)
    if len(shard_df) == 0:
        print(f"[shard {shard}] empty; nothing to do", flush=True)
        return

    shard_dir = out_root / f"shard_{shard:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Provenance hashes: compute once per shard, reused for every sample.
    provenance = {
        "stage1_ckpt": str(stage1_ckpt),
        "stage1_ckpt_sha256": _hash_file(Path(stage1_ckpt)),
        "stage2_ckpt": str(stage2_ckpt),
        "stage2_ckpt_sha256": _hash_file(Path(stage2_ckpt)),
        "sampling_steps": steps,
    }

    n_total = len(shard_df) if max_samples is None else min(len(shard_df), max_samples)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()

    for i, row in shard_df.head(n_total).iterrows():
        sample_id: str = row.sample_id
        sample_dir = shard_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        image_path = sample_dir / "image.nii.gz"
        mask_path  = sample_dir / "mask.nii.gz"
        meta_path  = sample_dir / "metadata.json"
        raw_mask_path = sample_dir / "_mask_raw.nii.gz"  # written by prepare_manifest.py

        if not overwrite and image_path.exists() and mask_path.exists() and meta_path.exists():
            n_skipped += 1
            # If the raw mask survived a previous failed run, clean it up now.
            try: raw_mask_path.unlink(missing_ok=True)
            except OSError: pass
            continue

        if not raw_mask_path.exists():
            print(f"[shard {shard}] {sample_id}  missing _mask_raw.nii.gz "
                  f"(run prepare_manifest.py with --out_root pointing here)",
                  file=sys.stderr, flush=True)
            n_failed += 1
            continue

        try:
            # Manifest may or may not carry molecular fields. -1 sentinel
            # means "no molecular conditioning requested" -> pass None to
            # the engine, which ignores the head entirely.
            row_idh  = int(row.get("idh",  -1)) if hasattr(row, "get") else int(getattr(row, "idh",  -1))
            row_mgmt = int(row.get("mgmt", -1)) if hasattr(row, "get") else int(getattr(row, "mgmt", -1))
            idh_arg  = None if row_idh  < 0 else row_idh
            mgmt_arg = None if row_mgmt < 0 else row_mgmt

            result: GenerationResult = engine.generate(
                prompt=row.prompt,
                mask_nifti_path=str(raw_mask_path),
                cfg=float(row.cfg),
                seed=int(row.ldm_seed),
                steps=steps,
                mode="text+mask",
                idh=idh_arg,
                mgmt=mgmt_arg,
            )
            image_np = result.images[0].detach().cpu().numpy()   # (4, X, Y, Z)

            # Save the mask in the same preprocessed space as the image so
            # downstream (mask, image) pairs are directly usable.
            from monai import transforms as T
            xforms = T.Compose([
                T.LoadImage(image_only=True),
                T.EnsureChannelFirst(channel_dim="no_channel"),
                T.Orientation(axcodes="LPS"),
                T.SpatialPad(spatial_size=(160, 224, 160), mode="constant"),
                T.CenterSpatialCrop(roi_size=(160, 224, 160)),
            ])
            mask_pp = xforms(str(raw_mask_path)).detach().cpu().numpy()
            if mask_pp.ndim == 4:
                mask_pp = mask_pp[0]
            mask_pp = mask_pp.astype(np.int16)

            _save_sample_4d(image_np, result.affine, image_path)
            _save_mask(mask_pp, result.affine, mask_path)

            metadata = {
                "sample_id": sample_id,
                "prompt": row.prompt,
                "prompt_source": row.get("prompt_source", "mask_derived"),
                "findings": row.get("findings", ""),
                "mask_source_path": row.mask_source_path,
                "mask_source_subj": row.mask_source_subj,
                "deform": {
                    "seed":            int(row.deform_seed),
                    "rotation_deg":    [float(row.deform_rot_deg_x),
                                        float(row.deform_rot_deg_y),
                                        float(row.deform_rot_deg_z)],
                    "translation_vox": [float(row.deform_trans_x),
                                        float(row.deform_trans_y),
                                        float(row.deform_trans_z)],
                    "scale":           float(row.deform_scale),
                    "attempts":        int(row.get("deform_attempts", 1)),
                    "ratio":           float(row.get("deform_ratio", 1.0)),
                },
                "ldm_seed": int(row.ldm_seed),
                "cfg": float(row.cfg),
                "idh":  row_idh,
                "mgmt": row_mgmt,
                "modalities": list(MODALITIES),
                "image_shape": list(image_np.shape),
                "image_dtype": "float32",
                "mask_dtype": "int16",
                **provenance,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))

            # Consumed: remove the raw mask so only the released files remain.
            try: raw_mask_path.unlink()
            except OSError: pass

            n_done += 1
            if n_done % 20 == 0 or n_done == 1:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (n_total - n_skipped - n_done - n_failed) / max(rate, 1e-6)
                print(f"[shard {shard}] done={n_done}  skipped={n_skipped}  "
                      f"failed={n_failed}  rate={rate*60:.1f}/min  eta={eta/60:.1f} min",
                      flush=True)
        except Exception as e:
            print(f"[shard {shard}] {sample_id}  FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            n_failed += 1

    # Shard-level completion marker only if we actually processed the FULL
    # shard, not just a subset via --max_samples. This prevents smoke tests
    # (e.g. --max_samples 3) from prematurely marking the shard as done.
    shard_total = len(shard_df)
    processed_full_shard = (max_samples is None) or (max_samples >= shard_total)
    if n_failed == 0 and (n_done + n_skipped) == n_total and processed_full_shard:
        (shard_dir / "_DONE").touch()
        print(f"[shard {shard}] complete: done={n_done}, skipped={n_skipped}, "
              f"total={n_total}, elapsed={ (time.time()-t0)/60:.1f} min",
              flush=True)
    elif n_failed == 0 and (n_done + n_skipped) == n_total:
        print(f"[shard {shard}] partial run OK: done={n_done}, skipped={n_skipped}, "
              f"total={n_total} (of {shard_total} in shard) — _DONE NOT written",
              flush=True)
    else:
        print(f"[shard {shard}] finished with failures: done={n_done}, "
              f"skipped={n_skipped}, failed={n_failed}, total={n_total}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=None,
                    help="Shard index (defaults to PBS_ARRAY_INDEX / SLURM_ARRAY_TASK_ID env var).")
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--stage1_config", type=Path, required=True)
    ap.add_argument("--stage2_config", type=Path, required=True)
    ap.add_argument("--stage1_ckpt",   type=Path, required=True)
    ap.add_argument("--stage2_ckpt",   type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--cache_dir", default=None, type=str)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Debug: cap the number of samples processed in this shard.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.shard is None:
        for env_key in ("PBS_ARRAY_INDEX", "SLURM_ARRAY_TASK_ID"):
            val = os.environ.get(env_key)
            if val is not None:
                args.shard = int(val)
                print(f"[env] using {env_key}={val} for --shard")
                break
    if args.shard is None:
        raise SystemExit("--shard is required (or set PBS_ARRAY_INDEX / SLURM_ARRAY_TASK_ID)")

    manifest = pd.read_csv(args.manifest)
    print(f"loaded manifest: {len(manifest)} rows, "
          f"{manifest.shard.nunique()} shards; running shard {args.shard}")

    engine = Text2GliomaEngine.from_paths(
        stage1_config=args.stage1_config,
        stage2_config=args.stage2_config,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    print(f"loaded engine on device={engine.device}, latent_ch={engine.stage1_latent_ch}, "
          f"latent_spatial={engine.latent_spatial}")

    process_shard(
        engine=engine,
        manifest=manifest,
        shard=args.shard,
        out_root=args.out_root,
        steps=args.steps,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
