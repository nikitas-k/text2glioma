"""Build the pre-generation manifest for the Text2Glioma synthetic release.

Given a training datalist and a target sample count, this script writes
a CSV where each row fully specifies one synthetic sample:

    sample_id            unique zero-padded id (e.g. "sample_007412")
    shard                integer 0..(num_shards-1)  (for parallel workers)
    prompt               VASARI impression text (either verbatim real or a novel recomposition)
    prompt_source        "real" | "novel"
    prompt_meta_json     JSON blob with pool_idx / swaps / base_pool_idx
    mask_source_path     path to the real training label used as the mask base
    mask_source_subj     subject_id from the datalist (traceability)
    deform_seed          int, drives mask_deformer.sample_deform_params
    deform_rot_deg_x/y/z three floats
    deform_trans_x/y/z   three floats
    deform_scale         float
    ldm_seed             int, drives DDIM latent initialisation
    cfg                  fixed to 1.0 by default (paper deployment setting)

Everything downstream (per-worker generation, memorisation audit,
final manifest aggregation) reads from this CSV, so the release is
fully reproducible from just this file + the model checkpoint.

Usage
-----
::

    python scripts/dataset_release/prepare_manifest.py \\
        --datalist datalist_N1510.json \\
        --n_samples 10000 \\
        --num_shards 20 \\
        --seed 12345 \\
        --out data/synth_release_10k/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo-relative import
sys.path.insert(0, str(Path(__file__).parent))
from prompt_sampler import (         # noqa: E402
    PromptSampler,
    load_training_impressions,
)
from mask_deformer import sample_deform_params  # noqa: E402


def _flatten_deform(d) -> dict:
    return {
        "deform_seed": d.seed,
        "deform_rot_deg_x": d.rotation_deg[0],
        "deform_rot_deg_y": d.rotation_deg[1],
        "deform_rot_deg_z": d.rotation_deg[2],
        "deform_trans_x": d.translation_vox[0],
        "deform_trans_y": d.translation_vox[1],
        "deform_trans_z": d.translation_vox[2],
        "deform_scale": d.scale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datalist", type=Path, required=True,
                    help="Training datalist providing the prompt + mask pool.")
    ap.add_argument("--split", default="training",
                    help="Split key to draw prompts and masks from.")
    ap.add_argument("--prompt_field", default="impression",
                    choices=["impression", "findings"],
                    help="Which VASARI field to use as the prompt.")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--novel_fraction", type=float, default=0.3,
                    help="Fraction of samples with novel-VASARI recomposed prompts.")
    ap.add_argument("--num_shards", type=int, default=20,
                    help="Number of independent worker shards to split into.")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Text CFG scale (paper deployment default = 1.0).")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Top-level manifest RNG seed (deterministic).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.datalist) as f:
        dl = json.load(f)
    if args.split not in dl:
        raise KeyError(f"split {args.split!r} not in {args.datalist}")
    training = dl[args.split]

    # Prompt pool (impressions from the same split).
    impressions = load_training_impressions(args.datalist, args.split, args.prompt_field)
    sampler = PromptSampler(impressions, seed=int(rng.integers(2**31 - 1)))

    # Mask pool (label paths + subject_ids)
    mask_pool: list[tuple[str, str]] = []
    for item in training:
        lbl = item.get("label")
        if lbl:
            mask_pool.append((lbl, item.get("subject_id", "?")))
    if not mask_pool:
        raise ValueError("no labels found in the datalist split")
    print(f"prompt pool: {len(impressions)} impressions "
          f"({sum(1 for i in impressions if i)} non-empty)")
    print(f"mask pool:   {len(mask_pool)} labels")
    print(f"target:      {args.n_samples} samples, "
          f"{args.novel_fraction:.0%} novel prompts, {args.num_shards} shards")

    # Draw prompts as a batch (interleaved real/novel).
    prompt_batch = sampler.sample_batch(args.n_samples, args.novel_fraction)

    # Draw mask indices with replacement.
    mask_indices = rng.integers(0, len(mask_pool), size=args.n_samples)

    # Per-sample deterministic seeds (deform + LDM).
    deform_seeds = rng.integers(0, 2**31 - 1, size=args.n_samples).astype(int)
    ldm_seeds    = rng.integers(0, 2**31 - 1, size=args.n_samples).astype(int)

    # Shard assignment (round-robin so workers get near-equal counts).
    shards = (np.arange(args.n_samples) % args.num_shards).astype(int)

    rows: list[dict] = []
    for i in range(args.n_samples):
        prompt, meta = prompt_batch[i]
        mask_path, mask_subj = mask_pool[int(mask_indices[i])]
        deform = sample_deform_params(int(deform_seeds[i]))

        row = {
            "sample_id":       f"sample_{i:07d}",
            "shard":           int(shards[i]),
            "prompt":          prompt,
            "prompt_source":   meta["source"],
            "prompt_meta_json": json.dumps(meta),
            "mask_source_path": mask_path,
            "mask_source_subj": mask_subj,
            **_flatten_deform(deform),
            "ldm_seed":        int(ldm_seeds[i]),
            "cfg":             float(args.cfg),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}   ({len(df)} rows, {df.shard.nunique()} shards)")

    # Summary stats
    print("\n== manifest summary ==")
    print(f"  prompt_source counts:\n{df.prompt_source.value_counts().to_string()}")
    print(f"\n  samples per shard: min={df.shard.value_counts().min()}, "
          f"max={df.shard.value_counts().max()}")
    print(f"  unique masks used: {df.mask_source_path.nunique()} "
          f"(each used ~{args.n_samples/df.mask_source_path.nunique():.1f}x on avg)")


if __name__ == "__main__":
    main()
