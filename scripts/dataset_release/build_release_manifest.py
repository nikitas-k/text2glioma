"""Aggregate per-sample metadata + memorisation-audit into the release manifest.

Walks the generation output directory, collects one row per sample from
its ``metadata.json`` and (if present) merges the corresponding
memorisation-audit rows to add a ``max_nn_ssim`` column and a
``memorisation_flag`` boolean.

Writes ``manifest_release.csv`` \u2014 the canonical file that ships with
the dataset. Users can filter on ``memorisation_flag == False`` or on
``max_nn_ssim < 0.85`` if they want a conservative subset.

Usage
-----
::

    python scripts/dataset_release/build_release_manifest.py \\
        --synth_root /path/to/synth_10k \\
        --audit_csv audit_all_shards.csv \\
        --out /path/to/synth_10k/manifest_release.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_MEMORISATION_THRESHOLD = 0.90


def _flatten_metadata(meta: dict, sample_dir: Path) -> dict:
    """Flatten one sample's metadata.json into a single row."""
    d = meta.get("deform", {})
    row = {
        "sample_id":        meta.get("sample_id", sample_dir.name),
        "shard":            int(sample_dir.parent.name.split("_")[-1]) if sample_dir.parent.name.startswith("shard_") else -1,
        "relpath_image":    str(sample_dir.relative_to(sample_dir.parent.parent) / "image.nii.gz"),
        "relpath_mask":     str(sample_dir.relative_to(sample_dir.parent.parent) / "mask.nii.gz"),
        "prompt":           meta.get("prompt", ""),
        "prompt_source":    meta.get("prompt_source", ""),
        "prompt_meta_json": json.dumps(meta.get("prompt_meta", {})),
        "mask_source_path": meta.get("mask_source_path", ""),
        "mask_source_subj": meta.get("mask_source_subj", ""),
        "deform_seed":      d.get("seed"),
        "deform_rot_x":     (d.get("rotation_deg") or [None, None, None])[0],
        "deform_rot_y":     (d.get("rotation_deg") or [None, None, None])[1],
        "deform_rot_z":     (d.get("rotation_deg") or [None, None, None])[2],
        "deform_trans_x":   (d.get("translation_vox") or [None, None, None])[0],
        "deform_trans_y":   (d.get("translation_vox") or [None, None, None])[1],
        "deform_trans_z":   (d.get("translation_vox") or [None, None, None])[2],
        "deform_scale":     d.get("scale"),
        "deform_valid_ratio": (meta.get("deform_validity") or {}).get("ratio"),
        "ldm_seed":         meta.get("ldm_seed"),
        "cfg":              meta.get("cfg"),
        "sampling_steps":   meta.get("sampling_steps"),
        "stage1_ckpt_sha":  (meta.get("stage1_ckpt_sha256") or "")[:16],
        "stage2_ckpt_sha":  (meta.get("stage2_ckpt_sha256") or "")[:16],
        "image_shape":      json.dumps(meta.get("image_shape", [])),
        "modalities":       ",".join(meta.get("modalities", [])),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synth_root", type=Path, required=True)
    ap.add_argument("--audit_csv",  type=Path, default=None,
                    help="Optional memorisation audit CSV (from memorisation_audit.py).")
    ap.add_argument("--memorisation_threshold", type=float,
                    default=DEFAULT_MEMORISATION_THRESHOLD)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # ---- Collect per-sample metadata ----
    rows: list[dict] = []
    n_missing = 0
    for meta_path in sorted(args.synth_root.glob("shard_*/sample_*/metadata.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            rows.append(_flatten_metadata(meta, meta_path.parent))
        except Exception as e:
            print(f"[warn] {meta_path}: {e}", file=sys.stderr)
            n_missing += 1

    df = pd.DataFrame(rows)
    print(f"collected {len(df)} sample metadata rows from {args.synth_root}")
    if n_missing:
        print(f"  ({n_missing} skipped due to load errors)")

    if df.empty:
        raise SystemExit("no samples found -- check --synth_root")

    # ---- Merge in memorisation audit (if provided) ----
    if args.audit_csv is not None and args.audit_csv.is_file():
        audit = pd.read_csv(args.audit_csv)
        print(f"loaded audit: {len(audit)} rows, "
              f"{audit.sample_id.nunique()} samples covered")
        # per-sample max nn_ssim across modalities
        sample_max = (audit.groupby("sample_id")["nn_ssim"].max()
                            .rename("max_nn_ssim").reset_index())
        # nearest training subject for the max-modality
        nn_subj = (audit.sort_values("nn_ssim", ascending=False)
                        .drop_duplicates("sample_id", keep="first")
                        [["sample_id", "nn_train_subj", "modality"]]
                        .rename(columns={"nn_train_subj": "nn_train_subj_at_max",
                                          "modality": "nn_modality_at_max"}))
        df = df.merge(sample_max, on="sample_id", how="left")
        df = df.merge(nn_subj,    on="sample_id", how="left")
        df["memorisation_flag"] = (df["max_nn_ssim"] > args.memorisation_threshold).fillna(False)
    else:
        print("no audit CSV supplied; skipping memorisation-flag columns")
        df["max_nn_ssim"] = pd.NA
        df["memorisation_flag"] = pd.NA
        df["nn_train_subj_at_max"] = pd.NA
        df["nn_modality_at_max"] = pd.NA

    # ---- Reorder columns for readability ----
    lead = ["sample_id", "shard", "relpath_image", "relpath_mask",
            "prompt", "prompt_source", "cfg", "ldm_seed", "deform_seed",
            "max_nn_ssim", "memorisation_flag",
            "nn_train_subj_at_max", "nn_modality_at_max",
            "mask_source_subj"]
    other = [c for c in df.columns if c not in lead]
    df = df[[c for c in lead if c in df.columns] + other]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}   ({len(df)} rows, {len(df.columns)} columns)")

    # ---- Release summary ----
    print("\n== release summary ==")
    src_counts = df.prompt_source.value_counts()
    for src, n in src_counts.items():
        print(f"  prompt_source={src}:  {n}")
    if df["memorisation_flag"].notna().any():
        n_flag = int(df["memorisation_flag"].sum())
        print(f"  memorisation_flag=True ({args.memorisation_threshold} threshold): {n_flag}/{len(df)} "
              f"({100*n_flag/len(df):.2f}%)")
        ssim = df["max_nn_ssim"].dropna()
        if len(ssim):
            print(f"  max_nn_ssim: mean={ssim.mean():.3f}  median={ssim.median():.3f}  "
                  f"p90={ssim.quantile(0.9):.3f}  p99={ssim.quantile(0.99):.3f}  max={ssim.max():.3f}")


if __name__ == "__main__":
    main()
