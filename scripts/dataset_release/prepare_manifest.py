"""Build the pre-generation manifest for the Text2Glioma synthetic release.

**Mask-first pipeline.** Every prompt in the release is derived from the
segmentation mask that will condition the diffusion sampler, using the
repo's own VASARI-auto pipeline (``text2glioma.preprocessing.utils.
compose_radiology_prompts``). This guarantees prompt-mask consistency by
construction — the text describes exactly what the mask geometry says.

Per-row logic:

    for i in range(N):
        pick a base mask from the training cohort         # 1187 unique masks
        pick a deform seed (deterministic)                # new geometry
        apply the affine deform to the mask
        (if the deformed mask fails volume-ratio check,    # rare with default bounds
         resample the deform seed up to K times)
        run VASARI-auto on the deformed mask              # geometric feature extraction
        compose the impression prompt                     # text from features
        record row: (sample_id, shard, prompt, mask_path, deform, ldm_seed, cfg)

Runs in parallel using multiprocessing.Pool. On Gadi, submit as a CPU-queue
PBS job (see launch_manifest_prep.pbs).

Usage
-----
::

    python scripts/dataset_release/prepare_manifest.py \\
        --datalist datalist_N1510.json \\
        --n_samples 10000 \\
        --num_shards 20 \\
        --seed 12345 \\
        --workers 16 \\
        --out data/synth_release_10k/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Repo-relative imports
sys.path.insert(0, str(Path(__file__).parent))
from mask_deformer import (         # noqa: E402
    apply_deformation,
    deformation_is_valid,
    sample_deform_params,
)

# Optional molecular conditioning (imported lazily inside main() to avoid
# a hard dependency for the vanilla release pipeline).


# ---------------------------------------------------------------------------
# Molecular class-assignment plans
# ---------------------------------------------------------------------------


def _plan_molecular(
    n_samples: int,
    balance_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample (idh_class, mgmt_class) arrays.

    Modes:
        * ``joint``: cycle deterministically through the four combinations
          (wt,unm), (wt,met), (mut,unm), (mut,met) then shuffle. Every
          bin gets exactly ``n_samples // 4`` samples (remainder assigned
          round-robin).
        * ``marginal``: sample IDH and MGMT independently 50/50.
        * ``natural``: sample from a fixed empirical prior roughly matching
          the training cohort (IDH: 80 wt / 20 mut; MGMT: 30 unm / 70 met).
        * ``all_unknown``: reserve for negative-control ablations (both
          fields set to UNKNOWN = 2 for every sample).
    """
    from text2glioma.training.molecular_conditioning import (
        IDH_WILDTYPE, IDH_MUTANT, IDH_UNKNOWN,
        MGMT_UNMETHYLATED, MGMT_METHYLATED, MGMT_UNKNOWN,
    )

    if balance_mode == "joint":
        # 4 cells cycled then shuffled
        cells = np.tile(
            [
                (IDH_WILDTYPE, MGMT_UNMETHYLATED),
                (IDH_WILDTYPE, MGMT_METHYLATED),
                (IDH_MUTANT,   MGMT_UNMETHYLATED),
                (IDH_MUTANT,   MGMT_METHYLATED),
            ],
            (n_samples // 4 + 1, 1),
        )[:n_samples]
        rng.shuffle(cells)
        idh_arr  = cells[:, 0].astype(int)
        mgmt_arr = cells[:, 1].astype(int)
    elif balance_mode == "marginal":
        idh_arr  = rng.choice([IDH_WILDTYPE, IDH_MUTANT],       size=n_samples)
        mgmt_arr = rng.choice([MGMT_UNMETHYLATED, MGMT_METHYLATED], size=n_samples)
    elif balance_mode == "natural":
        idh_arr  = rng.choice([IDH_WILDTYPE, IDH_MUTANT],
                              size=n_samples, p=[0.80, 0.20])
        mgmt_arr = rng.choice([MGMT_UNMETHYLATED, MGMT_METHYLATED],
                              size=n_samples, p=[0.30, 0.70])
    elif balance_mode == "all_unknown":
        idh_arr  = np.full(n_samples, IDH_UNKNOWN,  dtype=int)
        mgmt_arr = np.full(n_samples, MGMT_UNKNOWN, dtype=int)
    else:
        raise ValueError(f"unknown balance_mode: {balance_mode!r}")
    return idh_arr, mgmt_arr


# ---------------------------------------------------------------------------
# Worker: derive prompt from a deformed mask via VASARI-auto
# ---------------------------------------------------------------------------


def _run_one(job: dict) -> dict:
    """Process one manifest row: deform mask, run VASARI-auto, compose prompt,
    and persist the deformed raw mask to its final sample directory.

    Returns a dict with either a fully-populated row, or an error string in
    the ``error`` key. Errors are logged but not fatal — the worker retries
    with a fresh deform seed up to ``max_retries`` times.
    """
    from text2glioma.preprocessing.utils import compose_radiology_prompts

    idx: int = job["idx"]
    sample_id: str = job["sample_id"]
    shard: int = job["shard"]
    mask_path: str = job["mask_path"]
    subj: str = job["subj"]
    max_retries: int = job.get("max_retries", 5)
    deform_seed_start: int = job["deform_seed"]
    out_root = Path(job["out_root"])
    idh_class:  int = int(job.get("idh",  -1))   # -1 sentinel = not set
    mgmt_class: int = int(job.get("mgmt", -1))

    sample_dir = out_root / f"shard_{shard:04d}" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    final_mask_path = sample_dir / "_mask_raw.nii.gz"

    # Idempotency: if this sample already has a persisted mask AND we can
    # recover its metadata via the manifest, skip. But since we're building
    # the manifest fresh, we always regenerate to keep the returned row in
    # sync. We DO reuse the disk write path.

    try:
        lbl_nii = nib.load(mask_path)
        lbl = np.asanyarray(lbl_nii.dataobj).astype(np.int16)
    except Exception as e:
        return {"idx": idx, "error": f"load mask: {type(e).__name__}: {e}"}

    last_err: str | None = None
    for attempt in range(max_retries):
        d_seed = deform_seed_start + attempt
        deform = sample_deform_params(int(d_seed))
        try:
            deformed = apply_deformation(lbl, deform)
        except Exception as e:
            last_err = f"apply_deformation: {type(e).__name__}: {e}"
            continue

        ok, info = deformation_is_valid(lbl, deformed)
        if not ok:
            last_err = (f"deform invalid ({info.get('reason', '?')}, "
                        f"ratio={info.get('ratio', '?')})")
            continue

        # Write the deformed mask to a temp file so VASARI-auto can read it,
        # and to its final sample directory so generation can consume it.
        tmp_dir = job.get("tmp_dir") or tempfile.gettempdir()
        tmp_path = Path(tmp_dir) / f"deformed_pid{os.getpid()}_i{idx:07d}.nii.gz"
        try:
            nib.save(
                nib.Nifti1Image(deformed.astype(np.int16), lbl_nii.affine, lbl_nii.header),
                str(tmp_path),
            )
            prompts = compose_radiology_prompts(
                image_path=str(tmp_path),
                label_path=str(tmp_path),
                shuffle_order=True,
                seed=int(d_seed),
                verbose=False,
            )
        except Exception as e:
            last_err = f"compose: {type(e).__name__}: {e}"
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
            continue

        prompt = (prompts.get("short") or "").strip()
        if not prompt:
            last_err = "empty prompt"
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
            continue

        # SUCCESS: persist the deformed mask to the sample directory (this is
        # the file the generation driver will read). Then remove the temp.
        try:
            nib.save(
                nib.Nifti1Image(deformed.astype(np.int16), lbl_nii.affine, lbl_nii.header),
                str(final_mask_path),
            )
        except Exception as e:
            last_err = f"save final mask: {type(e).__name__}: {e}"
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
            continue
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass

        return {
            "idx": idx,
            "sample_id": sample_id,
            "shard": shard,
            "prompt": prompt,
            "findings": (prompts.get("long") or "").strip(),
            "mask_source_path": mask_path,
            "mask_source_subj": subj,
            "deform_seed": int(d_seed),
            "deform_rot_deg_x": deform.rotation_deg[0],
            "deform_rot_deg_y": deform.rotation_deg[1],
            "deform_rot_deg_z": deform.rotation_deg[2],
            "deform_trans_x": deform.translation_vox[0],
            "deform_trans_y": deform.translation_vox[1],
            "deform_trans_z": deform.translation_vox[2],
            "deform_scale": deform.scale,
            "deform_attempts": attempt + 1,
            "deform_ratio": float(info.get("ratio", 1.0)),
            "mask_persisted_path": str(final_mask_path),
            "idh":  idh_class,
            "mgmt": mgmt_class,
        }

    return {"idx": idx, "sample_id": sample_id, "shard": shard,
            "error": last_err or "no successful deform after retries"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_mask_pool(datalist_path: Path, split: str) -> list[tuple[str, str]]:
    """Return [(label_path, subject_id), ...] for the requested split."""
    dl = json.loads(datalist_path.read_text())
    if split not in dl:
        raise KeyError(f"split {split!r} not in {datalist_path}")
    out: list[tuple[str, str]] = []
    for item in dl[split]:
        lbl = item.get("label")
        if lbl:
            out.append((lbl, item.get("subject_id", item.get("subj", "?"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datalist", type=Path, required=True,
                    help="Training datalist providing the mask pool.")
    ap.add_argument("--split", default="training")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--num_shards", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Text CFG scale (paper deployment default = 1.0).")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Top-level manifest RNG seed (deterministic).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Number of parallel workers for VASARI-auto.")
    ap.add_argument("--tmp_dir", type=Path, default=None,
                    help="Directory for temporary deformed masks. "
                         "On Gadi set to $PBS_JOBFS for local SSD.")
    ap.add_argument("--out_root", type=Path, required=True,
                    help="Release root; deformed masks are persisted at "
                         "<out_root>/shard_XXXX/sample_YYYYYYY/_mask_raw.nii.gz "
                         "and the generation driver reads them from there.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Manifest CSV output path.")
    ap.add_argument("--use_molecular", action="store_true",
                    help="Assign IDH/MGMT class conditioning per sample. "
                         "Downstream generation will pass these through the "
                         "MolecularClassConditioning head. Requires the LDM "
                         "checkpoint to have a trained molecular_head.")
    ap.add_argument("--balance_mode", default="joint",
                    choices=["joint", "marginal", "natural", "all_unknown"],
                    help="How to distribute molecular classes across the release.")
    args = ap.parse_args()

    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    mask_pool = _iter_mask_pool(args.datalist, args.split)
    if not mask_pool:
        raise ValueError("no labels found in the datalist split")
    print(f"mask pool:   {len(mask_pool)} labels", file=sys.stderr, flush=True)
    print(f"target:      {args.n_samples} samples across {args.num_shards} shards",
          file=sys.stderr, flush=True)
    print(f"workers:     {args.workers}", file=sys.stderr, flush=True)

    mask_indices  = rng.integers(0, len(mask_pool), size=args.n_samples)
    # Reserve a small tail per row for retry seeds (deform_seed + attempt).
    deform_seeds  = rng.integers(0, 2**31 - 1_000, size=args.n_samples).astype(int)
    ldm_seeds     = rng.integers(0, 2**31 - 1,     size=args.n_samples).astype(int)

    # Molecular class assignment (deterministic from --seed).
    if args.use_molecular:
        idh_arr, mgmt_arr = _plan_molecular(args.n_samples, args.balance_mode, rng)
        from collections import Counter
        joint = Counter(zip(idh_arr.tolist(), mgmt_arr.tolist()))
        print(f"molecular:   mode={args.balance_mode}, joint counts:",
              file=sys.stderr, flush=True)
        for (i, m), n in sorted(joint.items()):
            print(f"    (idh={i}, mgmt={m}): {n}", file=sys.stderr, flush=True)
    else:
        idh_arr  = np.full(args.n_samples, -1, dtype=int)   # sentinel for "not set"
        mgmt_arr = np.full(args.n_samples, -1, dtype=int)

    jobs: list[dict] = []
    for i in range(args.n_samples):
        mp_idx = int(mask_indices[i])
        jobs.append({
            "idx": i,
            "sample_id":   f"sample_{i:07d}",
            "shard":       int(i % args.num_shards),
            "mask_path":   mask_pool[mp_idx][0],
            "subj":        mask_pool[mp_idx][1],
            "deform_seed": int(deform_seeds[i]),
            "tmp_dir":     str(args.tmp_dir) if args.tmp_dir else None,
            "out_root":    str(args.out_root),
            "idh":         int(idh_arr[i]),
            "mgmt":        int(mgmt_arr[i]),
        })

    results: list[dict | None] = [None] * args.n_samples
    n_done = 0
    n_err = 0

    def _accept(r: dict) -> None:
        nonlocal n_done, n_err
        results[r["idx"]] = r
        n_done += 1
        if "error" in r:
            n_err += 1
        if n_done % 200 == 0 or n_done == args.n_samples:
            print(f"  {n_done}/{args.n_samples}  (errors={n_err})",
                  file=sys.stderr, flush=True)

    if args.workers <= 1:
        for j in jobs:
            _accept(_run_one(j))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_run_one, jobs, chunksize=8):
                _accept(r)

    good = [r for r in results if r is not None and "error" not in r]
    bad  = [r for r in results if r is not None and "error" in r]
    print(f"successful: {len(good)}   failed: {len(bad)}",
          file=sys.stderr, flush=True)
    if bad:
        print("first few failures:", file=sys.stderr, flush=True)
        for r in bad[:5]:
            print(f"  idx={r['idx']}: {r['error']}", file=sys.stderr, flush=True)

    if not good:
        raise RuntimeError("no successful samples generated; aborting")

    good.sort(key=lambda r: r["idx"])
    n = len(good)

    # Note: sample_id and shard are assigned deterministically from idx in the
    # planning step (before workers run). Failures leave holes in the sample_id
    # sequence, which is the honest signal. No post-hoc renumbering.

    rows: list[dict] = []
    for r in good:
        rows.append({
            "sample_id":        r["sample_id"],
            "shard":            int(r["shard"]),
            "prompt":           r["prompt"],
            "findings":         r["findings"],
            "prompt_source":    "mask_derived",
            "mask_source_path": r["mask_source_path"],
            "mask_source_subj": r["mask_source_subj"],
            "mask_persisted_path": r["mask_persisted_path"],
            "deform_seed":      int(r["deform_seed"]),
            "deform_rot_deg_x": r["deform_rot_deg_x"],
            "deform_rot_deg_y": r["deform_rot_deg_y"],
            "deform_rot_deg_z": r["deform_rot_deg_z"],
            "deform_trans_x":   r["deform_trans_x"],
            "deform_trans_y":   r["deform_trans_y"],
            "deform_trans_z":   r["deform_trans_z"],
            "deform_scale":     r["deform_scale"],
            "deform_attempts":  int(r["deform_attempts"]),
            "deform_ratio":     float(r["deform_ratio"]),
            "ldm_seed":         int(ldm_seeds[r["idx"]]),
            "cfg":              float(args.cfg),
            "idh":              int(r.get("idh",  -1)),
            "mgmt":             int(r.get("mgmt", -1)),
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}   ({len(df)} rows, {df.shard.nunique()} shards)",
          file=sys.stderr, flush=True)

    if bad:
        bad_path = args.out.with_name(args.out.stem + ".failures.csv")
        pd.DataFrame(bad).to_csv(bad_path, index=False)
        print(f"failure log: {bad_path}", file=sys.stderr, flush=True)

    print("\n== manifest summary ==", file=sys.stderr, flush=True)
    print(f"  samples/shard: min={df.shard.value_counts().min()}, "
          f"max={df.shard.value_counts().max()}", file=sys.stderr, flush=True)
    print(f"  unique base masks used: {df.mask_source_path.nunique()} "
          f"(each ~{len(df)/df.mask_source_path.nunique():.1f}x)",
          file=sys.stderr, flush=True)
    print(f"  deform_attempts: mean={df.deform_attempts.mean():.2f} "
          f"(1.0 = no retries needed)", file=sys.stderr, flush=True)
    print(f"  deform_ratio: mean={df.deform_ratio.mean():.3f} "
          f"(p10={df.deform_ratio.quantile(0.10):.3f}, "
          f"p90={df.deform_ratio.quantile(0.90):.3f})",
          file=sys.stderr, flush=True)

    if args.use_molecular:
        joint_counts = df.groupby(["idh", "mgmt"]).size().to_dict()
        print(f"  molecular joint distribution: {joint_counts}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
