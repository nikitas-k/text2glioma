"""Quantify how much the text branch contributes to samples.

Given the outputs of `scripts/gadi_synthesise_nonsense.ipynb` for a single
mask source case, this computes three seed-paired divergence quantities:

  (1) Mask-off pairwise divergence   -- 1 - SSIM between every pair of
      mask-off samples across prompts. Non-zero <=> text branch encodes
      something prompt-specific.
  (2) Marginal mask contribution      -- 1 - SSIM between mask-on and
      mask-off for each prompt. Non-zero <=> mask channel is doing work.
  (3) In-mask L1 contribution         -- as (2) but restricted to the
      expert tumour mask (spatial support).

Outputs a per-model CSV and prints a compact summary.

Usage:
    python scripts/text_contribution_analysis.py \\
        --runs_root /Users/nk233/mhf/projects/text2glioma/runs \\
        --datalist  datalist_N1510.json \\
        --data_root data \\
        --case_idx  0 \\
        --split     validation \\
        --slugs     real,a_ood_vocab,b_impossible,c_contradictory,d_nonmedical,e_injection \\
        --out       paper/tables/text_contribution.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from skimage.metrics import structural_similarity as ssim

MODELS = {
    "BrainLDM-FT": "pinaya_decoder_only_v5_no_disc",
    "MaxFeat":     "stage1_overfit_ablate_kl1e6",
}
INTERNAL_SUBDIR = Path("data") / "cfg_sweep_text_only" / "cfg_1p00"
MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]


def _load4d(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 3:
        arr = arr[..., None]
    return arr.astype(np.float32), img


def _align_to(src: nib.Nifti1Image, tgt: nib.Nifti1Image) -> np.ndarray:
    """Resample src into tgt's grid; identity when shape+affine match."""
    if src.shape[:3] == tgt.shape[:3] and np.allclose(src.affine, tgt.affine):
        arr = np.asanyarray(src.dataobj).astype(np.float32)
    else:
        # Per-channel resample so we don't stack a 4D through resample_from_to.
        src_arr = np.asanyarray(src.dataobj).astype(np.float32)
        if src_arr.ndim == 3:
            src_arr = src_arr[..., None]
        chans = []
        for c in range(src_arr.shape[-1]):
            c_img = nib.Nifti1Image(src_arr[..., c], affine=src.affine)
            resampled = resample_from_to(c_img, (tgt.shape[:3], tgt.affine), order=1)
            chans.append(np.asanyarray(resampled.dataobj).astype(np.float32))
        arr = np.stack(chans, axis=-1)
    if arr.ndim == 3:
        arr = arr[..., None]
    return arr


def _norm01(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    pos = x[x > 0]
    if pos.size == 0:
        return np.clip(x, 0.0, 1.0)
    lo, hi = np.percentile(pos, [p_lo, p_hi])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def _ssim3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, data_range=1.0))


def _sample_path(runs_root: Path, model_dir: str, case_idx: int, slug: str,
                 nomask: bool) -> Path:
    parts = [slug] if slug != "real" else ["real"]
    if nomask:
        parts.append("nomask")
    suffix = "_" + "_".join(parts)
    return runs_root / model_dir / INTERNAL_SUBDIR / f"sample_cond_native_{case_idx:04d}{suffix}.nii.gz"


def _load_case(runs_root: Path, model_dir: str, case_idx: int, slug: str,
               nomask: bool, real_ref_img: nib.Nifti1Image) -> np.ndarray | None:
    p = _sample_path(runs_root, model_dir, case_idx, slug, nomask)
    if not p.is_file():
        return None
    src = nib.load(str(p))
    arr = _align_to(src, real_ref_img)  # (X,Y,Z,C)
    return np.stack([_norm01(arr[..., c]) for c in range(arr.shape[-1])], axis=-1)


def _load_real(datalist: Path, data_root: Path, split: str, case_idx: int):
    with open(datalist) as f:
        dl = json.load(f)
    item = dl[split][case_idx]
    img_path = data_root / Path(item["image"]).name
    lab_path = data_root.parent / "labelsTr" / Path(item["label"]).name
    real_img = nib.load(str(img_path))
    real_arr = np.asanyarray(real_img.dataobj).astype(np.float32)
    if real_arr.ndim == 3:
        real_arr = real_arr[..., None]
    real_arr = np.stack([_norm01(real_arr[..., c]) for c in range(real_arr.shape[-1])], axis=-1)
    lab = np.asanyarray(nib.load(str(lab_path)).dataobj).astype(np.float32)
    mask = (lab > 0).astype(np.float32)
    return item["subject_id"], real_img, real_arr, mask


def _pair_ssim_per_modality(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [_ssim3d(a[..., c], b[..., c]) for c in range(a.shape[-1])]


def _in_mask_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> list[float]:
    m = mask[..., None] if mask.ndim == 3 else mask
    denom = max(float(m.sum()), 1.0)
    return [float(np.abs((a[..., c] - b[..., c]) * m[..., 0]).sum() / denom)
            for c in range(a.shape[-1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, required=True)
    ap.add_argument("--datalist",  type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Directory containing imagesTr/ and labelsTr/.")
    ap.add_argument("--case_idx",  type=int, default=0)
    ap.add_argument("--split",     type=str, default="validation")
    ap.add_argument("--slugs",     type=str,
                    default="real,a_ood_vocab,b_impossible,c_contradictory,d_nonmedical,e_injection")
    ap.add_argument("--out",       type=Path, default=Path("paper/tables/text_contribution.csv"))
    args = ap.parse_args()

    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    imgs_dir = args.data_root / "imagesTr"
    subj, real_img, real_arr, mask = _load_real(args.datalist, imgs_dir, args.split, args.case_idx)
    print(f"case {args.case_idx} subj={subj} modalities={real_arr.shape[-1]} mask_voxels={int(mask.sum())}")

    rows: list[dict] = []
    for model_name, model_dir in MODELS.items():
        samples: dict[tuple[str, bool], np.ndarray] = {}
        for slug in slugs:
            for nomask in (False, True):
                arr = _load_case(args.runs_root, model_dir, args.case_idx, slug, nomask, real_img)
                if arr is None:
                    print(f"  [skip] {model_name} slug={slug} nomask={nomask} -- file missing")
                    continue
                samples[(slug, nomask)] = arr

        # (1) mask-off pairwise divergence between prompts.
        for si, sj in itertools.combinations(slugs, 2):
            if (si, True) not in samples or (sj, True) not in samples:
                continue
            per_mod = _pair_ssim_per_modality(samples[(si, True)], samples[(sj, True)])
            for mod, s in zip(MODALITIES, per_mod):
                rows.append({
                    "model": model_name, "quantity": "nomask_pair_1mSSIM",
                    "prompt_a": si, "prompt_b": sj, "modality": mod,
                    "value": 1.0 - s,
                })

        # (2) marginal mask contribution per prompt.
        for slug in slugs:
            if (slug, False) not in samples or (slug, True) not in samples:
                continue
            per_mod = _pair_ssim_per_modality(samples[(slug, False)], samples[(slug, True)])
            l1_in_mask = _in_mask_l1(samples[(slug, False)], samples[(slug, True)], mask)
            for mod, s, l in zip(MODALITIES, per_mod, l1_in_mask):
                rows.append({
                    "model": model_name, "quantity": "mask_marginal_1mSSIM",
                    "prompt_a": slug, "prompt_b": slug, "modality": mod,
                    "value": 1.0 - s,
                })
                rows.append({
                    "model": model_name, "quantity": "mask_marginal_L1_in_mask",
                    "prompt_a": slug, "prompt_b": slug, "modality": mod,
                    "value": l,
                })

        # (3) mask-on nonsense-vs-real divergence (upper bound on text effect w/ mask).
        if ("real", False) in samples:
            for slug in slugs:
                if slug == "real" or (slug, False) not in samples:
                    continue
                per_mod = _pair_ssim_per_modality(samples[("real", False)], samples[(slug, False)])
                for mod, s in zip(MODALITIES, per_mod):
                    rows.append({
                        "model": model_name, "quantity": "maskon_vs_real_1mSSIM",
                        "prompt_a": "real", "prompt_b": slug, "modality": mod,
                        "value": 1.0 - s,
                    })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")

    print("\n=== per-model, per-quantity mean over modalities & prompt pairs ===")
    if not df.empty:
        summary = (df.groupby(["model", "quantity"])["value"]
                     .agg(["mean", "std", "count"]).round(4))
        print(summary)


if __name__ == "__main__":
    main()
