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
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from skimage.metrics import structural_similarity as ssim

MODELS = {
    "BrainLDM-FT":     "pinaya_decoder_only_v5_no_disc",
    "MaxFeat":         "stage1_kl1e6_freebits_lc6",
}
INTERNAL_SUBDIR = Path("data") / "cfg_sweep_text_only"
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


def _cfg_slug(cfg: float) -> str:
    whole, frac = f"{cfg:.2f}".split(".")
    return f"cfg{whole}p{frac}"


def _candidate_paths(runs_root: Path, model_dir: str, case_idx: int, slug: str,
                     nomask: bool, cfg: float, real_slug: str) -> list[Path]:
    """Return candidate sample paths in preference order.

    Handles two directory conventions and two filename conventions:
      * dir per CFG (new default):        .../cfg_sweep_text_only/cfg_<W>p<FF>/
      * flat single dir (legacy overlay): .../cfg_sweep_text_only/
      * filename with cfg suffix:         sample_cond_native_XXXX_<slug>[_nomask]_cfg<W>p<FF>.nii.gz
      * filename without cfg suffix:      sample_cond_native_XXXX_<slug>[_nomask].nii.gz
    Real-prompt spelling: slug == real_slug is stored without a slug segment.
    """
    base_percfg = runs_root / model_dir / INTERNAL_SUBDIR / _cfg_dir_name(cfg)
    base_flat   = runs_root / model_dir / INTERNAL_SUBDIR
    parts_common: list[str] = []
    if slug != real_slug:
        parts_common.append(slug)
    if nomask:
        parts_common.append("nomask")
    parts_new = parts_common + [_cfg_slug(cfg)]
    suffix_new    = ("_" + "_".join(parts_new))    if parts_new    else ""
    suffix_legacy = ("_" + "_".join(parts_common)) if parts_common else ""
    return [
        base_percfg / f"sample_cond_native_{case_idx:04d}{suffix_legacy}.nii.gz",
        base_percfg / f"sample_cond_native_{case_idx:04d}{suffix_new}.nii.gz",
        base_flat   / f"sample_cond_native_{case_idx:04d}{suffix_new}.nii.gz",
        base_flat   / f"sample_cond_native_{case_idx:04d}{suffix_legacy}.nii.gz",
    ]


def _cfg_dir_name(cfg: float) -> str:
    """'cfg_1p00' style directory segment (matches offline_sample_stage2_compare output)."""
    whole, frac = f"{cfg:.2f}".split(".")
    return f"cfg_{whole}p{frac}"


def _load_case(runs_root: Path, model_dir: str, case_idx: int, slug: str,
               nomask: bool, cfg: float, real_slug: str,
               real_ref_img: nib.Nifti1Image) -> np.ndarray | None:
    for p in _candidate_paths(runs_root, model_dir, case_idx, slug, nomask, cfg, real_slug):
        if p.is_file():
            src = nib.load(str(p))
            arr = _align_to(src, real_ref_img)  # (X,Y,Z,C)
            return np.stack([_norm01(arr[..., c]) for c in range(arr.shape[-1])], axis=-1)
    return None


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


def _ssim3d_in_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """SSIM computed on the smallest axis-aligned bbox that contains the mask,
    after zeroing voxels outside the mask. Returns NaN if the mask is empty.

    Restricting to the bbox is important because skimage's SSIM uses a
    Gaussian window and would otherwise pool the (large, empty) background
    into the estimate, making the metric insensitive to changes inside a
    small tumour region.
    """
    if float(mask.sum()) < 1.0:
        return float("nan")
    idx = np.array(np.where(mask > 0))
    lo = idx.min(axis=1)
    hi = idx.max(axis=1) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    m = mask[sl]
    return float(ssim(a[sl] * m, b[sl] * m, data_range=1.0))


def _pair_ssim_in_mask_per_modality(a: np.ndarray, b: np.ndarray,
                                     mask: np.ndarray) -> list[float]:
    return [_ssim3d_in_mask(a[..., c], b[..., c], mask) for c in range(a.shape[-1])]


def _w1_in_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> list[float]:
    """Per-modality 1-Wasserstein distance between the *intensity histograms*
    of the two samples restricted to the mask region. Uses the analytic
    quantile formulation on the sorted values so no binning parameters are
    required.

    Interpretation: measures a shift/spread of the intensity distribution
    inside the tumour region, complementary to L1 (which is per-voxel and
    penalises spatial displacement) and SSIM (which is a structural
    correlation).
    """
    m = mask > 0 if mask.ndim == 3 else mask[..., 0] > 0
    n = int(m.sum())
    out: list[float] = []
    for c in range(a.shape[-1]):
        if n < 1:
            out.append(float("nan"))
            continue
        va = np.sort(a[..., c][m].astype(np.float64))
        vb = np.sort(b[..., c][m].astype(np.float64))
        out.append(float(np.mean(np.abs(va - vb))))
    return out


# Global switch: when True, skip the expensive whole-volume SSIM calls in
# both _emit_pair and _emit_interaction. Whole-volume 1-SSIM is dominated
# by background and is not used by any of the additive-decomposition
# estimators (||T||, ||M||, ||I||) or by Figures 5/6. Set via --skip_global_ssim
# on the CLI.
SKIP_GLOBAL_SSIM: bool = False


def _emit_pair(rows: list[dict], model_name: str, cfg: float, subject_id: str,
               case_idx: int, quantity: str,
               prompt_a: str, prompt_b: str, a: np.ndarray, b: np.ndarray,
               mask: np.ndarray) -> None:
    """Compute all four per-modality divergence metrics for one seed-paired
    (a, b) sample pair and append one row per modality per metric.

    Metrics emitted:
      * ``{quantity}_1mSSIM``          (whole-volume 1 - SSIM; skipped when
                                        SKIP_GLOBAL_SSIM is True -- rows are
                                        omitted entirely, not filled with NaN)
      * ``{quantity}_1mSSIM_in_mask``  (in-mask bbox 1 - SSIM)
      * ``{quantity}_L1_in_mask``      (per-voxel L1 restricted to mask)
      * ``{quantity}_W1_in_mask``      (per-modality 1-Wasserstein on the
                                        mask-region intensity distribution)
    """
    ssim_in   = _pair_ssim_in_mask_per_modality(a, b, mask)
    l1_in     = _in_mask_l1(a, b, mask)
    w1_in     = _w1_in_mask(a, b, mask)
    ssim_glob = ([float("nan")] * len(MODALITIES) if SKIP_GLOBAL_SSIM
                 else _pair_ssim_per_modality(a, b))
    for mod, sg, si, l, w in zip(MODALITIES, ssim_glob, ssim_in, l1_in, w1_in):
        base = dict(model=model_name, cfg=cfg, subject_id=subject_id,
                    case_idx=case_idx, prompt_a=prompt_a, prompt_b=prompt_b,
                    modality=mod)
        if not SKIP_GLOBAL_SSIM:
            rows.append({**base, "quantity": f"{quantity}_1mSSIM",     "value": 1.0 - sg if np.isfinite(sg) else float("nan")})
        rows.append({**base, "quantity": f"{quantity}_1mSSIM_in_mask", "value": 1.0 - si if np.isfinite(si) else float("nan")})
        rows.append({**base, "quantity": f"{quantity}_L1_in_mask",     "value": l})
        rows.append({**base, "quantity": f"{quantity}_W1_in_mask",     "value": w})


def _emit_interaction(rows: list[dict], model_name: str, cfg: float,
                      subject_id: str, case_idx: int, prompt: str,
                      x_pi_on: np.ndarray, x_pi_off: np.ndarray,
                      x_empty_on: np.ndarray, x_0: np.ndarray,
                      mask: np.ndarray) -> None:
    """Emit the 2x2 factorial interaction term for one prompt at mask-on.

    The additive decomposition x(pi, m) ~= x_0 + T(pi) + M(m) + I(pi, m) with
    T(empty) = 0, M(off) = 0, I(*, off) = 0, and I(empty, on) = 0 gives:

        I(pi, on) = x(pi, on) - x(pi, off) - x(empty, on) + x_0

    We measure ||I(pi, on)|| by three divergences between the actual sample
    x(pi, on) and its additive prediction x_add = x(pi, off) + x(empty, on) - x_0:

      * ``interaction_L1_in_mask``     = L1(x(pi, on), x_add)  restricted to mask.
      * ``interaction_1mSSIM_in_mask`` = 1 - SSIM(x(pi, on), x_add) on the mask bbox.
      * ``interaction_W1_in_mask``     = W1 between mask-region intensity histograms
                                         of x(pi, on) and x_add.

    L1 is the canonical norm-based estimator (proper metric on image space);
    SSIM/W1 are corroborators. The intensity of ``x_add`` may fall outside
    [0, 1] due to the subtraction; we clamp for SSIM but not for L1/W1.
    """
    x_add = x_pi_off + x_empty_on - x_0
    ssim_in   = _pair_ssim_in_mask_per_modality(x_pi_on, np.clip(x_add, 0.0, 1.0), mask)
    l1_in     = _in_mask_l1(x_pi_on, x_add, mask)
    w1_in     = _w1_in_mask(x_pi_on, x_add, mask)
    ssim_glob = ([float("nan")] * len(MODALITIES) if SKIP_GLOBAL_SSIM
                 else _pair_ssim_per_modality(x_pi_on, np.clip(x_add, 0.0, 1.0)))
    for mod, sg, si, l, w in zip(MODALITIES, ssim_glob, ssim_in, l1_in, w1_in):
        base = dict(model=model_name, cfg=cfg, subject_id=subject_id,
                    case_idx=case_idx, prompt_a=prompt, prompt_b=prompt,
                    modality=mod)
        if not SKIP_GLOBAL_SSIM:
            rows.append({**base, "quantity": "interaction_1mSSIM",     "value": 1.0 - sg if np.isfinite(sg) else float("nan")})
        rows.append({**base, "quantity": "interaction_1mSSIM_in_mask", "value": 1.0 - si if np.isfinite(si) else float("nan")})
        rows.append({**base, "quantity": "interaction_L1_in_mask",     "value": l})
        rows.append({**base, "quantity": "interaction_W1_in_mask",     "value": w})


def _parse_case_indices(spec: str) -> list[int]:
    """Parse '0-19' or '0,3,5,10-15' style specs into a flat list of ints."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, required=True)
    ap.add_argument("--datalist",  type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Directory containing imagesTr/ and labelsTr/.")
    ap.add_argument("--case_idx",  type=int, default=None,
                    help="Single case index (legacy). If given without "
                         "--case_indices, analysis runs on only this case.")
    ap.add_argument("--case_indices", type=str, default=None,
                    help="Comma-separated case indices or ranges (e.g. "
                         "'0,3,5,10-15' or '0-19'). Overrides --case_idx.")
    ap.add_argument("--split",     type=str, default="validation")
    ap.add_argument("--slugs",     type=str,
                    default="real,empty,a_ood_vocab,b_impossible,c_contradictory,d_nonmedical,e_injection")
    ap.add_argument("--empty_slug", type=str, default="empty",
                    help="Slug used for the empty-string uncond baseline "
                         "(the notebook writes 'f_empty' in older runs).")
    ap.add_argument("--real_slug", type=str, default="real",
                    help="Slug for the real-impression sample; on disk this row has no "
                         "slug segment (sample_cond_native_XXXX[_nomask][_cfg...].nii.gz).")
    ap.add_argument("--cfg_values", type=str, default="1.0,4.5,7.0",
                    help="Comma-separated CFG values whose sample files should be scanned.")
    ap.add_argument("--out",       type=Path, default=Path("paper/tables/text_contribution.csv"))
    args = ap.parse_args()

    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    cfg_values = [float(x) for x in args.cfg_values.split(",") if x.strip()]
    real_slug = args.real_slug
    empty_slug = args.empty_slug
    if empty_slug not in slugs:
        slugs = slugs + [empty_slug]

    if args.case_indices:
        case_indices = _parse_case_indices(args.case_indices)
    elif args.case_idx is not None:
        case_indices = [args.case_idx]
    else:
        raise SystemExit("supply either --case_idx N or --case_indices SPEC")

    imgs_dir = args.data_root / "imagesTr"
    rows: list[dict] = []

    for case_idx in case_indices:
        try:
            subj, real_img, real_arr, mask = _load_real(
                args.datalist, imgs_dir, args.split, case_idx)
        except (KeyError, IndexError) as exc:
            print(f"[skip] case {case_idx} unavailable in datalist: {exc}")
            continue
        print(f"\ncase {case_idx} subj={subj} modalities={real_arr.shape[-1]} "
              f"mask_voxels={int(mask.sum())}")

        for model_name, model_dir in MODELS.items():
            for cfg in cfg_values:
                samples: dict[tuple[str, bool], np.ndarray] = {}
                for slug in slugs:
                    for nomask in (False, True):
                        arr = _load_case(args.runs_root, model_dir, case_idx,
                                         slug, nomask, cfg, real_slug, real_img)
                        if arr is None:
                            print(f"  [skip] {model_name} case={case_idx} slug={slug} "
                                  f"nomask={nomask} cfg={cfg} -- file missing")
                            continue
                        samples[(slug, nomask)] = arr

                _analyse_one(rows, model_name, cfg, subj, case_idx,
                             samples, slugs, real_slug, empty_slug, mask)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")

    print("\n=== per-(model, cfg, quantity) mean over subjects x modalities x prompt pairs ===")
    if not df.empty:
        summary = (df.groupby(["model", "cfg", "quantity"])["value"]
                     .agg(["mean", "std", "count"]).round(4))
        print(summary)


def _analyse_one(rows: list[dict], model_name: str, cfg: float,
                 subj: str, case_idx: int,
                 samples: dict[tuple[str, bool], np.ndarray],
                 slugs: list[str], real_slug: str, empty_slug: str,
                 mask: np.ndarray) -> None:
    """All per-pair emits for one (subject, model, cfg) triple."""

    # ---- classic per-pair divergences (kept from the original design) ----

    # (1) Text-alone signal: real vs empty, mask-off, same seed.
    if (real_slug, True) in samples and (empty_slug, True) in samples:
        _emit_pair(rows, model_name, cfg, subj, case_idx,
                   "text_alone_real_vs_empty", real_slug, empty_slug,
                   samples[(real_slug, True)], samples[(empty_slug, True)], mask)

    # (2) Robustness to nonsense: nonsense vs empty, mask-off.
    if (empty_slug, True) in samples:
        for slug in slugs:
            if slug in (real_slug, empty_slug) or (slug, True) not in samples:
                continue
            _emit_pair(rows, model_name, cfg, subj, case_idx,
                       "nonsense_vs_empty", empty_slug, slug,
                       samples[(empty_slug, True)], samples[(slug, True)], mask)

    # (2b) vs_real_maskoff: each non-real prompt vs real, mask-off.
    if (real_slug, True) in samples:
        for slug in slugs:
            if slug == real_slug or (slug, True) not in samples:
                continue
            _emit_pair(rows, model_name, cfg, subj, case_idx,
                       "vs_real_maskoff", real_slug, slug,
                       samples[(real_slug, True)], samples[(slug, True)], mask)

    # (3) Marginal mask contribution per prompt: d(mask on, mask off | prompt).
    for slug in slugs:
        if (slug, False) not in samples or (slug, True) not in samples:
            continue
        _emit_pair(rows, model_name, cfg, subj, case_idx,
                   "mask_marginal", slug, slug,
                   samples[(slug, False)], samples[(slug, True)], mask)

    # (3b) Text-marginal ingredients per prompt, referenced against uncond.
    if (empty_slug, True) in samples:
        uncond = samples[(empty_slug, True)]
        for slug in slugs:
            if (slug, True) in samples:
                _emit_pair(rows, model_name, cfg, subj, case_idx,
                           "textonly_vs_uncond", empty_slug, slug,
                           uncond, samples[(slug, True)], mask)
            if (slug, False) in samples:
                _emit_pair(rows, model_name, cfg, subj, case_idx,
                           "textmask_vs_uncond", empty_slug, slug,
                           uncond, samples[(slug, False)], mask)

    # (4) Mask-on nonsense-vs-real divergence (supplementary).
    if (real_slug, False) in samples:
        for slug in slugs:
            if slug == real_slug or (slug, False) not in samples:
                continue
            _emit_pair(rows, model_name, cfg, subj, case_idx,
                       "maskon_vs_real", real_slug, slug,
                       samples[(real_slug, False)], samples[(slug, False)], mask)

    # ---- additive-decomposition norms (T, M, I) ----
    # For notational convenience:
    #   x_0        := x(empty, mask off)      = uncond
    #   x_pi_off   := x(prompt, mask off)      -> ||T(pi)|| = L1(x_pi_off, x_0)
    #   x_empty_on := x(empty, mask on)        -> ||M(on)|| = L1(x_empty_on, x_0)
    #   x_pi_on    := x(prompt, mask on)       -> ||I(pi, on)|| via 2x2 contrast

    if (empty_slug, True) in samples:
        x_0 = samples[(empty_slug, True)]

        # ||T(pi)|| for each prompt (mask off vs uncond).
        for slug in slugs:
            if (slug, True) not in samples:
                continue
            _emit_pair(rows, model_name, cfg, subj, case_idx,
                       "text_norm", empty_slug, slug,
                       x_0, samples[(slug, True)], mask)

        # ||M(on)|| once per (model, cfg, subject): empty prompt + mask on vs uncond.
        if (empty_slug, False) in samples:
            _emit_pair(rows, model_name, cfg, subj, case_idx,
                       "mask_norm", empty_slug, empty_slug,
                       x_0, samples[(empty_slug, False)], mask)

            # ||I(pi, on)|| for each prompt with all four samples available.
            x_empty_on = samples[(empty_slug, False)]
            for slug in slugs:
                if slug == empty_slug:
                    continue
                if (slug, True) not in samples or (slug, False) not in samples:
                    continue
                _emit_interaction(rows, model_name, cfg, subj, case_idx, slug,
                                  x_pi_on=samples[(slug, False)],
                                  x_pi_off=samples[(slug, True)],
                                  x_empty_on=x_empty_on,
                                  x_0=x_0,
                                  mask=mask)


if __name__ == "__main__":
    main()
