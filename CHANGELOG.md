# Changelog

All notable changes to **text2glioma** will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-02-12

### Added

- **Validation pipeline** (`text2glioma.validation`): seven new modules
  implementing the full 12-week validation plan as runnable Python code.
  - `image_quality` — per-modality FID (MedicalNet ResNet-50 features),
    3D MS-SSIM, pixel-level statistics (KS test, CNR, SNR).
  - `mask_fidelity` — round-trip segmentation Dice / HD95 for WT/TC/ET
    regions, mask-ablation comparison (dual vs text-only conditioning).
  - `text_alignment` — VASARI feature recovery (accuracy for categorical,
    quadratic-weighted κ for ordinal features), CLIP ViT-L/14 text–image
    cosine similarity.
  - `downstream_utility` — 5 data-regime × 4 task × 3 seed classifier
    grid (DenseNet-121) with AUROC, balanced accuracy, F1, and Wilcoxon
    signed-rank tests.
  - `ablation` — guidance scale sweep, conditioning mode ablation, DDIM
    steps sweep, dropout config generator for retraining experiments.
  - `diversity` — nearest-neighbour memorisation check (L2 in feature
    space), intra-prompt MS-SSIM diversity.
  - `radiologist_eval` — Turing test set preparation (randomised PNG
    triplets + blinded CSV forms), 5-point Likert quality rating forms,
    analysis utilities (Cohen's κ, per-criterion statistics).
- **CLI entry point** `text2glioma-validate` — runs all enabled steps from
  a single YAML config with per-step `enabled` flags and `--steps` filter.
- **Validation config** `configs/validation.yaml` — centralised path and
  parameter settings for the entire validation pipeline.
- **Validation tutorial** in `docs/validation_tutorial.md`.

## [0.2.0] — 2026-02-12

### Added

- **BraTS DecathlonDataset integration**: the tutorial and documentation
  now use `monai.apps.DecathlonDataset` with `task="Task01_BrainTumour"`
  for one-command data download.  Includes channel-reorder helper
  (MSD order FLAIR/T1/T1CE/T2 → pipeline order T1/T1CE/T2/FLAIR) and
  notes on the MSD label convention (1=edema, 2=non-enhancing vs the
  pipeline default 1=non-enhancing, 2=edema).
- **Multi-sequence MRI support**: the VAE (Stage 1) and full pipeline now
  accept 4-channel input (T1w, T1w post-contrast, T2w, FLAIR) instead of
  single-channel T2w.  Channel ordering is `[T1, T1CE, T2, FLAIR]`.
- **Mask conditioning via channel concatenation**: segmentation labels
  (background / non-enhancing / edema / enhancing) are one-hot encoded,
  downsampled to the VAE latent grid, and concatenated with the noisy
  latent for the LDM stage.  Independent mask dropout (default 20 %)
  enables classifier-free guidance at inference.
- **Dual classifier-free guidance (CFG)**: three-way formula supporting
  independent `guidance_scale_text` and `guidance_scale_mask` knobs.
  Falls back to standard text-only CFG when `guidance_scale_mask ≤ 0`.
- **Per-channel perceptual loss**: MedicalNet ResNet-50 expects 1-channel
  input; training now loops over each modality channel and averages.
- **Channel-wise intensity transforms**: `ScaleIntensityRangePercentilesd`
  and `RandShiftIntensityd` now use `channel_wise=True` so each MRI
  sequence is normalised independently.
- **Multi-channel visualisation**: `get_figure()` and
  `log_ldm_sample_unconditioned()` produce per-modality grids with
  yellow labels (T1 / T1CE / T2 / FLAIR).
- **4-channel NIfTI output**: `NiftiSaver` rescales per channel and saves
  4D NIfTI files `[D, H, W, C]` preserving all sequences.
- New CLI entry points: `sample` (batch inference) and `create_prompts`
  (VASARI-based prompt generation).
- Package metadata: classifiers, optional dependency groups
  (`eval`, `gpu-monitor`, `docs`, `dev`), project URLs.
- `__version__` exported from `text2glioma.__init__`.
- Full documentation tree under `docs/` with Sphinx + ReadTheDocs support.
- This CHANGELOG.

### Changed

- `configs/stage1.yaml`: `in_channels` / `out_channels` / discriminator
  `in_channels` changed from 1 → 4.
- `configs/ldm.yaml`: `in_channels` changed from 3 → 7
  (3 latent + 4 one-hot mask), added `mask:` section.
- `configs/inference.yaml`: single `guidance_scale` replaced by
  `guidance_scale_text` / `guidance_scale_mask`.
- `pyproject.toml` requires-python bumped from `>=3.8` to `>=3.9`;
  deduplicated dependencies; description updated.
- `setup.py` version synced with `pyproject.toml`.

### Fixed

- `train_ldm()`: `best_loss` was referenced but never initialised → added
  `best_loss = float("inf")`.
- `train_ldm()`: final checkpoint saved `model.state_dict()` (possibly
  DataParallel-wrapped) → now saves `raw_model.state_dict()`.
- `get_figure()`: `datetime.now()` crashed because `datetime` was the
  module, not the class.
- `train_stage1.py`: `args.n_epochs` → `args.num_epochs` (matching
  argparse definition).
- `train_stage1.py`: `--no_shuffle` default changed from `True` to
  `False` so training actually shuffles by default.
- `preprocessing/utils.py`: `from vasari_auto import …` →
  `from text2glioma.preprocessing.vasari_auto import …`.
- `preprocessing/utils.py`: `rng.shuffle()` return value was assigned
  (returns `None` in-place) → now correctly called in-place.
- `classification/run_experiments.py`: wrong import path fixed;
  `main(args)` → `main()`.
- `classification/experiments.py`: model construction fixed to use
  `getattr(nets, model_type)(**params)` instead of broken double-call.
- `classification/experiments.py`: `torch.save(model.state_dict, …)` →
  `torch.save(model.state_dict(), …)` (missing parentheses).
- `cfg_sample()`: removed dead code in `guidance_scale_mask <= 0` branch.
- `pynvml_utils` import guarded with `try/except` so the package loads
  without an Nvidia GPU or the optional library.
- `get_experiment_dataloaders`: `RandAffined spatial_size` corrected from
  `[160, 192, 96]` to `[160, 224, 160]` to match `Resized`.

## [0.1.0] — (unreleased)

Initial release (text-conditioned single-channel T2w latent diffusion).
