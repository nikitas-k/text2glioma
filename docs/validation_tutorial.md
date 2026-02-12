# Validation tutorial

This guide walks through the validation pipeline shipped in
`text2glioma.validation`.  The pipeline covers all seven evaluation
dimensions from the [validation plan](validation_plan.md) and is driven
by a single YAML config.

## Prerequisites

```bash
pip install -e ".[eval]"   # adds open-clip-torch, torchmetrics, bert-score
```

You will also need a trained model pipeline (VAE + LDM checkpoints) and a
set of generated synthetic volumes.  See the [main tutorial](tutorial.md)
for training instructions.

---

## Quick start — run everything

```bash
text2glioma-validate --config configs/validation.yaml
```

The CLI reads `configs/validation.yaml`, runs every step whose `enabled`
flag is `true`, and writes a combined summary to
`<output_dir>/validation_summary.json`.

### Run only selected steps

```bash
text2glioma-validate --config configs/validation.yaml \
    --steps image_quality diversity
```

### Override device or sample cap

```bash
text2glioma-validate --config configs/validation.yaml \
    --device cpu --max-n 10
```

---

## Configuration

All paths and parameters live in `configs/validation.yaml`.  Key sections:

```yaml
paths:
  real_dir: ./data/brats/imagesTr          # real 4-ch NIfTI volumes
  real_label_dir: ./data/brats/labelsTr    # real segmentation masks
  synth_dir: ./outputs/synth_volumes       # synthetic 4-ch NIfTI volumes
  synth_label_dir: ./outputs/synth_labels  # round-trip segmentation labels
  prompts_json: ./data/prompts.json        # text prompts (for CLIP scoring)
  atlas_dir: ./data/atlases                # atlas masks (for VASARI)
  output_dir: ./outputs/validation
```

Each validation step has an `enabled: true/false` flag and its own
`output_json` path.  Path references like `${paths.output_dir}` are
resolved automatically.

---

## Step-by-step guide

### §1 — Image quality metrics

Computes per-modality FID, 3D MS-SSIM, and pixel statistics (KS test,
CNR, SNR) between real and synthetic volumes.

```yaml
image_quality:
  enabled: true
  output_json: ${paths.output_dir}/image_quality.json
```

#### Python API

```python
from text2glioma.validation.image_quality import run_image_quality

results = run_image_quality(
    real_dir="./data/brats/imagesTr",
    synth_dir="./outputs/synth_volumes",
    output_json="./results/image_quality.json",
    device="cuda",
    max_n=100,
)
print(results["fid"])       # per-modality FID scores
print(results["ms_ssim"])   # mean/std MS-SSIM
print(results["pixel"])     # KS p-values, CNR, SNR
```

**Target values** (from validation plan):

| Metric | Target |
|--------|--------|
| FID per modality | < 50 |
| FID pooled | < 40 |
| MS-SSIM (intra-pair) | > 0.70 |
| MS-SSIM (diversity, random) | < 0.55 |
| Intensity KS test (per ch) | p > 0.05 |

---

### §2 — Mask fidelity

Tests whether synthetic images respect the conditioning segmentation
mask via a round-trip: generate → re-segment → compare to input mask.

```yaml
mask_fidelity:
  enabled: true
  segmenter_bundle: brats_mri_segmentation
  output_json: ${paths.output_dir}/mask_fidelity.json
```

#### Python API

```python
from text2glioma.validation.mask_fidelity import run_mask_fidelity

results = run_mask_fidelity(
    synth_dir="./outputs/synth_volumes",
    gt_label_dir="./data/brats/labelsTr",
    output_json="./results/mask_fidelity.json",
    device="cuda",
    max_n=50,
)
for region in ["WT", "TC", "ET"]:
    r = results["roundtrip"][region]
    print(f"{region}: Dice = {r['dice_mean']:.3f}, HD95 = {r['hd95_mean']:.1f} mm")
```

The module also provides `mask_ablation()` which compares dual-conditioned
vs text-only round-trip Dice to quantify the value of mask conditioning.

**Target values:**

| Region | Dice | HD95 |
|--------|------|------|
| Whole tumour (WT) | > 0.80 | < 10 mm |
| Tumour core (TC) | > 0.70 | — |
| Enhancing tumour (ET) | > 0.60 | — |

---

### §3 — Text–image alignment

Evaluates whether the generated images faithfully reflect the text prompt
through two complementary measures:

1. **VASARI feature recovery** — extract VASARI features from round-trip
   segmentations and compare to ground truth (accuracy for categorical
   features, quadratic-weighted κ for ordinal features).
2. **CLIP text–image score** — cosine similarity between CLIP ViT-L/14
   embeddings of the prompt and the mid-axial T1CE slice.

```yaml
text_alignment:
  enabled: true
  enhancing_label: 3      # MSD BraTS convention
  nonenhancing_label: 2
  oedema_label: 1
  output_json: ${paths.output_dir}/text_alignment.json
```

#### Python API

```python
from text2glioma.validation.text_alignment import (
    vasari_feature_recovery,
    compute_clip_score,
)

# VASARI recovery
vasari = vasari_feature_recovery(
    gt_labels=["./labels/BRATS_001.nii.gz", ...],
    synth_labels=["./synth_labels/BRATS_001.nii.gz", ...],
    atlas_dir="./data/atlases",
)
print(vasari)
# {'location_accuracy': 0.73, 'laterality_accuracy': 0.91, ...}

# CLIP score
clip = compute_clip_score(
    texts=["Large enhancing glioma in the right frontal lobe"],
    volumes=[synth_vol_array],   # [C, D, H, W] numpy arrays
    device="cuda",
)
print(clip["clip_score_mean"])   # target > 0.25
```

---

### §4 — Downstream utility

Trains DenseNet-121 classifiers on 4 binary tasks (MGMT, 1p/19q, IDH,
grade) across 5 data regimes to measure how synthetic data affects
downstream performance.

```yaml
downstream_utility:
  enabled: true
  tasks: [mgmt, 1p19q, idh, grade]
  regimes: [real_only, synth_only, augmented_50_50, augmented_25_75, low_data]
  seeds: [0, 1, 2]
  n_epochs: 200
  output_json: ${paths.output_dir}/downstream.json
```

> **Note:** This step is typically run separately as it trains
> 5 × 4 × 3 = 60 classifier models.

#### Python API

```python
from text2glioma.validation.downstream_utility import run_downstream_grid

summary = run_downstream_grid(
    datalist_dir="./data/datalists",    # datalist_mgmt.json, etc.
    run_dir="/runs/downstream",
    config_path="configs/cnn.yaml",
    tasks=["mgmt", "idh"],
    seeds=[0, 1, 2],
    device="cuda",
    output_json="./results/downstream.json",
)

# Per task × regime: AUROC mean ± std + Wilcoxon test vs real_only
for task, regimes in summary.items():
    if isinstance(regimes, dict):
        for regime, metrics in regimes.items():
            if isinstance(metrics, dict) and "auroc_mean" in metrics:
                print(f"{task}/{regime}: AUROC = {metrics['auroc_mean']:.3f} ± {metrics['auroc_std']:.3f}")
```

**Target values:**

| Condition | Expected Δ vs real-only |
|-----------|-------------------------|
| Synth-only | AUC within 5 % |
| Augmented 50:50 | AUC ↑ 2–5 % |
| Low-data (20 % real + 80 % synth) | AUC within 10 % of full real |

---

### §5 — Ablation studies

Three sweep types plus a dropout config generator:

#### 5.1 Conditioning mode ablation

```python
from text2glioma.validation.ablation import run_conditioning_ablation

results = run_conditioning_ablation(
    source_json="data/prompts.json",
    real_dir="./data/brats/imagesTr",
    gt_label_dir="./data/brats/labelsTr",
    config_path="configs/ldm.yaml",
    stage1_config="configs/stage1.yaml",
    stage1_uri="stage1.pth",
    model_ckpt="ldm.pth",
    output_dir="./ablation/",
    output_json="./results/conditioning_ablation.json",
)
```

Runs 5 settings: text-only, mask-only, dual (default), strong-mask,
strong-text.

#### 5.2 Guidance scale sweep

```python
from text2glioma.validation.ablation import run_guidance_sweep

results = run_guidance_sweep(
    ...,
    sweeps=[(3.0, 1.0), (5.0, 2.0), (7.5, 3.0), (10.0, 5.0), (12.0, 7.0)],
)
```

#### 5.3 DDIM steps sweep

```python
from text2glioma.validation.ablation import run_steps_sweep

results = run_steps_sweep(
    ...,
    steps_list=[25, 50, 100, 200],
)
# Each entry includes FID and wall_time_s / time_per_volume_s
```

#### 5.4 Dropout ablation (config generation)

Dropout ablation requires **retraining** the LDM for each dropout pair.
The helper generates YAML configs:

```python
from text2glioma.validation.ablation import generate_dropout_configs

configs = generate_dropout_configs(
    base_config_path="configs/ldm.yaml",
    output_dir="./dropout_configs/",
)
# Returns: ['dropout_configs/drop_t0.0_m0.0.yaml', ...]
```

Train each config with `train_stage2`, then evaluate with
`evaluate_dropout_checkpoints()`.

---

### §6 — Diversity & memorisation

#### 6.1 Nearest-neighbour memorisation check

Computes L2 distances between synthetic and real volumes in MedicalNet
feature space.  A low nearest-neighbour distance indicates potential
memorisation.

```python
from text2glioma.validation.diversity import run_memorisation_check

nn = run_memorisation_check(
    real_dir="./data/brats/imagesTr",
    synth_dir="./outputs/synth_volumes",
    device="cpu",
)
print(f"Median NN dist: {nn['median_nn_dist']:.4f}")
print(f"NN ratio: {nn['nn_ratio']:.3f}")        # target > 0.3
print(f"Near copies: {nn['pct_near_copies']:.1f}%")  # target < 2 %
```

#### 6.2 Intra-prompt diversity

Generate 10 samples from the same prompt and measure pairwise MS-SSIM:

```python
from text2glioma.validation.diversity import run_intra_prompt_diversity

diversity = run_intra_prompt_diversity({
    "prompt_1": "./outputs/prompt1_samples/",
    "prompt_2": "./outputs/prompt2_samples/",
})
# Text+mask: target 0.5–0.7; text-only: target 0.3–0.6
```

---

### §7 — Radiologist evaluation

#### 7.1 Visual Turing test preparation

Creates a randomised set of real + synthetic PNG slices with a blinded
rater form:

```python
from text2glioma.validation.radiologist_eval import prepare_turing_test

info = prepare_turing_test(
    real_dir="./data/brats/imagesTr",
    synth_dir="./outputs/synth_volumes",
    real_label_dir="./data/brats/labelsTr",
    synth_label_dir="./outputs/synth_labels",
    output_dir="./eval/turing_test/",
    n_each=50,
    channels=(1, 3),   # T1CE + FLAIR
)
# Outputs:
#   eval/turing_test/images/0000.png ... 0099.png
#   eval/turing_test/rater_form.csv      ← give to radiologists
#   eval/turing_test/answer_key.csv      ← keep hidden
```

#### 7.2 Quality rating form

```python
from text2glioma.validation.radiologist_eval import prepare_quality_rating

info = prepare_quality_rating(
    synth_dir="./outputs/synth_volumes",
    synth_label_dir="./outputs/synth_labels",
    output_dir="./eval/quality_rating/",
    n_samples=50,
)
# Outputs a CSV with columns:
#   id, filename, anatomical_plausibility, tumour_realism,
#   modality_consistency, artefact_free
# Raters fill each with a 1–5 Likert score.
```

#### Analysis

After collecting rater responses:

```python
from text2glioma.validation.radiologist_eval import (
    analyse_turing_test,
    analyse_quality_rating,
)

turing = analyse_turing_test(
    answer_key_csv="./eval/turing_test/answer_key.csv",
    rater_csvs=["./eval/rater1.csv", "./eval/rater2.csv"],
)
print(f"Mean accuracy: {turing['mean_accuracy']:.1%}")       # target < 65 %
print(f"Inter-rater κ: {turing['inter_rater_kappa']:.2f}")  # target > 0.4

quality = analyse_quality_rating(
    rating_csvs=["./eval/rater1_quality.csv", "./eval/rater2_quality.csv"],
)
print(f"Overall mean: {quality['overall']['mean']:.1f} / 5")  # target ≥ 3.5
```

---

## Module reference

| Module | Description | CLI step name |
|--------|-------------|---------------|
| `validation.image_quality` | FID, MS-SSIM, KS/CNR/SNR | `image_quality` |
| `validation.mask_fidelity` | Round-trip Dice/HD95 | `mask_fidelity` |
| `validation.text_alignment` | VASARI recovery, CLIP score | `text_alignment` |
| `validation.downstream_utility` | Classifier grid | `downstream_utility` |
| `validation.ablation` | Guidance / steps / dropout sweeps | `guidance_sweep`, `conditioning_ablation`, `steps_sweep`, `dropout_configs` |
| `validation.diversity` | NN memorisation, intra-prompt | `diversity` |
| `validation.radiologist_eval` | Turing test & quality rating | `turing_test`, `quality_rating` |

---

## Summary of target metrics

| Category | Key metric | Target |
|----------|-----------|--------|
| Quality | FID (per modality) | < 50 |
| Quality | MS-SSIM (intra-pair) | > 0.70 |
| Mask fidelity | Dice WT (round-trip) | > 0.80 |
| Mask fidelity | Dice ET (round-trip) | > 0.60 |
| Text fidelity | VASARI location accuracy | > 70 % |
| Text fidelity | CLIP score | > 0.25 |
| Downstream | AUC real+synth vs real-only | ↑ 2–5 % |
| Diversity | Memorisation rate | < 2 % |
| Clinical | Turing test accuracy | < 65 % |
| Clinical | Mean quality rating | ≥ 3.5 / 5 |
