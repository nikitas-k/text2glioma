# Validation plan — text2glioma v0.2

**Timeline:** 12 Feb – 12 May 2026 (12 weeks)
**Dataset:** MSD BraTS (Task01_BrainTumour) via `monai.apps.DecathlonDataset`
(388 training + 96 validation subjects, 4 MRI sequences each)

---

## Overview

```
Week  1–2   Data prep, training VAE + LDM, baseline generation
Week  3–5   Image quality & fidelity metrics
Week  6–8   Downstream utility experiments
Week  9–10  Ablation studies & conditioning analysis
Week 11–12  Radiologist evaluation, write-up, release
```

---

## 1 — Image quality metrics (weeks 3–4)

Generate **N = 1 000** synthetic volumes using prompts derived from the
real training set (one per real subject ×2–3 re-samples).

### 1.1 Fréchet Inception Distance (FID) — per modality

Compute FID between real and synthetic 2D slices independently for each
sequence (T1, T1CE, T2, FLAIR) using a MedicalNet ResNet-50 feature
extractor (the same backbone used for perceptual loss).

| Metric | Extractor | Slicing | Expected |
|--------|-----------|---------|----------|
| FID-T1 | MedicalNet-50 | axial mid-50 % | < 50 |
| FID-T1CE | MedicalNet-50 | axial mid-50 % | < 50 |
| FID-T2 | MedicalNet-50 | axial mid-50 % | < 50 |
| FID-FLAIR | MedicalNet-50 | axial mid-50 % | < 50 |
| FID-all | MedicalNet-50 | axial mid-50 % | < 40 |

**Implementation note:** extract 2048-d features from `layer4`, compute
mean + covariance per modality, report Fréchet distance.  Use the
`torcheval` or `torch-fidelity` library for Fréchet computation.

### 1.2 Multi-Scale Structural Similarity (MS-SSIM)

For each synthetic volume, find the nearest-neighbour real volume (by
CLIP embedding cosine similarity of the text prompt) and compute 3D
MS-SSIM across the 4 channels.

| Metric | Target |
|--------|--------|
| MS-SSIM (intra-pair) | > 0.70 |
| MS-SSIM (diversity, random pairs) | < 0.55 |

A high intra-pair score confirms prompt fidelity; a low random-pair
score confirms the model is not memorising the training set.

### 1.3 Pixel-level statistics

| Metric | How | Target |
|--------|-----|--------|
| Mean intensity per channel | histogram comparison (KS test) | p > 0.05 |
| Contrast-to-noise ratio (CNR) | tumour vs. normal WM | within ±15 % of real |
| Signal-to-noise ratio (SNR) | foreground / background std | within ±15 % of real |

---

## 2 — Mask fidelity (weeks 4–5)

Evaluate how well the generated images respect the conditioning
segmentation mask.  This answers: "if I give the model a tumour shape,
does it put the tumour there?"

### 2.1 Round-trip segmentation

1. Generate a synthetic volume conditioned on a **held-out** real mask.
2. Segment the synthetic volume with a pretrained BraTS segmentation
   model (e.g., `monai.bundle` → `brats_mri_segmentation`).
3. Compare the predicted segmentation to the conditioning mask.

| Metric | Regions | Target |
|--------|---------|--------|
| Dice | whole tumour (WT) | > 0.80 |
| Dice | tumour core (TC) | > 0.70 |
| Dice | enhancing tumour (ET) | > 0.60 |
| Hausdorff95 | whole tumour | < 10 mm |

```python
# pseudocode
from monai.metrics import DiceMetric, HausdorffDistanceMetric

dice = DiceMetric(include_background=False, reduction="mean_batch")
hd95 = HausdorffDistanceMetric(percentile=95, include_background=False)

for mask_gt, synth_vol in pairs:
    mask_pred = brats_segmenter(synth_vol)          # pretrained nnU-Net / SwinUNETR
    dice(mask_pred, mask_gt)
    hd95(mask_pred, mask_gt)

print(f"Dice WT: {dice.aggregate()}")
print(f"HD95 WT: {hd95.aggregate()}")
```

### 2.2 Mask-unconditional comparison (ablation)

Repeat the round-trip segmentation for **text-only** samples (mask
guidance off, `guidance_scale_mask = 0`).  The text-only Dice should be
significantly lower than the mask-conditioned Dice, demonstrating that
mask conditioning adds spatial control.

---

## 3 — Text–image alignment (weeks 5–6)

### 3.1 VASARI feature recovery

1. Generate volumes from **known** VASARI-derived prompts (from the real
   training set).
2. Run `vasari_auto.get_vasari_features()` on the synthetic label
   (obtained via round-trip segmentation from §2.1).
3. Compare the recovered VASARI features to the original ground truth.

| VASARI feature | Metric | Target |
|----------------|--------|--------|
| F1 Tumour location | Top-1 accuracy | > 70 % |
| F2 Laterality | Accuracy | > 85 % |
| F4 Enhancement quality | Accuracy | > 60 % |
| F5 Proportion enhancing | Ordinal κ | > 0.5 |
| F14 Proportion of oedema | Ordinal κ | > 0.5 |
| F9 Multifocal | Accuracy | > 70 % |

This is the primary measure of **text faithfulness**: do the radiological
features described in the free-text prompt actually appear in the
synthetic scan?

### 3.2 CLIP-based text–image score

Extract CLIP embeddings of the conditioning text and of a
radiology-captioned description of the generated image (via a medical
VLM or template matching of VASARI features).  Compute cosine similarity.

| Metric | Target |
|--------|--------|
| Mean cosine similarity (CLIP ViT-L/14) | > 0.25 |

---

## 4 — Downstream utility (weeks 6–8)

Train DenseNet-121 classifiers (3D, from `monai.networks.nets`) on
four binary tasks that BraTS data supports:

| Task | Labels source | Classes |
|------|---------------|---------|
| **IDH mutation** | `2025_cancer_dataset_v2.csv` or external | IDH-mut vs IDH-wt |
| **MGMT methylation** | `2025_cancer_dataset_v2.csv` or external | MGMT+ vs MGMT− |
| **1p/19q codeletion** | `2025_cancer_dataset_v2.csv` or external | Codel vs non-codel |
| **Grade** | `2025_cancer_dataset_v2.csv` or external | LGG vs HGG |

### 4.1 Experimental conditions

For each task, train under 5 data regimes and 3 seeds:

| Condition | Training data | Expected Δ vs Real-only |
|-----------|---------------|-------------------------|
| **Real-only** | 100 % real | baseline |
| **Synth-only** | 100 % synthetic | AUC within 5 % |
| **Augmented 50:50** | 50 % real + 50 % synth | AUC ↑ 2–5 % |
| **Augmented 25:75** | 25 % real + 75 % synth | AUC ↑ 1–3 % |
| **Low-data** | 20 % real + 80 % synth | AUC within 10 % of full real |

```bash
# example: augmented 50:50 on MGMT
python -m text2glioma.classification.run_experiments \
    ./data/brats_prepared/datalist_mgmt.json /runs/ \
    --config configs/cnn.yaml \
    --experiment mgmt \
    --exp_type real_synthetic \
    --ratio 0.5
```

### 4.2 Reporting

- Primary metric: **AUROC** (area under ROC curve).
- Secondary: balanced accuracy, F1, sensitivity, specificity.
- Report mean ± std over 3 seeds.
- Statistical test: paired Wilcoxon signed-rank (real-only vs augmented).

---

## 5 — Ablation studies (weeks 9–10)

### 5.1 Conditioning mode ablation

Fix `guidance_scale_text = 7.5` and sweep `guidance_scale_mask`:

| Experiment | `gs_text` | `gs_mask` | Measure |
|------------|-----------|-----------|---------|
| Text-only | 7.5 | 0.0 | FID, Dice, downstream AUC |
| Mask-only | 0.0 | 3.0 | FID, Dice, downstream AUC |
| Dual (default) | 7.5 | 3.0 | FID, Dice, downstream AUC |
| Strong mask | 7.5 | 7.5 | FID, Dice |
| Strong text | 12.0 | 0.0 | FID, Dice |

**Hypothesis:** dual conditioning achieves the best trade-off between
image quality (FID) and spatial accuracy (Dice).

### 5.2 Guidance scale sweep

Fix conditioning mode to dual and sweep both scales:

| `gs_text` | `gs_mask` | FID | Dice WT |
|-----------|-----------|-----|---------|
| 3.0 | 1.0 | – | – |
| 5.0 | 2.0 | – | – |
| 7.5 | 3.0 | – | – |
| 10.0 | 5.0 | – | – |
| 12.0 | 7.0 | – | – |

### 5.3 Sampling steps vs quality

| DDIM steps | FID | Wall time (s/vol) |
|------------|-----|-------------------|
| 25 | – | – |
| 50 | – | – |
| 100 | – | – |
| 200 | – | – |

### 5.4 Dropout ablation

| Text dropout | Mask dropout | FID | Dice |
|-------------|-------------|-----|------|
| 0.0 | 0.0 | – | – |
| 0.1 | 0.1 | – | – |
| 0.2 | 0.2 | – | – |
| 0.3 | 0.3 | – | – |
| 0.2 | 0.0 | – | – |
| 0.0 | 0.2 | – | – |

**Note:** dropout ablation requires **retraining** the LDM (6 runs ×
~2 days each on A100).  Schedule this first if GPU budget allows.

---

## 6 — Diversity & memorisation check (week 10)

### 6.1 Nearest-neighbour distance

For each synthetic volume, find the nearest real training volume by L2
distance in MedicalNet feature space.  Report the distribution of
nearest-neighbour distances.

| Metric | Target |
|--------|--------|
| Median NN distance | > 0.3 × mean inter-real distance |
| % of synth within 0.05 × min inter-real dist | < 2 % (no copies) |

### 6.2 Intra-prompt diversity

Generate 10 samples from the **same prompt** (with and without mask).
Compute pairwise MS-SSIM.

| Metric | Target |
|--------|--------|
| Intra-prompt MS-SSIM (text-only) | 0.3–0.6 |
| Intra-prompt MS-SSIM (text+mask) | 0.5–0.7 |

Text+mask should be higher (spatial layout is fixed by the mask), but
intensity/texture should still vary.

---

## 7 — Radiologist evaluation (weeks 11–12)

### 7.1 Visual Turing test

Present 2 radiologists with 50 real + 50 synthetic axial slice triplets
(T1CE + FLAIR + label overlay) in randomised order.  Ask:

> *"Is this image real or synthetic?"*  (binary, forced choice)

| Metric | Target |
|--------|--------|
| Mean accuracy | < 65 % (near chance) |
| Cohen's κ (inter-rater) | > 0.4 |

### 7.2 Quality rating

For 50 synthetic volumes, ask radiologists to rate on a 5-point Likert
scale:

1. **Anatomical plausibility** — are brain structures realistic?
2. **Tumour realism** — does the tumour look like a real glioma?
3. **Modality consistency** — are the 4 sequences internally consistent?
4. **Artefacts** — is the image free of obvious generative artefacts?

| Metric | Target |
|--------|--------|
| Mean score (all criteria) | ≥ 3.5 / 5 |
| % rated ≥ 3 on all criteria | > 80 % |

---

## 8 — Computational budget

| Stage | GPU | Time estimate |
|-------|-----|---------------|
| VAE training (300 ep) | 1× A100 80 GB | ~3 days |
| LDM training (250 ep) | 1× A100 80 GB | ~5 days |
| Dropout ablation (6× LDM) | 1× A100 80 GB | ~30 days |
| Sample generation (1 000 vols) | 1× A100 80 GB | ~8 hours |
| Round-trip segmentation | 1× A100 80 GB | ~4 hours |
| FID computation | CPU / 1× GPU | ~1 hour |
| Downstream classifiers (5 conditions × 4 tasks × 3 seeds) | 1× A100 | ~5 days |
| **Total GPU time** | | **~45 days single-GPU** |

With 2× A100s, the parallelisable stages (ablation training + classifier
experiments) fit within the 12-week window.

---

## 9 — Deliverables

| Week | Deliverable |
|------|-------------|
| 2 | Trained VAE + LDM checkpoints; 1 000 synthetic volumes |
| 5 | FID, MS-SSIM, round-trip Dice tables |
| 8 | Downstream classifier AUROCs (all 4 tasks, 5 conditions) |
| 10 | Ablation tables (guidance sweep, dropout, steps) |
| 12 | Radiologist evaluation scores; final validation report |

---

## 10 — Summary of targets

| Category | Key metric | Target |
|----------|-----------|--------|
| Quality | FID (per modality) | < 50 |
| Quality | MS-SSIM (intra-pair) | > 0.70 |
| Mask fidelity | Dice WT (round-trip) | > 0.80 |
| Mask fidelity | Dice ET (round-trip) | > 0.60 |
| Text fidelity | VASARI location accuracy | > 70 % |
| Downstream | AUC real+synth vs real-only | ↑ 2–5 % |
| Diversity | Memorisation rate | < 2 % |
| Clinical | Turing test accuracy | < 65 % |
| Clinical | Mean quality rating | ≥ 3.5 / 5 |
