# text2glioma

**Text- and mask-conditioned 3D latent diffusion for synthetic multi-sequence glioma MRI generation.**

[![Documentation](https://readthedocs.org/projects/text2glioma/badge/?version=latest)](https://text2glioma.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

text2glioma generates realistic synthetic 3D brain MRI volumes (T1, T1CE, T2, FLAIR) from
**free-text radiology prompts** and **segmentation masks**, using a two-stage
latent diffusion pipeline built on [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels)
and [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) text conditioning.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Stage 1 — VAE                        │
│  4-ch MRI (T1/T1CE/T2/FLAIR) ──► AutoencoderKL ──► z     │
│  160×224×160 → latent 16×24×16, 3 channels               │
└──────────────────────────────────────────────────────────┘
                          │  z
                          ▼
┌──────────────────────────────────────────────────────────┐
│                  Stage 2 — LDM                           │
│  Text prompt ──► CLIP encoder ──► cross-attention        │
│  Seg. mask   ──► one-hot + downsample ──► concat (7 ch)  │
│  DiffusionModelUNet  (v-prediction, DDIM)                │
└──────────────────────────────────────────────────────────┘
                          │  ẑ
                          ▼
                  VAE decoder ──► 4-ch synthetic MRI
```

### Key features

| Feature | Detail |
|---------|--------|
| **Multi-sequence** | T1, T1CE, T2, FLAIR — all generated simultaneously |
| **Dual conditioning** | Free-text via cross-attention + mask via channel concatenation |
| **Dual CFG** | Independent `guidance_scale_text` / `guidance_scale_mask` |
| **VASARI prompts** | Automated radiological feature extraction for prompt creation |
| **Per-channel perceptual loss** | MedicalNet ResNet-50 computed per modality |
| **4D NIfTI output** | All four sequences saved in a single file `[D,H,W,C]` |

---

## Installation

```bash
# Core install
pip install -e .

# With evaluation metrics (BERTScore, CLIP, etc.)
pip install -e ".[eval]"

# With GPU memory monitoring
pip install -e ".[gpu-monitor]"

# Full development install
pip install -e ".[dev,docs,eval,gpu-monitor]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 1.10, CUDA recommended.

---

## Quick start

### 1. Prepare your dataset

Organise your BraTS-style data so each subject has a 4D NIfTI image
(T1/T1CE/T2/FLAIR as the 4th dimension) and a segmentation label:

```
data/
  subj001_image.nii.gz   # shape (D, H, W, 4)
  subj001_label.nii.gz   # integer labels: 0=bg, 1=nCET, 2=edema, 3=ET
  ...
```

Create the datalist JSON with VASARI-based text prompts:

```bash
create_prompts \
    --input_dir data/ \
    --output_dir data/ \
    --label_dir data/ \
    --atlas_dir /path/to/atlas_masks/
```

This produces `data/datalist.json` with train/val splits, each entry
containing `image`, `label`, `impression` (short prompt), and `findings`
(detailed prompt) fields.

### 2. Train Stage 1 (VAE)

```bash
train_stage1 data/datalist.json \
    --config configs/stage1.yaml \
    --run_dir runs/ \
    --batch_size 2 \
    --num_epochs 300 \
    --device cuda
```

### 3. Train Stage 2 (LDM)

```bash
train_stage2 data/datalist.json \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri runs/text2glioma/autoencoder_stage1/output/final_model.pth \
    --run_dir runs/ \
    --batch_size 2 \
    --n_epochs 250 \
    --device cuda
```

### 4. Generate synthetic images

```bash
sample prompts.json output/ \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri runs/.../final_model.pth \
    --model_ckpt runs/.../best_model.pth \
    --n_samples 50 \
    --guidance_scale_text 7.5 \
    --guidance_scale_mask 3.0 \
    --ddim_steps 50
```

Or use the Python API directly:

```python
import torch
from text2glioma.utils import load_config, get_model, stage1_ify
from text2glioma.inference.inference_functions import GenericSampler
from transformers import AutoTokenizer, CLIPTextModel
from generative.networks.schedulers import DDIMScheduler

config = load_config("configs/ldm.yaml")
stage1_config = load_config("configs/stage1.yaml")

# Load models
stage1 = stage1_ify(get_model("AutoencoderKL", stage1_config, from_file="stage1.pth"))
model = get_model("DiffusionModelUNet", config)
model.load_state_dict(torch.load("ldm.pth", map_location="cpu"))

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder"
)
scheduler = DDIMScheduler(**config["scheduler"]["params"])

device = torch.device("cuda")
sampler = GenericSampler(stage1, model, scheduler, tokenizer, text_encoder, device=device)

# Generate
images = sampler.sample(
    steps=50,
    batch_size=1,
    latent_shape=(3, 10, 14, 10),
    texts=["Large enhancing mass in the right temporal lobe with moderate vasogenic edema"],
    guidance_scale_text=7.5,
    guidance_scale_mask=0.0,  # text-only generation
)
# images shape: [1, 4, 160, 224, 160] — (T1, T1CE, T2, FLAIR)
```

---

## Configuration files

| File | Purpose |
|------|---------|
| `configs/stage1.yaml` | VAE architecture, learning rates, loss weights |
| `configs/ldm.yaml` | LDM UNet, scheduler, text encoder, mask settings |
| `configs/inference.yaml` | Guidance scales, spatial dimensions, mask options |
| `configs/cnn.yaml` | Downstream classifier experiments |

---

## Project structure

```
text2glioma/
├── configs/                     # YAML configuration files
├── docs/                        # Sphinx documentation (ReadTheDocs)
├── src/text2glioma/
│   ├── __init__.py              # Package version
│   ├── utils.py                 # Models, data loaders, transforms, visualisation
│   ├── preprocessing/
│   │   ├── vasari_auto.py       # Automated VASARI feature extraction
│   │   ├── utils.py             # Prompt composer (short + long)
│   │   ├── create_text.py       # CLI: create datalist with prompts
│   │   └── data_conversion.py   # DICOM-to-NIfTI converter
│   ├── training/
│   │   ├── training_functions.py # Train/eval loops for VAE and LDM
│   │   ├── train_stage1.py      # CLI: train VAE
│   │   └── train_stage2.py      # CLI: train LDM
│   ├── inference/
│   │   ├── inference_functions.py # Dual CFG sampling, GenericSampler
│   │   ├── sampler.py           # CLI: batch inference
│   │   └── saver.py             # Multi-channel NIfTI saver
│   ├── classification/
│   │   ├── experiments.py       # Downstream classification training
│   │   └── run_experiments.py   # CLI: run classification experiments
│   └── testing/                 # (future) unit tests
├── tests/                       # Test suite
├── CHANGELOG.md
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Conditioning modes

text2glioma supports flexible conditioning at inference:

| Mode | `guidance_scale_text` | `guidance_scale_mask` | Mask input |
|------|----------------------|----------------------|------------|
| Text-only | 7.5 | 0.0 | None |
| Text + mask | 7.5 | 3.0 | Segmentation NIfTI |
| Mask-only | 0.0 | 3.0 | Segmentation NIfTI |
| Unconditional | 0.0 | 0.0 | None |

During training, both text and mask are independently dropped out (default
20 % each) to learn all four conditioning modes simultaneously.

---

## Citation

If you use this work, please cite:

```bibtex
@software{text2glioma2026,
  author = {Koussis, Nikitas},
  title  = {text2glioma: Text- and Mask-Conditioned 3D Latent Diffusion for Synthetic Glioma MRI},
  year   = {2026},
  url    = {https://github.com/nk233/text2glioma},
}
```

## License

[MIT](LICENSE) © 2025 Nikitas Koussis

