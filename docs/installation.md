# Installation

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.10 (CUDA recommended for training)
- MONAI ≥ 1.0 and MONAI Generative Models
- ~32 GB GPU VRAM for training at full resolution (160 × 224 × 160)

## From source (recommended)

```bash
git clone https://github.com/nk233/text2glioma.git
cd text2glioma
pip install -e .
```

## Optional dependency groups

| Group | Contents | Install command |
|-------|----------|-----------------|
| `eval` | torchmetrics, BERTScore, OpenCLIP | `pip install -e ".[eval]"` |
| `gpu-monitor` | pynvml, pynvml-utils | `pip install -e ".[gpu-monitor]"` |
| `docs` | Sphinx, RTD theme, MyST | `pip install -e ".[docs]"` |
| `dev` | pytest, ruff | `pip install -e ".[dev]"` |

Install everything at once:

```bash
pip install -e ".[dev,docs,eval,gpu-monitor]"
```

## Hugging Face models

The LDM uses the CLIP text encoder from Stable Diffusion 2.1.  On first run
the weights are downloaded from Hugging Face.  If your training node has no
internet access, pre-download the model:

```bash
python -c "
from transformers import AutoTokenizer, CLIPTextModel
AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='tokenizer')
CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='text_encoder')
"
```

Then pass `--cache_dir /path/to/huggingface/cache` and use
`local_files_only=True` (already the default in the CLI scripts).

## Verifying the installation

```bash
python -c "import text2glioma; print(text2glioma.__version__)"
# Expected output: 0.2.0
```
