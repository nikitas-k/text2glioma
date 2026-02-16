# Tutorial — end-to-end synthetic glioma generation

This tutorial walks through the complete workflow using the **BraTS
(Brain Tumour Segmentation)** dataset from the Medical Segmentation
Decathlon, downloaded directly through MONAI's `DecathlonDataset`.

## Prerequisites

```bash
pip install -e ".[gpu-monitor]"
```

MONAI is already a dependency, so `monai.apps.DecathlonDataset` is
available out of the box.

---

## Pre-caching models for HPC (no internet on compute nodes)

Most HPC clusters (e.g. NCI Gadi) block outbound internet access on
compute nodes.  All external model weights must be downloaded **on a
login node** before submitting batch jobs.  The commands below populate
the HuggingFace cache (`~/.cache/huggingface/hub` by default) so that
training and inference scripts can load models with
`local_files_only=True`, which is already the default in text2glioma.

### Set a shared cache directory (recommended)

Point the HuggingFace cache at a location on your scratch or project
filesystem so that all jobs (and all nodes in multi-node runs) share the
same cache:

```bash
# Add to your ~/.bashrc or PBS job script
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"
```

> All `from_pretrained` calls in text2glioma honour `HF_HOME`.
> You can also pass `--cache_dir /scratch/$USER/hf_cache` to the
> Stage 2 training and inference CLIs.

### CLIP (Stable Diffusion 2.1 — default text encoder)

Required by **Stage 2 training** and **inference** when using
`configs/ldm.yaml`:

```bash
python -c "
from transformers import AutoTokenizer, CLIPTextModel
AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='tokenizer')
CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='text_encoder')
print('CLIP tokenizer + text encoder cached ✓')
"
```

### RadBERT (radiology-specific text encoder)

Required by **Stage 2 training** and **inference** when using
`configs/ldm_radbert.yaml`:

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('StanfordAIMI/RadBERT')
AutoModel.from_pretrained('StanfordAIMI/RadBERT')
print('RadBERT tokenizer + model cached ✓')
"
```

### MedicalNet (perceptual loss — Stage 1)

The `PerceptualLoss` from monai-generative downloads
**MedicalNet ResNet-50** weights on first use.  Pre-cache by running a
throwaway forward pass on the login node:

```bash
python -c "
from generative.losses.perceptual import PerceptualLoss
import torch
p = PerceptualLoss(spatial_dims=3, network_type='medicalnet_resnet50_23datasets', is_fake_3d=False)
_ = p(torch.randn(1,1,32,32,32), torch.randn(1,1,32,32,32))
print('MedicalNet perceptual loss cached ✓')
"
```

### BraTS dataset (MONAI DecathlonDataset)

If you are using `DecathlonDataset` for DDP training (Step 2 / Step 3),
pre-download the data on the login node:

```bash
python -c "
from monai.apps import DecathlonDataset
DecathlonDataset(root_dir='/scratch/$USER/data', task='Task01_BrainTumour', section='training', download=True, transform=())
print('BraTS downloaded ✓')
"
```

### One-shot cache script

For convenience, you can cache everything in one go:

```bash
#!/usr/bin/env bash
# cache_models.sh — run on the login node before submitting jobs
set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
mkdir -p "$HF_HOME"

python - <<'EOF'
from transformers import AutoTokenizer, AutoModel, CLIPTextModel
from generative.losses.perceptual import PerceptualLoss
import torch

# ---- CLIP (default LDM encoder) ----
AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
print("[1/3] CLIP cached ✓")

# ---- RadBERT (optional radiology encoder) ----
AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT")
AutoModel.from_pretrained("StanfordAIMI/RadBERT")
print("[2/3] RadBERT cached ✓")

# ---- MedicalNet perceptual loss (Stage 1) ----
p = PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False)
_ = p(torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 32, 32, 32))
print("[3/3] MedicalNet cached ✓")
EOF

echo "All models pre-cached in $HF_HOME"
```

### Verifying the cache works offline

Set `HF_HUB_OFFLINE=1` to simulate a compute-node environment, then try
loading the models:

```bash
HF_HUB_OFFLINE=1 python -c "
from transformers import AutoTokenizer, CLIPTextModel, AutoModel
AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='tokenizer')
CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='text_encoder')
AutoTokenizer.from_pretrained('StanfordAIMI/RadBERT')
AutoModel.from_pretrained('StanfordAIMI/RadBERT')
print('Offline load OK ✓')
"
```

If this succeeds, your PBS jobs will work without internet.

---

## Step 0 — Download BraTS via MONAI DecathlonDataset

`DecathlonDataset` downloads and extracts **Task01_BrainTumour**
automatically.  Each subject contains a 4-channel NIfTI
`[D, H, W, 4]` (FLAIR / T1w / T1CE / T2w) and a matching integer
segmentation label.

```python
from monai.apps import DecathlonDataset

# First call downloads ~1.5 GB; subsequent calls use the cached data.
train_ds = DecathlonDataset(
    root_dir="./data",
    task="Task01_BrainTumour",
    section="training",
    download=True,
    seed=42,
    val_frac=0.2,
    transform=(),           # raw paths only; transforms applied later
)

val_ds = DecathlonDataset(
    root_dir="./data",
    task="Task01_BrainTumour",
    section="validation",
    download=False,          # already downloaded above
    seed=42,
    val_frac=0.2,
    transform=(),
)

# Check dataset properties
props = train_ds.get_properties(keys=["modality", "labels"])
print(props)
# {'modality': {'0': 'FLAIR', '1': 'T1w', '2': 't1gd', '3': 'T2w'},
#  'labels':   {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumour'}}
```

### Channel order

The MSD BraTS channel order is **FLAIR / T1 / T1CE / T2** (indices
`0, 1, 2, 3`), whereas text2glioma expects **T1 / T1CE / T2 / FLAIR**.
Reorder the channels before saving:

```python
import nibabel as nib
import numpy as np
from pathlib import Path

MSD_TO_T2G = [1, 2, 3, 0]   # T1, T1CE, T2, FLAIR

def reorder_brats(src_img_path, dst_img_path):
    """Reorder MSD BraTS channels → text2glioma convention."""
    nii = nib.load(str(src_img_path))
    data = nii.get_fdata()                      # [D, H, W, 4]
    data = data[..., MSD_TO_T2G]                # reorder → [D, H, W, 4]
    nib.save(nib.Nifti1Image(data, nii.affine), str(dst_img_path))
```

### Prepare the data directory

The following script downloads the dataset, reorders channels, and
constructs the file layout expected by text2glioma:

```python
import json, shutil

data_root   = Path("./data/Task01_BrainTumour")
output_root = Path("./data/brats_prepared")
img_dir     = output_root / "images"
lbl_dir     = output_root / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

# Iterate over the raw dataset_json produced by MSD
with open(data_root / "dataset.json") as f:
    msd_meta = json.load(f)

records = []
for entry in msd_meta["training"]:
    src_img = data_root / entry["image"]
    src_lbl = data_root / entry["label"]
    subj_id = Path(entry["image"]).stem.replace(".nii", "")

    dst_img = img_dir / f"{subj_id}.nii.gz"
    dst_lbl = lbl_dir / f"{subj_id}.nii.gz"

    # Reorder channels
    reorder_brats(src_img, dst_img)
    # Copy label as-is (MSD labels: 0=bg, 1=edema, 2=non-enh, 3=enh)
    shutil.copy2(src_lbl, dst_lbl)

    records.append({"image": str(dst_img), "label": str(dst_lbl), "subject_id": subj_id})

print(f"Prepared {len(records)} subjects in {output_root}")
```

### MSD BraTS label convention

| Value | MSD meaning | text2glioma default |
|-------|-------------|---------------------|
| 0 | Background | Background |
| 1 | Edema | Non-enhancing tumour |
| 2 | Non-enhancing tumour | Edema |
| 3 | Enhancing tumour | Enhancing tumour |

Labels 1 and 2 are **swapped** compared to the text2glioma default.
The `create_prompts` CLI and `vasari_auto` functions accept
`--nonenhancing_label` / `--oedema_label` flags to handle this, so
**no label remapping is needed** — just pass the MSD values.  The
one-hot mask conditioning is label-value-agnostic: it only requires
consistency between training and inference.

---

## Step 1 — Create the datalist and prompts

text2glioma includes a VASARI-auto based prompt composer that analyses each
segmentation to produce short "impression" and long "findings" prompts.

### CLI (with MSD BraTS label values)

```bash
create_prompts \
    --input_dir ./data/brats_prepared/images/ \
    --output_dir ./data/brats_prepared/ \
    --label_dir ./data/brats_prepared/labels/ \
    --file_extension .nii.gz \
    --train_split 0.8 \
    --seed 42 \
    --nonenhancing_label 2 \
    --oedema_label 1
```

> **Note:** `--nonenhancing_label 2 --oedema_label 1` tells the prompt
> composer to use the MSD label convention (label 1 → edema, label 2 →
> non-enhancing tumour).

### Output format

This creates `./data/brats_prepared/datalist.json`:

```json
{
  "training": [
    {
      "image": "./data/brats_prepared/images/BRATS_001.nii.gz",
      "label": "./data/brats_prepared/labels/BRATS_001.nii.gz",
      "subject_id": "BRATS_001",
      "impression": "Right-sided temporal lobe moderate-sized mass, contrast enhancement with thick rim, moderate vasogenic edema",
      "findings": "Location: right-sided temporal lobe. Lesion size class: moderate; volume ≈ 62.3 mL. Enhancement quality: enhancing component 33–67%. Edema: moderate vasogenic edema (~45 mL)."
    }
  ],
  "validation": [ ... ]
}
```

### Python API

```python
from text2glioma.preprocessing.utils import compose_radiology_prompts

result = compose_radiology_prompts(
    image_path="./data/brats_prepared/images/BRATS_001.nii.gz",
    label_path="./data/brats_prepared/labels/BRATS_001.nii.gz",
    nonenhancing_label=2,   # MSD convention
    edema_label=1,          # MSD convention
)
print(result["short"])   # one-line impression
print(result["long"])    # detailed findings
```

---

## Step 2 — Train the VAE (Stage 1)

The VAE compresses 4-channel MRI volumes from `160 × 224 × 160` to a
latent space of shape `3 × 10 × 14 × 10` (4 downsample levels).

### Config highlights (`configs/stage1.yaml`)

```yaml
model:
  params:
    in_channels: 4    # T1, T1CE, T2, FLAIR
    out_channels: 4
    latent_channels: 3
    num_channels: [32, 64, 128, 128]
```

### Run training

```bash
train_stage1 ./data/brats_prepared/datalist.json \
    --config configs/stage1.yaml \
    --run_dir /runs/ \
    --batch_size 2 \
    --num_epochs 300 \
    --val_interval 5 \
    --device cuda
```

### What to expect

- **Reconstruction loss (L1)** should drop below 0.02 by epoch 100.
- **Perceptual loss** is computed per channel (MedicalNet ResNet-50 expects
  single-channel input) and averaged — watch the `p_loss` scalar in
  TensorBoard.
- **Discriminator** starts contributing after the generator is reasonably
  trained (controlled by `adv_weight`).
- Reconstructions are logged to TensorBoard as 4-row grids (one row per
  modality).

### Monitor training

```bash
tensorboard --logdir /runs/text2glioma/autoencoder_stage1/output/logs/
```

---

## Step 3 — Train the LDM (Stage 2)

The LDM operates in the frozen VAE latent space.  It receives:

- **Text conditioning** via cross-attention (CLIP embeddings, 1024-d).
- **Mask conditioning** via channel concatenation (4-class one-hot
  downsampled to latent spatial dims → 4 extra channels).

Total input channels = 3 (latent) + 4 (mask) = **7**.

### Config highlights (`configs/ldm.yaml`)

```yaml
model:
  params:
    in_channels: 7     # 3 latent + 4 mask
    out_channels: 3    # predict noise in latent space
    cross_attention_dim: 1024

mask:
  num_classes: 4
  dropout_p: 0.2       # independent mask dropout for CFG

conditioning:
  dropout_p: 0.2       # independent text dropout for CFG
```

### Run training

```bash
train_stage2 ./data/brats_prepared/datalist.json \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri /runs/text2glioma/autoencoder_stage1/output/final_model.pth \
    --run_dir /runs/ \
    --batch_size 2 \
    --n_epochs 250 \
    --val_interval 5 \
    --train_spec impression \
    --device cuda
```

### Classifier-free guidance training

Both text and mask are independently dropped with probability 0.2 during
training.  This means the model learns four modes:

| Text | Mask | Probability |
|------|------|-------------|
| ✓ | ✓ | 0.64 (both kept) |
| ✓ | ✗ | 0.16 (mask dropped) |
| ✗ | ✓ | 0.16 (text dropped) |
| ✗ | ✗ | 0.04 (both dropped) |

### Monitor training

```bash
tensorboard --logdir /runs/text2glioma/ldm_stage2/output/logs/
```

Unconditional samples are logged every `2 × val_interval` epochs.

---

## Step 4 — Generate synthetic images

### CLI (batch generation)

Create a `prompts.json` file:

```json
[
  {
    "impression": "Large enhancing glioma in the right frontal lobe with severe mass effect",
    "label": "./data/brats_prepared/labels/BRATS_001.nii.gz"
  },
  {
    "impression": "Small non-enhancing lesion in the left temporal lobe, mild edema"
  }
]
```

```bash
sample prompts.json /output/ \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri /runs/.../final_model.pth \
    --model_ckpt /runs/.../best_model.pth \
    --n_samples 10 \
    --guidance_scale_text 7.5 \
    --guidance_scale_mask 3.0 \
    --ddim_steps 50 \
    --verbose
```

Output: one 4D NIfTI per sample in `/output/text2glioma/inference/output/`.

### Python API

```python
from text2glioma.utils import load_config, get_model, stage1_ify
from text2glioma.inference.inference_functions import GenericSampler
from text2glioma.inference.saver import NiftiSaver
from transformers import AutoTokenizer, CLIPTextModel
from generative.networks.schedulers import DDIMScheduler
import nibabel as nib
import numpy as np
import torch

# --- Load models ---
config = load_config("configs/ldm.yaml")
s1_config = load_config("configs/stage1.yaml")

stage1 = stage1_ify(get_model("AutoencoderKL", s1_config, from_file="stage1.pth"))
stage1.eval()

model = get_model("DiffusionModelUNet", config)
model.load_state_dict(torch.load("ldm.pth", map_location="cpu"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder"
)
scheduler = DDIMScheduler(**config["scheduler"]["params"])

device = torch.device("cuda")
for m in [stage1, model, text_encoder]:
    m.to(device)

sampler = GenericSampler(
    stage1, model, scheduler, tokenizer, text_encoder, device=device
)

# --- Text-only generation ---
images = sampler.sample(
    steps=50,
    batch_size=1,
    latent_shape=(3, 10, 14, 10),
    texts=["Heterogeneously enhancing mass in the right parietal lobe"],
    guidance_scale_text=7.5,
    guidance_scale_mask=0.0,
)
print(images.shape)  # torch.Size([1, 4, 160, 224, 160])

# --- Text + mask generation (using a BraTS label as spatial template) ---
label = nib.load("./data/brats_prepared/labels/BRATS_001.nii.gz").get_fdata().astype(np.float32)
mask = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
mask = torch.nn.functional.interpolate(mask, size=(160, 224, 160), mode="nearest")

images = sampler.sample(
    steps=50,
    batch_size=1,
    latent_shape=(3, 10, 14, 10),
    texts=["Heterogeneously enhancing mass in the right parietal lobe"],
    masks=mask.to(device),
    guidance_scale_text=7.5,
    guidance_scale_mask=3.0,
)

# --- Save ---
saver = NiftiSaver("output/")
saver.save(images[0].cpu(), "synthetic_001.nii.gz")
```

---

## Step 5 — Downstream classification (optional)

text2glioma includes a simple framework for evaluating the utility of
synthetic data in downstream tasks like MGMT methylation or IDH mutation
prediction.

```bash
python -m text2glioma.classification.run_experiments \
    ./data/brats_prepared/datalist_mgmt.json /runs/ \
    --config configs/cnn.yaml \
    --experiment mgmt \
    --exp_type real_synthetic \
    --ratio 0.5
```

This trains a DenseNet-121 classifier using 50 % synthetic + 50 % real data,
and logs accuracy on the real-only validation set.

---

## Distributed training with DecathlonDataset (DDP)

Two dedicated DDP CLIs are provided:

- **`train_stage1_ddp`** — Stage-1 VAE training
- **`train_stage2_ddp`** — Stage-2 LDM training (frozen VAE + CLIP)

Both CLIs:

1. Download **BraTS (Task01_BrainTumour)** via `monai.apps.DecathlonDataset`
   (only on rank 0 — other ranks wait).
2. Reorder channels from MSD order (FLAIR/T1/T1CE/T2) → pipeline order
   (T1/T1CE/T2/FLAIR) automatically.
3. Wrap trainable models in `DistributedDataParallel`.
4. Use `DistributedSampler` with per-epoch shuffling.

No JSON datalist is needed — the dataset is managed entirely by MONAI.

### Single-node, multi-GPU (interactive)

**Stage 1 (VAE):**

```bash
torchrun --nproc_per_node=4 \
    -m text2glioma.training.train_stage1_ddp \
    --config configs/stage1.yaml \
    --run_dir /runs/ \
    --data_dir ./data \
    --batch_size 2 \
    --num_epochs 300 \
    --val_interval 5
```

**Stage 2 (LDM):**

```bash
torchrun --nproc_per_node=4 \
    -m text2glioma.training.train_stage2_ddp \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri /runs/text2glioma/autoencoder_stage1/output/models/best_model.pth \
    --run_dir /runs/ \
    --data_dir ./data \
    --batch_size 2 \
    --num_epochs 250
```

Each GPU trains on its own shard of the dataset.  Effective batch size =
`batch_size × nproc_per_node` (e.g. 2 × 4 = 8).

### Multi-node with PBS

A `torchrun` wrapper is provided at `scripts/torchrun_hpc.sh`.
It auto-detects PBS environment variables (`$PBS_JOBID`, `$PBS_NODEFILE`)
and sets `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and `NODE_RANK`
accordingly.  It also works on bare-metal (no scheduler).

**Example `qsub` submission (Stage 1):**

```bash
qsub -v TRAIN_ARGS="--nproc_per_node 4 \
        -m text2glioma.training.train_stage1_ddp \
        --config configs/stage1.yaml \
        --run_dir /scratch/$USER/runs/ \
        --data_dir /scratch/$USER/data \
        --num_epochs 300" \
        scripts/torchrun_hpc.sh
```

**Example `qsub` submission (Stage 2):**

```bash
qsub -v TRAIN_ARGS="--nproc_per_node 4 \
        -m text2glioma.training.train_stage2_ddp \
        --config configs/ldm.yaml \
        --stage1_config configs/stage1.yaml \
        --stage1_uri /scratch/$USER/runs/text2glioma/autoencoder_stage1/output/models/best_model.pth \
        --run_dir /scratch/$USER/runs/ \
        --data_dir /scratch/$USER/data \
        --num_epochs 250" \
        scripts/torchrun_hpc.sh
```

Edit the `#PBS` directives at the top of `torchrun_hpc.sh` to match
your cluster (queue name, GPU type, wall-time, etc.):

```bash
#PBS -N t2g-train
#PBS -q gpu
#PBS -l nodes=1:ppn=16:gpus=4
#PBS -l mem=128gb
#PBS -l walltime=72:00:00
```

For **multi-node** jobs, set `-l nodes=N:ppn=…:gpus=…` and the script
will pass `--nnodes=N` to `torchrun`.  Each node must be able to reach
`MASTER_ADDR:MASTER_PORT`.

### Resuming a DDP run

```bash
torchrun --nproc_per_node=4 \
    -m text2glioma.training.train_stage1_ddp \
    --config configs/stage1.yaml \
    --run_dir /runs/ \
    --resume
```

The same `--resume` flag works for `train_stage2_ddp`.

The checkpoint (`checkpoint.pth`) stores the DDP-wrapped state dict,
optimiser states, and epoch counter, so all ranks resume consistently.

### Key CLI arguments — Stage 1 DDP

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Stage-1 YAML config |
| `--run_dir` | (required) | Root output directory |
| `--data_dir` | `./data` | DecathlonDataset download path |
| `--val_frac` | `0.2` | Train/val split ratio |
| `--batch_size` | `2` | Per-GPU batch size |
| `--num_epochs` | `300` | Training epochs |
| `--val_interval` | `5` | Validate every N epochs |
| `--resume` | `false` | Resume from checkpoint |
| `--dist_backend` | `nccl` | `nccl` (GPU) or `gloo` (CPU) |
| `--find_unused_parameters` | `false` | DDP flag for models with unused params |

### Key CLI arguments — Stage 2 DDP

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | LDM YAML config (`configs/ldm.yaml`) |
| `--stage1_config` | (required) | Stage-1 VAE YAML config |
| `--stage1_uri` | (required) | Path to pretrained Stage-1 checkpoint |
| `--run_dir` | (required) | Root output directory |
| `--data_dir` | `./data` | DecathlonDataset download path |
| `--val_frac` | `0.2` | Train/val split ratio |
| `--batch_size` | `2` | Per-GPU batch size |
| `--num_epochs` | `250` | Training epochs |
| `--val_interval` | `5` | Validate every N epochs |
| `--resume` | `false` | Resume from checkpoint |
| `--train_spec` | `impression` | Text field for conditioning (`impression` or `findings`) |
| `--scale_factor` | `1.0` | Latent scale factor |
| `--mask_dropout_p` | from config | Override mask dropout probability |
| `--text_dropout_p` | from config | Override text dropout probability |
| `--cache_dir` | `None` | HuggingFace model cache directory |

### Single-GPU fallback

If `torchrun` is not used (i.e. `RANK` / `WORLD_SIZE` env vars are not
set), both scripts fall back to single-process training automatically:

```bash
python -m text2glioma.training.train_stage1_ddp \
    --config configs/stage1.yaml \
    --run_dir /runs/

python -m text2glioma.training.train_stage2_ddp \
    --config configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri /path/to/best_model.pth \
    --run_dir /runs/
```

---

## Tips and best practices

### Guidance scale tuning

| Scale | Effect |
|-------|--------|
| `guidance_scale_text = 0` | Ignore text prompt |
| `guidance_scale_text = 3–5` | Mild text influence |
| `guidance_scale_text = 7.5` | Default; strong text adherence |
| `guidance_scale_text > 10` | Risk of artefacts / over-saturation |
| `guidance_scale_mask = 0` | Text-only mode |
| `guidance_scale_mask = 1–3` | Soft spatial guidance |
| `guidance_scale_mask = 5+` | Strong spatial constraint |

### Memory management

- **Batch size 1–2** for 32 GB GPUs at full resolution.
- Use `torch.cuda.amp` (already enabled in training scripts).
- The VAE is frozen during LDM training — its parameters use no gradient
  memory.

### Resuming training

Both `train_stage1` and `train_stage2` auto-detect `checkpoint.pth` in the
run directory and resume from the last epoch.  Pass `--resume` explicitly
for Stage 1.

### Multi-GPU

**Recommended:** use `train_stage1_ddp` / `train_stage2_ddp` with
`torchrun` for true distributed training (see the DDP section above).

For the legacy `train_stage1` CLI, pass `--use_parallel` to wrap models
in `DataParallel`, or `--distributed` for `DistributedDataParallel`.
The checkpoint saving logic correctly unwraps `model.module` before saving.
