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

Pass `--use_parallel` to wrap models in `DataParallel`.  The checkpoint
saving logic correctly unwraps `model.module` before saving.
