# text2glioma

Utilities for generating synthetic glioma volumes from textual descriptions
and for training lightweight downstream models. The snippets below outline a
minimal end‑to‑end workflow a newcomer can replicate.

## 1. Generate diseased images

`src/inference.py` provides `generate_images` which samples a volume from a
pathological prompt.  Passing a `healthy_prompt` in the configuration file
produces a reference scan and a difference map highlighting the lesion.

```python
from pathlib import Path
import torch
from src.inference import generate_images

image, diff = generate_images(
    model=model,                   # latent diffusion UNet
    stage1=autoencoder,            # VAE with a ``decode`` method
    scheduler=scheduler,
    text_encoder=text_encoder,
    prompt="enhancing glioma in the left frontal lobe",
    config_path=Path("configs/inference.yaml"),
    device=torch.device("cuda"),
)
```

Key sampling options are stored in `configs/inference.yaml`:

- `guidance_scale` – strength of classifier‑free guidance.
- `num_steps` – number of denoising steps.
- `depth`, `height`, `width` – spatial dimensions of the generated volume.
- `scale_factor` – latent scaling factor used by the autoencoder.
- `healthy_prompt` – optional baseline prompt to obtain a healthy reference.
- `difference_threshold` – minimum absolute difference kept in the mask.
- `save_difference` – when `true`, writes the difference map to
  `difference.pt`.

The returned `image`/`diff` pair can be saved to disk and reused for
downstream tasks.

## 2. Train downstream models with paired data

### 2a. MGMT and IDH status classifier

`src/status_classifier.py` implements simple training loops for binary
classification.  Dataloaders must yield dictionaries containing `"image"` and
`"label"` entries:

```python
from torch import optim
from dataloaders.brain_tumour_dataset import LoaderConfig, create_dataloaders
from src.status_classifier import (
    ClassifierTrainingConfig,
    train_classifier,
    evaluate_classifier,
)

files = [{"image": "image.npy", "label": 0}, {"image": "image2.npy", "label": 1}]
loaders = create_dataloaders(LoaderConfig(train_files=files, val_files=files))

model = MyModel()
optimizer = optim.Adam(model.parameters())
config = ClassifierTrainingConfig(n_epochs=5, device=torch.device("cuda"))

train_classifier(model, loaders["train"], optimizer, config=config)
accuracy = evaluate_classifier(model, loaders["val"], config.device)
```

Labels may encode MGMT promoter methylation or IDH mutation status depending on
the experiment.

### 2b. Segmentation for pathology localisation

Use `src/functions/localisation.py` or the convenience CLI to obtain lesion
masks by comparing healthy and diseased volumes and then train a small UNet:

```bash
python -m src.localisation_cli healthy_dir diseased_dir masks_dir --threshold 0.2 --epochs 5
```

`healthy_dir` and `diseased_dir` should contain paired `.npy` volumes. The
command saves masks to `masks_dir` before training a minimal segmentation model
on the diseased images and collected masks.

## 3. Inference and evaluation metrics

For inference on new prompts, call `generate_images` as in step 1.  The module
`src/evaluation.py` exposes a number of quality metrics which can be applied to
generated images or segmentations:

```python
from src.evaluation import ssim, fid, dice_coefficient

ssim_score = ssim(predictions, references)
fid_score = fid(fake_images, real_images)
dice = dice_coefficient(pred_mask, gt_mask)
```

Additional helpers such as `bertscore` and `biomedclip_accuracy` quantify text
alignment and image–text retrieval performance.

`src/prompt.py` also exposes `generate_prompt` with optional `mgmt_status` and
`idh_status` arguments to embed molecular information into prompts used for
generation, classification or reporting.

