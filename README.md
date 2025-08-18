# text2glioma

Utilities for working with glioma imaging data.

## Lesion localisation and segmentation

The repository includes a simple two-stage workflow to derive lesion masks
and train a toy segmentation network.

1. **Mask collection** – compare a healthy reference image with a diseased
   scan using `localise_pathology` to obtain a binary lesion mask. The
   function resides in `src/functions/localisation.py`.
2. **Segmentation training** – use the generated masks to train a small
   network. A demonstration CLI is provided:

```bash
python -m src.localisation_cli healthy_dir diseased_dir masks_dir --threshold 0.2 --epochs 5
```

`healthy_dir` and `diseased_dir` should contain paired `.npy` volumes. The
command saves masks to `masks_dir` before training a minimal segmentation
model on the diseased images and collected masks.

## MGMT and IDH status classification

The repository contains lightweight helpers to train binary classifiers for
MGMT promoter methylation and IDH mutation status.  The training utilities are
implemented in `src/status_classifier.py` and operate on generic PyTorch
models.  Dataloaders are expected to yield dictionaries with `"image"` and
`"label"` entries.

```python
from src.status_classifier import (
    ClassifierTrainingConfig,
    train_classifier,
    evaluate_classifier,
)

config = ClassifierTrainingConfig(n_epochs=5, device=torch.device("cuda"))
train_classifier(model, train_loader, optimizer, config=config)
accuracy = evaluate_classifier(model, val_loader, config.device)
```

`src/prompt.py` also exposes `generate_prompt` with optional `mgmt_status` and
`idh_status` arguments to embed molecular information into text prompts used
for data generation or reporting.
