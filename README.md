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
