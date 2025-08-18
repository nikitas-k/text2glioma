"""Evaluation metrics for text-to-image models."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def bertscore(predictions: Sequence[str], references: Sequence[str], lang: str = "en") -> float:
    """Return the mean BERTScore F1 between ``predictions`` and ``references``."""

    from bert_score import score as bert_score

    _, _, f1 = bert_score(list(predictions), list(references), lang=lang)
    return f1.mean().item()


def ssim(predictions: Tensor, references: Tensor) -> float:
    """Compute Structural Similarity (SSIM) for two batches.

    ``predictions`` and ``references`` may be ``(N, C, H, W)`` images or
    ``(N, C, D, H, W)`` volumes with values in ``[0, 1]``.
    """

    from torchmetrics.functional.image import structural_similarity_index_measure

    return structural_similarity_index_measure(predictions, references).item()


def fid(fake: Tensor, real: Tensor) -> float:
    """Compute the Fréchet Inception Distance between two image sets.

    ``fake`` and ``real`` may be 4D tensors ``(N, C, H, W)`` or 5D tensors
    ``(N, C, D, H, W)`` containing values in ``[0, 1]``.
    """

    from torchmetrics.image.fid import FrechetInceptionDistance

    def to4d(x: Tensor) -> Tensor:
        if x.ndim == 5:
            n, c, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
        elif x.ndim != 4:
            raise ValueError("Input must have 4 or 5 dimensions")
        return x

    metric = FrechetInceptionDistance(feature=64)
    metric.update(to4d(real) * 255, real=True)
    metric.update(to4d(fake) * 255, real=False)
    return metric.compute().item()


def biomedclip_accuracy(images: Sequence["Image.Image"], prompts: Sequence[str], device: torch.device) -> float:
    """Return retrieval accuracy of ``images`` against ``prompts`` using BiomedCLIP.

    The function embeds both images and their textual descriptions and reports
    the top‑1 retrieval accuracy.
    """

    import open_clip

    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        "hf-hub:medicalai/biomedclip-vit-base-patch16",
        pretrained="hf-hub:medicalai/biomedclip-vit-base-patch16",
    )
    model.to(device).eval()

    image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
    text_inputs = tokenizer(list(prompts)).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = image_features @ text_features.t()
    preds = similarity.argmax(dim=1)
    targets = torch.arange(len(prompts), device=device)
    return (preds == targets).float().mean().item()
