"""Smoke test for inpainting_masks.py — no GPU, no real data needed.

Runs three checks:
  1. Synthetic case with a small central tumour exercises every scenario
     deterministically and prints the realised counts.
  2. Tumour-free case → tumour-dependent scenarios must fall back to
     `healthy_blob`.
  3. Mask + masked-image shapes and value ranges sanity-check.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text2glioma.preprocessing.inpainting_masks import (  # noqa: E402
    DEFAULT_WEIGHTS,
    apply_inpainting_mask,
    mix_text_for_scenario,
    sample_inpainting_mask,
)


def _make_dummy_case(with_tumor: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image (4, 64, 64, 64), label (1, 64, 64, 64))."""
    D = H = W = 64
    # ellipsoidal "brain": filled volume in centre
    zz, yy, xx = np.ogrid[:D, :H, :W]
    brain = (((zz - 32) / 28) ** 2 + ((yy - 32) / 28) ** 2 + ((xx - 32) / 28) ** 2) <= 1.0
    image = np.zeros((4, D, H, W), dtype=np.float32)
    image[:, brain] = 0.5  # uniform "tissue"

    label = np.zeros((1, D, H, W), dtype=np.int32)
    if with_tumor:
        tumor = (((zz - 36) / 6) ** 2 + ((yy - 36) / 6) ** 2 + ((xx - 32) / 6) ** 2) <= 1.0
        label[0, tumor & brain] = 1
    return torch.from_numpy(image), torch.from_numpy(label)


def check_distribution(n: int = 500) -> None:
    image, label = _make_dummy_case(with_tumor=True)
    rng = np.random.default_rng(0)
    counts: Counter = Counter()
    for _ in range(n):
        _, scen = sample_inpainting_mask(image, label, rng=rng)
        counts[scen] += 1
    print(f"\n[Test 1] Scenario distribution over {n} draws (with tumour):")
    for k, p in DEFAULT_WEIGHTS.items():
        print(f"  {k:<14s}  drawn={counts[k]:>3d} / {n}   target={p:.2f}")


def check_fallback() -> None:
    image, label = _make_dummy_case(with_tumor=False)
    rng = np.random.default_rng(1)
    realised: Counter = Counter()
    for _ in range(200):
        _, scen = sample_inpainting_mask(image, label, rng=rng)
        realised[scen] += 1
    print("\n[Test 2] Realised scenarios on tumour-free case (200 draws):")
    for k, v in realised.items():
        print(f"  {k:<14s}  {v}")
    bad = {k for k in realised if k in ("tumor_dilated", "tumor_eroded", "tumor_exact")}
    assert not bad, f"Tumour-dependent scenario leaked: {bad}"
    print("  -> all tumour-dependent scenarios correctly fell back. OK")


def check_shapes_and_text() -> None:
    image, label = _make_dummy_case(with_tumor=True)
    rng = np.random.default_rng(2)
    mask, scen = sample_inpainting_mask(image, label, rng=rng)
    masked = apply_inpainting_mask(image, mask)
    print("\n[Test 3] Shapes/values:")
    print(f"  scenario          : {scen}")
    print(f"  mask shape        : {tuple(mask.shape)}  dtype={mask.dtype}")
    print(f"  mask voxel count  : {int(mask.sum())}")
    print(f"  masked image shape: {tuple(masked.shape)}")
    print(f"  masked: max in ROI={float((masked * mask).max()):.3f} "
          f"(should be 0)")
    text = mix_text_for_scenario("Left frontal mass with rim enhancement.", scen, rng=rng)
    print(f"  text for {scen}   : {text!r}")
    null_text = mix_text_for_scenario("...", "tumor_exact", rng=rng)
    print(f"  text for tumor_exact: {null_text!r}")
    assert mask.shape == (1, 64, 64, 64)
    assert masked.shape == image.shape
    assert float((masked * mask).max()) == 0.0
    print("  -> all assertions passed.")


if __name__ == "__main__":
    check_distribution()
    check_fallback()
    check_shapes_and_text()
    print("\nAll smoke tests passed.")
