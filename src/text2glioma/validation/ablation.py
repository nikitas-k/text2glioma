"""Ablation studies: guidance sweeps, DDIM steps, dropout ablation."""

from __future__ import annotations

import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default sweep grids from validation plan §5
CONDITIONING_ABLATION: List[Dict[str, float]] = [
    {"label": "text_only",  "gs_text": 7.5,  "gs_mask": 0.0},
    {"label": "mask_only",  "gs_text": 0.0,  "gs_mask": 3.0},
    {"label": "dual",       "gs_text": 7.5,  "gs_mask": 3.0},
    {"label": "strong_mask","gs_text": 7.5,  "gs_mask": 7.5},
    {"label": "strong_text","gs_text": 12.0, "gs_mask": 0.0},
]

GUIDANCE_SWEEP: List[Tuple[float, float]] = [
    (3.0,  1.0),
    (5.0,  2.0),
    (7.5,  3.0),
    (10.0, 5.0),
    (12.0, 7.0),
]

DDIM_STEPS_SWEEP: List[int] = [25, 50, 100, 200]

DROPOUT_ABLATION: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (0.1, 0.1),
    (0.2, 0.2),
    (0.3, 0.3),
    (0.2, 0.0),
    (0.0, 0.2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_samples(
    source_json: str,
    output_dir: str,
    config_path: str,
    stage1_config: str,
    stage1_uri: str,
    model_ckpt: str,
    n_samples: int,
    guidance_scale_text: float,
    guidance_scale_mask: float,
    ddim_steps: int,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 2,
) -> float:
    """Generate samples via the sampler CLI and return wall-clock time."""
    import subprocess
    import sys

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "text2glioma.inference.sampler",
        source_json, str(out),
        "--config", config_path,
        "--stage1_config", stage1_config,
        "--stage1_uri", stage1_uri,
        "--model_ckpt", model_ckpt,
        "--n_samples", str(n_samples),
        "--guidance_scale_text", str(guidance_scale_text),
        "--guidance_scale_mask", str(guidance_scale_mask),
        "--ddim_steps", str(ddim_steps),
        "--device", device,
        "--seed", str(seed),
        "--batch_size", str(batch_size),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - t0
    return elapsed


def _eval_quality(
    real_dir: str,
    synth_dir: str,
    max_n: int = 50,
) -> Dict[str, Any]:
    """Compute FID (per modality) on generated vs real."""
    from text2glioma.validation.image_quality import run_image_quality

    results = run_image_quality(
        real_dir=real_dir,
        synth_dir=synth_dir,
        output_json="/dev/null",
        max_n=max_n,
    )
    return results


def _eval_mask(
    synth_dir: str,
    gt_label_dir: str,
    max_n: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Round-trip Dice / HD95."""
    from text2glioma.validation.mask_fidelity import run_mask_fidelity

    results = run_mask_fidelity(
        synth_dir=synth_dir,
        gt_label_dir=gt_label_dir,
        output_json="/dev/null",
        device=device,
        max_n=max_n,
    )
    return results


# ---------------------------------------------------------------------------
# §5.1 / §5.2 — Guidance scale sweeps
# ---------------------------------------------------------------------------

def run_guidance_sweep(
    source_json: str,
    real_dir: str,
    gt_label_dir: str,
    config_path: str,
    stage1_config: str,
    stage1_uri: str,
    model_ckpt: str,
    output_dir: str,
    sweeps: Optional[List[Tuple[float, float]]] = None,
    n_samples: int = 50,
    ddim_steps: int = 50,
    device: str = "cuda",
    output_json: str = "guidance_sweep.json",
) -> List[Dict[str, Any]]:
    """Sweep guidance_scale_text × guidance_scale_mask."""
    sweeps = sweeps or GUIDANCE_SWEEP
    results: List[Dict[str, Any]] = []

    for gs_text, gs_mask in sweeps:
        tag = f"gs_t{gs_text}_m{gs_mask}"
        synth_dir = str(Path(output_dir) / tag)
        logger.info("Guidance sweep: gs_text=%.1f, gs_mask=%.1f", gs_text, gs_mask)

        elapsed = _generate_samples(
            source_json=source_json,
            output_dir=synth_dir,
            config_path=config_path,
            stage1_config=stage1_config,
            stage1_uri=stage1_uri,
            model_ckpt=model_ckpt,
            n_samples=n_samples,
            guidance_scale_text=gs_text,
            guidance_scale_mask=gs_mask,
            ddim_steps=ddim_steps,
            device=device,
        )

        quality = _eval_quality(real_dir, synth_dir, max_n=n_samples)
        mask = _eval_mask(synth_dir, gt_label_dir, max_n=n_samples, device=device)

        entry = {
            "gs_text": gs_text,
            "gs_mask": gs_mask,
            "wall_time_s": elapsed,
            "quality": quality,
            "mask": mask,
        }
        results.append(entry)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Guidance sweep saved to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# §5.1 — Conditioning mode ablation (convenience wrapper)
# ---------------------------------------------------------------------------

def run_conditioning_ablation(
    source_json: str,
    real_dir: str,
    gt_label_dir: str,
    config_path: str,
    stage1_config: str,
    stage1_uri: str,
    model_ckpt: str,
    output_dir: str,
    n_samples: int = 50,
    ddim_steps: int = 50,
    device: str = "cuda",
    output_json: str = "conditioning_ablation.json",
) -> List[Dict[str, Any]]:
    """Run the 5-point conditioning mode ablation (§5.1)."""
    sweeps = [(e["gs_text"], e["gs_mask"]) for e in CONDITIONING_ABLATION]
    return run_guidance_sweep(
        source_json=source_json,
        real_dir=real_dir,
        gt_label_dir=gt_label_dir,
        config_path=config_path,
        stage1_config=stage1_config,
        stage1_uri=stage1_uri,
        model_ckpt=model_ckpt,
        output_dir=output_dir,
        sweeps=sweeps,
        n_samples=n_samples,
        ddim_steps=ddim_steps,
        device=device,
        output_json=output_json,
    )


# ---------------------------------------------------------------------------
# §5.3 — DDIM steps sweep
# ---------------------------------------------------------------------------

def run_steps_sweep(
    source_json: str,
    real_dir: str,
    gt_label_dir: str,
    config_path: str,
    stage1_config: str,
    stage1_uri: str,
    model_ckpt: str,
    output_dir: str,
    steps_list: Optional[List[int]] = None,
    n_samples: int = 50,
    gs_text: float = 7.5,
    gs_mask: float = 3.0,
    device: str = "cuda",
    output_json: str = "steps_sweep.json",
) -> List[Dict[str, Any]]:
    """Sweep DDIM steps at fixed guidance scales."""
    steps_list = steps_list or DDIM_STEPS_SWEEP
    results: List[Dict[str, Any]] = []

    for steps in steps_list:
        tag = f"steps_{steps}"
        synth_dir = str(Path(output_dir) / tag)
        logger.info("Steps sweep: %d", steps)

        elapsed = _generate_samples(
            source_json=source_json,
            output_dir=synth_dir,
            config_path=config_path,
            stage1_config=stage1_config,
            stage1_uri=stage1_uri,
            model_ckpt=model_ckpt,
            n_samples=n_samples,
            guidance_scale_text=gs_text,
            guidance_scale_mask=gs_mask,
            ddim_steps=steps,
            device=device,
        )
        per_vol = elapsed / max(n_samples, 1)

        quality = _eval_quality(real_dir, synth_dir, max_n=n_samples)

        entry = {
            "ddim_steps": steps,
            "wall_time_s": elapsed,
            "time_per_volume_s": per_vol,
            "quality": quality,
        }
        results.append(entry)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Steps sweep saved to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# §5.4 — Dropout ablation (requires retraining; this is a config generator)
# ---------------------------------------------------------------------------

def generate_dropout_configs(
    base_config_path: str,
    output_dir: str,
    dropout_grid: Optional[List[Tuple[float, float]]] = None,
) -> List[str]:
    """Generate LDM training configs for each dropout setting.

    Returns list of generated YAML paths.  Actual training must be
    launched externally (each run is ~2 days on A100).
    """
    from text2glioma.utils import load_config
    import yaml

    dropout_grid = dropout_grid or DROPOUT_ABLATION
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(base_config_path)
    paths: List[str] = []

    for text_drop, mask_drop in dropout_grid:
        tag = f"drop_t{text_drop}_m{mask_drop}"
        cfg = dict(config)
        cfg.setdefault("training", {})
        cfg["training"]["text_dropout"] = text_drop
        cfg["training"]["mask_dropout"] = mask_drop
        cfg_path = out / f"{tag}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        paths.append(str(cfg_path))
        logger.info("Wrote dropout config: %s", cfg_path)

    return paths


def evaluate_dropout_checkpoints(
    checkpoint_dirs: Dict[str, str],
    source_json: str,
    real_dir: str,
    gt_label_dir: str,
    config_path: str,
    stage1_config: str,
    stage1_uri: str,
    output_dir: str,
    n_samples: int = 50,
    ddim_steps: int = 50,
    device: str = "cuda",
    output_json: str = "dropout_ablation.json",
) -> List[Dict[str, Any]]:
    """Evaluate pre-trained checkpoints for each dropout setting.

    Parameters
    ----------
    checkpoint_dirs : dict mapping dropout tag → model checkpoint path
                      e.g. {"drop_t0.2_m0.2": "/runs/drop_t0.2_m0.2/model.pt"}
    """
    results: List[Dict[str, Any]] = []

    for tag, ckpt in checkpoint_dirs.items():
        synth_dir = str(Path(output_dir) / tag)
        logger.info("Dropout ablation: %s", tag)

        elapsed = _generate_samples(
            source_json=source_json,
            output_dir=synth_dir,
            config_path=config_path,
            stage1_config=stage1_config,
            stage1_uri=stage1_uri,
            model_ckpt=ckpt,
            n_samples=n_samples,
            guidance_scale_text=7.5,
            guidance_scale_mask=3.0,
            ddim_steps=ddim_steps,
            device=device,
        )

        quality = _eval_quality(real_dir, synth_dir, max_n=n_samples)
        mask = _eval_mask(synth_dir, gt_label_dir, max_n=n_samples, device=device)

        results.append({
            "tag": tag,
            "checkpoint": ckpt,
            "wall_time_s": elapsed,
            "quality": quality,
            "mask": mask,
        })

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Dropout ablation saved to %s", out_path)
    return results
