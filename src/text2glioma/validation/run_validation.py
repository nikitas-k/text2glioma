"""CLI entry point for the validation pipeline.

Usage::

    text2glioma-validate --config configs/validation.yaml
    python -m text2glioma.validation.run_validation --config configs/validation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve(val: str, paths: Dict[str, str]) -> str:
    """Resolve ``${paths.key}`` placeholders in config strings."""
    if not isinstance(val, str):
        return val
    import re

    def _repl(m):
        key = m.group(1)
        return str(paths.get(key, m.group(0)))

    return re.sub(r"\$\{paths\.(\w+)\}", _repl, val)


def _resolve_dict(d: Any, paths: Dict[str, str]) -> Any:
    """Recursively resolve path references in a config dict."""
    if isinstance(d, dict):
        return {k: _resolve_dict(v, paths) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_dict(v, paths) for v in d]
    if isinstance(d, str):
        return _resolve(d, paths)
    return d


def parse_args():
    parser = argparse.ArgumentParser(description="Run text2glioma validation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to validation YAML config.")
    parser.add_argument("--steps", nargs="*", default=None,
                        help="Run only these steps (e.g. image_quality mask_fidelity). Default: all enabled.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--max-n", type=int, default=None, help="Override max_n sample cap.")
    return parser.parse_args()


def main():
    args = parse_args()
    from text2glioma.utils import load_config

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    cfg = _resolve_dict(cfg, paths)

    device = args.device or cfg.get("device", "cpu")
    max_n = args.max_n if args.max_n is not None else cfg.get("max_n")
    steps = args.steps  # None → run all enabled

    results: Dict[str, Any] = {}

    # ---------------------------------------------------------------
    # §1 — Image quality
    # ---------------------------------------------------------------
    iq = cfg.get("image_quality", {})
    if iq.get("enabled") and (steps is None or "image_quality" in steps):
        logger.info("=== §1 Image Quality ===")
        from text2glioma.validation.image_quality import run_image_quality

        results["image_quality"] = run_image_quality(
            real_dir=paths["real_dir"],
            synth_dir=paths["synth_dir"],
            output_json=iq["output_json"],
            device=device,
            max_n=max_n,
        )

    # ---------------------------------------------------------------
    # §2 — Mask fidelity
    # ---------------------------------------------------------------
    mf = cfg.get("mask_fidelity", {})
    if mf.get("enabled") and (steps is None or "mask_fidelity" in steps):
        logger.info("=== §2 Mask Fidelity ===")
        from text2glioma.validation.mask_fidelity import run_mask_fidelity

        results["mask_fidelity"] = run_mask_fidelity(
            synth_dir=paths["synth_dir"],
            gt_label_dir=paths["real_label_dir"],
            output_json=mf["output_json"],
            device=device,
            max_n=max_n,
        )

    # ---------------------------------------------------------------
    # §3 — Text alignment
    # ---------------------------------------------------------------
    ta = cfg.get("text_alignment", {})
    if ta.get("enabled") and (steps is None or "text_alignment" in steps):
        logger.info("=== §3 Text Alignment ===")
        from text2glioma.validation.text_alignment import run_text_alignment

        results["text_alignment"] = run_text_alignment(
            gt_label_dir=paths["real_label_dir"],
            synth_label_dir=paths.get("synth_label_dir", paths["real_label_dir"]),
            atlas_dir=paths["atlas_dir"],
            prompts_json=paths["prompts_json"],
            synth_image_dir=paths["synth_dir"],
            output_json=ta["output_json"],
            enhancing_label=ta.get("enhancing_label", 3),
            nonenhancing_label=ta.get("nonenhancing_label", 2),
            oedema_label=ta.get("oedema_label", 1),
            device=device,
            max_n=max_n,
        )

    # ---------------------------------------------------------------
    # §4 — Downstream utility
    # ---------------------------------------------------------------
    du = cfg.get("downstream_utility", {})
    if du.get("enabled") and (steps is None or "downstream_utility" in steps):
        logger.info("=== §4 Downstream Utility ===")
        from text2glioma.validation.downstream_utility import run_downstream_grid

        results["downstream_utility"] = run_downstream_grid(
            datalist_dir=paths["datalist_dir"],
            run_dir=str(Path(paths["output_dir"]) / "downstream_runs"),
            config_path=paths["cnn_config"],
            tasks=du.get("tasks"),
            regimes=du.get("regimes"),
            seeds=du.get("seeds"),
            n_epochs=du.get("n_epochs", 200),
            val_interval=du.get("val_interval", 10),
            device=device,
            output_json=du["output_json"],
        )

    # ---------------------------------------------------------------
    # §5 — Ablation studies
    # ---------------------------------------------------------------
    abl = cfg.get("ablation", {})
    ablation_common = {
        "source_json": paths.get("source_json", ""),
        "real_dir": paths["real_dir"],
        "gt_label_dir": paths["real_label_dir"],
        "config_path": paths.get("ldm_config", ""),
        "stage1_config": paths.get("stage1_config", ""),
        "stage1_uri": paths.get("stage1_uri", ""),
        "model_ckpt": paths.get("model_ckpt", ""),
        "output_dir": str(Path(paths["output_dir"]) / "ablation"),
        "device": device,
    }

    gs = abl.get("guidance_sweep", {})
    if gs.get("enabled") and (steps is None or "guidance_sweep" in steps):
        logger.info("=== §5.2 Guidance Sweep ===")
        from text2glioma.validation.ablation import run_guidance_sweep

        results["guidance_sweep"] = run_guidance_sweep(
            **ablation_common,
            n_samples=gs.get("n_samples", 50),
            ddim_steps=gs.get("ddim_steps", 50),
            output_json=gs["output_json"],
        )

    ca = abl.get("conditioning_ablation", {})
    if ca.get("enabled") and (steps is None or "conditioning_ablation" in steps):
        logger.info("=== §5.1 Conditioning Ablation ===")
        from text2glioma.validation.ablation import run_conditioning_ablation

        results["conditioning_ablation"] = run_conditioning_ablation(
            **ablation_common,
            n_samples=ca.get("n_samples", 50),
            ddim_steps=ca.get("ddim_steps", 50),
            output_json=ca["output_json"],
        )

    ss = abl.get("steps_sweep", {})
    if ss.get("enabled") and (steps is None or "steps_sweep" in steps):
        logger.info("=== §5.3 Steps Sweep ===")
        from text2glioma.validation.ablation import run_steps_sweep

        results["steps_sweep"] = run_steps_sweep(
            **ablation_common,
            n_samples=ss.get("n_samples", 50),
            gs_text=ss.get("gs_text", 7.5),
            gs_mask=ss.get("gs_mask", 3.0),
            output_json=ss["output_json"],
        )

    dp = abl.get("dropout", {})
    if dp.get("enabled") and (steps is None or "dropout_configs" in steps):
        logger.info("=== §5.4 Dropout Config Generation ===")
        from text2glioma.validation.ablation import generate_dropout_configs

        dropout_paths = generate_dropout_configs(
            base_config_path=paths.get("ldm_config", ""),
            output_dir=dp.get("output_dir", "./dropout_configs"),
        )
        results["dropout_configs"] = dropout_paths

    va = abl.get("vae_ablation", {})
    if va.get("enabled") and (steps is None or "vae_ablation" in steps):
        logger.info("=== §5.5 VAE Ablation (pretrained vs frozen vs from-scratch) ===")
        from text2glioma.validation.ablation import run_vae_ablation

        results["vae_ablation"] = run_vae_ablation(
            conditions=va.get("conditions", []),
            source_json=paths.get("source_json", ""),
            real_dir=paths["real_dir"],
            gt_label_dir=paths["real_label_dir"],
            output_dir=str(Path(paths["output_dir"]) / "vae_ablation"),
            n_samples=va.get("n_samples", 50),
            ddim_steps=va.get("ddim_steps", 50),
            gs_text=va.get("gs_text", 7.5),
            gs_mask=va.get("gs_mask", 3.0),
            device=device,
            output_json=va["output_json"],
        )

    # ---------------------------------------------------------------
    # §6 — Diversity & memorisation
    # ---------------------------------------------------------------
    dv = cfg.get("diversity", {})
    if dv.get("enabled") and (steps is None or "diversity" in steps):
        logger.info("=== §6 Diversity & Memorisation ===")
        from text2glioma.validation.diversity import run_diversity

        results["diversity"] = run_diversity(
            real_dir=paths["real_dir"],
            synth_dir=paths["synth_dir"],
            prompt_dirs=dv.get("prompt_dirs"),
            device=device,
            max_n=max_n,
            output_json=dv["output_json"],
        )

    # ---------------------------------------------------------------
    # §7 — Radiologist evaluation
    # ---------------------------------------------------------------
    re_cfg = cfg.get("radiologist_eval", {})

    tt = re_cfg.get("turing_test", {})
    if tt.get("enabled") and (steps is None or "turing_test" in steps):
        logger.info("=== §7.1 Turing Test Prep ===")
        from text2glioma.validation.radiologist_eval import prepare_turing_test

        results["turing_test"] = prepare_turing_test(
            real_dir=paths["real_dir"],
            synth_dir=paths["synth_dir"],
            real_label_dir=paths.get("real_label_dir"),
            synth_label_dir=paths.get("synth_label_dir"),
            output_dir=tt["output_dir"],
            n_each=tt.get("n_each", 50),
            channels=tuple(tt.get("channels", [1, 3])),
            seed=cfg.get("seed", 42),
        )

    qr = re_cfg.get("quality_rating", {})
    if qr.get("enabled") and (steps is None or "quality_rating" in steps):
        logger.info("=== §7.2 Quality Rating Prep ===")
        from text2glioma.validation.radiologist_eval import prepare_quality_rating

        results["quality_rating"] = prepare_quality_rating(
            synth_dir=paths["synth_dir"],
            synth_label_dir=paths.get("synth_label_dir"),
            output_dir=qr["output_dir"],
            n_samples=qr.get("n_samples", 50),
            channels=tuple(qr.get("channels", [1, 3])),
            seed=cfg.get("seed", 42),
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary_path = Path(paths.get("output_dir", ".")) / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Validation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
