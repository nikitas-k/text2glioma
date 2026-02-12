"""Downstream utility: classifier experiments across data regimes."""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Tasks and data regimes from validation plan §4
TASKS = ("mgmt", "1p19q", "idh", "grade")
REGIMES = {
    "real_only":       {"exp_type": "real",           "ratio": 1.0},
    "synth_only":      {"exp_type": "synthetic",      "ratio": 1.0},
    "augmented_50_50": {"exp_type": "real_synthetic",  "ratio": 0.5},
    "augmented_25_75": {"exp_type": "real_synthetic",  "ratio": 0.25},
    "low_data":        {"exp_type": "real_synthetic",  "ratio": 0.2},
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def _compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    """Run a trained model on a dataloader and return classification metrics."""
    import torch

    model.eval()
    all_labels: list = []
    all_probs: list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(class=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "auroc": _compute_auroc(y_true, y_score),
        "balanced_accuracy": _compute_balanced_accuracy(y_true, y_pred),
        "f1": _compute_f1(y_true, y_pred),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# Single experiment wrapper
# ---------------------------------------------------------------------------

def run_single_experiment(
    datalist_path: str,
    run_dir: str,
    config_path: str,
    task: str,
    regime_name: str,
    seed: int,
    n_epochs: int = 200,
    val_interval: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train one classifier and return val metrics.

    Wraps ``text2glioma.classification.experiments.run_experiment``
    with AUROC / balanced-acc / F1 evaluation on the best checkpoint.
    """
    import torch
    from monai.utils import set_determinism

    from text2glioma.classification.experiments import run_experiment
    from text2glioma.utils import load_config, get_experiment_dataloaders, get_model

    regime = REGIMES[regime_name]
    config = load_config(config_path)
    config["datalist"] = datalist_path
    config["seed"] = seed
    config["n_epochs"] = n_epochs
    config["val_interval"] = val_interval
    config["device"] = device
    config["data_ratio"] = regime["ratio"]
    config["experiment_name"] = f"{task}_{regime_name}_s{seed}"
    set_determinism(seed)

    exp_type = regime["exp_type"]
    exp_name = config["experiment_name"]
    logger.info("Running %s / %s / seed=%d", task, regime_name, seed)

    # Train
    run_experiment(
        run_dir=run_dir,
        config=config,
        experiment_name=exp_name,
        exp_type=exp_type,
        debug=False,
        resume=False,
    )

    # Load best checkpoint and evaluate
    model_path = (
        Path(run_dir) / "text2glioma" / "experiments" / exp_name / exp_type
        / "output" / "models" / "best_model.pth"
    )
    if not model_path.exists():
        model_path = model_path.parent / "final_model.pth"

    model_cfg = config.get("model", {})
    from monai.networks import nets
    model_type = model_cfg.get("name", "densenet121")
    model = getattr(nets, model_type)(**model_cfg.get("params", {}))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # Get val loader for evaluation
    with open(datalist_path) as f:
        datalist = json.load(f)

    cache_dir = Path(run_dir) / "cache" / exp_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = get_experiment_dataloaders(
        datalist=datalist,
        cache_dir=cache_dir,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", False),
        shuffle=False,
        model_type=model_type,
        ratio=regime["ratio"],
    )

    metrics = evaluate_model(model, val_loader, dev)
    metrics.update({"task": task, "regime": regime_name, "seed": seed})
    return metrics


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_downstream_grid(
    datalist_dir: str,
    run_dir: str,
    config_path: str,
    tasks: Optional[List[str]] = None,
    regimes: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_epochs: int = 200,
    val_interval: int = 10,
    device: str = "cuda",
    output_json: str = "downstream_results.json",
) -> Dict[str, Any]:
    """Run the full downstream utility grid: tasks × regimes × seeds.

    Parameters
    ----------
    datalist_dir : directory containing ``datalist_<task>.json`` files
    run_dir : root experiment directory
    config_path : path to CNN YAML config
    """
    from scipy.stats import wilcoxon

    tasks = tasks or list(TASKS)
    regimes = regimes or list(REGIMES.keys())
    seeds = seeds or [0, 1, 2]

    all_results: List[Dict] = []

    for task, regime, seed in product(tasks, regimes, seeds):
        datalist_path = str(Path(datalist_dir) / f"datalist_{task}.json")
        if not Path(datalist_path).exists():
            logger.warning("Datalist %s not found — skipping", datalist_path)
            continue
        result = run_single_experiment(
            datalist_path=datalist_path,
            run_dir=run_dir,
            config_path=config_path,
            task=task,
            regime_name=regime,
            seed=seed,
            n_epochs=n_epochs,
            val_interval=val_interval,
            device=device,
        )
        all_results.append(result)

    # Aggregate
    summary = _aggregate(all_results, tasks, regimes)

    # Statistical tests: real_only vs each augmented regime
    stat_tests = _run_stat_tests(all_results, tasks)
    summary["statistical_tests"] = stat_tests

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"raw": all_results, "summary": summary}, f, indent=2, default=str)
    logger.info("Downstream results saved to %s", out_path)
    return summary


def _aggregate(results: List[Dict], tasks, regimes) -> Dict:
    """Compute mean ± std per task × regime."""
    import pandas as pd

    df = pd.DataFrame(results)
    summary: Dict[str, Any] = {}
    for task in tasks:
        summary[task] = {}
        for regime in regimes:
            subset = df[(df["task"] == task) & (df["regime"] == regime)]
            if subset.empty:
                continue
            summary[task][regime] = {
                "auroc_mean": float(subset["auroc"].mean()),
                "auroc_std": float(subset["auroc"].std()),
                "balanced_accuracy_mean": float(subset["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(subset["balanced_accuracy"].std()),
                "f1_mean": float(subset["f1"].mean()),
                "f1_std": float(subset["f1"].std()),
                "n_runs": len(subset),
            }
    return summary


def _run_stat_tests(results: List[Dict], tasks) -> Dict:
    """Wilcoxon signed-rank: real_only vs augmented regimes."""
    from scipy.stats import wilcoxon

    import pandas as pd

    df = pd.DataFrame(results)
    tests: Dict[str, Dict] = {}

    for task in tasks:
        tests[task] = {}
        real = df[(df["task"] == task) & (df["regime"] == "real_only")]
        if real.empty:
            continue
        for regime in REGIMES:
            if regime == "real_only":
                continue
            aug = df[(df["task"] == task) & (df["regime"] == regime)]
            if aug.empty or len(aug) < 3:
                continue
            try:
                stat, pval = wilcoxon(
                    real["auroc"].values[: len(aug)],
                    aug["auroc"].values[: len(real)],
                    alternative="two-sided",
                )
                tests[task][regime] = {"statistic": float(stat), "p_value": float(pval)}
            except ValueError:
                tests[task][regime] = {"statistic": float("nan"), "p_value": float("nan")}

    return tests
