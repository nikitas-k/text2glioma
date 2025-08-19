import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "mgmt_classifier", Path(__file__).resolve().parents[1] / "src" / "mgmt_classifier.py"
)
mgmt_classifier = importlib.util.module_from_spec(spec)
sys.modules["mgmt_classifier"] = mgmt_classifier  # required for dataclasses
spec.loader.exec_module(mgmt_classifier)
train_mgmt_classifiers = mgmt_classifier.train_mgmt_classifiers
evaluate = mgmt_classifier.evaluate


def _make_dataset(n_samples: int, offset: float, rng: np.random.Generator):
    X = rng.normal(loc=offset, scale=1.0, size=(n_samples, 3))
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


def test_train_and_evaluate():
    rng = np.random.default_rng(0)
    real = _make_dataset(50, 0.0, rng)
    synthetic = _make_dataset(50, 0.5, rng)

    models = train_mgmt_classifiers(real, synthetic)
    val_X, val_y = _make_dataset(20, 0.25, rng)

    for name, model in models.items():
        acc = evaluate(model, val_X, val_y)
        assert 0.7 <= acc <= 1.0, f"low accuracy for {name}: {acc}"
