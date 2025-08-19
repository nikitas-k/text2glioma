"""Lightweight MGMT status classifier using logistic regression.

This module provides a minimal implementation of logistic regression for
predicting MGMT promoter methylation from tabular features.  It offers
utility functions to train separate models on real data, synthetic data and
their combination.

The implementation avoids heavy dependencies such as PyTorch and relies only
on NumPy, making it suitable for quick experiments or environments where
installing large frameworks is impractical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "LogisticRegression",
    "train_mgmt_classifiers",
    "evaluate",
]


@dataclass
class LogisticRegression:
    """Simple logistic regression binary classifier.

    Parameters
    ----------
    n_features:
        Number of input features.
    lr:
        Learning rate used during gradient descent optimisation.
    n_epochs:
        Number of passes over the training data.
    """

    n_features: int
    lr: float = 0.1
    n_epochs: int = 100

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=float)
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier on features ``X`` and labels ``y``."""
        for _ in range(self.n_epochs):
            logits = X @ self.weights + self.bias
            probs = self._sigmoid(logits)
            error = probs - y
            grad_w = (X.T @ error) / X.shape[0]
            grad_b = error.mean()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(N, 2)`` array of class probabilities for ``X``."""
        logits = X @ self.weights + self.bias
        probs = self._sigmoid(logits)
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for ``X``."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """Return classification accuracy of ``model`` on ``(X, y)``."""
    preds = model.predict(X)
    return (preds == y).mean().item()


def _train_single(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(n_features=X.shape[1])
    model.fit(X, y)
    return model


def train_mgmt_classifiers(
    real: Tuple[np.ndarray, np.ndarray],
    synthetic: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, LogisticRegression]:
    """Train MGMT classifiers on real, synthetic and combined datasets.

    Parameters
    ----------
    real:
        Tuple ``(X, y)`` with real features and labels.
    synthetic:
        Tuple ``(X, y)`` with synthetic features and labels.

    Returns
    -------
    dict
        Mapping with keys ``"real"``, ``"real+synthetic"`` and ``"synthetic"``
        corresponding to the respective trained models.
    """

    X_real, y_real = real
    X_syn, y_syn = synthetic

    models = {
        "real": _train_single(X_real, y_real),
        "synthetic": _train_single(X_syn, y_syn),
    }

    X_comb = np.concatenate([X_real, X_syn])
    y_comb = np.concatenate([y_real, y_syn])
    models["real+synthetic"] = _train_single(X_comb, y_comb)
    return models
