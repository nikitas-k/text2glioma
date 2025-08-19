from pathlib import Path
import importlib.util

import pytest

# Torch is an optional dependency; skip these tests if it is unavailable to
# allow the remainder of the suite to run on lightweight environments.
torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:  # pragma: no cover - best effort when torch missing
    pytest.skip("torch not installed", allow_module_level=True)

import torch

spec = importlib.util.spec_from_file_location(
    "evaluation", Path(__file__).resolve().parents[1] / "src" / "evaluation.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
dice_coefficient = evaluation.dice_coefficient


def test_dice_coefficient_perfect_overlap():
    pred = torch.ones((1, 1, 2, 2))
    gt = torch.ones((1, 1, 2, 2))
    assert dice_coefficient(pred, gt) == pytest.approx(1.0)


def test_dice_coefficient_partial_overlap():
    pred = torch.tensor([[[[1, 1], [0, 0]]]])
    gt = torch.tensor([[[[1, 0], [1, 0]]]])
    assert dice_coefficient(pred, gt) == pytest.approx(0.5)
