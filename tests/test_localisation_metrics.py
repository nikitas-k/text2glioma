from pathlib import Path
import importlib.util

import torch
import pytest

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
