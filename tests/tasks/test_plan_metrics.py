import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "cup_to_container"))
from plan_metrics import plan_stats  # noqa: E402


def test_perfect_plans_score_zero():
    target = torch.rand(3, 24, 7)
    stats = plan_stats(target[:, None].repeat(1, 6, 1, 1), target)
    assert stats == pytest.approx({"plan_error_mm": 0, "merged_plan_error_mm": 0, "plan_spread_mm": 0}, abs=1e-3)


def test_shared_offset_is_error_without_spread():
    target = torch.zeros(2, 24, 7)
    plans = target[:, None].repeat(1, 6, 1, 1)
    plans[..., 0] = 0.01
    stats = plan_stats(plans, target)
    assert stats["plan_error_mm"] == pytest.approx(10)
    assert stats["merged_plan_error_mm"] == pytest.approx(10)
    assert stats["plan_spread_mm"] == pytest.approx(0, abs=1e-3)


def test_averaging_cancels_opposite_plans():
    target = torch.zeros(1, 24, 7)
    plans = target[:, None].repeat(1, 2, 1, 1)
    plans[:, 0, :, 1] = 0.02
    plans[:, 1, :, 1] = -0.02
    stats = plan_stats(plans, target)
    assert stats["plan_error_mm"] == pytest.approx(20)
    assert stats["merged_plan_error_mm"] == pytest.approx(0, abs=1e-3)
    assert stats["plan_spread_mm"] == pytest.approx(20)


def test_only_position_counts():
    target = torch.zeros(1, 24, 7)
    plans = target[:, None].repeat(1, 6, 1, 1)
    plans[..., 3:] = 1.0
    assert plan_stats(plans, target)["plan_error_mm"] == pytest.approx(0, abs=1e-3)
