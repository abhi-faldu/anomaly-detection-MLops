"""Tests for the PR-AUC gate in the training pipeline."""

from __future__ import annotations

from src.pipeline import passes_gate


def _results(*pr_aucs: float) -> list[dict]:
    return [{"kind": f"m{i}", "metrics": {"pr_auc": v}} for i, v in enumerate(pr_aucs)]


def test_gate_passes_when_best_meets_floor():
    ok, best = passes_gate(_results(0.40, 0.82), floor=0.75)
    assert ok is True
    assert best == 0.82


def test_gate_fails_when_all_below_floor():
    ok, best = passes_gate(_results(0.40, 0.60), floor=0.75)
    assert ok is False
    assert best == 0.60


def test_gate_passes_on_exact_floor():
    ok, _ = passes_gate(_results(0.75), floor=0.75)
    assert ok is True


def test_gate_uses_configured_floor_by_default():
    # With the project default floor (0.75), a 0.90 model must pass.
    ok, _ = passes_gate(_results(0.90))
    assert ok is True
