"""Tests for registry promotion logic.

The decision logic is exercised with a fake client so no MLflow server is
needed; a separate end-to-end test registers against a temporary store.
"""

from __future__ import annotations

import mlflow
import pytest

from src.registry.promote import (
    CHAMPION_ALIAS,
    METRIC_TAG,
    get_champion,
    is_better,
    promote_best,
    promote_if_better,
    version_metric,
)


class FakeVersion:
    def __init__(self, version, pr_auc=None):
        self.version = version
        self.tags = {} if pr_auc is None else {METRIC_TAG: str(pr_auc)}


class FakeClient:
    """Minimal stand-in for MlflowClient's alias methods."""

    def __init__(self, champion: FakeVersion | None = None):
        self._champion = champion
        self.assignments: list[tuple[str, str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        if self._champion is None:
            raise RuntimeError("no champion")
        return self._champion

    def set_registered_model_alias(self, name, alias, version):
        self.assignments.append((name, alias, version))
        self._champion = FakeVersion(version)


def test_version_metric_parses_tag():
    assert version_metric(FakeVersion("1", 0.83)) == pytest.approx(0.83)


def test_version_metric_missing_tag_is_neg_inf():
    assert version_metric(FakeVersion("1")) == float("-inf")


def test_is_better_ties_go_to_candidate():
    assert is_better(0.80, 0.80) is True
    assert is_better(0.81, 0.80) is True
    assert is_better(0.79, 0.80) is False


def test_get_champion_returns_none_when_unset():
    assert get_champion(FakeClient(champion=None), "m") is None


def test_promote_when_no_champion_exists():
    client = FakeClient(champion=None)
    promoted = promote_if_better(client, "m", candidate_version="3", candidate_metric=0.5)
    assert promoted is True
    assert client.assignments == [("m", CHAMPION_ALIAS, "3")]


def test_promote_when_candidate_beats_champion():
    client = FakeClient(champion=FakeVersion("1", 0.70))
    promoted = promote_if_better(client, "m", candidate_version="2", candidate_metric=0.85)
    assert promoted is True
    assert client.assignments[-1] == ("m", CHAMPION_ALIAS, "2")


def test_no_promotion_when_candidate_is_worse():
    client = FakeClient(champion=FakeVersion("1", 0.90))
    promoted = promote_if_better(client, "m", candidate_version="2", candidate_metric=0.85)
    assert promoted is False
    assert client.assignments == []


def test_promote_best_end_to_end(tmp_path):
    """Register a real trained model into a temp registry and promote it."""
    from src.data.load import load_raw
    from src.train.train import train_model

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("promote-e2e")

    results = [train_model(load_raw(), "baseline", log_to_mlflow=True)]
    outcome = promote_best(results, model_name="test-classifier")

    assert outcome["promoted"] is True
    assert outcome["pr_auc"] == pytest.approx(results[0]["metrics"]["pr_auc"])

    # The champion alias resolves to the version we just registered.
    from mlflow.tracking import MlflowClient

    champ = MlflowClient().get_model_version_by_alias("test-classifier", CHAMPION_ALIAS)
    assert champ.version == outcome["version"]
