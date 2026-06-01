"""Tests for training, evaluation metrics, and diagnostic plots."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import settings
from src.data.load import load_raw
from src.data.preprocess import FEATURES
from src.train.evaluate import compute_metrics, save_confusion_matrix, save_pr_curve
from src.train.train import build_model, train_model


@pytest.fixture(scope="module")
def df():
    if not settings.data_path_abs.exists():
        pytest.skip("real AI4I dataset not present (see README)")
    return load_raw()


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(y_true, y_proba)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_build_model_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_model("randomforest", y_train=np.array([0, 1]))


def test_train_without_mlflow_returns_pipeline(df):
    result = train_model(df, "baseline", log_to_mlflow=False)
    assert result["run_id"] is None
    assert result["model"].named_steps["model"] is not None
    assert {"pr_auc", "roc_auc", "precision", "recall", "f1"} <= result["metrics"].keys()


def test_xgboost_meets_pr_auc_floor(df):
    """The main model must clear the CI gate floor (deterministic seed)."""
    result = train_model(df, "xgboost", log_to_mlflow=False)
    assert result["metrics"]["pr_auc"] >= settings.pr_auc_floor


def test_save_pr_curve_creates_png(df, tmp_path):
    result = train_model(df, "baseline", log_to_mlflow=False)
    # Reuse the fitted pipeline to score the same data for a quick plot smoke test.
    y = df["Machine failure"].to_numpy()
    proba = result["model"].predict_proba(df[FEATURES])[:, 1]
    path = save_pr_curve(y, proba, tmp_path / "pr.png")
    assert path.exists() and path.stat().st_size > 0


def test_save_confusion_matrix_creates_png(tmp_path):
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    path = save_confusion_matrix(y, proba, tmp_path / "cm.png")
    assert path.exists() and path.stat().st_size > 0


def test_train_logs_run_to_mlflow(df, tmp_path):
    """Logging path creates a real MLflow run with metrics and a model artifact."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-experiment")

    result = train_model(df, "baseline", log_to_mlflow=True)
    assert result["run_id"] is not None

    client = MlflowClient()
    run = client.get_run(result["run_id"])
    assert run.data.metrics["pr_auc"] == pytest.approx(result["metrics"]["pr_auc"])
    # Diagnostic plots live under the run; the model is a separate MLflow 3
    # logged-model entity linked back to the run via source_run_id.
    artifacts = [a.path for a in client.list_artifacts(result["run_id"])]
    assert "plots" in artifacts
    logged = mlflow.search_logged_models(output_format="list")
    assert any(m.source_run_id == result["run_id"] for m in logged)
