"""Tests for the FastAPI prediction service using a stubbed model."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.serve.main import app, get_model

VALID_PAYLOAD = {
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0,
    "type": "L",
}


class DummyModel:
    """Returns a fixed positive-class probability."""

    def __init__(self, proba: float):
        self._proba = proba

    def predict_proba(self, x):
        assert list(x.columns)  # receives a dataframe
        return np.array([[1 - self._proba, self._proba]])


def _force_no_model(monkeypatch):
    """Make startup find no champion, regardless of any local mlflow.db."""
    def _raise():
        raise RuntimeError("no champion registered")

    monkeypatch.setattr("src.serve.main.load_champion", _raise)


def test_health_degraded_without_model(monkeypatch):
    _force_no_model(monkeypatch)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_predict_returns_503_without_model(monkeypatch):
    _force_no_model(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503


def test_predict_with_injected_model():
    app.dependency_overrides[get_model] = lambda: DummyModel(proba=0.92)
    try:
        with TestClient(app) as client:
            resp = client.post("/predict", json=VALID_PAYLOAD)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["failure_probability"] == 0.92
    assert body["prediction"] == 1
    assert body["threshold"] == 0.5


def test_predict_below_threshold_predicts_no_failure():
    app.dependency_overrides[get_model] = lambda: DummyModel(proba=0.10)
    try:
        with TestClient(app) as client:
            resp = client.post("/predict", json=VALID_PAYLOAD)
    finally:
        app.dependency_overrides.clear()
    assert resp.json()["prediction"] == 0


def test_predict_rejects_invalid_type():
    bad = {**VALID_PAYLOAD, "type": "X"}
    app.dependency_overrides[get_model] = lambda: DummyModel(proba=0.5)
    try:
        with TestClient(app) as client:
            resp = client.post("/predict", json=bad)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "torque_nm"}
    app.dependency_overrides[get_model] = lambda: DummyModel(proba=0.5)
    try:
        with TestClient(app) as client:
            resp = client.post("/predict", json=incomplete)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422


def test_request_to_frame_uses_training_columns():
    from src.serve.main import PredictionRequest

    frame = PredictionRequest(**VALID_PAYLOAD).to_frame()
    assert list(frame.columns) == [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]
    assert frame.shape == (1, 6)


def test_metrics_endpoint():
    with TestClient(app) as client:
        body = client.get("/metrics").json()
    assert body["champion_alias"] == "champion"
    assert body["decision_threshold"] == 0.5
