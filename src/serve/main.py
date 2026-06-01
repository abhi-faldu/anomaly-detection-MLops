"""FastAPI prediction service.

Loads the ``champion`` model from the MLflow registry at startup and serves
failure-probability predictions.

Run:  uvicorn src.serve.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import settings
from src.data.preprocess import NUMERIC_FEATURES
from src.registry.promote import CHAMPION_ALIAS

DECISION_THRESHOLD = 0.5

# Maps tidy request fields to the exact training feature column names.
FIELD_TO_COLUMN = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "type": "Type",
}


class PredictionRequest(BaseModel):
    """One machine's sensor reading."""

    air_temperature_k: float = Field(..., examples=[300.0])
    process_temperature_k: float = Field(..., examples=[310.0])
    rotational_speed_rpm: float = Field(..., examples=[1500.0])
    torque_nm: float = Field(..., examples=[40.0])
    tool_wear_min: float = Field(..., examples=[100.0])
    type: Literal["L", "M", "H"] = Field(..., examples=["L"])

    def to_frame(self) -> pd.DataFrame:
        """Build a single-row dataframe with the model's expected columns."""
        row = {col: getattr(self, field) for field, col in FIELD_TO_COLUMN.items()}
        return pd.DataFrame([row])[[*NUMERIC_FEATURES, "Type"]]


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    threshold: float
    model_version: str | None


def champion_uri() -> str:
    """Registry URI for the champion model."""
    return f"models:/{settings.mlflow_registered_model_name}@{CHAMPION_ALIAS}"


def load_champion():
    """Load the champion model and its version from the registry.

    Returns (model, version). Raises if no champion is registered.
    """
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(champion_uri())
    version = MlflowClient().get_model_version_by_alias(
        settings.mlflow_registered_model_name, CHAMPION_ALIAS
    ).version
    return model, str(version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once at startup; degrade gracefully if absent."""
    try:
        app.state.model, app.state.model_version = load_champion()
    except Exception as exc:  # noqa: BLE001 — startup must not crash the server
        app.state.model, app.state.model_version = None, None
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="AI4I Failure Classifier", version="0.1.0", lifespan=lifespan)


def get_model(request: Request):
    """Dependency that yields the loaded model or raises 503 if unavailable."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health")
def health(request: Request) -> dict:
    loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": getattr(request.app.state, "model_version", None),
    }


@app.get("/metrics")
def metrics(request: Request) -> dict:
    return {
        "model_name": settings.mlflow_registered_model_name,
        "model_version": getattr(request.app.state, "model_version", None),
        "champion_alias": CHAMPION_ALIAS,
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    model: Annotated[object, Depends(get_model)],
):
    proba = float(model.predict_proba(payload.to_frame())[:, 1][0])
    return PredictionResponse(
        failure_probability=proba,
        prediction=int(proba >= DECISION_THRESHOLD),
        threshold=DECISION_THRESHOLD,
        model_version=getattr(request.app.state, "model_version", None),
    )
