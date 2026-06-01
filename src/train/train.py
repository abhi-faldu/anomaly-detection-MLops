"""Train the baseline and main failure classifiers, logging everything to MLflow.

Run:  python -m src.train.train

Trains two models on a stratified split:
  * baseline — LogisticRegression with balanced class weights
  * xgboost  — XGBClassifier with scale_pos_weight for the ~3.4% positive rate

Each model is wrapped in a sklearn Pipeline (preprocessing + estimator) and
logged to MLflow with params, metrics (PR-AUC primary), and diagnostic plots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import settings
from src.data.load import load_raw
from src.data.preprocess import NUMERIC_FEATURES, build_preprocessor, make_splits
from src.train.evaluate import compute_metrics, save_confusion_matrix, save_pr_curve

MODEL_KINDS = ("baseline", "xgboost")


def _scale_pos_weight(y) -> float:
    """neg/pos ratio used to counter class imbalance in XGBoost."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos if pos else 1.0


def build_model(kind: str, y_train) -> Pipeline:
    """Build a preprocessing + estimator pipeline for the given model kind."""
    if kind == "baseline":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=settings.random_seed,
        )
    elif kind == "xgboost":
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            scale_pos_weight=_scale_pos_weight(y_train),
            random_state=settings.random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (expected one of {MODEL_KINDS})")

    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def train_model(
    df: pd.DataFrame,
    kind: str,
    log_to_mlflow: bool = True,
    register: bool = False,
) -> dict:
    """Fit one model and (optionally) log it to MLflow.

    Returns a dict with the model kind, fitted pipeline, test metrics, and run id.
    """
    x_train, x_test, y_train, y_test = make_splits(df)
    pipeline = build_model(kind, y_train)
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_proba)

    result = {"kind": kind, "model": pipeline, "metrics": metrics, "run_id": None}
    if not log_to_mlflow:
        return result

    params = {
        "model_kind": kind,
        "test_size": settings.test_size,
        "random_seed": settings.random_seed,
        **{f"clf__{k}": v for k, v in pipeline.named_steps["model"].get_params().items()},
    }

    with mlflow.start_run(run_name=kind) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            pr_path = save_pr_curve(y_test.to_numpy(), y_proba, Path(tmp) / "pr_curve.png")
            cm_path = save_confusion_matrix(
                y_test.to_numpy(), y_proba, Path(tmp) / "confusion_matrix.png"
            )
            mlflow.log_artifact(str(pr_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
        # Cast integer numerics to float so the logged signature tolerates float
        # inputs at serving time (avoids MLflow schema-enforcement errors).
        input_example = x_train.head(3).astype({c: "float64" for c in NUMERIC_FEATURES})
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            registered_model_name=(
                settings.mlflow_registered_model_name if register else None
            ),
        )
        result["run_id"] = run.info.run_id

    return result


def run_training(register_best: bool = False) -> list[dict]:
    """Train all model kinds, log to MLflow, and print a comparison."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = load_raw()
    results = [train_model(df, kind, log_to_mlflow=True) for kind in MODEL_KINDS]

    print(f"\n{'model':<12}{'PR-AUC':>10}{'ROC-AUC':>10}{'recall':>10}{'precision':>12}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: x["metrics"]["pr_auc"], reverse=True):
        m = r["metrics"]
        print(
            f"{r['kind']:<12}{m['pr_auc']:>10.4f}{m['roc_auc']:>10.4f}"
            f"{m['recall']:>10.4f}{m['precision']:>12.4f}"
        )
    return results


if __name__ == "__main__":
    run_training()
