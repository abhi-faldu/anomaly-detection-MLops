"""Register a trained model and promote it to ``champion`` if it beats the incumbent.

MLflow 3 deprecates registry *stages* (Staging/Production) in favour of
*aliases*. This module uses a single ``champion`` alias to mark the model the
serving layer should load. Each registered version carries its PR-AUC as a tag
so promotion decisions need no extra lookups.

Run:  python -m src.registry.promote   (trains, registers, and promotes the best)
"""

from __future__ import annotations

from typing import Protocol

import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings

CHAMPION_ALIAS = "champion"
METRIC_TAG = "pr_auc"


class _RegistryClient(Protocol):
    """Subset of MlflowClient used here — lets tests inject a fake."""

    def get_model_version_by_alias(self, name: str, alias: str): ...
    def set_registered_model_alias(self, name: str, alias: str, version: str): ...


def version_metric(version) -> float:
    """Read the PR-AUC tag off a model version (missing/blank → -inf)."""
    raw = (version.tags or {}).get(METRIC_TAG)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def get_champion(client: _RegistryClient, model_name: str):
    """Return the current champion model version, or None if unset."""
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:
        return None


def is_better(candidate_metric: float, champion_metric: float) -> bool:
    """Promotion rule: candidate wins ties so retrains can refresh the champion."""
    return candidate_metric >= champion_metric


def promote_if_better(
    client: _RegistryClient,
    model_name: str,
    candidate_version: str,
    candidate_metric: float,
) -> bool:
    """Point the champion alias at the candidate when it is at least as good.

    Returns True if the alias was (re)assigned to the candidate.
    """
    champion = get_champion(client, model_name)
    if champion is not None and not is_better(candidate_metric, version_metric(champion)):
        return False
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, candidate_version)
    return True


def register_candidate(
    client: MlflowClient,
    run_id: str,
    metric: float,
    model_name: str | None = None,
) -> str:
    """Register the model logged under ``run_id`` and tag it with its PR-AUC.

    Returns the new registered model version string.
    """
    model_name = model_name or settings.mlflow_registered_model_name
    logged = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run_id}'", output_format="list"
    )
    if not logged:
        raise ValueError(f"No logged model found for run {run_id}")

    mv = mlflow.register_model(f"models:/{logged[0].model_id}", model_name)
    client.set_model_version_tag(model_name, mv.version, METRIC_TAG, str(metric))
    return mv.version


def promote_best(results: list[dict], model_name: str | None = None) -> dict:
    """Register the best run by PR-AUC and promote it if it beats the champion.

    Args:
        results: train_model outputs, each with ``run_id`` and ``metrics``.
    """
    model_name = model_name or settings.mlflow_registered_model_name
    client = MlflowClient()

    best = max(results, key=lambda r: r["metrics"]["pr_auc"])
    metric = best["metrics"]["pr_auc"]
    version = register_candidate(client, best["run_id"], metric, model_name)
    promoted = promote_if_better(client, model_name, version, metric)

    return {
        "kind": best["kind"],
        "version": version,
        "pr_auc": metric,
        "promoted": promoted,
        "model_name": model_name,
    }


if __name__ == "__main__":
    from src.train.train import run_training

    training_results = run_training()
    outcome = promote_best(training_results)
    verb = "promoted to champion" if outcome["promoted"] else "registered (champion unchanged)"
    print(
        f"\n{outcome['kind']} v{outcome['version']} "
        f"(PR-AUC {outcome['pr_auc']:.4f}) {verb}."
    )
