"""Data-drift monitoring with Evidently (0.7 API).

Compares a reference dataset (e.g. the training distribution) against a current
dataset and writes an HTML report plus a JSON metric summary. The summary
carries a single ``dataset_drift`` boolean the CI/monitoring job can gate on.

Run:  python -m src.monitoring.drift_report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import PROJECT_ROOT, settings
from src.data.load import load_raw
from src.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MONITORED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
DRIFT_SHARE_THRESHOLD = 0.5  # dataset drift if >= this share of columns drift


def build_data_definition() -> DataDefinition:
    """Evidently schema covering the model's input features."""
    return DataDefinition(
        numerical_columns=list(NUMERIC_FEATURES),
        categorical_columns=list(CATEGORICAL_FEATURES),
    )


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Run the Evidently DataDrift report and return the result object."""
    definition = build_data_definition()
    ref_ds = Dataset.from_pandas(reference[MONITORED_COLUMNS], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[MONITORED_COLUMNS], data_definition=definition)
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)])
    return report.run(reference_data=ref_ds, current_data=cur_ds)


def summarize(result) -> dict:
    """Distil the Evidently result into a compact, gate-friendly summary."""
    metrics = result.dict()["metrics"]
    n_drifted = 0
    drift_share = 0.0
    column_pvalues: dict[str, float] = {}

    for m in metrics:
        mtype = m["config"]["type"]
        if mtype.endswith("DriftedColumnsCount"):
            n_drifted = int(m["value"]["count"])
            drift_share = float(m["value"]["share"])
        elif mtype.endswith("ValueDrift"):
            column_pvalues[m["config"]["column"]] = float(m["value"])

    return {
        "n_drifted_columns": n_drifted,
        "n_monitored_columns": len(MONITORED_COLUMNS),
        "drift_share": drift_share,
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "dataset_drift": drift_share >= DRIFT_SHARE_THRESHOLD,
        "column_pvalues": column_pvalues,
    }


def generate_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Run drift detection and persist ``drift_report.html`` + ``drift_summary.json``.

    Returns the summary dict (also written to disk).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_drift_report(reference, current)
    result.save_html(str(output_dir / "drift_report.html"))

    summary = summarize(result)
    (output_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def simulate_drift(
    df: pd.DataFrame,
    temperature_shift: float = 1.5,
    wear_scale: float = 1.4,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with injected covariate shift, for demos/tests."""
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)
    shifted = df.copy()
    shifted["Air temperature [K]"] = shifted["Air temperature [K]"] + temperature_shift
    shifted["Process temperature [K]"] = shifted["Process temperature [K]"] + temperature_shift
    shifted["Tool wear [min]"] = shifted["Tool wear [min]"] * wear_scale
    # Light noise so distributions widen rather than merely translate.
    shifted["Torque [Nm]"] = shifted["Torque [Nm]"] + rng.normal(0, 3, len(shifted))
    return shifted


def main() -> dict:
    """Reference = full training data; current = a drifted copy → report."""
    df = load_raw()
    reference = df
    current = simulate_drift(df)
    output_dir = PROJECT_ROOT / "reports"
    summary = generate_report(reference, current, output_dir)
    print(
        f"Dataset drift: {summary['dataset_drift']} "
        f"({summary['n_drifted_columns']}/{summary['n_monitored_columns']} columns, "
        f"share={summary['drift_share']:.2f}). Report -> {output_dir / 'drift_report.html'}"
    )
    return summary


if __name__ == "__main__":
    main()
