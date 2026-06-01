"""Tests for the Evidently drift-monitoring module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.monitoring.drift_report import (
    generate_report,
    run_drift_report,
    simulate_drift,
    summarize,
)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.integers(1200, 2800, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n),
            "Type": rng.choice(["L", "M", "H"], n),
        }
    )


def test_no_drift_when_reference_equals_current():
    df = _frame()
    summary = summarize(run_drift_report(df, df.copy()))
    assert summary["dataset_drift"] is False
    assert summary["n_drifted_columns"] == 0
    assert summary["n_monitored_columns"] == 6


def test_drift_detected_when_current_is_shifted():
    df = _frame()
    shifted = simulate_drift(df, temperature_shift=10.0, wear_scale=2.0, seed=1)
    summary = summarize(run_drift_report(df, shifted))
    assert summary["n_drifted_columns"] >= 1
    assert summary["dataset_drift"] is True


def test_simulate_drift_shifts_distribution():
    df = _frame()
    shifted = simulate_drift(df, temperature_shift=5.0, wear_scale=1.5, seed=0)
    assert shifted["Air temperature [K]"].mean() > df["Air temperature [K]"].mean()
    assert shifted["Tool wear [min]"].mean() > df["Tool wear [min]"].mean()


def test_generate_report_writes_html_and_json(tmp_path):
    df = _frame()
    summary = generate_report(df, simulate_drift(df, seed=2), tmp_path)

    html = tmp_path / "drift_report.html"
    js = tmp_path / "drift_summary.json"
    assert html.exists() and html.stat().st_size > 0
    assert js.exists()

    on_disk = json.loads(js.read_text())
    assert on_disk == summary
    assert "dataset_drift" in on_disk
    assert set(on_disk["column_pvalues"]) <= {
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    }
