"""Tests for data loading and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.data.load import EXPECTED_COLUMNS, SchemaError, load_raw
from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    make_splits,
    split_features_target,
)


def _toy_df(n: int = 40) -> pd.DataFrame:
    """Small synthetic frame matching the AI4I schema with both classes present."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.integers(1200, 2800, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n),
            "Machine failure": ([0, 1] * (n // 2))[:n],
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )
    return df


@pytest.mark.skipif(
    not settings.data_path_abs.exists(),
    reason="real AI4I dataset not present (see README)",
)
def test_load_raw_reads_real_dataset():
    df = load_raw()
    assert len(df) == 10000
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_bad_schema_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"UDI": [1], "Type": ["L"]}).to_csv(bad, index=False)
    with pytest.raises(SchemaError):
        load_raw(bad)


def test_split_features_target_excludes_leakage_and_ids():
    x, y = split_features_target(_toy_df())
    # Only the intended features survive.
    assert list(x.columns) == FEATURES
    # No identifier, leakage, or target column leaks into X.
    for col in [*LEAKAGE_COLUMNS, "UDI", "Product ID", TARGET]:
        assert col not in x.columns
    assert set(y.unique()) <= {0, 1}


def test_make_splits_are_stratified_and_disjoint():
    df = _toy_df(40)
    x_train, x_test, y_train, y_test = make_splits(df, test_size=0.25, random_state=42)
    assert len(x_train) + len(x_test) == len(df)
    # Stratification keeps the positive rate stable across splits.
    assert pytest.approx(y_train.mean(), abs=0.05) == y_test.mean()
    # No row index overlap between train and test.
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_preprocessor_shape_and_no_nan():
    df = _toy_df(40)
    x, _ = split_features_target(df)
    pre = build_preprocessor()
    transformed = pre.fit_transform(x)
    # 5 scaled numerics + one-hot of 3 Type categories = 8 columns.
    n_type = df["Type"].nunique()
    assert transformed.shape == (len(df), len(NUMERIC_FEATURES) + n_type)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unknown_category():
    df = _toy_df(40)
    x, _ = split_features_target(df)
    pre = build_preprocessor()
    pre.fit(x)
    # An unseen Type must not blow up (handle_unknown="ignore").
    unseen = x.iloc[[0]].copy()
    unseen.loc[:, CATEGORICAL_FEATURES[0]] = "Z"
    out = pre.transform(unseen)
    assert out.shape[0] == 1
