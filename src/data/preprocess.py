"""Feature/target separation, train-test split, and the sklearn preprocessing pipeline.

Leakage guard: the per-cause failure flags (TWF/HDF/PWF/OSF/RNF) are downstream
indicators of the target and the identifier columns carry no signal, so all of
them are dropped from the feature matrix.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings

TARGET: str = "Machine failure"

NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES: list[str] = ["Type"]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Identifiers (no signal) + per-cause flags (target leakage) — never features.
ID_COLUMNS: list[str] = ["UDI", "Product ID"]
LEAKAGE_COLUMNS: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y), dropping identifier, leakage, and target columns from X."""
    y = df[TARGET].astype(int)
    x = df[FEATURES].copy()
    return x, y


def make_splits(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the (imbalanced) target.

    Returns:
        X_train, X_test, y_train, y_test
    """
    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=test_size if test_size is not None else settings.test_size,
        random_state=random_state if random_state is not None else settings.random_seed,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: standard-scale numerics, one-hot encode Type."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
