"""Load and schema-validate the AI4I 2020 predictive-maintenance dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings

# Columns the raw CSV must contain. The file ships with a UTF-8 BOM, so it is
# read with utf-8-sig to keep the first column name clean.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)


class SchemaError(ValueError):
    """Raised when the loaded dataframe does not match the expected schema."""


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the AI4I CSV and validate its schema.

    Args:
        path: CSV path. Defaults to the configured dataset path.

    Returns:
        The raw dataframe with all original columns.

    Raises:
        FileNotFoundError: if the CSV is missing.
        SchemaError: if expected columns are absent.
    """
    csv_path = Path(path) if path is not None else settings.data_path_abs
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. See README for download instructions."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Dataset is missing expected columns: {missing}")

    return df
