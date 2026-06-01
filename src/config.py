"""Central configuration, driven by environment variables / .env.

All paths are resolved relative to the project root so the code behaves the
same whether invoked from the repo root, a test, or inside Docker.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed settings loaded from environment / .env (see .env.example)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data ---
    data_path: Path = Path("data/ai4i2020.csv")

    # --- MLflow ---
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "ai4i-predictive-maintenance"
    mlflow_registered_model_name: str = "ai4i-failure-classifier"

    # --- Training ---
    random_seed: int = 42
    test_size: float = 0.2
    pr_auc_floor: float = 0.70

    # --- Serving ---
    model_stage: str = "Production"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def data_path_abs(self) -> Path:
        """Absolute path to the dataset, anchored at the project root."""
        p = self.data_path
        return p if p.is_absolute() else PROJECT_ROOT / p


settings = Settings()
