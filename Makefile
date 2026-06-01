# anomaly-detection-mlops — common entrypoints
# On Windows, run targets via `make <target>` (Git Bash / WSL) or copy the command.

PYTHON ?= python

.PHONY: help install install-dev test lint train serve drift mlflow clean

help:
	@echo "Targets:"
	@echo "  install      install runtime deps"
	@echo "  install-dev  install runtime + dev deps"
	@echo "  test         run pytest with coverage"
	@echo "  lint         run ruff"
	@echo "  train        train models and log to MLflow"
	@echo "  serve        run the FastAPI prediction service"
	@echo "  drift        generate the Evidently drift report"
	@echo "  mlflow       launch the local MLflow UI"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

train:
	$(PYTHON) -m src.train.train

serve:
	$(PYTHON) -m uvicorn src.serve.main:app --host 0.0.0.0 --port 8000

drift:
	$(PYTHON) -m src.monitoring.drift_report

mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ */__pycache__
