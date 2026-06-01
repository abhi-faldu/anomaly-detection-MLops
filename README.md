# anomaly-detection-mlops

> Production-style **MLOps pipeline** for predictive maintenance — train, track, register, serve, and monitor a failure-classification model on the **AI4I 2020** dataset.

Third project in a 3-part Industry 4.0 portfolio:

1. [robotic-bearing-pdm](https://github.com/abhi-faldu/robotic-bearing-pdm) — LSTM autoencoder anomaly detection (NASA IMS bearings)
2. [iiot-smart-factory-sim](https://github.com/abhi-faldu/iiot-smart-factory-sim) — MQTT IIoT simulator + real-time Z-score detection
3. **anomaly-detection-mlops** — the MLOps layer (this repo)

The focus here is **MLOps engineering**, not model novelty: reproducible training, experiment tracking, model-registry promotion, drift monitoring, testing, and CI/CD automation.

## Stack

| Layer | Technology |
|---|---|
| Modeling | scikit-learn (LogisticRegression baseline) + XGBoost (main) |
| Experiment tracking & registry | MLflow (SQLite backend, local artifacts) |
| Drift / data quality | Evidently |
| Serving | FastAPI + Uvicorn |
| Testing | pytest + pytest-cov |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose |

## Dataset — AI4I 2020 Predictive Maintenance

~10,000 rows of tabular sensor data; binary `Machine failure` target (~3.4% positive — class imbalance handled, PR-AUC reported as the primary metric).

The dataset is **gitignored**. Download it and place at `data/ai4i2020.csv`:

```bash
kaggle datasets download stephanmatzka/predictive-maintenance-dataset-ai4i-2020
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt

make train                       # train + log to MLflow
make mlflow                      # browse runs at http://localhost:5000
make serve                       # FastAPI at http://localhost:8000/docs
make drift                       # Evidently drift report → reports/
make test                        # pytest + coverage
```

## Status

🚧 Under construction — see [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the phased build plan.

## License

MIT
