# anomaly-detection-mlops — Execution Plan

> Self-contained plan for the next session. Read this top-to-bottom before starting.

## Portfolio context

3-project Industry 4.0 portfolio targeting German automotive internships (BMW, Audi, Mercedes-Benz):

1. **robotic-bearing-pdm** ✅ — LSTM Autoencoder anomaly detection on NASA IMS bearing data (`C:/Users/ASUS/robotic-bearing-pdm`)
2. **iiot-smart-factory-sim** ✅ — MQTT IIoT simulator + real-time Z-score detection (`C:/Users/ASUS/iiot-smart-factory-sim`)
3. **anomaly-detection-mlops** ⬅️ THIS PROJECT — the MLOps layer

**Local path:** `C:/Users/ASUS/anomaly-detection-mlops`
**GitHub (to create):** `https://github.com/abhi-faldu/anomaly-detection-mlops`

## Goal

A production-style **MLOps pipeline** for predictive maintenance: train a supervised failure-classification model, track every run in MLflow, register/promote the best model in the MLflow Model Registry, serve predictions via FastAPI, monitor data/concept drift with Evidently, and gate the whole thing through GitHub Actions CI/CD. Fully Dockerized.

The point of this project (vs. Projects 1 & 2) is **MLOps engineering**, not model novelty. The model can be a solid gradient-boosted classifier — the value is in reproducibility, tracking, registry promotion, drift monitoring, testing, and automation.

## Locked decisions (chosen 2026-05-28)

| Decision | Choice |
|---|---|
| Dataset | **AI4I 2020 Predictive Maintenance** (Kaggle) — reused from Project 2's `replay.py` |
| ML task | **Supervised failure classification** — predict machine failure from sensor features |
| Stack | **Full**: GitHub Actions + Docker + FastAPI + MLflow Model Registry + scheduled Evidently drift monitoring |

## Dataset — AI4I 2020

- Source: `kaggle datasets download stephanmatzka/predictive-maintenance-dataset-ai4i-2020`
- Save as `data/ai4i2020.csv` (gitignored — download instructions in README).
- ~10,000 rows, tabular. Features: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Type (L/M/H).
- Target: `Machine failure` (binary). Sub-failure flags: TWF, HDF, PWF, OSF, RNF (can support multi-label later; start binary).
- **Class imbalance:** failures are ~3.4% — must handle (class weights / SMOTE / stratified split) and report PR-AUC, not just accuracy.

## Proposed stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Modeling | scikit-learn (baseline LogisticRegression) + XGBoost or LightGBM (main model) |
| Experiment tracking | MLflow (tracking server + Model Registry) |
| Drift / data quality | Evidently AI |
| Serving | FastAPI + Uvicorn (prediction endpoint loading the registered model) |
| Orchestration (light) | Makefile / Python entrypoints; optional Prefect later (out of scope v1) |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose (mlflow server + api) |
| Config | pydantic-settings / YAML + env vars |
| Testing | pytest + pytest-cov |

## Target architecture

```
data/ai4i2020.csv
      │  (ingest + validate)
      ▼
src/data/        ── load, clean, split (stratified), feature pipeline (sklearn ColumnTransformer)
      ▼
src/train/       ── train baseline + main model; log params/metrics/artifacts to MLflow
      ▼
MLflow Tracking + Model Registry  ── compare runs; promote best to "Staging"/"Production"
      ▼
src/serve/ (FastAPI)  ── loads Production model from registry; /predict, /health
      ▼
src/monitoring/ (Evidently)  ── reference vs current dataset → drift + data-quality report (HTML + metrics)
      ▼
GitHub Actions  ── lint → test → train-gate (min PR-AUC) → build Docker → (optional) push
```

## Proposed folder structure

```
anomaly-detection-mlops/
├── data/                          ← ai4i2020.csv (gitignored)
├── src/
│   ├── config.py                  ← paths, MLflow URI, model params, thresholds (env-driven)
│   ├── data/
│   │   ├── load.py                ← read CSV, basic schema checks
│   │   └── preprocess.py          ← ColumnTransformer: scale numerics, encode Type
│   ├── train/
│   │   ├── train.py               ← train + MLflow logging (params, metrics, model, plots)
│   │   └── evaluate.py            ← PR-AUC, ROC-AUC, confusion matrix, classification report
│   ├── registry/
│   │   └── promote.py             ← compare latest run vs Production; promote if better
│   ├── serve/
│   │   └── main.py                ← FastAPI: /predict (loads registry model), /health, /metrics
│   └── monitoring/
│       └── drift_report.py        ← Evidently reference vs current → HTML + JSON metrics
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py              ← trains on a small slice, asserts metric floor
│   ├── test_promote.py
│   └── test_serve.py              ← FastAPI TestClient, mock/registry model
├── .github/workflows/
│   ├── ci.yml                     ← lint + test on PR/push
│   └── train.yml                  ← manual/scheduled train + drift-report gate
├── mlruns/                        ← local MLflow store (gitignored) OR use sqlite backend
├── reports/                       ← Evidently HTML output (gitignored except a sample)
├── Dockerfile.api
├── Dockerfile.mlflow
├── docker-compose.yml             ← mlflow server + api (+ optional postgres backend)
├── Makefile                       ← make data | train | serve | drift | test
├── requirements.txt               ← pinned versions
├── requirements-dev.txt
├── .env.example
├── .gitignore                     ← copy/adapt from iiot-smart-factory-sim
├── pytest.ini
└── README.md                      ← portfolio README, robotic-bearing-pdm / iiot style
```

## Phased execution plan (each phase = a few one-file commits)

**Phase 0 — Scaffold**
- `git init`, create `.gitignore` (adapt iiot's: ignore `data/`, `mlruns/`, `reports/`, `.venv/`, `__pycache__/`, `.env`, `*.png` except diagram), `.env.example`, `requirements*.txt` (pinned), `pytest.ini`, empty `src/` package skeleton.
- Create GitHub repo `anomaly-detection-mlops`, push initial scaffold to `main`.

**Phase 1 — Data**
- `src/data/load.py` + schema checks; `src/data/preprocess.py` (ColumnTransformer).
- Tests for preprocessing (shape, no leakage, encoded columns).

**Phase 2 — Train + MLflow tracking**
- `src/train/train.py`: baseline LogisticRegression + main XGBoost/LightGBM; stratified split; class imbalance handling; log params/metrics/model/plots to MLflow (local sqlite backend `mlflow.db`).
- `src/train/evaluate.py`: PR-AUC (primary), ROC-AUC, confusion matrix, classification report; save plots as artifacts.
- Test: train on small slice, assert PR-AUC ≥ floor.

**Phase 3 — Model Registry promotion**
- `src/registry/promote.py`: register model, compare candidate vs current Production by PR-AUC, transition stage if better.
- Test promotion logic with a stubbed MLflow client.

**Phase 4 — Serving (FastAPI)**
- `src/serve/main.py`: load Production model from registry on startup; `/predict` (validate input with pydantic), `/health`, `/metrics`.
- Tests with FastAPI TestClient.

**Phase 5 — Drift monitoring (Evidently)**
- `src/monitoring/drift_report.py`: reference (train) vs current (holdout/simulated shifted) → Evidently DataDrift + DataQuality report → HTML + JSON metric summary.
- Test that report generates and drift flag is computed.

**Phase 6 — Docker**
- `Dockerfile.api`, `Dockerfile.mlflow`, `docker-compose.yml` (mlflow server + api; optional postgres + minio for artifacts — keep v1 to sqlite + local volume).

**Phase 7 — CI/CD (GitHub Actions)**
- `ci.yml`: ruff/flake8 lint + pytest with coverage on PR/push to main.
- `train.yml`: workflow_dispatch + schedule (cron) → run training + drift report, gate on PR-AUC floor, upload report artifact.

**Phase 8 — README + polish**
- Portfolio README matching the style of the other two projects (badges, problem statement, architecture diagram, quickstart, MLflow/Evidently screenshots, CI badge).
- FINDINGS.md: model comparison, PR curves, drift analysis.

## Workflow rules (carry over from Projects 1 & 2)

- **Commit to `main` directly**, push to `origin/main`. No feature/worktree branches.
- **One file per commit.** Commit messages **PAST TENSE, plain imperative, no conventional prefixes** (e.g. `added preprocessing pipeline with ColumnTransformer`).
- **No `Co-Authored-By: Claude` trailer** on commits.
- Show proposed commit messages and **wait for a green flag** before committing — unless the user grants autonomous/green-flag mode.
- Frontend (if any dashboard later): write a Claude Design prompt first; don't implement directly.
- Pin dependency versions in `requirements.txt`.

## Resume trigger for next session

When the user says "let's start Project 3" / "let's continue":
1. Read this file fully.
2. `cd C:/Users/ASUS/anomaly-detection-mlops` and `git status` / `git log --oneline` (if initialized).
3. Confirm Python venv + dataset presence (`data/ai4i2020.csv`).
4. Start at the lowest unfinished Phase above.

## Open questions to confirm at kickoff

- Main model: **XGBoost vs LightGBM** (default: XGBoost unless user prefers LightGBM).
- MLflow backend for v1: **sqlite + local artifacts** (simplest) vs full **postgres + minio** (more "production"). Default: sqlite, note the upgrade path.
- PR-AUC gate threshold (set after seeing baseline numbers).
- Whether to add a small **Streamlit/HTML dashboard** for drift reports (stretch goal).
</content>
