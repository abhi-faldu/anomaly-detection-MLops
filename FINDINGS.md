# Findings

Results from the training and monitoring pipeline on the **AI4I 2020** dataset
(10,000 rows, binary `Machine failure`, ~3.39% positive). All runs use a fixed
seed (42) and an 80/20 stratified split. PR-AUC (average precision) is the
headline metric because accuracy and ROC-AUC flatter a classifier on data this
imbalanced.

## Model comparison

| Model | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| LogisticRegression (baseline, balanced) | 0.382 | 0.907 | 0.824 | 0.142 |
| **XGBoost** (scale_pos_weight) | **0.828** | **0.968** | 0.779 | 0.697 |

**Read-through:**

- The baseline's ROC-AUC (0.907) looks respectable, but its **PR-AUC of 0.382**
  exposes the truth: at the 0.5 threshold it flags failures with only **14%
  precision** — ~6 false alarms for every true catch. This is exactly the trap
  imbalanced data sets for accuracy/ROC-based evaluation.
- XGBoost more than **doubles PR-AUC to 0.828** and lifts precision to **0.70**
  at comparable recall — far more actionable for a maintenance team. It is the
  model registered and promoted to the `champion` alias.
- Class imbalance is handled without resampling: `class_weight="balanced"` for
  the baseline and `scale_pos_weight = n_neg / n_pos` for XGBoost.

## Leakage guard

The dataset ships with per-cause failure flags (`TWF`, `HDF`, `PWF`, `OSF`,
`RNF`). These are **downstream of the target** — using them as features would
leak the label and inflate metrics. They are dropped from the feature matrix
along with the identifier columns (`UDI`, `Product ID`). Only the five sensor
numerics and the `Type` category feed the model.

## CI gate

The training pipeline (`src/pipeline.py`) enforces a **PR-AUC floor of 0.75**.
XGBoost (0.828) clears it with margin; a regression below the floor exits
non-zero and fails the `train.yml` workflow before any promotion happens.

## Drift monitoring

`src/monitoring/drift_report.py` compares a reference distribution against a
current one with Evidently (0.7 API) and emits an HTML report plus a compact
JSON summary with a single `dataset_drift` boolean.

To demonstrate detection, `simulate_drift` injects covariate shift (+1.5 K on
both temperatures, ×1.4 tool wear, noise on torque). The report then flags:

- **3 of 6 monitored columns drifted** (both temperatures and tool wear),
- **drift share 0.50**, meeting the 0.50 dataset-drift threshold → `dataset_drift: true`.

Rotational speed, torque, and `Type` correctly show no drift, confirming the
detector responds to the shifted features rather than firing indiscriminately.

## Reproduce

```bash
pip install -r requirements-dev.txt
# dataset → data/ai4i2020.csv (see README)
python -m src.train.train          # prints the comparison table above
python -m src.monitoring.drift_report
pytest                              # 38 tests, ~90% coverage
```
