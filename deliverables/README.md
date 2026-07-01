# Deliverables — LinkedIn / portfolio assets

Ready-to-post images for the `anomaly-detection-mlops` project. These
**complement** the images already posted on LinkedIn (the real Precision-Recall
curve comparison and the champion confusion matrix) rather than repeat them.

| # | File | What it shows | Source |
|---|------|---------------|--------|
| 1 | `01_architecture.png` | The full train → track → register → promote → serve → monitor loop, gated by GitHub Actions | Rendered from the README Mermaid diagram |
| 2 | `02_model_comparison.png` | PR-AUC / ROC-AUC / Recall / Precision, baseline vs XGBoost champion — adds the ROC-AUC-vs-PR-AUC contrast at a glance | Real metrics from `FINDINGS.md` |
| 3 | `03_drift_monitoring.png` | Evidently drift result: 3 of 6 columns flagged → `dataset_drift: true` | Illustrates the documented drift outcome (per-column bar heights are indicative) |
| 4 | `04_mlops_loop_summary.png` | One-tile "cover" card — pipeline stages + headline numbers | Real metrics + stack summary |

## Already on LinkedIn (not duplicated here)

- **Precision-Recall curve comparison** — Baseline vs Champion, PR-AUC 0.38 → 0.83
- **Confusion matrix (champion)** — TN 1909 / FP 23 / FN 15 / TP 53
  (→ recall 0.779, precision 0.697)

## Suggested carousel order

Lead with **1** (architecture, the hero), then **2** (metrics), then your two
existing PR-AUC images, then **3** (monitoring) and **4** (summary) — so the
scroll reads *here's the system → here's the result → here's the proof it runs*.

## Regenerating

```bash
# Architecture diagram (mermaid-cli)
mmdc -i arch.mmd -o 01_architecture.png -b transparent -s 3 -w 1400
```

> Tip: swap the indicative drift bars in `03` for a real Evidently run via
> `python -m src.monitoring.drift_report` (screenshot `reports/drift_report.html`),
> and add live MLflow UI / FastAPI `/docs` screenshots once the stack is running
> (`docker compose up`).
