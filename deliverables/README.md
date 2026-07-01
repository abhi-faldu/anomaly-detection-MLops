# Deliverables — LinkedIn / portfolio assets

Ready-to-post images for the `anomaly-detection-mlops` project. Suggested
carousel order tells the story: *here's the system → here's the result →
here's the proof it's real*.

| # | File | What it shows | Source |
|---|------|---------------|--------|
| 1 | `01_architecture.png` | The full train → track → register → promote → serve → monitor loop, gated by GitHub Actions | Rendered from the README Mermaid diagram |
| 2 | `02_model_comparison.png` | PR-AUC / ROC-AUC / Recall / Precision, baseline vs XGBoost champion | Real metrics from `FINDINGS.md` |
| 3 | `03_pr_auc_trap.png` | Why PR-AUC (not accuracy/ROC-AUC) drives promotion — the precision gap | Real metrics from `FINDINGS.md` |
| 4 | `04_drift_monitoring.png` | Evidently drift result: 3 of 6 columns flagged → `dataset_drift: true` | Illustrates the documented drift outcome (per-column bar heights are indicative) |
| 5 | `05_mlops_loop_summary.png` | One-tile "cover" card — pipeline stages + headline numbers | Real metrics + stack summary |

## Suggested post

Full LinkedIn copy is in the project chat. Short version: lead with image **1**
(architecture) as the hero, follow with **2**/**3** (the PR-AUC insight), then
**4** (monitoring) and **5** (summary).

## Regenerating

```bash
# Charts (matplotlib)
python scripts/make_deliverables.py    # if wired into the repo; otherwise see chat

# Architecture diagram (mermaid-cli)
mmdc -i arch.mmd -o 01_architecture.png -b transparent -s 3 -w 1400
```

> Tip: for images 3–5 you can swap the indicative drift bars in `04` for a real
> Evidently run by executing `python -m src.monitoring.drift_report` and
> screenshotting `reports/drift_report.html`, and add live MLflow UI / FastAPI
> `/docs` screenshots once the stack is running (`docker compose up`).
