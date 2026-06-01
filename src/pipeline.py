"""Gated training pipeline entrypoint for CI / scheduled runs.

Trains all models, enforces the PR-AUC floor (exit 1 on failure so CI blocks),
then registers and promotes the best model.

Run:  python -m src.pipeline
"""

from __future__ import annotations

import sys

from src.config import settings
from src.registry.promote import promote_best
from src.train.train import run_training


def passes_gate(results: list[dict], floor: float | None = None) -> tuple[bool, float]:
    """Return (passed, best_pr_auc) against the PR-AUC floor."""
    floor = settings.pr_auc_floor if floor is None else floor
    best = max(r["metrics"]["pr_auc"] for r in results)
    return best >= floor, best


def main() -> None:
    results = run_training()
    ok, best = passes_gate(results)
    if not ok:
        print(f"PR-AUC gate FAILED: best {best:.4f} < floor {settings.pr_auc_floor:.4f}")
        sys.exit(1)
    print(f"PR-AUC gate passed: best {best:.4f} >= floor {settings.pr_auc_floor:.4f}")

    outcome = promote_best(results)
    verb = "promoted to champion" if outcome["promoted"] else "registered (champion kept)"
    print(f"{outcome['kind']} v{outcome['version']} (PR-AUC {outcome['pr_auc']:.4f}) {verb}.")


if __name__ == "__main__":
    main()
