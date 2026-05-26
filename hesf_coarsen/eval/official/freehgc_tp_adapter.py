from __future__ import annotations

from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.external_baselines_tp import plan_external_tp_rows


def freehgc_dependency_available(freehgc_root: str | Path | None) -> bool:
    return freehgc_root is not None and Path(freehgc_root).exists()


def freehgc_missing_dependency_row(
    *,
    dataset: str,
    budget: float,
    graph_seed: int,
    training_seed: int,
    freehgc_root: str | Path | None = None,
) -> dict[str, Any]:
    return plan_external_tp_rows(
        dataset=dataset,
        methods=["FreeHGC-TP"],
        budgets=[budget],
        graph_seeds=[graph_seed],
        training_seeds=[training_seed],
        freehgc_root=freehgc_root,
    )[0]
