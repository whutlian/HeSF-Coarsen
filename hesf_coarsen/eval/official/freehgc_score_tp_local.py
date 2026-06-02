from __future__ import annotations

from typing import Any, Iterable

from hesf_coarsen.eval.official.external_tp_baseline_impl import build_gate21_16_external_tp_rows


def build_freehgc_score_tp_local_rows(*, datasets: Iterable[str], mode: str = "smoke") -> list[dict[str, Any]]:
    rows = build_gate21_16_external_tp_rows(datasets=datasets, mode=mode)
    return [row for row in rows if row.get("method") == "FreeHGC-score-TP"]


def freehgc_local_score_formula() -> dict[str, float]:
    return {
        "target_receptive_field_coverage": 1.0,
        "metapath_reachability_gain": 0.8,
        "feature_diversity_score": 0.5,
        "trainval_label_proxy_purity": 0.5,
        "redundancy_to_selected_centers": -0.3,
        "hub_overrepresentation_penalty": -0.2,
    }
