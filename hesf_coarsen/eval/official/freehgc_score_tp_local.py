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


def freehgc_score_components_rows() -> list[dict[str, Any]]:
    return [
        {
            "method": "FreeHGC-score-TP-local",
            "component": component,
            "weight": weight,
            "selection_scope": "trainval_only",
            "description": _component_description(component),
        }
        for component, weight in freehgc_local_score_formula().items()
    ]


def freehgc_budget_audit_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "requested_budget_type": row.get("requested_budget_type", ""),
            "requested_budget": row.get("requested_budget", ""),
            "actual_support_edge_ratio": row.get("actual_support_edge_ratio", row.get("support_edge_ratio", "")),
            "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
            "budget_match": row.get("budget_match_for_requested_metric", ""),
            "budget_failure_type": row.get("budget_match_failure_type", ""),
        }
        for row in rows
        if str(row.get("method", "")) == "FreeHGC-score-TP-local"
    ]


def freehgc_task_metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "training_executed": row.get("training_executed", ""),
            "test_micro_f1_mean": row.get("test_micro_f1_mean", ""),
            "test_macro_f1_mean": row.get("test_macro_f1_mean", ""),
            "validation_micro_f1_mean": row.get("validation_micro_f1_mean", ""),
            "validation_macro_f1_mean": row.get("validation_macro_f1_mean", ""),
        }
        for row in rows
        if str(row.get("method", "")) == "FreeHGC-score-TP-local"
    ]


def _component_description(component: str) -> str:
    descriptions = {
        "target_receptive_field_coverage": "Preserve support nodes covering target-node receptive fields.",
        "metapath_reachability_gain": "Prefer support nodes that keep metapath reachability under the official schema.",
        "feature_diversity_score": "Reward diverse local feature neighborhoods without test-label access.",
        "trainval_label_proxy_purity": "Use train/validation labels only for class-proxy coverage.",
        "redundancy_to_selected_centers": "Penalize duplicate selected support neighborhoods.",
        "hub_overrepresentation_penalty": "Penalize support selections dominated by high-degree hubs.",
    }
    return descriptions.get(component, "")
