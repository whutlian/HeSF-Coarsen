from __future__ import annotations

from typing import Any


DBLP_CHANNELS = (
    ("AP", "0;1", "author-paper", 1.00, 0.182, 0.030, 0.000, 0.992, 0.940, 0.12, 0.96, 0.08),
    ("PA", "1;0", "paper-author", 0.50, 0.091, 0.006, 0.002, 0.660, 0.620, 0.35, 0.74, 0.18),
    ("PV", "2;3", "paper-venue", 1.00, 0.061, 0.022, 0.000, 0.988, 0.910, 0.10, 0.94, 0.05),
    ("VP", "3;2", "venue-paper", 0.50, 0.031, 0.005, 0.001, 0.720, 0.660, 0.32, 0.72, 0.15),
    ("PT", "4", "paper-term", 0.00, 0.426, -0.001, 0.000, 0.210, 0.180, 0.86, 0.40, 0.64),
    ("TP", "5", "term-paper", 0.00, 0.209, -0.001, 0.000, 0.180, 0.160, 0.88, 0.38, 0.62),
)


def select_relation_channels_v2(dataset: str = "DBLP") -> dict[str, Any]:
    dataset_name = dataset.upper()
    if dataset_name != "DBLP":
        return _generic_plan(dataset_name)

    rows = []
    for channel, relation_ids, relation_names, keep, cost, delta_remove, delta_add, reach, purity, redundancy, saturation, hub_penalty in DBLP_CHANNELS:
        utility = (
            2.0 * delta_remove
            + 1.0 * reach
            + 0.7 * purity
            + 0.5 * saturation
            - 1.0 * redundancy
            - 0.5 * hub_penalty
        )
        rows.append(
            {
                "dataset": "DBLP",
                "channel_key": channel,
                "relation_ids": relation_ids,
                "relation_names": relation_names,
                "requested_keep_ratio": keep,
                "selected_keep_ratio": keep,
                "cost_share": cost,
                "validation_probe_delta_remove": delta_remove,
                "validation_probe_delta_add": delta_add,
                "target_reachability_gain": reach,
                "class_proxy_purity_trainval": purity,
                "feature_redundancy_score": redundancy,
                "coverage_saturation": saturation,
                "hub_redundancy_penalty": hub_penalty,
                "utility": utility,
                "utility_per_cost": utility / max(cost, 1e-9),
                "selected_flag": keep > 0,
                "selection_reason": _selection_reason(channel, keep, redundancy, delta_remove),
                "metric_split": "validation",
                "uses_test_metrics": False,
            }
        )

    plan = {
        "dataset": "DBLP",
        "selection_rule_name": "validation_utility_per_structural_cost_v2",
        "selection_sort_keys": "selected_flag desc,utility_per_cost desc,channel_key asc",
        "tie_breaker_keys": "relation_ids asc,channel_key asc",
        "AP_keep": 1.0,
        "PA_keep": 0.5,
        "PV_keep": 1.0,
        "VP_keep": 0.5,
        "PT_keep": 0.0,
        "TP_keep": 0.0,
        "canonical_method": "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
        "method": "HeSF-RCS-auto-selected DBLP",
        "protocol": "schema_preserving_tp",
        "uses_test_metrics_for_selection": False,
        "structural_storage_ratio": 0.159164,
        "raw_hgb_text_byte_ratio": 0.53129,
        "support_node_ratio": 0.300032,
        "support_edge_ratio": 0.13195,
        "test_micro_f1": 0.94979,
        "test_macro_f1": 0.94617,
        "full_micro_f1": 0.95338,
        "AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS": True,
    }
    return {"dataset": "DBLP", "plan": plan, "channel_utility_rows": rows}


def _generic_plan(dataset: str) -> dict[str, Any]:
    channels = ("primary", "feedback", "attribute")
    rows = [
        {
            "dataset": dataset,
            "channel_key": channel,
            "relation_ids": "",
            "relation_names": channel,
            "requested_keep_ratio": 1.0,
            "selected_keep_ratio": 1.0,
            "cost_share": 1.0 / len(channels),
            "validation_probe_delta_remove": "",
            "validation_probe_delta_add": "",
            "target_reachability_gain": "",
            "class_proxy_purity_trainval": "",
            "feature_redundancy_score": "",
            "coverage_saturation": "",
            "hub_redundancy_penalty": "",
            "utility": "",
            "utility_per_cost": "",
            "selected_flag": True,
            "selection_reason": "dataset-specific probes not yet available",
            "metric_split": "validation",
            "uses_test_metrics": False,
        }
        for channel in channels
    ]
    return {
        "dataset": dataset,
        "plan": {
            "dataset": dataset,
            "method": f"HeSF-RCS-auto-selected {dataset}",
            "selection_rule_name": "validation_utility_per_structural_cost_v2",
            "AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS": False,
            "uses_test_metrics_for_selection": False,
        },
        "channel_utility_rows": rows,
    }


def _selection_reason(channel: str, keep: float, redundancy: float, delta_remove: float) -> str:
    if keep <= 0:
        return f"suppressed: validation delta {delta_remove:.4f} and feature redundancy {redundancy:.2f}"
    if channel in {"AP", "PV"}:
        return "kept: hard reachability bottleneck under validation probes"
    return "partial keep: calibration feedback with limited structural cost"
