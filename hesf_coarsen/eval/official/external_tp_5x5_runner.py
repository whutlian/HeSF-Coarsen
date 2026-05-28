from __future__ import annotations

from typing import Any
from statistics import pstdev

from hesf_coarsen.eval.official.gate21_9_decision import REQUIRED_EXTERNAL_TP_5X5


EXTERNAL_TP_BUDGETS = (
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
    ("structural_storage_ratio", 0.12),
    ("structural_storage_ratio", 0.16),
    ("structural_storage_ratio", 0.20),
    ("structural_storage_ratio", 0.30),
)


def build_external_tp_5x5_grid(graph_seeds: list[int], training_seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "protocol": "schema_preserving_tp",
            "budget_type": budget_type,
            "requested_budget": budget,
            "graph_seed": graph_seed,
            "training_seed": training_seed,
        }
        for method in REQUIRED_EXTERNAL_TP_5X5
        for budget_type, budget in EXTERNAL_TP_BUDGETS
        for graph_seed in graph_seeds
        for training_seed in training_seeds
    ]


def summarize_gate21_11_external_tp(
    runs: list[dict[str, Any]],
    *,
    required_methods: tuple[str, ...] = REQUIRED_EXTERNAL_TP_5X5,
    expected_run_count: int = 25,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in runs:
        key = (
            str(row.get("dataset", "DBLP")),
            str(row.get("method", "")),
            str(row.get("budget_family", row.get("budget_type", ""))),
            str(row.get("requested_budget", "")),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        dataset, method, budget_family, requested_budget = key
        ready = [row for row in group if _ready(row)]
        out.append(_summary_row(dataset, method, budget_family, requested_budget, group, ready, expected_run_count))
    for method in required_methods:
        if not any(row["method"] == method for row in out):
            out.append(_summary_row("DBLP", method, "", "", [], [], expected_run_count))
    return out


def _summary_row(
    dataset: str,
    method: str,
    budget_family: str,
    requested_budget: str,
    group: list[dict[str, Any]],
    ready: list[dict[str, Any]],
    expected_run_count: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "budget_family": budget_family,
        "requested_budget": requested_budget,
        "ready_run_count": len(ready),
        "expected_run_count": int(expected_run_count),
        "success_count": len(ready),
        "failure_count": len(group) - len(ready),
        "test_micro_f1_mean": _mean(ready, "test_micro_f1"),
        "test_micro_f1_std": _std(ready, "test_micro_f1"),
        "test_macro_f1_mean": _mean(ready, "test_macro_f1"),
        "test_macro_f1_std": _std(ready, "test_macro_f1"),
        "actual_structural_storage_ratio_mean": _mean(ready, "actual_structural_storage_ratio"),
        "actual_structural_storage_ratio_std": _std(ready, "actual_structural_storage_ratio"),
        "raw_hgb_text_byte_ratio_mean": _mean(ready, "raw_hgb_text_byte_ratio"),
        "support_node_ratio_mean": _mean(ready, "actual_support_node_ratio"),
        "support_edge_ratio_mean": _mean(ready, "actual_support_edge_ratio"),
        "budget_match_rate": _rate(ready, "budget_matched_within_tolerance"),
        "training_executed_rate": _rate(group, "training_executed"),
        "official_hgb_exported_rate": _rate(group, "official_hgb_exported"),
        "preprocess_time_seconds_mean": _mean(ready, "preprocess_time_seconds"),
        "train_time_seconds_mean": _mean(ready, "train_time_seconds"),
        "peak_cpu_rss_mb_mean": _mean(ready, "peak_cpu_rss_mb"),
        "peak_gpu_memory_mb_mean": _mean(ready, "peak_gpu_memory_mb"),
        "eligible_for_main_comparison": len(ready) >= int(expected_run_count) and all(_bool(row.get("budget_matched_within_tolerance")) for row in ready),
    }


def _ready(row: dict[str, Any]) -> bool:
    return bool(
        _bool(row.get("training_executed"))
        and _bool(row.get("success", True))
        and _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and _finite(row.get("test_micro_f1"))
        and _finite(row.get("test_macro_f1"))
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float | str:
    vals = [_float(row.get(field)) for row in rows]
    finite = [val for val in vals if val is not None]
    return "NaN" if not finite else sum(finite) / len(finite)


def _std(rows: list[dict[str, Any]], field: str) -> float | str:
    vals = [_float(row.get(field)) for row in rows]
    finite = [val for val in vals if val is not None]
    return "NaN" if not finite else pstdev(finite)


def _rate(rows: list[dict[str, Any]], field: str) -> float | str:
    if not rows:
        return "NaN"
    return sum(1 for row in rows if _bool(row.get(field))) / len(rows)


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _float(value: Any) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
