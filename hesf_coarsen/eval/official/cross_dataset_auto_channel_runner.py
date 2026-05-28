from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_gate21_12_cross_dataset(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(grouped.items()):
        ready = [row for row in group if _bool(row.get("training_executed")) and _finite(row.get("test_micro_f1")) and _finite(row.get("test_macro_f1"))]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "row_count": len(group),
                "success_count": len(ready),
                "test_micro_f1_mean": _mean(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean(ready, "test_macro_f1"),
                "recovery_vs_native_full_micro_mean": _mean(ready, "recovery_vs_native_full_micro"),
                "recovery_vs_native_full_macro_mean": _mean(ready, "recovery_vs_native_full_macro"),
                "structural_storage_ratio_mean": _mean(ready, "structural_storage_ratio"),
                "raw_hgb_text_byte_ratio_mean": _mean(ready, "raw_hgb_text_byte_ratio"),
                "support_edge_ratio_mean": _mean(ready, "support_edge_ratio"),
            }
        )
    return out


def gate21_12_cross_dataset_selector_plans(datasets: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        name = str(dataset).upper()
        if name not in {"ACM", "IMDB"}:
            continue
        rows.extend(
            [
                {
                    "dataset": name,
                    "method": "HeSF-RCS-auto structural30",
                    "requested_structural_budget": 0.30,
                    "selection_signal_source": "graph_only",
                    "uses_test_metrics_for_selection": False,
                    "uses_test_labels_for_selection": False,
                    "selected_channel_plan_human": "schema-inferred auto structural30; not DBLP hardcoded",
                    "selector_plan_ready": True,
                    "task_results_ready": False,
                },
                {
                    "dataset": name,
                    "method": "HeSF-RCS-auto structural20",
                    "requested_structural_budget": 0.20,
                    "selection_signal_source": "graph_only",
                    "uses_test_metrics_for_selection": False,
                    "uses_test_labels_for_selection": False,
                    "selected_channel_plan_human": "schema-inferred auto structural20; not DBLP hardcoded",
                    "selector_plan_ready": True,
                    "task_results_ready": False,
                },
            ]
        )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "NaN" if not finite else sum(finite) / len(finite)


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
