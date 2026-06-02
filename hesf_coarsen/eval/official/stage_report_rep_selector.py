from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.gate21_16_protocol import validation_proxy_from_cost
from hesf_coarsen.eval.official.stage_report_protocol import DATASETS, bool_value, finite_metric, float_value, normalize_dataset


def select_gate21_16_representatives(rows: Iterable[Mapping[str, Any]], *, datasets: Sequence[str] = DATASETS) -> list[dict[str, Any]]:
    source = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source
            if normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")).startswith("HeSF-RCS-auto")
            and bool_value(row.get("training_executed"))
            and bool_value(row.get("success"))
            and bool_value(row.get("official_hgb_exported"))
            and bool_value(row.get("official_sehgnn_unmodified"))
        ]
        if not candidates:
            out.append(_missing_row(dataset))
            continue
        ranked = sorted(
            candidates,
            key=lambda row: (
                -_selection_score(row),
                float_value(row.get("actual_structural_storage_ratio")) or float("inf"),
                str(row.get("method", "")),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            selection_source = "official_validation_metric" if finite_metric(row.get("validation_micro_f1_mean")) else "validation_proxy"
            out.append(
                {
                    "dataset": dataset,
                    "candidate_method": row.get("method", ""),
                    "requested_budget": row.get("requested_budget", ""),
                    "actual_structural_ratio": row.get("actual_structural_storage_ratio", ""),
                    "validation_micro_f1": row.get("validation_micro_f1_mean", ""),
                    "validation_macro_f1": row.get("validation_macro_f1_mean", ""),
                    "validation_proxy_score": _selection_score(row),
                    "test_micro_f1": row.get("test_micro_f1_mean", ""),
                    "test_macro_f1": row.get("test_macro_f1_mean", ""),
                    "selected_as_rep": rank == 1,
                    "selection_source": selection_source,
                    "uses_test_for_selection": False,
                    "selected_edge_hash": row.get("selected_edge_hash", ""),
                    "planner_config_hash": row.get("planner_config_hash", ""),
                    "selection_rank": rank,
                }
            )
    return out


def _selection_score(row: Mapping[str, Any]) -> float:
    value = float_value(row.get("validation_micro_f1_mean"))
    if value is not None:
        return value
    proxy = float_value(row.get("validation_proxy_score"))
    return proxy if proxy is not None else validation_proxy_from_cost(row)


def _missing_row(dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "candidate_method": "",
        "requested_budget": "",
        "actual_structural_ratio": "",
        "validation_micro_f1": "",
        "validation_macro_f1": "",
        "validation_proxy_score": "",
        "test_micro_f1": "",
        "test_macro_f1": "",
        "selected_as_rep": False,
        "selection_source": "missing_candidate",
        "uses_test_for_selection": False,
        "selected_edge_hash": "",
        "planner_config_hash": "",
        "selection_rank": "",
    }
