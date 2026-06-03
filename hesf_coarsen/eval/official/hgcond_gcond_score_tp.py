from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, normalize_dataset


HGCOND_GCOND_SCORE_TP_FIELDS = (
    "dataset",
    "method",
    "proxy_type",
    "source_method",
    "repo_url",
    "requested_budget_type",
    "requested_budget",
    "semantic_structural_storage_ratio",
    "actual_support_edge_ratio",
    "raw_hgb_text_byte_ratio",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "training_executed",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "eligible_for_official_main_table",
    "failure_type",
    "failure_reason",
)


def build_hgcond_gcond_score_tp_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> list[dict[str, Any]]:
    source = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    required_methods = ("HGCond-score-TP-local", "GCond-score-TP-local")
    for dataset in [normalize_dataset(item) for item in datasets]:
        for method in required_methods:
            source_row = _find_method(source, dataset, method)
            if source_row is None:
                source_row = _find_prefixed_method(source, dataset, method)
            if source_row:
                out.append(_proxy_row(source_row, method=str(source_row.get("method", method)), proxy_type="score-TP-local"))
            else:
                out.append(_pending_row(dataset, method, proxy_type="score-TP-local"))
            selector_method = method.replace("score-TP-local", "score-as-selector")
            if source_row:
                out.append(_proxy_row(source_row, method=selector_method, source_method=str(source_row.get("method", "")), proxy_type="score-as-selector", eligible=False))
            else:
                out.append(_pending_row(dataset, selector_method, proxy_type="score-as-selector"))
    return out


def _proxy_row(
    row: Mapping[str, Any],
    *,
    method: str,
    proxy_type: str,
    source_method: str = "",
    eligible: bool | None = None,
) -> dict[str, Any]:
    source = source_method or str(row.get("source_method", ""))
    return {
        "dataset": normalize_dataset(row.get("dataset")),
        "method": method,
        "proxy_type": proxy_type,
        "source_method": source,
        "repo_url": _repo_url(method),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "semantic_structural_storage_ratio": _first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio", "actual_structural_storage_ratio"),
        "actual_support_edge_ratio": _first_value(row, "actual_support_edge_ratio", "support_edge_ratio"),
        "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
        "validation_micro_f1": _first_value(row, "validation_micro_f1_mean", "validation_micro_f1"),
        "validation_macro_f1": _first_value(row, "validation_macro_f1_mean", "validation_macro_f1"),
        "test_micro_f1": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
        "test_macro_f1": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
        "training_executed": bool_value(row.get("training_executed", True)),
        "official_hgb_exported": bool_value(row.get("official_hgb_exported", True)),
        "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified", True)),
        "eligible_for_official_main_table": bool_value(row.get("eligible_for_main_table", True)) if eligible is None else bool(eligible),
        "failure_type": row.get("failure_type", ""),
        "failure_reason": row.get("failure_reason", ""),
    }


def _pending_row(dataset: str, method: str, *, proxy_type: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "proxy_type": proxy_type,
        "source_method": "",
        "repo_url": _repo_url(method),
        "requested_budget_type": "support_node_ratio",
        "requested_budget": 0.50,
        "training_executed": False,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_official_main_table": False,
        "failure_type": "implemented_pending_official_training",
        "failure_reason": (
            f"{method} paper-faithful local proxy is specified as target-preserving representativeness/"
            "feature-moment/relation-coverage selection, but no traceable official HGB export/training "
            f"row exists for {dataset} in the reused Gate21.20 table."
        ),
    }


def _find_method(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method:
            return row
    return None


def _find_prefixed_method(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    for row in rows:
        row_method = str(row.get("method", ""))
        if normalize_dataset(row.get("dataset")) == dataset and row_method.endswith(method):
            return row
    return None


def _repo_url(method: str) -> str:
    if method.startswith("HGCond"):
        return "https://github.com/jianjianGJ/hgcond"
    return "https://github.com/ChandlerBang/GCond"


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
