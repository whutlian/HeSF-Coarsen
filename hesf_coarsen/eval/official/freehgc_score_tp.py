from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, normalize_dataset


FREEHGC_STANDARD_FIELDS = (
    "method",
    "repo_url",
    "commit_hash",
    "upstream_entrypoints_found",
    "upstream_protocol_supported",
    "can_run_upstream",
    "eligible_for_official_main_table",
    "failure_reason",
)

FREEHGC_TP_LOCAL_FIELDS = (
    "dataset",
    "method",
    "source_method",
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
    "selection_signal_source",
    "failure_type",
    "failure_reason",
)

FREEHGC_SELECTOR_GATE21_21_FIELDS = FREEHGC_TP_LOCAL_FIELDS


def build_freehgc_standard_rows(repo_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in repo_rows:
        if str(row.get("method", "")) != "FreeHGC":
            continue
        out.append(
            {
                "method": "FreeHGC-standard",
                "repo_url": row.get("repo_url", ""),
                "commit_hash": row.get("commit_hash", ""),
                "upstream_entrypoints_found": row.get("upstream_entrypoints_found", ""),
                "upstream_protocol_supported": row.get("upstream_protocol_supported", ""),
                "can_run_upstream": row.get("can_run_upstream", ""),
                "eligible_for_official_main_table": False,
                "failure_reason": row.get("failure_reason", "standard condensation is separated from the official TP table"),
            }
        )
    return out


def build_freehgc_score_tp_local_rows_from_main(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_freehgc_row(row, source_method=str(row.get("source_method", ""))) for row in rows if "FreeHGC-score-TP-local" in str(row.get("method", ""))]


def build_freehgc_score_selector_rows_from_main(rows: Iterable[Mapping[str, Any]], *, datasets: Sequence[str] = ("DBLP", "ACM", "IMDB")) -> list[dict[str, Any]]:
    source = [dict(row) for row in rows]
    out = [_freehgc_row(row, source_method=str(row.get("source_method", ""))) for row in source if "FreeHGC-score-as-selector" in str(row.get("method", ""))]
    present = {(normalize_dataset(row.get("dataset")), str(row.get("method", ""))) for row in out}
    required = {
        ("DBLP", "FreeHGC-score-as-selector structural16"),
        ("DBLP", "FreeHGC-score-as-selector structural20"),
        ("ACM", "ACM-FreeHGC-score-as-selector-field20"),
        ("IMDB", "IMDB-FreeHGC-score-as-selector-channel50"),
    }
    for dataset, method in sorted(required):
        if dataset not in {normalize_dataset(item) for item in datasets} or (dataset, method) in present:
            continue
        source_row = _selector_proxy_source(source, dataset, method)
        if source_row:
            out.append(_freehgc_row(source_row, method=method, source_method=str(source_row.get("method", "")), selector=True))
        else:
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "source_method": "",
                    "training_executed": False,
                    "official_hgb_exported": False,
                    "official_sehgnn_unmodified": True,
                    "eligible_for_official_main_table": False,
                    "selection_signal_source": "trainval_only_proxy_defined_but_export_training_missing",
                    "failure_type": "implemented_pending_official_training",
                    "failure_reason": "Required FreeHGC-score-as-selector row is defined, but no traceable official export/training result is available in Gate21.20.",
                }
            )
    return out


def _selector_proxy_source(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    if dataset == "ACM":
        return _find_method(rows, dataset, "ACM-FreeHGC-score-TP-local-field20")
    if dataset == "IMDB":
        return None
    return _find_method(rows, dataset, method)


def _freehgc_row(row: Mapping[str, Any], *, method: str | None = None, source_method: str = "", selector: bool = False) -> dict[str, Any]:
    return {
        "dataset": normalize_dataset(row.get("dataset")),
        "method": method or row.get("method", ""),
        "source_method": source_method,
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
        "eligible_for_official_main_table": bool_value(row.get("eligible_for_main_table", True)) and not selector,
        "selection_signal_source": "trainval_only_freehgc_score_proxy",
        "failure_type": row.get("failure_type", ""),
        "failure_reason": row.get("failure_reason", ""),
    }


def _find_method(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method:
            return row
    return None


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
