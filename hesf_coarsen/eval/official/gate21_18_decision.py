from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_18_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "BUDGET_METRIC_SEMANTICS_PASS",
    "NO_MIXED_ACTUAL_STRUCTURAL_RATIO_PASS",
    "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE",
    "FULL_HASH_ROWS_ONLY_IN_SANITY_TABLE",
    "REQUESTED_BUDGET_MATCH_PASS_BY_ROW",
    "DBLP_EDGE_BASELINE_SUPPORT_EDGE20_READY",
    "DBLP_EXTERNAL_TP_SMOKE_READY",
    "ACM_REAL_COMPRESSED_ROW_READY",
    "IMDB_REAL_COMPRESSED_ROW_READY",
    "HESF_RCS_REP_ACTUAL_VALIDATION_READY",
    "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE",
    "FREEHGC_SCORE_TP_LOCAL_READY",
    "STAGE_REPORT_SMOKE_READY",
    "STAGE_REPORT_BUDGET_TRUTH_READY",
)


DBLP_EDGE20_METHODS = (
    "Random-edge-relwise",
    "Degree-edge-relwise",
    "Proportional-relation-budget",
)


def gate21_18_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    fallback_rows: Iterable[Mapping[str, Any]] = (),
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> dict[str, Any]:
    rows = [annotate_budget_truth(row) for row in main_rows]
    sanity_rows = [dict(row) for row in fallback_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]
    row_budget_detail = [_row_budget_detail(row) for row in rows]

    full_detail = {dataset: _ready(rows, dataset, "Full-native-SeHGNN", allow_full=True) for dataset in dataset_names}
    export_detail = {dataset: _ready(rows, dataset, "Export-full-SeHGNN", allow_full=True) for dataset in dataset_names}
    fallback_free = not any(bool_value(row.get("constraint_safe_fallback")) and bool_value(row.get("eligible_for_main_table", True)) for row in rows)
    full_hash_rows_only_sanity = not any(
        _looks_full_hash(row) and not _is_full_fidelity_anchor(row)
        for row in rows
        if bool_value(row.get("eligible_for_main_table", True))
    ) and all(
        bool_value(row.get("constraint_safe_fallback")) for row in sanity_rows if _looks_full_hash(row)
    )
    no_mixed = all(_no_mixed_actual_structural(row) for row in rows)
    budget_semantics = all(str(row.get("budget_metric_used_for_match", "")) for row in rows if _is_compression_row(row))
    budget_match_detail = {
        _row_key(row): bool_value(row.get("budget_match_for_requested_metric")) or str(row.get("budget_match_failure_type", "")) == "budget_infeasible"
        for row in rows
        if _is_compression_row(row)
    }

    dblp_edge_ready = all(
        _ready(
            rows,
            "DBLP",
            method,
            budget_type="support_edge_ratio",
            budget=0.20,
            require_budget_truth=True,
        )
        for method in DBLP_EDGE20_METHODS
    )
    dblp_external_ready = any(_ready(rows, "DBLP", method, allow_budget_infeasible=True) for method in ("Herding-HG-TP", "HGCond-score-TP-local", "GCond-score-TP-local"))
    freehgc_ready = _ready(rows, "DBLP", "FreeHGC-score-TP-local", allow_budget_infeasible=True)
    acm_real_ready = any(_real_compressed_ready(row, dataset="ACM") for row in rows)
    imdb_real_ready = any(_real_compressed_ready(row, dataset="IMDB") for row in rows)
    rep_actual_validation = any(
        normalize_dataset(row.get("dataset")) in dataset_names
        and str(row.get("method", "")) == "HeSF-RCS-Rep-Validated"
        and str(row.get("selection_source", "")) == "actual_validation"
        and not bool_value(row.get("uses_test_for_selection"))
        for row in rows
    )
    rep_no_leak = not any(str(row.get("method", "")) == "HeSF-RCS-Rep-Validated" and bool_value(row.get("uses_test_for_selection")) for row in rows)

    smoke_ready = bool(
        all(full_detail.values())
        and all(export_detail.values())
        and dblp_edge_ready
        and dblp_external_ready
        and freehgc_ready
        and acm_real_ready
        and imdb_real_ready
        and fallback_free
        and no_mixed
    )
    budget_truth_ready = bool(
        budget_semantics
        and no_mixed
        and fallback_free
        and dblp_edge_ready
        and acm_real_ready
        and imdb_real_ready
    )
    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "BUDGET_METRIC_SEMANTICS_PASS": bool(budget_semantics),
        "NO_MIXED_ACTUAL_STRUCTURAL_RATIO_PASS": bool(no_mixed),
        "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE": bool(fallback_free),
        "FULL_HASH_ROWS_ONLY_IN_SANITY_TABLE": bool(full_hash_rows_only_sanity),
        "REQUESTED_BUDGET_MATCH_PASS_BY_ROW": all(budget_match_detail.values()) if budget_match_detail else False,
        "REQUESTED_BUDGET_MATCH_DETAIL_BY_ROW": budget_match_detail,
        "REQUESTED_BUDGET_AUDIT_DETAIL": row_budget_detail,
        "DBLP_EDGE_BASELINE_SUPPORT_EDGE20_READY": bool(dblp_edge_ready),
        "DBLP_EXTERNAL_TP_SMOKE_READY": bool(dblp_external_ready),
        "ACM_REAL_COMPRESSED_ROW_READY": bool(acm_real_ready),
        "IMDB_REAL_COMPRESSED_ROW_READY": bool(imdb_real_ready),
        "HESF_RCS_REP_ACTUAL_VALIDATION_READY": bool(rep_actual_validation),
        "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE": bool(rep_no_leak),
        "FREEHGC_SCORE_TP_LOCAL_READY": bool(freehgc_ready),
        "STAGE_REPORT_SMOKE_READY": bool(smoke_ready),
        "STAGE_REPORT_BUDGET_TRUTH_READY": bool(budget_truth_ready),
    }
    return {name: flags.get(name, False) for name in GATE21_18_DECISION_FLAGS} | {k: v for k, v in flags.items() if k not in GATE21_18_DECISION_FLAGS}


def _ready(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    method: str,
    *,
    budget_type: str | None = None,
    budget: float | None = None,
    require_budget_truth: bool = False,
    allow_full: bool = False,
    allow_budget_infeasible: bool = False,
) -> bool:
    for row in rows:
        if normalize_dataset(row.get("dataset")) != dataset or str(row.get("method", "")) != method:
            continue
        if budget_type is not None and str(row.get("requested_budget_type", "")) != budget_type:
            continue
        if budget is not None:
            actual_budget = float_value(row.get("requested_budget"))
            if actual_budget is None or abs(actual_budget - budget) > 1e-9:
                continue
        if not _row_ready(row, allow_full=allow_full):
            continue
        if require_budget_truth and not bool_value(row.get("budget_match_for_requested_metric")):
            continue
        if not allow_budget_infeasible and str(row.get("budget_match_failure_type", "")) == "budget_infeasible":
            continue
        return True
    return False


def _row_ready(row: Mapping[str, Any], *, allow_full: bool = False) -> bool:
    if bool_value(row.get("constraint_safe_fallback")) and not allow_full:
        return False
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("target_preserving", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and finite_metric(row.get("test_micro_f1_mean"))
        and finite_metric(row.get("test_macro_f1_mean"))
    )


def _real_compressed_ready(row: Mapping[str, Any], *, dataset: str) -> bool:
    if normalize_dataset(row.get("dataset")) != dataset or not _row_ready(row):
        return False
    edge_ratio = float_value(row.get("actual_support_edge_ratio"))
    structural_ratio = float_value(row.get("semantic_structural_storage_ratio"))
    keyword_ratio = float_value(row.get("keyword_feature_ratio"))
    pk_ratio = float_value(row.get("PK_edge_ratio"))
    return bool(
        (edge_ratio is not None and edge_ratio < 1.0)
        or (structural_ratio is not None and structural_ratio < 1.0)
        or (keyword_ratio is not None and keyword_ratio < 1.0)
        or (pk_ratio is not None and pk_ratio < 1.0)
    )


def _is_compression_row(row: Mapping[str, Any]) -> bool:
    return bool(not bool_value(row.get("constraint_safe_fallback")) and str(row.get("method", "")) not in {"Full-native-SeHGNN", "Export-full-SeHGNN"})


def _is_full_fidelity_anchor(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN"}


def _no_mixed_actual_structural(row: Mapping[str, Any]) -> bool:
    if str(row.get("requested_budget_type", "")) != "structural_storage_ratio":
        return True
    return str(row.get("budget_metric_used_for_match", "")) == "semantic_structural_storage_ratio"


def _looks_full_hash(row: Mapping[str, Any]) -> bool:
    value = str(row.get("selected_edge_hash", "")).lower()
    return "full" in value or value == str(row.get("full_edge_hash", "")).lower()


def _row_budget_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row": _row_key(row),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "metric": row.get("budget_metric_used_for_match", ""),
        "match": row.get("budget_match_for_requested_metric", ""),
        "failure_type": row.get("budget_match_failure_type", ""),
    }


def _row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            normalize_dataset(row.get("dataset")),
            str(row.get("method", "")),
            str(row.get("requested_budget_type", "")),
            str(row.get("requested_budget", "")),
        ]
    )
