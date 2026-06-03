from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_19_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "BUDGET_METRIC_SEMANTICS_PASS",
    "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE",
    "DBLP_FRONTIER_READY",
    "ACM_CLOSURE_FRONTIER_READY",
    "IMDB_CHANNEL_FRONTIER_READY",
    "ACM_VALIDATION_GREEDY_READY",
    "IMDB_VALIDATION_GREEDY_READY",
    "EXTERNAL_TP_LOCAL_BASELINES_READY",
    "HESF_RCS_REP_VALIDATED_READY",
    "HESF_RCS_REP_NO_TEST_LEAKAGE",
    "STAGE_REPORT_SMOKE_READY",
    "STAGE_REPORT_QUICK_ROBUSTNESS_READY",
)


def gate21_19_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    fallback_rows: Iterable[Mapping[str, Any]] = (),
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
    mode: str = "smoke",
) -> dict[str, Any]:
    rows = [annotate_budget_truth(row) if _is_budget_checked_row(row) else dict(row) for row in main_rows]
    sanity_rows = [dict(row) for row in fallback_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]

    full_detail = {dataset: _ready(rows, dataset, "Full-native-SeHGNN", allow_full=True) for dataset in dataset_names}
    export_detail = {dataset: _ready(rows, dataset, "Export-full-SeHGNN", allow_full=True) for dataset in dataset_names}
    fallback_free = _no_fallback_in_main(rows) and _fallback_rows_only_sanity(sanity_rows)
    budget_semantics = _budget_metric_semantics_pass(rows)
    dblp_ready = _dblp_frontier_ready(rows) if "DBLP" in dataset_names else True
    acm_ready = _acm_closure_frontier_ready(rows) if "ACM" in dataset_names else True
    imdb_ready = _imdb_channel_frontier_ready(rows) if "IMDB" in dataset_names else True
    acm_validation = _has_ready_method(rows, "ACM", "ValidationGreedy")
    imdb_validation = _has_ready_method(rows, "IMDB", "ValidationGreedy")
    external_ready = _external_tp_ready(rows, dataset_names)
    rep_validated = any(
        str(row.get("method", "")) == "HeSF-RCS-Rep-Validated"
        and str(row.get("selection_source", "")) == "actual_validation"
        and not bool_value(row.get("uses_test_for_selection"))
        and _row_ready(row)
        for row in rows
    )
    rep_no_leak = not any(str(row.get("method", "")) == "HeSF-RCS-Rep-Validated" and bool_value(row.get("uses_test_for_selection")) for row in rows)
    quick_ready = _quick_robustness_ready(rows) if str(mode).lower() == "quick" else False

    smoke_ready = bool(
        all(full_detail.values())
        and all(export_detail.values())
        and budget_semantics
        and fallback_free
        and dblp_ready
        and acm_ready
        and imdb_ready
        and acm_validation
        and imdb_validation
        and external_ready
        and rep_validated
        and rep_no_leak
    )
    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "BUDGET_METRIC_SEMANTICS_PASS": bool(budget_semantics),
        "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE": bool(fallback_free),
        "DBLP_FRONTIER_READY": bool(dblp_ready),
        "ACM_CLOSURE_FRONTIER_READY": bool(acm_ready),
        "IMDB_CHANNEL_FRONTIER_READY": bool(imdb_ready),
        "ACM_VALIDATION_GREEDY_READY": bool(acm_validation),
        "IMDB_VALIDATION_GREEDY_READY": bool(imdb_validation),
        "EXTERNAL_TP_LOCAL_BASELINES_READY": bool(external_ready),
        "HESF_RCS_REP_VALIDATED_READY": bool(rep_validated),
        "HESF_RCS_REP_NO_TEST_LEAKAGE": bool(rep_no_leak),
        "STAGE_REPORT_SMOKE_READY": bool(smoke_ready),
        "STAGE_REPORT_QUICK_ROBUSTNESS_READY": bool(quick_ready),
        "ROW_BUDGET_DETAIL": [_row_budget_detail(row) for row in rows if _is_budget_checked_row(row)],
    }
    return {name: flags.get(name, False) for name in GATE21_19_DECISION_FLAGS} | {
        key: value for key, value in flags.items() if key not in GATE21_19_DECISION_FLAGS
    }


def _ready(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    method: str,
    *,
    allow_full: bool = False,
) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == dataset
        and str(row.get("method", "")) == method
        and _row_ready(row, allow_full=allow_full)
        for row in rows
    )


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


def _is_full_fidelity_anchor(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN"}


def _is_compression_row(row: Mapping[str, Any]) -> bool:
    return not _is_full_fidelity_anchor(row) and not bool_value(row.get("uses_test_for_selection"))


def _no_fallback_in_main(rows: Sequence[Mapping[str, Any]]) -> bool:
    return not any(
        bool_value(row.get("constraint_safe_fallback"))
        and (
            bool_value(row.get("eligible_for_main_table", True))
            or bool_value(row.get("eligible_for_compression_claim", False))
        )
        for row in rows
    )


def _fallback_rows_only_sanity(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        bool_value(row.get("constraint_safe_fallback")) and not bool_value(row.get("eligible_for_main_table", False))
        for row in rows
        if bool_value(row.get("constraint_safe_fallback"))
    )


def _budget_metric_semantics_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    checked = 0
    for row in rows:
        if not _is_budget_checked_row(row) or bool_value(row.get("constraint_safe_fallback")):
            continue
        checked += 1
        budget_type = str(row.get("requested_budget_type", "")).strip()
        metric = str(row.get("budget_metric_used_for_match", "")).strip()
        if not budget_type or not metric:
            return False
        if budget_type == "raw_hgb_text_byte_ratio":
            return False
        failure_type = str(row.get("budget_match_failure_type", ""))
        if failure_type in {"unsupported_budget_type", "missing_requested_budget_type", "missing_requested_budget", "metric_missing"}:
            return False
    return checked > 0


def _dblp_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready_dblp = [row for row in rows if normalize_dataset(row.get("dataset")) == "DBLP" and _row_ready(row)]
    return bool(
        any("HeSF-RCS-auto" in str(row.get("method", "")) for row in ready_dblp)
        and any("Random-edge-relwise" in str(row.get("method", "")) for row in ready_dblp)
        and len([row for row in ready_dblp if str(row.get("method_family", "")) == "external_tp_baseline"]) >= 2
    )


def _acm_closure_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "ACM-HeSF-RCS-auto-field20",
        "ACM-Degree-field20",
        "ACM-Random-field20",
        "ACM-ValidationGreedy-field20",
    }
    methods = {str(row.get("method", "")) for row in rows if normalize_dataset(row.get("dataset")) == "ACM" and _row_ready(row)}
    return required.issubset(methods)


def _imdb_channel_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "IMDB-HeSF-RCS-auto structural20",
        "IMDB-HeSF-RCS-auto structural30",
        "IMDB-Random-channel20",
        "IMDB-Degree-channel20",
        "IMDB-MDfull-MA20-MK50",
        "IMDB-ValidationGreedy-channel30",
    }
    methods = {str(row.get("method", "")) for row in rows if normalize_dataset(row.get("dataset")) == "IMDB" and _row_ready(row)}
    return required.issubset(methods)


def _has_ready_method(rows: Sequence[Mapping[str, Any]], dataset: str, needle: str) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == dataset
        and needle in str(row.get("method", ""))
        and _row_ready(row)
        for row in rows
    )


def _external_tp_ready(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> bool:
    external_by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("method_family", "")) != "external_tp_baseline" or not _row_ready(row):
            continue
        external_by_dataset.setdefault(normalize_dataset(row.get("dataset")), []).append(row)
    dblp_rows = external_by_dataset.get("DBLP", [])
    if "DBLP" in datasets and len(dblp_rows) < 2:
        return False
    for dataset in ("ACM", "IMDB"):
        rows_for_dataset = external_by_dataset.get(dataset, [])
        if rows_for_dataset and len(rows_for_dataset) < 2:
            return False
    return bool(dblp_rows or any(external_by_dataset.values()))


def _quick_robustness_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    robust = [
        row
        for row in rows
        if _row_ready(row)
        and (
            (float_value(row.get("graph_seed_count")) or 0) >= 3
            or (float_value(row.get("training_seed_count")) or 0) >= 3
        )
    ]
    return len(robust) >= 3


def _row_budget_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row": "|".join(
            [
                normalize_dataset(row.get("dataset")),
                str(row.get("method", "")),
                str(row.get("requested_budget_type", "")),
                str(row.get("requested_budget", "")),
            ]
        ),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "metric": row.get("budget_metric_used_for_match", ""),
        "match": row.get("budget_match_for_requested_metric", ""),
        "failure_type": row.get("budget_match_failure_type", ""),
    }


def _is_budget_checked_row(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    if method in {"HeSF-RCS-Rep-Validated", "HeSF-RCS-TestOracleRep"}:
        return False
    return _is_compression_row(row)
