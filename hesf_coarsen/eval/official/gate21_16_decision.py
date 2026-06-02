from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.gate21_16_protocol import GATE21_16_DECISION_FLAGS, gate21_16_row_ready
from hesf_coarsen.eval.official.stage_report_protocol import (
    DATASETS,
    EXTERNAL_TP_BASELINES,
    STRUCTURAL_BASELINES,
    bool_value,
    finite_metric,
    normalize_dataset,
)


def gate21_16_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    acm_consistency_rows: Iterable[Mapping[str, Any]],
    imdb_consistency_rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]],
    structural_rows: Iterable[Mapping[str, Any]],
    external_tp_rows: Iterable[Mapping[str, Any]],
    freehgc_score_rows: Iterable[Mapping[str, Any]],
    datasets: Sequence[str] = DATASETS,
    mode: str = "quick",
) -> dict[str, Any]:
    rows = [dict(row) for row in main_rows]
    acm_rows = [dict(row) for row in acm_consistency_rows]
    imdb_rows = [dict(row) for row in imdb_consistency_rows]
    reps = [dict(row) for row in rep_rows]
    structural = [dict(row) for row in structural_rows]
    external_tp = [dict(row) for row in external_tp_rows]
    freehgc_score = [dict(row) for row in freehgc_score_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]

    full_detail = {dataset: _has_ready(rows, dataset, {"Full-native-SeHGNN"}) for dataset in dataset_names}
    export_detail = {dataset: _has_ready(rows, dataset, {"Export-full-SeHGNN"}) for dataset in dataset_names}
    structural_detail = {dataset: _has_any_success(structural, dataset, set(STRUCTURAL_BASELINES)) for dataset in dataset_names}
    external_smoke_detail = {dataset: _has_any_implemented_or_success(external_tp, dataset, set(EXTERNAL_TP_BASELINES)) for dataset in dataset_names}
    external_quick_detail = {dataset: _has_any_success(external_tp, dataset, set(EXTERNAL_TP_BASELINES)) for dataset in dataset_names}
    hesf_auto_detail = {dataset: any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")).startswith("HeSF-RCS-auto") and gate21_16_row_ready(row) for row in rows) for dataset in dataset_names}
    rep_selected_detail = {
        dataset: any(normalize_dataset(row.get("dataset")) == dataset and bool_value(row.get("selected_as_rep")) and not bool_value(row.get("uses_test_for_selection")) for row in reps)
        for dataset in dataset_names
    }
    rep_task_detail = {dataset: _has_ready(rows, dataset, {"HeSF-RCS-Rep"}) for dataset in dataset_names}

    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "ACM_EXPORT_CONSISTENCY_PASS": bool(acm_rows) and all(bool_value(row.get("official_loader_preflight_pass")) for row in acm_rows),
        "IMDB_EXPORT_CONSISTENCY_PASS": bool(imdb_rows) and all(bool_value(row.get("official_loader_preflight_pass")) for row in imdb_rows),
        "STRUCTURAL_BASELINES_EXECUTED_BY_DATASET": all(structural_detail.values()),
        "STRUCTURAL_BASELINES_EXECUTED_DETAIL_BY_DATASET": structural_detail,
        "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET": all(external_smoke_detail.values()),
        "EXTERNAL_TP_SMOKE_EXECUTED_DETAIL_BY_DATASET": external_smoke_detail,
        "EXTERNAL_TP_QUICK_READY_BY_DATASET": all(external_quick_detail.values()),
        "EXTERNAL_TP_QUICK_READY_DETAIL_BY_DATASET": external_quick_detail,
        "FREEHGC_STANDARD_ATTEMPTED": any(str(row.get("method", "")) == "FreeHGC-standard" for row in rows) or bool(freehgc_score),
        "FREEHGC_SCORE_TP_EXECUTED": any(gate21_16_row_ready(row) for row in freehgc_score),
        "HESF_RCS_AUTO_EXECUTED_BY_DATASET": all(hesf_auto_detail.values()),
        "HESF_RCS_AUTO_EXECUTED_DETAIL_BY_DATASET": hesf_auto_detail,
        "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE": all(rep_selected_detail.values()),
        "HESF_RCS_REP_SELECTED_DETAIL_BY_DATASET": rep_selected_detail,
        "HESF_RCS_REP_TASK_RESULTS_READY": all(rep_task_detail.values()),
        "HESF_RCS_REP_TASK_READY_DETAIL_BY_DATASET": rep_task_detail,
        "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE": _no_diagnostic_rows(rows),
        "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS": _no_placeholder_success(rows),
    }
    flags["STAGE_REPORT_SMOKE_READY"] = bool(
        flags["FULL_NATIVE_READY_BY_DATASET"]
        and flags["EXPORT_FULL_FIDELITY_PASS_BY_DATASET"]
        and flags["ACM_EXPORT_CONSISTENCY_PASS"]
        and flags["IMDB_EXPORT_CONSISTENCY_PASS"]
        and flags["EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET"]
        and flags["HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE"]
        and flags["NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS"]
    )
    flags["STAGE_REPORT_QUICK_READY"] = bool(
        flags["STAGE_REPORT_SMOKE_READY"]
        and flags["STRUCTURAL_BASELINES_EXECUTED_BY_DATASET"]
        and flags["EXTERNAL_TP_QUICK_READY_BY_DATASET"]
        and flags["HESF_RCS_AUTO_EXECUTED_BY_DATASET"]
        and flags["HESF_RCS_REP_TASK_RESULTS_READY"]
    )
    return {name: flags.get(name, False) for name in GATE21_16_DECISION_FLAGS} | {k: v for k, v in flags.items() if k not in GATE21_16_DECISION_FLAGS}


def _has_ready(rows: list[dict[str, Any]], dataset: str, methods: set[str]) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) in methods and gate21_16_row_ready(row) for row in rows)


def _has_any_success(rows: list[dict[str, Any]], dataset: str, methods: set[str]) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) in methods and gate21_16_row_ready(row) for row in rows)


def _has_any_implemented_or_success(rows: list[dict[str, Any]], dataset: str, methods: set[str]) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == dataset
        and str(row.get("method", "")) in methods
        and (gate21_16_row_ready(row) or str(row.get("failure_type", "")) == "implemented_pending_official_training")
        for row in rows
    )


def _no_diagnostic_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        family = str(row.get("method_family", "")).lower()
        if "adapter" in family or "storage" in family or bool_value(row.get("diagnostic_only")):
            return False
    return True


def _no_placeholder_success(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if not bool_value(row.get("success")):
            continue
        if not finite_metric(row.get("test_micro_f1_mean")) or not finite_metric(row.get("test_macro_f1_mean")):
            return False
    return True
