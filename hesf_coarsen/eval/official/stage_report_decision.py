from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import (
    DATASETS,
    EXTERNAL_TP_BASELINES,
    STRUCTURAL_BASELINES,
    bool_value,
    finite_metric,
    main_row_success_ready,
    normalize_dataset,
    REQUIRED_DECISION_FLAGS,
)


def gate21_15_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]],
    external_repo_rows: Iterable[Mapping[str, Any]],
    budget_audit_rows: Iterable[Mapping[str, Any]],
    export_fidelity_rows: Iterable[Mapping[str, Any]],
    structural_rows: Iterable[Mapping[str, Any]] = (),
    external_tp_rows: Iterable[Mapping[str, Any]] = (),
    freehgc_standard_rows: Iterable[Mapping[str, Any]] = (),
    datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    rows = [dict(row) for row in main_rows]
    reps = [dict(row) for row in rep_rows]
    repos = [dict(row) for row in external_repo_rows]
    budget_rows = [dict(row) for row in budget_audit_rows]
    export_rows = [dict(row) for row in export_fidelity_rows]
    structural = [dict(row) for row in structural_rows]
    external_tp = [dict(row) for row in external_tp_rows]
    freehgc_standard = [dict(row) for row in freehgc_standard_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]

    full_by_dataset = {
        dataset: any(_is_method(row, dataset, {"Full-native-SeHGNN", "full-native-SeHGNN", "full-native"}) and main_row_success_ready(row) for row in rows)
        for dataset in dataset_names
    }
    export_by_dataset = {
        dataset: any(_is_method(row, dataset, {"Export-full-SeHGNN", "export-full-SeHGNN", "export-full"}) and main_row_success_ready(row) for row in rows)
        for dataset in dataset_names
    }

    rep_selected = {
        dataset: any(
            normalize_dataset(row.get("dataset")) == dataset
            and bool_value(row.get("selected_as_rep"))
            and not bool_value(row.get("uses_test_for_selection"))
            for row in reps
        )
        for dataset in dataset_names
    }
    rep_task_ready = {
        dataset: any(_is_method(row, dataset, {"HeSF-RCS-Rep"}) and main_row_success_ready(row) for row in rows)
        for dataset in dataset_names
    }

    repo_ready = bool(repos) and all(bool_value(row.get("clone_success")) or bool_value(row.get("adapter_implemented")) for row in repos)
    structural_ready = _all_required_method_rows_ready(structural, required=STRUCTURAL_BASELINES, datasets=dataset_names)
    external_tp_ready = _all_required_method_rows_ready(external_tp, required=EXTERNAL_TP_BASELINES, datasets=dataset_names)
    freehgc_hard_failure = any(str(row.get("failure_type", "")) for row in freehgc_standard)
    freehgc_success = any(bool_value(row.get("success")) and finite_metric(row.get("test_micro_f1_mean", row.get("mean_micro"))) for row in freehgc_standard)
    freehgc_score_ready = any(str(row.get("method", "")) == "FreeHGC-score-TP" and bool_value(row.get("success")) for row in external_tp)

    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_by_dataset.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_by_dataset,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_by_dataset.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_by_dataset,
        "MAIN_TABLE_HAS_DBLP_ACM_IMDB": set(dataset_names).issubset({normalize_dataset(row.get("dataset")) for row in rows}),
        "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE": all(rep_selected.values()),
        "HESF_RCS_REP_SELECTED_DETAIL_BY_DATASET": rep_selected,
        "HESF_RCS_REP_TASK_RESULTS_READY": all(rep_task_ready.values()),
        "HESF_RCS_REP_TASK_READY_DETAIL_BY_DATASET": rep_task_ready,
        "STRUCTURAL_BASELINES_READY": structural_ready,
        "EXTERNAL_TP_BASELINES_CLONED_OR_IMPLEMENTED": repo_ready,
        "EXTERNAL_TP_TASK_RESULTS_READY": external_tp_ready,
        "FREEHGC_STANDARD_READY_OR_HARD_FAILURE_RECORDED": bool(freehgc_success or freehgc_hard_failure),
        "FREEHGC_SCORE_TP_READY": freehgc_score_ready,
        "BUDGET_MATCH_AUDIT_PASS": bool(budget_rows) and all(bool_value(row.get("budget_match_pass", True)) for row in budget_rows if bool_value(row.get("success"))),
        "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE": _no_adapter_or_diagnostic_rows(rows),
        "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS": _no_placeholder_success_metrics(rows),
    }
    flags["STAGE_REPORT_TABLE_READY"] = all(bool(flags[name]) for name in REQUIRED_DECISION_FLAGS if name != "STAGE_REPORT_TABLE_READY")
    return flags


def _is_method(row: Mapping[str, Any], dataset: str, names: set[str]) -> bool:
    return normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) in names


def _all_required_method_rows_ready(rows: list[dict[str, Any]], *, required: Sequence[str], datasets: Sequence[str]) -> bool:
    if not rows:
        return False
    for dataset in datasets:
        for method in required:
            if not any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method and bool_value(row.get("success")) for row in rows):
                return False
    return True


def _no_adapter_or_diagnostic_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        family = str(row.get("method_family", "")).lower()
        method = str(row.get("method", "")).lower()
        if "adapter" in family or "adapter" in method or bool_value(row.get("diagnostic_only")):
            return False
        if "storage-only" in family or "standard_condensation" in family:
            return False
    return True


def _no_placeholder_success_metrics(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if not bool_value(row.get("success")):
            continue
        for field in ("test_micro_f1_mean", "test_macro_f1_mean"):
            if not finite_metric(row.get(field)):
                return False
    return True
