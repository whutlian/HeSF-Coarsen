from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import DATASETS, bool_value, finite_metric, float_value, normalize_dataset
from hesf_coarsen.eval.official.stage_report_table import gate21_17_row_ready


GATE21_17_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "ACM_EXPORT_CONSISTENCY_PASS",
    "IMDB_EXPORT_CONSISTENCY_PASS",
    "STRUCTURAL_BASELINES_SMOKE_EXECUTED_BY_DATASET",
    "STRUCTURAL_BASELINES_QUICK_READY_BY_DATASET",
    "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET",
    "EXTERNAL_TP_QUICK_READY_BY_DATASET",
    "FREEHGC_SCORE_TP_SMOKE_EXECUTED_BY_DATASET",
    "CONDENSATION_SCORE_TP_SMOKE_EXECUTED_BY_DATASET",
    "HESF_RCS_AUTO_EXECUTED_BY_DATASET",
    "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE",
    "HESF_RCS_REP_ACTUAL_VALIDATION_READY",
    "STAGE_REPORT_SMOKE_READY",
    "STAGE_REPORT_QUICK_READY",
    "NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE",
    "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE",
    "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS",
)

STRUCTURAL_SMOKE_METHODS = ("Random-edge-relwise", "Degree-edge-relwise", "Proportional-relation-budget")
EXTERNAL_SMOKE_METHODS = ("Herding-HG-TP", "FreeHGC-score-TP", "FreeHGC-score-TP-local")
CONDENSATION_SMOKE_METHODS = ("HGCond-score-TP-local", "GCond-score-TP-local", "HGCond-score-TP", "GCond-score-TP")


def gate21_17_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    datasets: Sequence[str] = DATASETS,
    mode: str = "smoke",
    acm_consistency_rows: Iterable[Mapping[str, Any]] = (),
    imdb_consistency_rows: Iterable[Mapping[str, Any]] = (),
    rep_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    rows = [dict(row) for row in main_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]
    acm_rows = [dict(row) for row in acm_consistency_rows]
    imdb_rows = [dict(row) for row in imdb_consistency_rows]
    reps = [dict(row) for row in rep_rows]

    full_detail = {dataset: _ready(rows, dataset, "Full-native-SeHGNN") for dataset in dataset_names}
    export_detail = {dataset: _ready(rows, dataset, "Export-full-SeHGNN") for dataset in dataset_names}
    structural_smoke_detail = {
        dataset: _all_methods_ready(rows, dataset, STRUCTURAL_SMOKE_METHODS, budget=0.20)
        if dataset == "DBLP"
        else _ready(rows, dataset, "Random-edge-relwise", budget_type="structural_storage_ratio", budget=0.20)
        for dataset in dataset_names
    }
    structural_quick_detail = {
        dataset: all(_all_methods_ready(rows, dataset, STRUCTURAL_SMOKE_METHODS, budget=budget) for budget in (0.30, 0.20, 0.16))
        for dataset in dataset_names
    }
    external_smoke_detail = {dataset: _external_smoke_ready(rows, dataset) for dataset in dataset_names}
    external_quick_detail = {dataset: _any_ready_family(rows, dataset, "external_tp_baseline") for dataset in dataset_names}
    freehgc_detail = {dataset: _freehgc_smoke_ready(rows, dataset) if dataset == "DBLP" else "not_required_for_smoke" for dataset in dataset_names}
    condensation_detail = {dataset: any(_ready(rows, dataset, method) for method in CONDENSATION_SMOKE_METHODS) for dataset in dataset_names}
    hesf_auto_detail = {
        dataset: any(
            normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")).startswith("HeSF-RCS-auto")
            and gate21_17_row_ready(row)
            for row in rows
        )
        for dataset in dataset_names
    }
    rep_selected_detail = {
        dataset: any(
            normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")) == "HeSF-RCS-Rep"
            and not bool_value(row.get("uses_test_for_selection"))
            for row in reps + rows
        )
        for dataset in dataset_names
    }
    rep_validation_detail = {
        dataset: any(
            normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")) == "HeSF-RCS-Rep"
            and str(row.get("selection_source", "")) == "actual_validation"
            for row in reps + rows
        )
        for dataset in dataset_names
    }

    no_pending = not any(
        bool_value(row.get("eligible_for_main_table", True)) and str(row.get("failure_type", "")) == "implemented_pending_official_training"
        for row in rows
    )
    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "ACM_EXPORT_CONSISTENCY_PASS": bool(acm_rows) and all(bool_value(row.get("official_loader_preflight_pass")) for row in acm_rows),
        "IMDB_EXPORT_CONSISTENCY_PASS": bool(imdb_rows) and all(bool_value(row.get("official_loader_preflight_pass")) for row in imdb_rows),
        "STRUCTURAL_BASELINES_SMOKE_EXECUTED_BY_DATASET": all(structural_smoke_detail.values()),
        "STRUCTURAL_BASELINES_SMOKE_DETAIL_BY_DATASET": structural_smoke_detail,
        "STRUCTURAL_BASELINES_QUICK_READY_BY_DATASET": all(structural_quick_detail.values()),
        "STRUCTURAL_BASELINES_QUICK_DETAIL_BY_DATASET": structural_quick_detail,
        "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET": all(external_smoke_detail.values()),
        "EXTERNAL_TP_SMOKE_DETAIL_BY_DATASET": external_smoke_detail,
        "EXTERNAL_TP_QUICK_READY_BY_DATASET": all(external_quick_detail.values()),
        "EXTERNAL_TP_QUICK_DETAIL_BY_DATASET": external_quick_detail,
        "FREEHGC_SCORE_TP_SMOKE_EXECUTED_BY_DATASET": bool(_freehgc_smoke_ready(rows, "DBLP")),
        "FREEHGC_SCORE_TP_DETAIL_BY_DATASET": freehgc_detail,
        "CONDENSATION_SCORE_TP_SMOKE_EXECUTED_BY_DATASET": all(condensation_detail.values()),
        "CONDENSATION_SCORE_TP_DETAIL_BY_DATASET": condensation_detail,
        "HESF_RCS_AUTO_EXECUTED_BY_DATASET": all(hesf_auto_detail.values()),
        "HESF_RCS_AUTO_DETAIL_BY_DATASET": hesf_auto_detail,
        "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE": all(rep_selected_detail.values()),
        "HESF_RCS_REP_SELECTED_DETAIL_BY_DATASET": rep_selected_detail,
        "HESF_RCS_REP_ACTUAL_VALIDATION_READY": all(rep_validation_detail.values()),
        "HESF_RCS_REP_ACTUAL_VALIDATION_DETAIL_BY_DATASET": rep_validation_detail,
        "NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE": no_pending,
        "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE": _no_diagnostics(rows),
        "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS": _no_placeholder_success(rows),
    }
    smoke_required = bool(
        flags["FULL_NATIVE_READY_BY_DATASET"]
        and flags["EXPORT_FULL_FIDELITY_PASS_BY_DATASET"]
        and structural_smoke_detail.get("DBLP", False)
        and any(external_smoke_detail.get(dataset, False) for dataset in dataset_names if dataset == "DBLP")
        and (_dataset_has_acm_metric(rows) if "ACM" in dataset_names else True)
        and (_dataset_has_imdb_metric(rows) if "IMDB" in dataset_names else True)
        and flags["NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE"]
    )
    flags["STAGE_REPORT_SMOKE_READY"] = smoke_required
    flags["STAGE_REPORT_QUICK_READY"] = bool(smoke_required and flags["STRUCTURAL_BASELINES_QUICK_READY_BY_DATASET"] and flags["EXTERNAL_TP_QUICK_READY_BY_DATASET"])
    return {name: flags.get(name, False) for name in GATE21_17_DECISION_FLAGS} | {k: v for k, v in flags.items() if k not in GATE21_17_DECISION_FLAGS}


def _ready(rows: list[dict[str, Any]], dataset: str, method: str, *, budget_type: str | None = None, budget: float | None = None) -> bool:
    for row in rows:
        if normalize_dataset(row.get("dataset")) != dataset or str(row.get("method", "")) != method:
            continue
        if budget_type is not None and str(row.get("requested_budget_type", "")) != budget_type:
            continue
        if budget is not None:
            value = float_value(row.get("requested_budget"))
            if value is None or abs(value - budget) > 1e-9:
                continue
        if gate21_17_row_ready(row):
            return True
    return False


def _all_methods_ready(rows: list[dict[str, Any]], dataset: str, methods: Sequence[str], *, budget: float) -> bool:
    return all(_ready(rows, dataset, method, budget_type="structural_storage_ratio", budget=budget) for method in methods)


def _external_smoke_ready(rows: list[dict[str, Any]], dataset: str) -> bool:
    herding_ready = _ready(rows, dataset, "Herding-HG-TP", budget_type="support_node_ratio", budget=0.50) or _ready(
        rows, dataset, "Herding-HG-TP-local", budget_type="support_node_ratio", budget=0.50
    )
    if dataset != "DBLP":
        return herding_ready
    return herding_ready and _freehgc_smoke_ready(rows, dataset)


def _freehgc_smoke_ready(rows: list[dict[str, Any]], dataset: str) -> bool:
    return _ready(rows, dataset, "FreeHGC-score-TP", budget_type="structural_storage_ratio", budget=0.20) or _ready(
        rows, dataset, "FreeHGC-score-TP-local", budget_type="structural_storage_ratio", budget=0.20
    )


def _any_ready_family(rows: list[dict[str, Any]], dataset: str, family: str) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method_family", "")) == family and gate21_17_row_ready(row) for row in rows)


def _dataset_has_acm_metric(rows: list[dict[str, Any]]) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == "ACM"
        and str(row.get("method", "")) in {"H6-node30", "HeSF-RCS-auto structural20", "HeSF-RCS-auto structural30"}
        and gate21_17_row_ready(row)
        for row in rows
    )


def _dataset_has_imdb_metric(rows: list[dict[str, Any]]) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == "IMDB"
        and str(row.get("method", "")) in {"HeSF-RCS-auto structural20", "HeSF-RCS-auto structural30", "H6-node30"}
        and gate21_17_row_ready(row)
        for row in rows
    )


def _no_diagnostics(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if bool_value(row.get("eligible_for_main_table", True)) is False:
            continue
        family = str(row.get("method_family", "")).lower()
        method = str(row.get("method", "")).lower()
        if "adapter" in family or "diagnostic" in family or "oracle" in method:
            return False
    return True


def _no_placeholder_success(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if not bool_value(row.get("success")):
            continue
        if not finite_metric(row.get("test_micro_f1_mean")) or not finite_metric(row.get("test_macro_f1_mean")):
            return False
    return True
