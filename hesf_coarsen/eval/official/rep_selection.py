from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


REP_SELECTION_FIELDS = (
    "dataset",
    "rep_type",
    "candidate_pool",
    "selected_method",
    "selected_method_family",
    "selection_metric",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "uses_test_for_selection",
    "eligible_for_main_decision",
    "selection_reason",
)


def select_gate21_20_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        compressed = [row for row in source_rows if normalize_dataset(row.get("dataset")) == dataset and _eligible_compressed(row)]
        hesf_pool = [row for row in compressed if _is_hesf_rcs_candidate(row)]
        out.append(
            _selection_row(
                dataset=dataset,
                rep_type="HeSF-RCS-Rep-Validated",
                candidate_pool="hesf_rcs_only",
                selected=_select_by_validation(hesf_pool),
                uses_test=False,
                missing_reason="missing_real_validation_metric",
            )
        )
        out.append(
            _selection_row(
                dataset=dataset,
                rep_type="Best-Compressed-Validated",
                candidate_pool="all_compressed",
                selected=_select_by_validation(compressed),
                uses_test=False,
                missing_reason="missing_real_validation_metric",
            )
        )
        out.append(
            _selection_row(
                dataset=dataset,
                rep_type="TestOracle-Best",
                candidate_pool="all_methods_test_diagnostic",
                selected=_select_by_test(compressed),
                uses_test=True,
                missing_reason="missing_test_metric",
            )
        )
    return out


def resolve_validation_metrics(row: Mapping[str, Any], *, training_runs: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    out = dict(row)
    if float_value(out.get("validation_micro_f1_mean")) is not None and float_value(out.get("validation_macro_f1_mean")) is not None:
        out["validation_resolution_source"] = "row"
        return out
    dataset = normalize_dataset(out.get("dataset"))
    method = str(out.get("method", ""))
    for run in training_runs:
        if normalize_dataset(run.get("dataset")) != dataset or str(run.get("method", "")) != method:
            continue
        micro = float_value(run.get("validation_micro_f1") or run.get("validation_micro_f1_mean"))
        macro = float_value(run.get("validation_macro_f1") or run.get("validation_macro_f1_mean"))
        if micro is not None and macro is not None:
            out["validation_micro_f1_mean"] = micro
            out["validation_macro_f1_mean"] = macro
            out["validation_resolution_source"] = "training_runs"
            return out
    out["validation_resolution_source"] = "missing"
    return out


def _eligible_compressed(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
        return False
    if "Rep" in method or "TestOracle" in method:
        return False
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and not bool_value(row.get("constraint_safe_fallback"))
    )


def _is_hesf_rcs_candidate(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    planner_mode = str(row.get("planner_mode", ""))
    if "external" in family.lower():
        return False
    if any(token in method for token in ("Random", "Degree", "Proportional", "ValidationGreedy", "MDfull", "FreeHGC-score-as-selector")):
        return False
    return bool("HeSF-RCS" in method or family in {"schema_preserving_rcs", "hesf_rcs"} or planner_mode.startswith("hesf"))


def _select_by_validation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if float_value(row.get("validation_micro_f1_mean")) is not None and float_value(row.get("validation_macro_f1_mean")) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float_value(row.get("validation_micro_f1_mean")) or -1.0,
            float_value(row.get("validation_macro_f1_mean")) or -1.0,
            -_cost(row),
        ),
    )


def _select_by_test(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if float_value(row.get("test_micro_f1_mean")) is not None and float_value(row.get("test_macro_f1_mean")) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float_value(row.get("test_micro_f1_mean")) or -1.0,
            float_value(row.get("test_macro_f1_mean")) or -1.0,
            -_cost(row),
        ),
    )


def _selection_row(
    *,
    dataset: str,
    rep_type: str,
    candidate_pool: str,
    selected: Mapping[str, Any] | None,
    uses_test: bool,
    missing_reason: str,
) -> dict[str, Any]:
    if selected is None:
        return {
            "dataset": dataset,
            "rep_type": rep_type,
            "candidate_pool": candidate_pool,
            "selected_method": "",
            "selected_method_family": "",
            "selection_metric": "test_micro_f1/test_macro_f1" if uses_test else "validation_micro_f1/validation_macro_f1",
            "validation_micro_f1": "",
            "validation_macro_f1": "",
            "test_micro_f1": "",
            "test_macro_f1": "",
            "uses_test_for_selection": bool(uses_test),
            "eligible_for_main_decision": False,
            "selection_reason": missing_reason,
        }
    return {
        "dataset": dataset,
        "rep_type": rep_type,
        "candidate_pool": candidate_pool,
        "selected_method": selected.get("method", ""),
        "selected_method_family": selected.get("method_family", ""),
        "selection_metric": "test_micro_f1/test_macro_f1" if uses_test else "validation_micro_f1/validation_macro_f1",
        "validation_micro_f1": selected.get("validation_micro_f1_mean", ""),
        "validation_macro_f1": selected.get("validation_macro_f1_mean", ""),
        "test_micro_f1": selected.get("test_micro_f1_mean", ""),
        "test_macro_f1": selected.get("test_macro_f1_mean", ""),
        "uses_test_for_selection": bool(uses_test),
        "eligible_for_main_decision": not uses_test,
        "selection_reason": "selected_by_test_metric" if uses_test else "selected_by_real_validation_metric",
    }


def _cost(row: Mapping[str, Any]) -> float:
    for key in ("semantic_structural_storage_ratio", "actual_support_edge_ratio", "channel_edge_ratio", "keyword_feature_ratio", "support_node_ratio"):
        value = float_value(row.get(key))
        if value is not None:
            return value
    return 999.0
