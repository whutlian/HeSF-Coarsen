from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import DATASETS, bool_value, float_value, normalize_dataset


def select_gate21_17_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source_rows
            if normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")).startswith("HeSF-RCS-auto")
            and bool_value(row.get("eligible_for_main_table", True))
            and bool_value(row.get("success"))
            and bool_value(row.get("training_executed"))
        ]
        main = _select_main_rep(candidates)
        oracle = _select_test_oracle(candidates)
        if main is None:
            out.append(
                {
                    "dataset": dataset,
                    "method": "HeSF-RCS-Rep",
                    "source_method": "",
                    "selected_as_rep": False,
                    "selection_source": "validation_metric_missing",
                    "rep_selection_confidence": "missing",
                    "uses_test_for_selection": False,
                    "eligible_for_main_table": False,
                    "eligible_for_decision": False,
                    "failure_type": "validation_metric_missing",
                    "failure_reason": "No successful HeSF-RCS-auto row has actual validation metrics or validation proxy.",
                }
            )
        else:
            source, selection_source = main
            out.append(_rep_row(source, method="HeSF-RCS-Rep", selection_source=selection_source, diagnostic=False))
        if oracle is not None:
            out.append(_rep_row(oracle, method="HeSF-RCS-TestOracleRep", selection_source="test_oracle_diagnostic_only", diagnostic=True))
    return out


def select_gate21_18_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source_rows
            if normalize_dataset(row.get("dataset")) == dataset
            and "HeSF-RCS-auto" in str(row.get("method", ""))
            and bool_value(row.get("eligible_for_main_table", True))
            and bool_value(row.get("success"))
            and bool_value(row.get("training_executed"))
        ]
        actual = [
            row
            for row in candidates
            if float_value(row.get("validation_micro_f1_mean")) is not None and float_value(row.get("validation_macro_f1_mean")) is not None
        ]
        if actual:
            selected = max(
                actual,
                key=lambda row: (
                    float_value(row.get("validation_micro_f1_mean")) or -1.0,
                    float_value(row.get("validation_macro_f1_mean")) or -1.0,
                    -(float_value(row.get("semantic_structural_storage_ratio")) or float_value(row.get("actual_structural_storage_ratio")) or 999.0),
                ),
            )
            out.append(_rep_row(selected, method="HeSF-RCS-Rep-Validated", selection_source="actual_validation", diagnostic=False))
        else:
            out.append(
                {
                    "dataset": dataset,
                    "method": "HeSF-RCS-Rep-Validated",
                    "source_method": "",
                    "selected_as_rep": False,
                    "selection_source": "validation_missing",
                    "rep_selection_confidence": "missing",
                    "uses_test_for_selection": False,
                    "eligible_for_main_table": False,
                    "eligible_for_decision": False,
                    "failure_type": "validation_missing",
                    "failure_reason": "Gate21.18 does not allow proxy-only HeSF-RCS representative selection.",
                }
            )
        oracle = _select_test_oracle(candidates)
        if oracle is not None:
            out.append(_rep_row(oracle, method="HeSF-RCS-TestOracleRep", selection_source="test_oracle_diagnostic_only", diagnostic=True))
    return out


def select_gate21_19_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    """Select a Gate21.19 representative using validation metrics only.

    Gate21.19 compares multiple real planner families per dataset, so the
    representative selector is no longer limited to rows whose method contains
    ``HeSF-RCS-auto``.  Test metrics remain diagnostic-only.
    """

    source_rows = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source_rows
            if normalize_dataset(row.get("dataset")) == dataset
            and _gate21_19_rep_candidate(row)
        ]
        actual = [
            row
            for row in candidates
            if float_value(row.get("validation_micro_f1_mean")) is not None
            and float_value(row.get("validation_macro_f1_mean")) is not None
        ]
        if actual:
            selected = max(
                actual,
                key=lambda row: (
                    float_value(row.get("validation_micro_f1_mean")) or -1.0,
                    float_value(row.get("validation_macro_f1_mean")) or -1.0,
                    -_gate21_19_cost(row),
                ),
            )
            out.append(_rep_row(selected, method="HeSF-RCS-Rep-Validated", selection_source="actual_validation", diagnostic=False))
        else:
            out.append(
                {
                    "dataset": dataset,
                    "method": "HeSF-RCS-Rep-Validated",
                    "source_method": "",
                    "selected_as_rep": False,
                    "selection_source": "validation_missing",
                    "rep_selection_confidence": "missing",
                    "uses_test_for_selection": False,
                    "eligible_for_main_table": False,
                    "eligible_for_decision": False,
                    "failure_type": "validation_missing",
                    "failure_reason": "Gate21.19 representative selection requires actual validation metrics.",
                }
            )
        oracle = _select_test_oracle(candidates)
        if oracle is not None:
            out.append(_rep_row(oracle, method="HeSF-RCS-TestOracleRep", selection_source="test_oracle_diagnostic_only", diagnostic=True))
    return out


def _select_main_rep(candidates: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str] | None:
    with_actual = [row for row in candidates if float_value(row.get("validation_micro_f1_mean")) is not None and float_value(row.get("validation_macro_f1_mean")) is not None]
    if with_actual:
        return (
            max(
                with_actual,
                key=lambda row: (
                    float_value(row.get("validation_micro_f1_mean")) or -1.0,
                    float_value(row.get("validation_macro_f1_mean")) or -1.0,
                    -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
                ),
            ),
            "actual_validation",
        )
    with_proxy = [row for row in candidates if float_value(row.get("validation_proxy_score")) is not None]
    if with_proxy:
        return (
            max(
                with_proxy,
                key=lambda row: (
                    float_value(row.get("validation_proxy_score")) or -1.0,
                    -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
                ),
            ),
            "validation_proxy",
        )
    return None


def _gate21_19_rep_candidate(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
        return False
    if bool_value(row.get("constraint_safe_fallback")):
        return False
    if bool_value(row.get("uses_test_for_selection")):
        return False
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("eligible_for_compression_claim", True))
        and bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
    )


def _gate21_19_cost(row: Mapping[str, Any]) -> float:
    for key in (
        "semantic_structural_storage_ratio",
        "actual_support_edge_ratio",
        "keyword_feature_ratio",
        "channel_edge_ratio",
        "support_node_ratio",
        "raw_hgb_text_byte_ratio",
    ):
        value = float_value(row.get(key))
        if value is not None:
            return value
    return 999.0


def _select_test_oracle(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    with_test = [row for row in candidates if float_value(row.get("test_micro_f1_mean")) is not None and float_value(row.get("test_macro_f1_mean")) is not None]
    if not with_test:
        return None
    return max(
        with_test,
        key=lambda row: (
            float_value(row.get("test_micro_f1_mean")) or -1.0,
            float_value(row.get("test_macro_f1_mean")) or -1.0,
            -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
        ),
    )


def _rep_row(source: Mapping[str, Any], *, method: str, selection_source: str, diagnostic: bool) -> dict[str, Any]:
    return {
        "dataset": normalize_dataset(source.get("dataset")),
        "method": method,
        "source_method": source.get("method", ""),
        "method_family": "schema_preserving_rcs_diagnostic" if diagnostic else "schema_preserving_rcs",
        "planner_backend": source.get("planner_backend", ""),
        "planner_mode": source.get("planner_mode", ""),
        "requested_budget_type": source.get("requested_budget_type", ""),
        "requested_budget": source.get("requested_budget", ""),
        "actual_structural_storage_ratio": source.get("actual_structural_storage_ratio", ""),
        "actual_edge_ratio": source.get("actual_edge_ratio", ""),
        "actual_support_edge_ratio": source.get("actual_support_edge_ratio", ""),
        "actual_support_node_ratio": source.get("actual_support_node_ratio", ""),
        "semantic_structural_storage_ratio": source.get("semantic_structural_storage_ratio", ""),
        "support_node_ratio": source.get("support_node_ratio", ""),
        "support_edge_ratio": source.get("support_edge_ratio", ""),
        "raw_hgb_text_byte_ratio": source.get("raw_hgb_text_byte_ratio", ""),
        "static_inference_package_ratio": source.get("static_inference_package_ratio", ""),
        "reconstructable_package_ratio": source.get("reconstructable_package_ratio", ""),
        "keyword_feature_ratio": source.get("keyword_feature_ratio", ""),
        "PK_edge_ratio": source.get("PK_edge_ratio", ""),
        "actor_channel_ratio": source.get("actor_channel_ratio", ""),
        "keyword_channel_ratio": source.get("keyword_channel_ratio", ""),
        "channel_edge_ratio": source.get("channel_edge_ratio", ""),
        "test_micro_f1_mean": source.get("test_micro_f1_mean", ""),
        "test_macro_f1_mean": source.get("test_macro_f1_mean", ""),
        "validation_micro_f1_mean": source.get("validation_micro_f1_mean", ""),
        "validation_macro_f1_mean": source.get("validation_macro_f1_mean", ""),
        "validation_proxy_score": source.get("validation_proxy_score", ""),
        "selected_edge_hash": source.get("selected_edge_hash", ""),
        "planner_config_hash": source.get("planner_config_hash", ""),
        "source_path": source.get("source_path", ""),
        "training_executed": source.get("training_executed", True),
        "success": source.get("success", True),
        "official_hgb_exported": source.get("official_hgb_exported", True),
        "official_sehgnn_unmodified": source.get("official_sehgnn_unmodified", True),
        "schema_compatible": source.get("schema_compatible", True),
        "target_preserving": source.get("target_preserving", True),
        "selection_source": selection_source,
        "rep_selection_confidence": "actual_validation" if selection_source == "actual_validation" else "proxy_only" if selection_source == "validation_proxy" else "diagnostic_only",
        "uses_test_for_selection": bool(diagnostic),
        "eligible_for_main_table": not diagnostic,
        "eligible_for_decision": not diagnostic,
        "selected_as_rep": True,
    }
