from __future__ import annotations

from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


ALL_METHODS_EXPERIMENT_FIELDS = (
    "dataset",
    "method_group",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "node_compression_ratio",
    "edge_compression_ratio",
    "semantic_structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "static_inference_package_ratio",
    "reconstructable_package_ratio",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "training_seed_count",
    "graph_seed_count",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
)


def build_all_methods_experiment_table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table = [_experiment_row(row) for row in rows if _eligible(row)]
    return sorted(table, key=lambda row: (_dataset_order(row["dataset"]), _group_order(row["method_group"]), _cost(row), str(row["method"])))


def experiment_table_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    fields = (
        "dataset",
        "method_group",
        "method",
        "node_compression_ratio",
        "edge_compression_ratio",
        "semantic_structural_storage_ratio",
        "test_micro_f1_mean",
        "test_macro_f1_mean",
        "recovery_vs_native_full_micro",
    )
    lines = ["# Gate21.22 All-Methods Experiment Table", "", "|" + "|".join(fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_md(row.get(field, "")) for field in fields) + "|")
    return "\n".join(lines) + "\n"


def _experiment_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": normalize_dataset(row.get("dataset")),
        "method_group": _method_group(row),
        "method": row.get("method", ""),
        "method_family": row.get("method_family", ""),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "node_compression_ratio": _numeric(_first_value(row, "actual_support_node_ratio", "support_node_ratio", "total_node_ratio"), default=1.0),
        "edge_compression_ratio": _numeric(_first_value(row, "actual_support_edge_ratio", "support_edge_ratio", "channel_edge_ratio"), default=1.0),
        "semantic_structural_storage_ratio": _numeric(_first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio", "actual_structural_storage_ratio", "keyword_feature_ratio", "channel_edge_ratio"), default=1.0),
        "raw_hgb_text_byte_ratio": _numeric(_first_value(row, "raw_hgb_text_byte_ratio", "hgb_raw_file_byte_ratio", "official_text_hgb_byte_ratio"), default=1.0),
        "static_inference_package_ratio": _numeric(_first_value(row, "static_inference_package_ratio", "preprocessed_cache_byte_ratio", "raw_hgb_text_byte_ratio"), default=1.0),
        "reconstructable_package_ratio": _numeric(_first_value(row, "reconstructable_package_ratio", "transform_recipe_package_ratio", "raw_hgb_text_byte_ratio"), default=1.0),
        "test_micro_f1_mean": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
        "test_micro_f1_std": row.get("test_micro_f1_std", ""),
        "test_macro_f1_mean": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
        "test_macro_f1_std": row.get("test_macro_f1_std", ""),
        "validation_micro_f1_mean": _first_value(row, "validation_micro_f1_mean", "validation_micro_f1"),
        "validation_macro_f1_mean": _first_value(row, "validation_macro_f1_mean", "validation_macro_f1"),
        "recovery_vs_native_full_micro": _first_value(row, "recovery_vs_native_full_micro", "recovery_micro"),
        "recovery_vs_native_full_macro": _first_value(row, "recovery_vs_native_full_macro", "recovery_macro"),
        "training_seed_count": row.get("training_seed_count", ""),
        "graph_seed_count": row.get("graph_seed_count", ""),
        "official_hgb_exported": bool_value(row.get("official_hgb_exported", True)),
        "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified", True)),
    }


def _eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and not bool_value(row.get("constraint_safe_fallback"))
        and not bool_value(row.get("full_fallback"))
        and str(row.get("method", ""))
    )


def _method_group(row: Mapping[str, Any]) -> str:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
        return "full_anchor"
    if family in {"schema_preserving_rcs", "hesf_rcs", "hesf_dataset_planner"} or "HeSF-RCS" in method:
        return "ours_hesf"
    if family in {"condensation_score_tp_proxy", "condensation_score_as_selector"}:
        return "external_condensation_score_proxy"
    if family in {"external_tp_baseline", "local_coreset_tp_baseline", "local_graph_sparsify_tp_baseline"}:
        return "external_tp_baseline"
    return "structural_or_dataset_baseline"


def _dataset_order(dataset: str) -> int:
    return {"DBLP": 0, "ACM": 1, "IMDB": 2}.get(normalize_dataset(dataset), 99)


def _group_order(group: str) -> int:
    return {
        "full_anchor": 0,
        "ours_hesf": 1,
        "external_tp_baseline": 2,
        "external_condensation_score_proxy": 3,
        "structural_or_dataset_baseline": 4,
    }.get(group, 99)


def _cost(row: Mapping[str, Any]) -> float:
    for field in ("node_compression_ratio", "semantic_structural_storage_ratio", "edge_compression_ratio", "raw_hgb_text_byte_ratio"):
        value = float_value(row.get(field))
        if value is not None:
            return value
    return 999.0


def _numeric(value: Any, *, default: float) -> float:
    parsed = float_value(value)
    return float(parsed) if parsed is not None else float(default)


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None, "induced_schema_preserving", "all_target_preserved"}:
            return value
    return ""


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
