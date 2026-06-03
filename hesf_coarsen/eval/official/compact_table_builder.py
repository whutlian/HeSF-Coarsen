from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_22_COMPACT_FIELDS = (
    "dataset",
    "row_category",
    "method",
    "method_family",
    "method_source",
    "requested_budget_type",
    "requested_budget",
    "semantic_structural_storage_ratio",
    "actual_support_edge_ratio",
    "actual_support_node_ratio",
    "raw_hgb_text_byte_ratio",
    "test_micro_f1_mean_std",
    "test_macro_f1_mean_std",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "recovery_micro",
    "recovery_macro",
    "training_seed_count",
    "graph_seed_count",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "notes",
)

GATE21_22_CATEGORIES = (
    "Full-native-SeHGNN",
    "Export-full-SeHGNN",
    "HeSF-RCS-Rep-Validated",
    "Best-edge-or-structural-baseline",
    "Best-external-TP-baseline",
    "Best-condensation-score-TP-baseline",
    "Best-dataset-specific-baseline",
    "Best-Compressed-Validated diagnostic",
    "TestOracle-Best-Diagnostic",
)


def build_gate21_22_compact_table(rows: Iterable[Mapping[str, Any]], *, datasets: Sequence[str] = ("DBLP", "ACM", "IMDB")) -> list[dict[str, Any]]:
    source = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        dataset_rows = [row for row in source if normalize_dataset(row.get("dataset")) == dataset]
        out.append(_compact_row(dataset, "Full-native-SeHGNN", _find_method(dataset_rows, "Full-native-SeHGNN"), notes="full native anchor"))
        out.append(_compact_row(dataset, "Export-full-SeHGNN", _find_method(dataset_rows, "Export-full-SeHGNN"), notes="official exported full graph anchor"))
        out.append(_compact_row(dataset, "HeSF-RCS-Rep-Validated", _best_by_validation([row for row in dataset_rows if _is_hesf_candidate(row)]), notes="validation-only HeSF-RCS family selection"))
        out.append(_compact_row(dataset, "Best-edge-or-structural-baseline", _best_by_validation([row for row in dataset_rows if _is_edge_structural(row)]), notes="edge/structural/closure/channel baseline category"))
        out.append(_compact_row(dataset, "Best-external-TP-baseline", _best_by_validation([row for row in dataset_rows if _is_external_tp(row)]), notes="Random/Herding/KCenter/GraphSparsify/Coarsening TP category"))
        out.append(_compact_row(dataset, "Best-condensation-score-TP-baseline", _best_by_validation([row for row in dataset_rows if _is_condensation(row)]), notes="FreeHGC/HGCond/GCond/GCondenser score proxy category"))
        out.append(_compact_row(dataset, "Best-dataset-specific-baseline", _best_by_validation([row for row in dataset_rows if _is_dataset_specific(dataset, row)]), notes="dataset-specific closure/channel baseline category"))
        out.append(_compact_row(dataset, "Best-Compressed-Validated diagnostic", _best_by_validation([row for row in dataset_rows if _is_compressed(row)]), eligible=False, notes="diagnostic only; selected by validation across all compressed rows"))
        out.append(_compact_row(dataset, "TestOracle-Best-Diagnostic", _best_by_test([row for row in dataset_rows if _is_compressed(row)]), eligible=False, notes="diagnostic only; selected by test metric"))
    return out


def compact_table_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    fields = (
        "dataset",
        "row_category",
        "method",
        "method_family",
        "semantic_structural_storage_ratio",
        "actual_support_edge_ratio",
        "test_micro_f1_mean_std",
        "test_macro_f1_mean_std",
        "eligible_for_main_decision",
        "notes",
    )
    lines = ["# Gate21.22 Final Compact Table", "", "|" + "|".join(fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_md(row.get(field, "")) for field in fields) + "|")
    return "\n".join(lines) + "\n"


def _compact_row(dataset: str, category: str, row: Mapping[str, Any] | None, *, notes: str, eligible: bool | None = None) -> dict[str, Any]:
    if row is None:
        return {
            "dataset": dataset,
            "row_category": category,
            "method": "",
            "eligible_for_main_decision": False,
            "notes": f"missing source row: {notes}",
        }
    main_eligible = _row_ready(row) if eligible is None else bool(eligible)
    micro = _first_value(row, "test_micro_f1_mean", "test_micro_f1")
    macro = _first_value(row, "test_macro_f1_mean", "test_macro_f1")
    micro_std = row.get("test_micro_f1_std", "")
    macro_std = row.get("test_macro_f1_std", "")
    return {
        "dataset": dataset,
        "row_category": category,
        "method": row.get("method", ""),
        "method_family": row.get("method_family", ""),
        "method_source": row.get("method_source", row.get("source_method", "")),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "semantic_structural_storage_ratio": _first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio"),
        "actual_support_edge_ratio": _first_value(row, "actual_support_edge_ratio", "support_edge_ratio"),
        "actual_support_node_ratio": _first_value(row, "actual_support_node_ratio", "support_node_ratio"),
        "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
        "test_micro_f1_mean_std": _mean_std(micro, micro_std),
        "test_macro_f1_mean_std": _mean_std(macro, macro_std),
        "test_micro_f1_mean": micro,
        "test_micro_f1_std": micro_std,
        "test_macro_f1_mean": macro,
        "test_macro_f1_std": macro_std,
        "recovery_micro": _first_value(row, "recovery_micro", "recovery_vs_native_full_micro"),
        "recovery_macro": _first_value(row, "recovery_macro", "recovery_vs_native_full_macro"),
        "training_seed_count": row.get("training_seed_count", ""),
        "graph_seed_count": row.get("graph_seed_count", ""),
        "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified", True)),
        "eligible_for_main_decision": main_eligible,
        "notes": notes,
    }


def _best_by_validation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _row_ready(row) and float_value(_first_value(row, "validation_micro_f1_mean", "validation_micro_f1")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float_value(_first_value(row, "validation_micro_f1_mean", "validation_micro_f1")) or -1.0, float_value(_first_value(row, "validation_macro_f1_mean", "validation_macro_f1")) or -1.0, -_cost(row)))


def _best_by_test(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _row_ready(row)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float_value(_first_value(row, "test_micro_f1_mean", "test_micro_f1")) or -1.0, float_value(_first_value(row, "test_macro_f1_mean", "test_macro_f1")) or -1.0, -_cost(row)))


def _row_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and not bool_value(row.get("constraint_safe_fallback"))
        and not bool_value(row.get("full_fallback"))
        and finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
        and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    )


def _is_hesf_candidate(row: Mapping[str, Any]) -> bool:
    family = str(row.get("method_family", ""))
    method = str(row.get("method", ""))
    banned = ("FreeHGC-score", "HGCond-score", "GCond-score", "GCondenser-score", "Herding", "KCenter", "GraphSparsify", "Degree", "Random", "ValidationGreedy")
    return family in {"schema_preserving_rcs", "hesf_rcs", "hesf_dataset_planner"} and not any(token in method for token in banned)


def _is_edge_structural(row: Mapping[str, Any]) -> bool:
    family = str(row.get("method_family", ""))
    method = str(row.get("method", ""))
    if _is_external_tp(row) or _is_condensation(row):
        return False
    return family in {"relation_structural_baseline", "edge_sparsification_baseline", "closure_field_baseline", "channel_baseline"} or any(token in method for token in ("Random-edge", "Degree-edge", "Proportional", "Degree-field", "ValidationGreedy-field", "Degree-channel", "ValidationGreedy-channel"))


def _is_external_tp(row: Mapping[str, Any]) -> bool:
    family = str(row.get("method_family", ""))
    method = str(row.get("method", ""))
    if _is_condensation(row) or "score-" in method:
        return False
    return family in {"external_tp_baseline", "local_coreset_tp_baseline", "local_graph_sparsify_tp_baseline"} and any(token in method for token in ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP"))


def _is_condensation(row: Mapping[str, Any]) -> bool:
    return str(row.get("method_family", "")) in {"condensation_score_tp_proxy", "condensation_score_as_selector"}


def _is_dataset_specific(dataset: str, row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    if family in {"dataset_specific_closure_baseline", "dataset_specific_channel_baseline"}:
        return True
    if dataset == "ACM":
        return method in {"ACM-Degree-field20", "ACM-ValidationGreedy-field20"}
    if dataset == "IMDB":
        return method in {"IMDB-MDfull-MA50-MK50", "IMDB-ValidationGreedy-channel50"}
    if dataset == "DBLP":
        return method in {"Degree-edge-relwise", "Random-edge-relwise", "Proportional-relation-budget"}
    return False


def _is_compressed(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")) not in {"Full-native-SeHGNN", "Export-full-SeHGNN"} and not bool_value(row.get("constraint_safe_fallback"))


def _find_method(rows: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if str(row.get("method", "")) == method), None)


def _cost(row: Mapping[str, Any]) -> float:
    for field in ("semantic_structural_storage_ratio", "actual_support_edge_ratio", "raw_hgb_text_byte_ratio"):
        value = float_value(row.get(field))
        if value is not None:
            return value
    return 999.0


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""


def _mean_std(mean: Any, std: Any) -> str:
    if mean in {"", None}:
        return ""
    return f"{mean} +/- {std}" if std not in {"", None} else str(mean)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
