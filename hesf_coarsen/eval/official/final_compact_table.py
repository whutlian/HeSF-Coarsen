from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_21_FINAL_COMPACT_FIELDS = (
    "dataset",
    "row_category",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "semantic_structural_storage_ratio",
    "actual_support_edge_ratio",
    "raw_hgb_text_byte_ratio",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "recovery_micro",
    "recovery_macro",
    "training_seed_count",
    "graph_seed_count",
    "official_sehgnn_unmodified",
    "schema_compatible",
    "eligible_for_main_decision",
    "notes",
)

ROW_CATEGORIES = (
    "Full-native-SeHGNN",
    "Export-full-SeHGNN",
    "HeSF-RCS-Rep-Validated",
    "Best-edge-or-structural-baseline",
    "Best-external-TP-baseline",
    "Best-dataset-specific-baseline",
    "Best-Compressed-Validated diagnostic",
)


def build_gate21_21_final_compact_table(
    rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    reps = [dict(row) for row in rep_rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        dataset_rows = [row for row in source_rows if normalize_dataset(row.get("dataset")) == dataset]
        out.append(_compact_row(dataset, "Full-native-SeHGNN", _find_method(dataset_rows, "Full-native-SeHGNN"), notes="full native anchor"))
        out.append(_compact_row(dataset, "Export-full-SeHGNN", _find_method(dataset_rows, "Export-full-SeHGNN"), notes="official exported full graph anchor"))
        hesf_rep = _rep_method(reps, dataset, "HeSF-RCS-Rep-Validated")
        out.append(_compact_row(dataset, "HeSF-RCS-Rep-Validated", _find_method(dataset_rows, hesf_rep), notes="selected by validation metrics inside HeSF-RCS pool"))
        edge = _best_by_validation([row for row in dataset_rows if _is_edge_or_structural_baseline(row)])
        out.append(_compact_row(dataset, "Best-edge-or-structural-baseline", edge, notes="best validation baseline among edge/field/channel structural rows"))
        external = _best_by_validation([row for row in dataset_rows if str(row.get("method_family", "")) == "external_tp_baseline"])
        out.append(_compact_row(dataset, "Best-external-TP-baseline", external, notes="best validation target-preserving external/local proxy row"))
        dataset_specific = _best_dataset_specific(dataset, dataset_rows)
        out.append(_compact_row(dataset, "Best-dataset-specific-baseline", dataset_specific, notes="dataset-specific non-HeSF official-compatible baseline"))
        best_compressed = _rep_method(reps, dataset, "Best-Compressed-Validated")
        out.append(
            _compact_row(
                dataset,
                "Best-Compressed-Validated diagnostic",
                _find_method(dataset_rows, best_compressed),
                eligible_override=False,
                notes="diagnostic; can select any compressed method by validation",
            )
        )
    return out


def compact_table_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    fields = (
        "dataset",
        "row_category",
        "method",
        "semantic_structural_storage_ratio",
        "actual_support_edge_ratio",
        "test_micro_f1_mean",
        "test_macro_f1_mean",
        "eligible_for_main_decision",
        "notes",
    )
    lines = ["# Gate21.21 Final Compact Table", "", "|" + "|".join(fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_md(row.get(field, "")) for field in fields) + "|")
    return "\n".join(lines) + "\n"


def _compact_row(
    dataset: str,
    category: str,
    row: Mapping[str, Any] | None,
    *,
    eligible_override: bool | None = None,
    notes: str,
) -> dict[str, Any]:
    if row is None:
        return {
            "dataset": dataset,
            "row_category": category,
            "method": "",
            "eligible_for_main_decision": False,
            "notes": f"missing source row: {notes}",
        }
    eligible = _row_ready(row) if eligible_override is None else bool(eligible_override)
    return {
        "dataset": dataset,
        "row_category": category,
        "method": row.get("method", ""),
        "method_family": row.get("method_family", ""),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "semantic_structural_storage_ratio": _first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio", "actual_structural_storage_ratio"),
        "actual_support_edge_ratio": _first_value(row, "actual_support_edge_ratio", "support_edge_ratio"),
        "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
        "test_micro_f1_mean": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
        "test_micro_f1_std": row.get("test_micro_f1_std", ""),
        "test_macro_f1_mean": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
        "test_macro_f1_std": row.get("test_macro_f1_std", ""),
        "recovery_micro": _first_value(row, "recovery_vs_native_full_micro", "recovery_micro"),
        "recovery_macro": _first_value(row, "recovery_vs_native_full_macro", "recovery_macro"),
        "training_seed_count": row.get("training_seed_count", ""),
        "graph_seed_count": row.get("graph_seed_count", ""),
        "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified", True)),
        "schema_compatible": bool_value(row.get("schema_compatible", True)),
        "eligible_for_main_decision": eligible,
        "notes": notes,
    }


def _best_dataset_specific(dataset: str, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if dataset == "ACM":
        return _best_by_validation([row for row in rows if str(row.get("method", "")) in {"ACM-Degree-field20", "ACM-ValidationGreedy-field20"}])
    if dataset == "IMDB":
        return _best_by_validation([row for row in rows if str(row.get("method", "")) in {"IMDB-ValidationGreedy-channel50", "IMDB-MDfull-MA50-MK50"}])
    return _best_by_validation([row for row in rows if _is_edge_or_structural_baseline(row) or str(row.get("method_family", "")) == "external_tp_baseline"])


def _best_by_validation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _row_ready(row) and float_value(_first_value(row, "validation_micro_f1_mean", "validation_micro_f1")) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float_value(_first_value(row, "validation_micro_f1_mean", "validation_micro_f1")) or -1.0,
            float_value(_first_value(row, "validation_macro_f1_mean", "validation_macro_f1")) or -1.0,
            -(float_value(_first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio", "actual_support_edge_ratio")) or 999.0),
        ),
    )


def _row_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and bool_value(row.get("training_executed", True))
        and not bool_value(row.get("constraint_safe_fallback"))
        and finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
        and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    )


def _is_edge_or_structural_baseline(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    if family == "external_tp_baseline" or method.endswith("-HG-TP") or "-HG-TP-" in method:
        return False
    return bool(
        family in {"relation_structural_baseline", "channel_baseline"}
        or any(token in method for token in ("Random", "Degree", "Proportional", "ValidationGreedy", "MDfull"))
    )


def _find_method(rows: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any] | None:
    if not method:
        return None
    for row in rows:
        if str(row.get("method", "")) == method:
            return row
    return None


def _rep_method(rows: Sequence[Mapping[str, Any]], dataset: str, rep_type: str) -> str:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("rep_type", "")) == rep_type:
            return str(row.get("selected_method", ""))
    return ""


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
