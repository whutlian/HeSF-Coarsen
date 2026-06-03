from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.compact_table_builder import GATE21_22_CATEGORIES
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_22_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE",
    "BUDGET_METRIC_SEMANTICS_PASS",
    "HESF_REP_CANDIDATE_POOL_PASS",
    "HESF_REP_NO_TEST_LEAKAGE",
    "DBLP_CONDENSATION_SCORE_BASELINES_READY",
    "ACM_CONDENSATION_SCORE_BASELINES_READY",
    "IMDB_CONDENSATION_SCORE_BASELINES_READY",
    "GCONDENSER_PROXY_READY",
    "FREEHGC_STANDARD_PROTOCOL_HANDLED",
    "FREEHGC_TP_PROXY_READY",
    "HGCOND_TP_PROXY_READY",
    "GCOND_TP_PROXY_READY",
    "GCONDENSER_TP_PROXY_READY",
    "COMPACT_TABLE_HAS_BEST_CONDENSATION_SCORE_CATEGORY",
    "COMPACT_TABLE_EXTERNAL_CATEGORIES_SEPARATED",
    "STAGE_COMPACT_TABLE_READY",
    "PAPER_FINAL_EXTERNAL_BASELINES_READY",
    "PAPER_FINAL_TABLE_READY",
)

CONDENSATION_BASELINES = ("FreeHGC", "HGCond", "GCond", "GCondenser")


def gate21_22_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    compact_rows: Iterable[Mapping[str, Any]],
    repo_rows: Iterable[Mapping[str, Any]],
    standard_rows: Iterable[Mapping[str, Any]],
    condensation_tp_rows: Iterable[Mapping[str, Any]],
    condensation_selector_rows: Iterable[Mapping[str, Any]],
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in datasets)
    main = [dict(row) for row in main_rows]
    compact = [dict(row) for row in compact_rows]
    repos = [dict(row) for row in repo_rows]
    standard = [dict(row) for row in standard_rows]
    condensation = [dict(row) for row in condensation_tp_rows] + [dict(row) for row in condensation_selector_rows]
    blockers: list[str] = []

    full_ready = {dataset: _ready(_find_method(main, dataset, "Full-native-SeHGNN")) for dataset in datasets}
    export_ready = {dataset: _ready(_find_method(main, dataset, "Export-full-SeHGNN")) for dataset in datasets}
    _add_missing(blockers, "full native", full_ready)
    _add_missing(blockers, "export full", export_ready)

    no_full_fallback = not any(bool_value(row.get("constraint_safe_fallback")) or bool_value(row.get("full_fallback")) for row in main if _compression_eligible(row))
    if not no_full_fallback:
        blockers.append("One or more compression rows are still marked as full fallback.")

    budget_semantics = _budget_semantics_pass(main)
    if not budget_semantics:
        blockers.append("At least one successful non-full row is missing cost/budget semantic fields.")

    hesf_candidates = [row for row in compact if str(row.get("row_category", "")) == "HeSF-RCS-Rep-Validated"]
    hesf_pool_pass = bool(hesf_candidates) and all(_compact_ready(row) and not _banned_hesf_method(str(row.get("method", ""))) for row in hesf_candidates)
    if not hesf_pool_pass:
        blockers.append("HeSF-RCS representative row is missing, not ready, or mixed with external score/baseline methods.")
    hesf_no_test = all(not bool_value(row.get("uses_test_for_selection")) for row in main if str(row.get("method_family", "")) in {"schema_preserving_rcs", "hesf_rcs", "hesf_dataset_planner"})
    if not hesf_no_test:
        blockers.append("A HeSF representative candidate is marked as using test labels/metrics for selection.")

    by_dataset_baseline = {
        dataset: {baseline: _baseline_ready(condensation, dataset, baseline) for baseline in CONDENSATION_BASELINES}
        for dataset in datasets
    }
    condensation_ready_by_dataset = {dataset: all(by_dataset_baseline[dataset].values()) for dataset in datasets}
    for dataset, ready in condensation_ready_by_dataset.items():
        if not ready:
            missing = [baseline for baseline, state in by_dataset_baseline[dataset].items() if not state]
            blockers.append(f"{dataset} condensation-score proxy baselines missing official metrics: {','.join(missing)}")

    proxy_ready_by_baseline = {
        baseline: all(_baseline_ready(condensation, dataset, baseline) for dataset in datasets) for baseline in CONDENSATION_BASELINES
    }
    for baseline, ready in proxy_ready_by_baseline.items():
        if not ready:
            blockers.append(f"{baseline} TP/selector proxy is not ready on every requested dataset.")

    standard_handled = bool(standard) and all(not bool_value(row.get("eligible_for_official_main_table")) for row in standard)
    if not standard_handled:
        blockers.append("FreeHGC standard protocol rows are missing or still eligible for the official main table.")

    has_condensation_category = any(str(row.get("row_category", "")) == "Best-condensation-score-TP-baseline" and str(row.get("method", "")) for row in compact)
    separated = _compact_categories_separated(compact)
    if not has_condensation_category:
        blockers.append("Compact table has no populated Best-condensation-score-TP-baseline row.")
    if not separated:
        blockers.append("Compact table external TP and condensation-score categories are not cleanly separated.")

    compact_matrix = {
        dataset: {
            category: bool(_compact_ready(_find_compact(compact, dataset, category)))
            for category in GATE21_22_CATEGORIES
        }
        for dataset in datasets
    }
    stage_ready = all(all(category_ready.values()) for category_ready in compact_matrix.values())
    if not stage_ready:
        for dataset, category_ready in compact_matrix.items():
            missing = [category for category, ready in category_ready.items() if not ready]
            if missing:
                blockers.append(f"{dataset} compact table missing ready rows: {','.join(missing)}")

    external_ready = bool(repos) and all(proxy_ready_by_baseline.values()) and all(condensation_ready_by_dataset.values())
    paper_ready = all(full_ready.values()) and all(export_ready.values()) and no_full_fallback and budget_semantics and hesf_pool_pass and hesf_no_test and standard_handled and has_condensation_category and separated and stage_ready and external_ready

    decision: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_ready.values()),
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_ready.values()),
        "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE": no_full_fallback,
        "BUDGET_METRIC_SEMANTICS_PASS": budget_semantics,
        "HESF_REP_CANDIDATE_POOL_PASS": hesf_pool_pass,
        "HESF_REP_NO_TEST_LEAKAGE": hesf_no_test,
        "DBLP_CONDENSATION_SCORE_BASELINES_READY": condensation_ready_by_dataset.get("DBLP", False),
        "ACM_CONDENSATION_SCORE_BASELINES_READY": condensation_ready_by_dataset.get("ACM", False),
        "IMDB_CONDENSATION_SCORE_BASELINES_READY": condensation_ready_by_dataset.get("IMDB", False),
        "GCONDENSER_PROXY_READY": proxy_ready_by_baseline.get("GCondenser", False),
        "FREEHGC_STANDARD_PROTOCOL_HANDLED": standard_handled,
        "FREEHGC_TP_PROXY_READY": proxy_ready_by_baseline.get("FreeHGC", False),
        "HGCOND_TP_PROXY_READY": proxy_ready_by_baseline.get("HGCond", False),
        "GCOND_TP_PROXY_READY": proxy_ready_by_baseline.get("GCond", False),
        "GCONDENSER_TP_PROXY_READY": proxy_ready_by_baseline.get("GCondenser", False),
        "COMPACT_TABLE_HAS_BEST_CONDENSATION_SCORE_CATEGORY": has_condensation_category,
        "COMPACT_TABLE_EXTERNAL_CATEGORIES_SEPARATED": separated,
        "STAGE_COMPACT_TABLE_READY": stage_ready,
        "PAPER_FINAL_EXTERNAL_BASELINES_READY": external_ready,
        "PAPER_FINAL_TABLE_READY": paper_ready,
        "READY_BY_DATASET_AND_BASELINE": by_dataset_baseline,
        "COMPACT_MATRIX": compact_matrix,
        "FULL_NATIVE_DETAIL": full_ready,
        "EXPORT_FULL_DETAIL": export_ready,
        "PAPER_FINAL_TABLE_BLOCKERS": blockers,
    }
    return decision


def decision_flag_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"flag": flag, "value": bool_value(decision.get(flag)), "detail": decision.get(flag)} for flag in GATE21_22_DECISION_FLAGS]


def _baseline_ready(rows: Sequence[Mapping[str, Any]], dataset: str, baseline: str) -> bool:
    candidates = [
        row
        for row in rows
        if normalize_dataset(row.get("dataset")) == dataset
        and str(row.get("method", "")).startswith(f"{baseline}-score-")
    ]
    return bool(candidates) and all(_ready(row) and bool_value(row.get("eligible_for_external_baseline_table", True)) for row in candidates)


def _ready(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row
        and bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("target_preserving", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
        and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    )


def _compact_ready(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    if str(row.get("method", "")) == "":
        return False
    if str(row.get("row_category", "")) in {"Best-Compressed-Validated diagnostic", "TestOracle-Best-Diagnostic"}:
        return finite_metric(row.get("test_micro_f1_mean")) and finite_metric(row.get("test_macro_f1_mean"))
    return bool_value(row.get("eligible_for_main_decision")) and finite_metric(row.get("test_micro_f1_mean")) and finite_metric(row.get("test_macro_f1_mean"))


def _compact_categories_separated(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        category = str(row.get("row_category", ""))
        method_family = str(row.get("method_family", ""))
        method = str(row.get("method", ""))
        if category == "Best-external-TP-baseline" and (method_family in {"condensation_score_tp_proxy", "condensation_score_as_selector"} or "score-" in method):
            return False
        if category == "Best-condensation-score-TP-baseline" and method_family not in {"condensation_score_tp_proxy", "condensation_score_as_selector"}:
            return False
    return True


def _budget_semantics_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = (
        "requested_budget_type",
        "requested_budget",
        "semantic_structural_storage_ratio",
        "actual_support_edge_ratio",
        "actual_support_node_ratio",
        "raw_hgb_text_byte_ratio",
        "static_inference_package_ratio",
        "reconstructable_package_ratio",
    )
    for row in rows:
        if not _ready(row) or str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
            continue
        for field in required:
            value = row.get(field, "")
            if value in {"", None}:
                return False
            if field != "requested_budget_type" and float_value(value) is None:
                return False
    return True


def _compression_eligible(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")) not in {"Full-native-SeHGNN", "Export-full-SeHGNN"} and bool_value(row.get("eligible_for_main_table", row.get("eligible_for_official_main_table", True)))


def _find_method(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method), None)


def _find_compact(rows: Sequence[Mapping[str, Any]], dataset: str, category: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if normalize_dataset(row.get("dataset")) == dataset and str(row.get("row_category", "")) == category), None)


def _add_missing(blockers: list[str], label: str, readiness: Mapping[str, bool]) -> None:
    missing = [dataset for dataset, ready in readiness.items() if not ready]
    if missing:
        blockers.append(f"Missing ready {label} rows: {','.join(missing)}")


def _banned_hesf_method(method: str) -> bool:
    return any(token in method for token in ("FreeHGC-score", "HGCond-score", "GCond-score", "GCondenser-score", "Herding", "KCenter", "GraphSparsify", "Random", "Degree", "ValidationGreedy"))


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
