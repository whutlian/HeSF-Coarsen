from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_21_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE",
    "BUDGET_METRIC_SEMANTICS_PASS",
    "DBLP_FINAL_COMPARISON_READY",
    "ACM_FINAL_COMPARISON_READY",
    "IMDB_FINAL_COMPARISON_READY",
    "HESF_REP_CANDIDATE_POOL_PASS",
    "HESF_REP_VALIDATION_SELECTION_PASS",
    "HESF_REP_NO_TEST_LEAKAGE",
    "EXTERNAL_BASELINES_INCLUDED_IN_COMPACT_TABLE",
    "EXTERNAL_REPOS_CLONED_OR_LOCAL_PROXY_IMPLEMENTED",
    "FREEHGC_HANDLED_BY_STANDARD_OR_LOCAL_PROXY",
    "PARETO_FLAGS_RECOMPUTED",
    "FINAL_COMPACT_TABLE_READY",
    "PAPER_FINAL_TABLE_READY",
)


def gate21_21_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]],
    compact_rows: Iterable[Mapping[str, Any]],
    frontier_rows: Iterable[Mapping[str, Any]],
    external_repo_rows: Iterable[Mapping[str, Any]],
    freehgc_standard_rows: Iterable[Mapping[str, Any]],
    freehgc_tp_rows: Iterable[Mapping[str, Any]],
    freehgc_selector_rows: Iterable[Mapping[str, Any]],
    acm_overlap_rows: Iterable[Mapping[str, Any]],
    imdb_planner_rows: Iterable[Mapping[str, Any]],
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> dict[str, Any]:
    rows = [dict(row) for row in main_rows]
    reps = [dict(row) for row in rep_rows]
    compact = [dict(row) for row in compact_rows]
    frontiers = [dict(row) for row in frontier_rows]
    repos = [dict(row) for row in external_repo_rows]
    freehgc_standard = [dict(row) for row in freehgc_standard_rows]
    freehgc_tp = [dict(row) for row in freehgc_tp_rows]
    freehgc_selector = [dict(row) for row in freehgc_selector_rows]
    overlap = [dict(row) for row in acm_overlap_rows]
    imdb_planner = [dict(row) for row in imdb_planner_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]

    full_detail = {dataset: _method_ready(rows, dataset, "Full-native-SeHGNN", allow_full=True) for dataset in dataset_names}
    export_detail = {dataset: _method_ready(rows, dataset, "Export-full-SeHGNN", allow_full=True) for dataset in dataset_names}
    final_detail = {dataset: _compact_dataset_ready(compact, dataset) for dataset in dataset_names}
    external_detail = {dataset: _compact_category_ready(compact, dataset, "Best-external-TP-baseline") for dataset in dataset_names}
    edge_detail = {dataset: _compact_category_ready(compact, dataset, "Best-edge-or-structural-baseline") for dataset in dataset_names}
    hesf_rep_detail = {dataset: _rep_ready(reps, dataset) for dataset in dataset_names}
    paper_detail = {
        dataset: bool(final_detail.get(dataset) and external_detail.get(dataset) and edge_detail.get(dataset) and hesf_rep_detail.get(dataset))
        for dataset in dataset_names
    }

    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE": not any(bool_value(row.get("constraint_safe_fallback")) and bool_value(row.get("eligible_for_main_table", True)) for row in rows),
        "BUDGET_METRIC_SEMANTICS_PASS": _budget_metric_semantics_pass(rows),
        "DBLP_FINAL_COMPARISON_READY": final_detail.get("DBLP", True),
        "ACM_FINAL_COMPARISON_READY": final_detail.get("ACM", True),
        "IMDB_FINAL_COMPARISON_READY": final_detail.get("IMDB", True),
        "HESF_REP_CANDIDATE_POOL_PASS": _hesf_candidate_pool_pass(reps),
        "HESF_REP_VALIDATION_SELECTION_PASS": _hesf_validation_selection_pass(reps, dataset_names),
        "HESF_REP_NO_TEST_LEAKAGE": not any(str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated" and bool_value(row.get("uses_test_for_selection")) for row in reps),
        "EXTERNAL_BASELINES_INCLUDED_IN_COMPACT_TABLE": all(external_detail.values()),
        "EXTERNAL_REPOS_CLONED_OR_LOCAL_PROXY_IMPLEMENTED": bool(repos) and all(bool_value(row.get("required_files_present")) or bool_value(row.get("fallback_local_proxy_required")) for row in repos),
        "FREEHGC_HANDLED_BY_STANDARD_OR_LOCAL_PROXY": bool(freehgc_standard) and (any(_row_ready(row) for row in freehgc_tp + freehgc_selector) or any(bool_value(row.get("fallback_local_proxy_required")) for row in repos if str(row.get("method", "")) == "FreeHGC")),
        "PARETO_FLAGS_RECOMPUTED": bool(frontiers) and all("pareto_by_micro_macro_joint" in row for row in frontiers),
        "FINAL_COMPACT_TABLE_READY": all(final_detail.values()),
        "PAPER_FINAL_TABLE_READY": False,
        "FINAL_COMPACT_READY_DETAIL_BY_DATASET": final_detail,
        "PAPER_READY_DETAIL_BY_DATASET": paper_detail,
        "EXTERNAL_BASELINE_DETAIL_BY_DATASET": external_detail,
        "EDGE_BASELINE_DETAIL_BY_DATASET": edge_detail,
        "HESF_REP_DETAIL_BY_DATASET": hesf_rep_detail,
        "ACM_CLOSURE_COMPRESSION_READY": bool(overlap),
        "ACM_HEFS_SELECTOR_DISTINCT_AND_BETTER": _acm_distinct_and_better(overlap),
        "ACM_HEFS_DEGENERATES_TO_DEGREE_SELECTOR": any(bool_value(row.get("selector_degeneracy_flag")) and "Degree" in str(row.get("method_b", "")) for row in overlap),
        "IMDB_CHANNEL_PLANNER_READY": any(_row_ready(row) for row in imdb_planner),
        "PAPER_FINAL_TABLE_BLOCKERS": [],
    }
    flags["PAPER_FINAL_TABLE_READY"] = bool(
        flags["FINAL_COMPACT_TABLE_READY"]
        and flags["FULL_NATIVE_READY_BY_DATASET"]
        and flags["EXPORT_FULL_FIDELITY_PASS_BY_DATASET"]
        and flags["NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"]
        and flags["HESF_REP_CANDIDATE_POOL_PASS"]
        and flags["HESF_REP_VALIDATION_SELECTION_PASS"]
        and flags["HESF_REP_NO_TEST_LEAKAGE"]
        and flags["EXTERNAL_BASELINES_INCLUDED_IN_COMPACT_TABLE"]
        and flags["PARETO_FLAGS_RECOMPUTED"]
        and all(paper_detail.values())
    )
    if not flags["PAPER_FINAL_TABLE_READY"]:
        flags["PAPER_FINAL_TABLE_BLOCKERS"] = _paper_blockers(flags, compact, reps, dataset_names)
    return {name: flags.get(name, False) for name in GATE21_21_DECISION_FLAGS} | {key: value for key, value in flags.items() if key not in GATE21_21_DECISION_FLAGS}


def decision_flag_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"flag": name, "value": decision.get(name, False)} for name in GATE21_21_DECISION_FLAGS]


def _method_ready(rows: Sequence[Mapping[str, Any]], dataset: str, method: str, *, allow_full: bool = False) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method and _row_ready(row, allow_full=allow_full) for row in rows)


def _row_ready(row: Mapping[str, Any], *, allow_full: bool = False) -> bool:
    if bool_value(row.get("constraint_safe_fallback")) and not allow_full:
        return False
    return bool(
        bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and not bool_value(row.get("uses_weighted_superedges"))
        and not bool_value(row.get("uses_synthetic_target_nodes"))
        and finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
        and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    )


def _compact_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    dataset_rows = [row for row in rows if normalize_dataset(row.get("dataset")) == dataset]
    categories = {str(row.get("row_category", "")): row for row in dataset_rows}
    required = {
        "Full-native-SeHGNN",
        "Export-full-SeHGNN",
        "HeSF-RCS-Rep-Validated",
        "Best-edge-or-structural-baseline",
        "Best-external-TP-baseline",
        "Best-dataset-specific-baseline",
        "Best-Compressed-Validated diagnostic",
    }
    return bool(required.issubset(categories) and all(str(categories[category].get("method", "")) for category in required))


def _compact_category_ready(rows: Sequence[Mapping[str, Any]], dataset: str, category: str) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("row_category", "")) == category and str(row.get("method", "")) and finite_metric(row.get("test_micro_f1_mean")) for row in rows)


def _rep_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    return any(
        normalize_dataset(row.get("dataset")) == dataset
        and str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated"
        and str(row.get("selected_method", ""))
        and not bool_value(row.get("uses_test_for_selection"))
        and str(row.get("selection_status", "")) == "selected_by_real_validation_metric"
        for row in rows
    )


def _budget_metric_semantics_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    checked = 0
    required = (
        "requested_budget_type",
        "requested_budget",
        "actual_support_node_ratio",
        "actual_support_edge_ratio",
        "semantic_structural_storage_ratio",
        "raw_hgb_text_byte_ratio",
        "static_inference_package_ratio",
        "reconstructable_package_ratio",
    )
    for row in rows:
        method = str(row.get("method", ""))
        if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
            continue
        if not bool_value(row.get("eligible_for_main_table", True)) or bool_value(row.get("constraint_safe_fallback")):
            continue
        checked += 1
        for field in required:
            if row.get(field, "") in {"", None}:
                return False
    return checked > 0


def _hesf_candidate_pool_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    bad_tokens = ("FreeHGC", "HGCond", "GCond", "Herding", "KCenter", "Random", "Degree", "ValidationGreedy", "GCond", "MDfull")
    hesf_rows = [row for row in rows if str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated"]
    if not hesf_rows:
        return False
    for row in hesf_rows:
        selected = str(row.get("selected_method", ""))
        if not selected:
            return False
        if any(token in selected for token in bad_tokens):
            return False
        if "HeSF-RCS" not in selected:
            return False
    return True


def _hesf_validation_selection_pass(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> bool:
    for dataset in datasets:
        if not _rep_ready(rows, dataset):
            return False
    return True


def _acm_distinct_and_better(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if "Degree" not in str(row.get("method_b", "")):
            continue
        if (float_value(row.get("micro_gap")) or 0.0) >= 0.002 and (float_value(row.get("selected_keyword_jaccard")) or 1.0) <= 0.90:
            return True
    return False


def _paper_blockers(flags: Mapping[str, Any], compact: Sequence[Mapping[str, Any]], reps: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> list[str]:
    blockers: list[str] = []
    for name in GATE21_21_DECISION_FLAGS:
        if name != "PAPER_FINAL_TABLE_READY" and not bool_value(flags.get(name)):
            blockers.append(name)
    for dataset in datasets:
        if not _compact_category_ready(compact, dataset, "Best-external-TP-baseline"):
            blockers.append(f"{dataset}_missing_external_baseline_in_compact_table")
        if not _rep_ready(reps, dataset):
            blockers.append(f"{dataset}_missing_valid_hesf_rep")
    return blockers


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
