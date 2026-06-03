from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_20_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "BUDGET_METRIC_SEMANTICS_PASS",
    "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE",
    "DBLP_FRONTIER_READY",
    "ACM_FRONTIER_READY",
    "IMDB_FRONTIER_READY",
    "HESF_RCS_REP_VALIDATED_READY",
    "HESF_RCS_REP_CANDIDATE_POOL_PASS",
    "HESF_RCS_REP_NO_TEST_LEAKAGE",
    "FREEHGC_SCORE_AS_SELECTOR_READY",
    "ACM_SELECTOR_OVERLAP_READY",
    "IMDB_HEFS_UPGRADED_PLANNER_READY",
    "STAGE_REPORT_SMOKE_READY",
    "STAGE_REPORT_QUICK_ROBUSTNESS_READY",
    "STAGE_REPORT_FINAL_TABLE_READY",
)


def gate21_20_decision(
    *,
    main_rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]] = (),
    robustness_rows: Iterable[Mapping[str, Any]] = (),
    acm_overlap_rows: Iterable[Mapping[str, Any]] = (),
    imdb_upgrade_rows: Iterable[Mapping[str, Any]] = (),
    freehgc_selector_rows: Iterable[Mapping[str, Any]] = (),
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> dict[str, Any]:
    rows = [dict(row) for row in main_rows]
    reps = [dict(row) for row in rep_rows]
    robust = [dict(row) for row in robustness_rows]
    overlap = [dict(row) for row in acm_overlap_rows]
    imdb = [dict(row) for row in imdb_upgrade_rows]
    freehgc = [dict(row) for row in freehgc_selector_rows]
    dataset_names = [normalize_dataset(item) for item in datasets]

    full_detail = {dataset: _method_ready(rows, dataset, "Full-native-SeHGNN", allow_full=True) for dataset in dataset_names}
    export_detail = {dataset: _method_ready(rows, dataset, "Export-full-SeHGNN", allow_full=True) for dataset in dataset_names}
    fallback_free = _no_full_fallback_in_main(rows)
    budget_semantics = _budget_semantics_pass(rows)
    dblp_ready = _dblp_frontier_ready(rows) if "DBLP" in dataset_names else True
    acm_ready = _acm_frontier_ready(rows) if "ACM" in dataset_names else True
    imdb_ready = _imdb_frontier_ready(rows) if "IMDB" in dataset_names else True
    rep_ready = _rep_ready(reps, dataset_names)
    rep_pool = _rep_candidate_pool_pass(reps)
    rep_no_leak = _rep_no_test_leakage(reps)
    freehgc_ready = _freehgc_ready(rows, freehgc)
    acm_overlap_ready = bool(overlap) and all(row.get("field_ratio", "") not in {"", None} for row in overlap)
    imdb_ready_upgrade = _imdb_upgrade_ready(rows, imdb)
    quick_ready = _quick_robustness_ready(robust)
    smoke_ready = bool(
        all(full_detail.values())
        and all(export_detail.values())
        and budget_semantics
        and fallback_free
        and rep_ready
        and rep_pool
        and rep_no_leak
        and freehgc_ready
        and acm_overlap_ready
        and imdb_ready_upgrade
    )
    final_ready = bool(smoke_ready and quick_ready and dblp_ready and acm_ready and imdb_ready)

    flags: dict[str, Any] = {
        "FULL_NATIVE_READY_BY_DATASET": all(full_detail.values()),
        "FULL_NATIVE_READY_DETAIL_BY_DATASET": full_detail,
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET": all(export_detail.values()),
        "EXPORT_FULL_FIDELITY_DETAIL_BY_DATASET": export_detail,
        "BUDGET_METRIC_SEMANTICS_PASS": budget_semantics,
        "NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE": fallback_free,
        "DBLP_FRONTIER_READY": dblp_ready,
        "ACM_FRONTIER_READY": acm_ready,
        "IMDB_FRONTIER_READY": imdb_ready,
        "HESF_RCS_REP_VALIDATED_READY": rep_ready,
        "HESF_RCS_REP_CANDIDATE_POOL_PASS": rep_pool,
        "HESF_RCS_REP_NO_TEST_LEAKAGE": rep_no_leak,
        "FREEHGC_SCORE_AS_SELECTOR_READY": freehgc_ready,
        "ACM_SELECTOR_OVERLAP_READY": acm_overlap_ready,
        "IMDB_HEFS_UPGRADED_PLANNER_READY": imdb_ready_upgrade,
        "STAGE_REPORT_SMOKE_READY": smoke_ready,
        "STAGE_REPORT_QUICK_ROBUSTNESS_READY": quick_ready,
        "STAGE_REPORT_FINAL_TABLE_READY": final_ready,
        "ROBUSTNESS_READY_DETAIL": _robustness_detail(robust),
        "HESF_RCS_REP_DETAIL": [row for row in reps if str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated"],
    }
    return {name: flags.get(name, False) for name in GATE21_20_DECISION_FLAGS} | {
        key: value for key, value in flags.items() if key not in GATE21_20_DECISION_FLAGS
    }


def _method_ready(rows: Sequence[Mapping[str, Any]], dataset: str, method: str, *, allow_full: bool = False) -> bool:
    return any(normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method and _row_ready(row, allow_full=allow_full) for row in rows)


def _row_ready(row: Mapping[str, Any], *, allow_full: bool = False) -> bool:
    if bool_value(row.get("constraint_safe_fallback")) and not allow_full:
        return False
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("target_preserving", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and finite_metric(row.get("test_micro_f1_mean", row.get("test_micro_f1")))
        and finite_metric(row.get("test_macro_f1_mean", row.get("test_macro_f1")))
    )


def _no_full_fallback_in_main(rows: Sequence[Mapping[str, Any]]) -> bool:
    return not any(
        bool_value(row.get("constraint_safe_fallback"))
        and (bool_value(row.get("eligible_for_main_table", True)) or bool_value(row.get("eligible_for_compression_claim", False)))
        for row in rows
    )


def _budget_semantics_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    checked = 0
    for row in rows:
        method = str(row.get("method", ""))
        if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"} or "Rep" in method:
            continue
        if not bool_value(row.get("eligible_for_main_table", True)) or bool_value(row.get("constraint_safe_fallback")):
            continue
        checked += 1
        budget_type = str(row.get("requested_budget_type", "")).strip()
        if not budget_type or row.get("requested_budget", "") in {"", None}:
            return False
        if budget_type == "support_node_ratio" and float_value(row.get("actual_support_edge_ratio", row.get("support_edge_ratio"))) is None:
            return False
        if not any(
            float_value(row.get(field)) is not None
            for field in (
                "semantic_structural_storage_ratio",
                "actual_semantic_structural_ratio",
                "actual_support_edge_ratio",
                "support_edge_ratio",
                "raw_hgb_text_byte_ratio",
                "keyword_feature_ratio",
                "channel_edge_ratio",
            )
        ):
            return False
    return checked > 0


def _dblp_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in rows if normalize_dataset(row.get("dataset")) == "DBLP" and _row_ready(row)]
    methods = {str(row.get("method", "")) for row in ready}
    return bool(any(method.startswith("HeSF-RCS-auto") for method in methods) and any("FreeHGC-score-as-selector" in method or "Random-edge-relwise" in method or "Proportional" in method for method in methods))


def _acm_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = {str(row.get("method", "")) for row in rows if normalize_dataset(row.get("dataset")) == "ACM" and _row_ready(row)}
    return bool("ACM-HeSF-RCS-auto-field20" in ready and ("ACM-Degree-field20" in ready or "ACM-ValidationGreedy-field20" in ready) and "ACM-Random-field20" in ready)


def _imdb_frontier_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = {str(row.get("method", "")) for row in rows if normalize_dataset(row.get("dataset")) == "IMDB" and _row_ready(row)}
    return bool("IMDB-HeSF-RCS-channel50" in ready and ("IMDB-ValidationGreedy-channel50" in ready or "IMDB-MDfull-MA50-MK50" in ready))


def _rep_ready(rep_rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> bool:
    for dataset in datasets:
        matches = [row for row in rep_rows if normalize_dataset(row.get("dataset")) == dataset and str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated"]
        if not matches:
            continue
        row = matches[0]
        if not str(row.get("selected_method", "")) or not bool_value(row.get("eligible_for_main_decision")):
            return False
    return any(str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated" and str(row.get("selected_method", "")) for row in rep_rows)


def _rep_candidate_pool_pass(rep_rows: Sequence[Mapping[str, Any]]) -> bool:
    hesf_rows = [row for row in rep_rows if str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated" and str(row.get("selected_method", ""))]
    if not hesf_rows:
        return False
    for row in hesf_rows:
        family = str(row.get("selected_method_family", "")).lower()
        method = str(row.get("selected_method", ""))
        if "external" in family:
            return False
        if not ("HeSF-RCS" in method or family in {"schema_preserving_rcs", "hesf_rcs"}):
            return False
    return True


def _rep_no_test_leakage(rep_rows: Sequence[Mapping[str, Any]]) -> bool:
    return not any(str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated" and bool_value(row.get("uses_test_for_selection")) for row in rep_rows)


def _freehgc_ready(main_rows: Sequence[Mapping[str, Any]], selector_rows: Sequence[Mapping[str, Any]]) -> bool:
    methods = {str(row.get("method", "")) for row in main_rows + selector_rows}
    return {"FreeHGC-score-as-selector structural16", "FreeHGC-score-as-selector structural20"}.issubset(methods)


def _imdb_upgrade_ready(main_rows: Sequence[Mapping[str, Any]], upgrade_rows: Sequence[Mapping[str, Any]]) -> bool:
    rows = main_rows + upgrade_rows
    return any(
        normalize_dataset(row.get("dataset")) == "IMDB"
        and str(row.get("method", "")) == "IMDB-HeSF-RCS-channel50"
        and bool_value(row.get("constraint_pass", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and bool_value(row.get("success", True))
        for row in rows
    )


def _quick_robustness_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        int(float_value(row.get("training_executed_count")) or 0) >= 3
        and int(float_value(row.get("training_seed_count")) or 0) >= 3
        and int(float_value(row.get("failure_count")) or 0) == 0
        and bool_value(row.get("robustness_ready", True))
        for row in rows
    )


def _robustness_detail(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": normalize_dataset(row.get("dataset")),
            "method": row.get("method", ""),
            "training_executed_count": row.get("training_executed_count", ""),
            "training_seed_count": row.get("training_seed_count", ""),
            "graph_seed_count": row.get("graph_seed_count", ""),
            "failure_count": row.get("failure_count", ""),
            "robustness_mode": row.get("robustness_mode", ""),
            "robustness_ready": row.get("robustness_ready", ""),
        }
        for row in rows
    ]
