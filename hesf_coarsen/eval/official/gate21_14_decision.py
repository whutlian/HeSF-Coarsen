from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REQUIRED_DECISION_FLAGS = (
    "OFFICIAL_DBLP_APV12_ANCHOR_PASS",
    "OFFICIAL_DBLP_APV16_ANCHOR_PASS",
    "BUDGETED_SELECTOR_HASH_AUDIT_PASS",
    "BUDGETED_SELECTOR_LINKAGE_PASS",
    "BUDGETED_SELECTOR_NO_TEST_LEAKAGE_PASS",
    "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_TP_SELECTION_READY",
    "FREEHGC_SCORE_SELECTOR_READY",
    "FEATURE_ABLATION_TASK_RESULTS_READY",
    "PAPER_FEATURE_REDUNDANCY_TESTED",
    "SUPPORT_FEATURE_REDUNDANCY_TESTED",
    "METAPATH_TENSOR_DUMP_READY",
    "CACHE_HASH_REAL_PASS",
    "COVERAGE_DISTRIBUTIONAL_MECHANISM_READY",
    "APV12_RP64_ADAPTER_RESTORED",
    "APV16_RP64_ADAPTER_READY",
    "SYSTEM_WORKLOAD_COST_READY",
    "CROSS_DATASET_ACM_TASK_RESULTS_READY",
    "CROSS_DATASET_IMDB_TASK_RESULTS_READY",
    "PARETO_FRONTIER_READY",
    "ICDE_EVIDENCE_READY",
)

REQUIRED_EXTERNAL_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
    "FreeHGC-score-TP",
)
REQUIRED_EXTERNAL_BUDGETS = {
    ("structural_storage_ratio", 0.12),
    ("structural_storage_ratio", 0.16),
    ("structural_storage_ratio", 0.20),
    ("structural_storage_ratio", 0.30),
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
}
REQUIRED_FREEHGC_RATIOS = {0.012, 0.024, 0.048, 0.096, 0.120}
REQUIRED_FEATURE_METHODS = {"full/export-full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"}
REQUIRED_FEATURE_TRANSFORMS = {
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-venue-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-only-preserve-original-dims",
    "term-only-preserve-original-dims",
    "venue-only-preserve-original-dims",
    "paper-random-projection64",
    "paper-pca64",
}
REQUIRED_LABEL_GRAPH_SETTINGS = {
    "default",
    "no_label_feats",
    "num_feature_hops_0",
    "num_label_hops_0",
    "feature_only_mlp_adapter",
    "no_label_feats+zero-all-support-preserve-dim",
    "num_feature_hops_0+zero-all-support-preserve-dim",
}
REQUIRED_METAPATH_METHODS = {
    "full/export-full",
    "H6-node30",
    "H6-APV-skeleton",
    "HeSF-RCS-APV12",
    "HeSF-RCS-APV16",
    "APV12-PTTP10",
    "APV12-PV75",
}
REQUIRED_COVERAGE_METHODS = {"H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16", "APV12-PV75", "APV12-PTTP10"}
REQUIRED_CROSS_METHODS = {
    "full-native",
    "export-full",
    "H6-node30",
    "random-edge-relation-wise",
    "HeSF-RCS-auto structural30",
    "HeSF-RCS-auto structural20",
    "best available external TP baseline",
}


def gate21_14_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    budgeted_selector_rows: Sequence[Mapping[str, Any]] = (),
    selector_hash_audit: Sequence[Mapping[str, Any]] = (),
    external_tp_runs: Sequence[Mapping[str, Any]] = (),
    external_tp_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_protocol_audit: Sequence[Mapping[str, Any]] = (),
    freehgc_score_selector_by_method: Sequence[Mapping[str, Any]] = (),
    feature_ablation_runs: Sequence[Mapping[str, Any]] = (),
    feature_ablation_by_method: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_assertions: Sequence[Mapping[str, Any]] = (),
    coverage_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    adapter_audit: Sequence[Mapping[str, Any]] = (),
    system_cost_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
    pareto_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, bool]:
    flags = {name: False for name in REQUIRED_DECISION_FLAGS}
    flags["OFFICIAL_DBLP_APV12_ANCHOR_PASS"] = _official_anchor_ready(official_rows, "APV12")
    flags["OFFICIAL_DBLP_APV16_ANCHOR_PASS"] = _official_anchor_ready(official_rows, "APV16")
    flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] = _selector_hash_ready(selector_hash_audit)
    flags["BUDGETED_SELECTOR_LINKAGE_PASS"] = _selector_linkage_ready(budgeted_selector_rows)
    flags["BUDGETED_SELECTOR_NO_TEST_LEAKAGE_PASS"] = bool(budgeted_selector_rows) and all(not _bool(row.get("uses_test_metrics_for_selection")) for row in budgeted_selector_rows)
    flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] = _external_ready(external_tp_runs, external_tp_by_method)
    flags["FREEHGC_STANDARD_5SEED_READY"] = _freehgc_standard_ready(freehgc_standard_by_method)
    flags["FREEHGC_TP_SELECTION_READY"] = _freehgc_tp_ready(freehgc_tp_by_method, "selection")
    flags["FREEHGC_SCORE_SELECTOR_READY"] = _freehgc_score_ready(freehgc_score_selector_by_method)
    flags["FEATURE_ABLATION_TASK_RESULTS_READY"] = _feature_ablation_ready(feature_ablation_runs, feature_ablation_by_method)
    flags["PAPER_FEATURE_REDUNDANCY_TESTED"] = _feature_redundancy_tested(feature_ablation_runs, "zero-paper-preserve-dim")
    flags["SUPPORT_FEATURE_REDUNDANCY_TESTED"] = _feature_redundancy_tested(feature_ablation_runs, "zero-all-support-preserve-dim")
    flags["METAPATH_TENSOR_DUMP_READY"] = _metapath_ready(metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = _cache_ready(cache_assertions)
    flags["COVERAGE_DISTRIBUTIONAL_MECHANISM_READY"] = _coverage_ready(coverage_rows)
    flags["APV12_RP64_ADAPTER_RESTORED"] = _adapter_ready(adapter_rows, "APV12", "random_projection_dim64") and _adapter_semantics_pass(adapter_rows, adapter_audit)
    flags["APV16_RP64_ADAPTER_READY"] = _adapter_ready(adapter_rows, "APV16", "random_projection_dim64") and _adapter_semantics_pass(adapter_rows, adapter_audit)
    flags["SYSTEM_WORKLOAD_COST_READY"] = _system_cost_ready(system_cost_rows)
    flags["CROSS_DATASET_ACM_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "ACM")
    flags["CROSS_DATASET_IMDB_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "IMDB")
    flags["PARETO_FRONTIER_READY"] = _pareto_ready(pareto_rows)
    flags["ICDE_EVIDENCE_READY"] = (
        flags["OFFICIAL_DBLP_APV12_ANCHOR_PASS"]
        and flags["OFFICIAL_DBLP_APV16_ANCHOR_PASS"]
        and flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"]
        and flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"]
        and flags["FEATURE_ABLATION_TASK_RESULTS_READY"]
        and flags["METAPATH_TENSOR_DUMP_READY"]
        and flags["SYSTEM_WORKLOAD_COST_READY"]
        and flags["CROSS_DATASET_ACM_TASK_RESULTS_READY"]
        and flags["CROSS_DATASET_IMDB_TASK_RESULTS_READY"]
    )
    return flags


def decision_status(flags: Mapping[str, Any]) -> str:
    if _bool(flags.get("ICDE_EVIDENCE_READY")):
        return "ICDE_EVIDENCE_READY"
    if (
        _bool(flags.get("OFFICIAL_DBLP_APV12_ANCHOR_PASS"))
        and _bool(flags.get("OFFICIAL_DBLP_APV16_ANCHOR_PASS"))
        and _bool(flags.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS"))
        and (_bool(flags.get("CROSS_DATASET_ACM_TASK_RESULTS_READY")) or _bool(flags.get("CROSS_DATASET_IMDB_TASK_RESULTS_READY")))
    ):
        return "GATE21_14_PARTIAL_REAL_EXECUTION_EVIDENCE"
    if _bool(flags.get("OFFICIAL_DBLP_APV12_ANCHOR_PASS")) and _bool(flags.get("OFFICIAL_DBLP_APV16_ANCHOR_PASS")):
        return "GATE21_14_ANCHORS_PRESERVED_NOT_ICDE_READY"
    return "NOT_READY"


def _official_anchor_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(
        str(row.get("dataset", "")).upper() == "DBLP"
        and token in _method(row)
        and str(row.get("row_kind", "direct_task_result")) != "planner_plan"
        and _bool(row.get("schema_compatible", True))
        and _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and not _bool(row.get("uses_weighted_superedges"))
        and not _bool(row.get("uses_synthetic_nodes"))
        and not _bool(row.get("uses_adapter_loader", row.get("uses_feature_adapter", False)))
        and _bool(row.get("eligible_for_official_main_table"))
        and not _bool(row.get("uses_test_metrics_for_selection"))
        and _task_ready(row)
        for row in rows
    )


def _selector_hash_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "APV12_selected_edge_hash != APV16_selected_edge_hash",
        "budget12_selected_edge_hash == official_main_APV12_selected_edge_hash",
        "budget16_selected_edge_hash == official_main_APV16_selected_edge_hash",
        "same_input_different_graph_seed_same_selected_edge_hash_for_deterministic_plans",
        "same_selected_edge_hash_same_export_hash",
        "planner_rows_not_marked_official_main_eligible",
        "linked_task_rows_marked_official_main_eligible_if_unmodified",
    }
    present = {str(row.get("assertion_name", "")) for row in rows if _bool(row.get("pass"))}
    return required.issubset(present)


def _selector_linkage_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    required_budgets = {0.12, 0.16, 0.20, 0.30, 0.50}
    ready_budgets = set()
    for row in rows:
        budget = _round(row.get("requested_structural_budget"))
        if budget not in required_budgets:
            continue
        if not _bool(row.get("planner_row")):
            continue
        if _bool(row.get("eligible_for_official_main_table")):
            return False
        if not _bool(row.get("eligible_for_planner_decision")):
            return False
        if not str(row.get("linked_official_task_method", "")).strip():
            return False
        if not str(row.get("linked_task_result_hash", "")).strip():
            return False
        ready_budgets.add(budget)
    return ready_budgets == required_budgets


def _external_ready(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(
            any(
                _method(row) == method
                and _budget_key(row) == budget_type
                and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001
                and (_float(row.get("success_count")) or 0.0) >= 25
                and _bool(row.get("ready_5x5"))
                and _bool(row.get("all_required_metrics_present"))
                and (_float(row.get("budget_match_rate")) or 0.0) >= 1.0
                for row in by_method
            )
            for method in REQUIRED_EXTERNAL_METHODS
            for budget_type, budget in REQUIRED_EXTERNAL_BUDGETS
        )
    return all(_external_cell_ready(runs, method, budget_type, budget) for method in REQUIRED_EXTERNAL_METHODS for budget_type, budget in REQUIRED_EXTERNAL_BUDGETS)


def _external_cell_ready(rows: Sequence[Mapping[str, Any]], method: str, budget_type: str, budget: float) -> bool:
    ready = [
        row
        for row in rows
        if _method(row) == method
        and _budget_key(row) == budget_type
        and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001
        and _bool(row.get("eligible_for_external_tp_table"))
        and _bool(row.get("no_test_leakage", row.get("no_test_leakage")))
        and _bool(row.get("budget_matched_within_tolerance"))
        and _task_ready(row)
    ]
    return len({(str(row.get("graph_seed")), str(row.get("training_seed"))) for row in ready}) >= 25


def _freehgc_standard_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        any(
            abs((_float(row.get("ratio", row.get("reduction_rate"))) or -1.0) - ratio) <= 0.0005
            and (_float(row.get("success_count")) or 0.0) >= 5
            and _bool(row.get("ready_5seed"))
            and _finite(row.get("mean_micro"))
            for row in rows
        )
        for ratio in REQUIRED_FREEHGC_RATIOS
    )


def _freehgc_tp_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(token in _method(row).lower() and _bool(row.get("official_hgb_exported")) and _bool(row.get("training_executed")) and _bool(row.get("ready")) for row in rows)


def _freehgc_score_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("freehgc" in _method(row).lower() and (_float(row.get("success_count")) or 0.0) >= 25 and _bool(row.get("ready_5x5")) for row in rows)


def _feature_ablation_ready(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(
            any(
                str(row.get("base_method")) == method
                and str(row.get("feature_transform")) == transform
                and str(row.get("label_graph_setting")) == setting
                and (_float(row.get("success_count")) or 0.0) >= 5
                and _bool(row.get("all_required_metrics_present"))
                for row in by_method
            )
            for method in REQUIRED_FEATURE_METHODS
            for transform in REQUIRED_FEATURE_TRANSFORMS
            for setting in REQUIRED_LABEL_GRAPH_SETTINGS
        )
    return False


def _feature_redundancy_tested(rows: Sequence[Mapping[str, Any]], transform: str) -> bool:
    methods = {"HeSF-RCS-APV12", "HeSF-RCS-APV16", "H6-APV-skeleton"}
    return all(
        any(
            str(row.get("base_method")) == method
            and str(row.get("feature_transform")) == transform
            and str(row.get("label_graph_setting")) == "default"
            and _task_ready(row)
            and _bool(row.get("shape_safe_pass"))
            for row in rows
        )
        for method in methods
    )


def _metapath_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        any(
            str(row.get("method")) == method
            and _bool(row.get("real_tensor_dumped"))
            and _bool(row.get("tensor_key_dumped"))
            and _real_hash(row.get("feature_tensor_hash"))
            and _positive(row.get("feature_tensor_bytes"))
            for row in rows
        )
        for method in REQUIRED_METAPATH_METHODS
    )


def _cache_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "full_vs_APV12_hash_diff",
        "APV12_vs_APV16_hash_diff",
        "APV12_vs_APV12_PTTP10_hash_diff",
        "APV12_vs_APV12_PV75_hash_diff",
        "cache_hash_not_empty_sha256",
        "feature_tensor_hash_not_nan",
        "label_tensor_hash_not_nan_if_labels_enabled",
    }
    present = {str(row.get("assertion_name", "")) for row in rows if _bool(row.get("pass"))}
    return required.issubset(present)


def _coverage_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    distribution_fields = (
        "AP_PV_path_multiplicity_mean",
        "AP_PV_path_multiplicity_p50",
        "AP_PV_path_multiplicity_p90",
        "paper_venue_entropy_mean",
        "venue_class_proxy_purity_trainval",
        "paper_class_proxy_purity_trainval",
    )
    return all(
        any(
            str(row.get("method")) == method
            and not _bool(row.get("uses_test_labels_for_proxy"))
            and all(_finite(row.get(field)) for field in distribution_fields)
            for row in rows
        )
        for method in REQUIRED_COVERAGE_METHODS
    )


def _adapter_ready(rows: Sequence[Mapping[str, Any]], base_token: str, adapter: str) -> bool:
    return any(
        base_token in str(row.get("base_method", row.get("method", "")))
        and str(row.get("adapter_method", "")) == adapter
        and _bool(row.get("eligible_for_adapter_table"))
        and not _bool(row.get("eligible_for_official_main_table"))
        and _finite(row.get("static_inference_package_ratio"))
        and _task_ready(row)
        for row in rows
    )


def _adapter_semantics_pass(rows: Sequence[Mapping[str, Any]], audit: Sequence[Mapping[str, Any]]) -> bool:
    if audit and not all(_bool(row.get("package_semantics_pass", True)) for row in audit):
        return False
    return all(
        not _bool(row.get("eligible_for_official_main_table"))
        and _finite(row.get("static_inference_package_ratio"))
        and str(row.get("static_inference_package_ratio")) != str(row.get("transform_recipe_package_ratio", ""))
        for row in rows
        if _bool(row.get("eligible_for_adapter_table"))
    )


def _system_cost_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    tokens = ("raw_hgb_text", "HeSF-RCS-APV12 official text", "HeSF-RCS-APV16 official text", "HeSF-RCS-APV12 + RP64 adapter", "external TP")
    return all(
        any(
            token.lower() in _method(row).lower()
            and _task_ready(row)
            and _positive(row.get("official_sehgnn_preprocess_time_seconds"))
            and _positive(row.get("training_time_seconds"))
            and _positive(row.get("peak_cpu_rss_mb"))
            and _positive(row.get("preprocessed_cache_bytes"))
            for row in rows
        )
        for token in tokens
    )


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    return all(
        any(
            str(row.get("dataset", "")).upper() == dataset
            and required.lower() in _method(row).lower()
            and _bool(row.get("official_hgb_exported"))
            and _bool(row.get("official_sehgnn_unmodified"))
            and not _bool(row.get("uses_test_metrics_for_selection"))
            and _finite(row.get("recovery_vs_native_full_micro"))
            and _finite(row.get("recovery_vs_native_full_macro"))
            and _task_ready(row)
            for row in rows
        )
        for required in REQUIRED_CROSS_METHODS
    )


def _pareto_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    dblp = [row for row in rows if str(row.get("dataset", "")).upper() == "DBLP" and _bool(row.get("eligible_for_main_table")) and _finite(row.get("validation_micro")) and _finite(row.get("test_micro"))]
    methods = {str(row.get("selected_plan_name", "")) for row in dblp}
    modes = {str(row.get("selector_mode", "")) for row in dblp}
    return "HeSF-RCS-APV12" in methods and "HeSF-RCS-APV16" in methods and bool(modes - {"bottleneck_first", "feedback_aware"}) and any(_bool(row.get("is_pareto_optimal")) for row in dblp)


def _task_ready(row: Mapping[str, Any]) -> bool:
    return _bool(row.get("training_executed")) and _bool(row.get("success", True)) and _finite(row.get("test_micro_f1")) and _finite(row.get("test_macro_f1")) and not str(row.get("failure_type", "")).strip()


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("base_method", row.get("selected_canonical_method", ""))))


def _budget_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("budget_type", row.get("budget_family", "")))
    return "structural_storage_ratio" if "structural" in key else "support_node_ratio" if "support" in key else key


def _real_hash(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() != "nan" and text != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _float(value: Any) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _round(value: Any) -> float:
    parsed = _float(value)
    return -1.0 if parsed is None else round(parsed, 2)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ready"}
