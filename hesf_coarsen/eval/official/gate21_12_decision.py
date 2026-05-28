from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REQUIRED_DECISION_FLAGS = (
    "OFFICIAL_MAIN_DBLP_APV12_READY",
    "OFFICIAL_MAIN_DBLP_APV16_READY",
    "OFFICIAL_MAIN_BUDGETED_SELECTOR_READY",
    "BUDGETED_SELECTOR_HASH_AUDIT_PASS",
    "APV16_DETERMINISTIC_PROOF_PASS",
    "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
    "EXTERNAL_TP_REQUIRED_METHODS_READY",
    "EXTERNAL_TP_BUDGET_FAIRNESS_PASS",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED",
    "FREEHGC_STANDARD_SPLIT_VERIFIED",
    "FREEHGC_TP_SELECTION_TASK_READY",
    "FREEHGC_TP_SYNTHETIC_TASK_READY",
    "FREEHGC_TP_HARD_FAILURE_PROOF_READY",
    "METAPATH_TENSOR_DUMP_READY",
    "CACHE_HASH_REAL_PASS",
    "APV12_APV16_CACHE_DIFF_PASS",
    "PTTP_CACHE_DIFF_PASS",
    "FEATURE_ABLATION_TASK_METRICS_READY",
    "FEATURE_ABLATION_SHAPE_SAFE_PASS",
    "FEATURE_ABLATION_OFFICIAL_VS_ADAPTER_SEPARATED",
    "ADAPTER_APV16_TASK_RESULTS_READY",
    "ADAPTER_BY_METHOD_RATIO_MERGE_PASS",
    "ADAPTER_FAILED_ROWS_EXCLUDED_FROM_NUMERIC_SUMMARY",
    "PCA_REPRODUCIBLE_PACKAGE_COMPLETE",
    "SYSTEM_COST_END_TO_END_READY",
    "SYSTEM_COST_PREPROCESS_TIME_READY",
    "SYSTEM_COST_TRAINING_TIME_READY",
    "SYSTEM_COST_MEMORY_READY",
    "STORAGE_ONLY_BASELINES_CONTEXTUALIZED",
    "CROSS_DATASET_ACM_TASK_RESULTS_READY",
    "CROSS_DATASET_IMDB_TASK_RESULTS_READY",
    "CROSS_DATASET_AUTO_SELECTOR_NO_TEST_LEAKAGE_PASS",
    "COVERAGE_REACHABILITY_TABLE_EMITTED",
    "COVERAGE_REACHABILITY_SANITY_PASS",
    "COVERAGE_DISTRIBUTIONAL_TABLE_EMITTED",
    "COVERAGE_DISTRIBUTIONAL_MECHANISM_READY",
    "ICDE_SUBMISSION_EVIDENCE_READY",
)

REQUIRED_EXTERNAL_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
)
REQUIRED_EXTERNAL_BUDGETS = {
    ("structural_ratio", 0.12),
    ("structural_ratio", 0.16),
    ("structural_ratio", 0.20),
    ("structural_ratio", 0.30),
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
}
REQUIRED_FREEHGC_RATIOS = {0.012, 0.024, 0.048, 0.096, 0.120}


def gate21_12_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    budgeted_selector_rows: Sequence[Mapping[str, Any]] = (),
    selector_hash_audit: Sequence[Mapping[str, Any]] = (),
    apv16_deterministic_proof: Mapping[str, Any] | None = None,
    external_tp_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_runs: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_env: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_rows: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    adapter_by_method: Sequence[Mapping[str, Any]] = (),
    system_cost_rows: Sequence[Mapping[str, Any]] = (),
    storage_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_selector_plans: Sequence[Mapping[str, Any]] = (),
    coverage_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    flags = {name: False for name in REQUIRED_DECISION_FLAGS}
    flags["OFFICIAL_MAIN_DBLP_APV12_READY"] = _official_method_ready(official_rows, "APV12")
    flags["OFFICIAL_MAIN_DBLP_APV16_READY"] = _official_method_ready(official_rows, "APV16")
    flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] = any(_bool(row.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS")) for row in selector_hash_audit) or any(_bool(row.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS")) for row in budgeted_selector_rows)
    flags["APV16_DETERMINISTIC_PROOF_PASS"] = _bool((apv16_deterministic_proof or {}).get("deterministic_proof_pass"))
    flags["OFFICIAL_MAIN_BUDGETED_SELECTOR_READY"] = (
        flags["OFFICIAL_MAIN_DBLP_APV12_READY"]
        and flags["OFFICIAL_MAIN_DBLP_APV16_READY"]
        and flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"]
        and _selector_plan_ready(budgeted_selector_rows, 0.12, "APV12")
        and _selector_plan_ready(budgeted_selector_rows, 0.16, "APV16")
        and _no_selection_leakage(budgeted_selector_rows)
    )

    flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] = _external_5x5_ready(external_tp_rows, external_tp_by_method)
    flags["EXTERNAL_TP_REQUIRED_METHODS_READY"] = all(_external_method_ready(external_tp_rows, external_tp_by_method, method) for method in REQUIRED_EXTERNAL_METHODS)
    flags["EXTERNAL_TP_BUDGET_FAIRNESS_PASS"] = _external_budget_fair(external_tp_rows, external_tp_by_method)

    flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"] = any(_bool(row.get("upstream_config_verified")) and _bool(row.get("required_files_present")) for row in freehgc_env)
    flags["FREEHGC_STANDARD_SPLIT_VERIFIED"] = any(_bool(row.get("split_matches_hgb_official", row.get("split_matches_official_or_documented"))) for row in freehgc_env)
    flags["FREEHGC_STANDARD_5SEED_READY"] = flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"] and flags["FREEHGC_STANDARD_SPLIT_VERIFIED"] and _freehgc_standard_ready(freehgc_standard_runs, freehgc_standard_by_method)
    flags["FREEHGC_TP_SELECTION_TASK_READY"] = _freehgc_tp_ready(freehgc_tp_rows, "selection")
    flags["FREEHGC_TP_SYNTHETIC_TASK_READY"] = _freehgc_tp_ready(freehgc_tp_rows, "synthetic")
    flags["FREEHGC_TP_HARD_FAILURE_PROOF_READY"] = any(_freehgc_hard_proof(row) for row in freehgc_tp_rows) or any(str(row.get("hard_failure_reason", "")).strip() for row in freehgc_env)

    flags["METAPATH_TENSOR_DUMP_READY"] = bool(metapath_rows) and all(_real_hash(row.get("feature_tensor_hash")) and _positive(row.get("feature_tensor_bytes")) for row in metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_rows) and all(_real_hash(row.get("cache_file_hash", row.get("cache_hash"))) and _bool(row.get("assertion_pass", True)) for row in cache_rows)
    flags["APV12_APV16_CACHE_DIFF_PASS"] = any(_bool(row.get("APV12_APV16_CACHE_DIFF_PASS", row.get("apv12_apv16_cache_diff_pass"))) for row in cache_rows)
    flags["PTTP_CACHE_DIFF_PASS"] = any(_bool(row.get("PTTP_CACHE_DIFF_PASS", row.get("pttp_cache_diff_pass"))) for row in cache_rows)

    flags["FEATURE_ABLATION_TASK_METRICS_READY"] = _feature_ablation_ready(feature_ablation_rows)
    flags["FEATURE_ABLATION_SHAPE_SAFE_PASS"] = bool(feature_ablation_rows) and all(_blank_or_true(row.get("feature_shape_safe", row.get("shape_safe_pass"))) for row in feature_ablation_rows)
    flags["FEATURE_ABLATION_OFFICIAL_VS_ADAPTER_SEPARATED"] = bool(feature_ablation_rows) and all(not (_bool(row.get("official_sehgnn_unmodified")) and str(row.get("adapter_family", "")).strip()) for row in feature_ablation_rows)

    flags["ADAPTER_APV16_TASK_RESULTS_READY"] = _adapter_ready(adapter_rows, "APV16", "random_projection_dim64") or _adapter_ready(adapter_rows, "APV16", "int8") or _adapter_ready(adapter_rows, "APV16", "fp16")
    flags["ADAPTER_BY_METHOD_RATIO_MERGE_PASS"] = bool(adapter_rows or adapter_by_method) and all(_adapter_row_ok(row) for row in adapter_rows)
    flags["ADAPTER_FAILED_ROWS_EXCLUDED_FROM_NUMERIC_SUMMARY"] = bool(adapter_rows) and all(_failed_adapter_clean(row) for row in adapter_rows)
    flags["PCA_REPRODUCIBLE_PACKAGE_COMPLETE"] = any("pca" in _adapter_name(row).lower() and _bool(row.get("reproducible_transform_package_complete", row.get("pca_reproducible_package_complete"))) for row in adapter_rows)

    flags["SYSTEM_COST_END_TO_END_READY"] = _system_cost_ready(system_cost_rows)
    flags["SYSTEM_COST_PREPROCESS_TIME_READY"] = _system_field_ready(system_cost_rows, "official_sehgnn_preprocess_time_seconds")
    flags["SYSTEM_COST_TRAINING_TIME_READY"] = _system_field_ready(system_cost_rows, "training_time_seconds")
    flags["SYSTEM_COST_MEMORY_READY"] = _system_field_ready(system_cost_rows, "peak_cpu_rss_mb")
    flags["STORAGE_ONLY_BASELINES_CONTEXTUALIZED"] = any(_bool(row.get("archive_only_compression")) or _bool(row.get("requires_loader_adapter")) for row in [*storage_rows, *system_cost_rows])

    flags["CROSS_DATASET_ACM_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "ACM")
    flags["CROSS_DATASET_IMDB_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "IMDB")
    flags["CROSS_DATASET_AUTO_SELECTOR_NO_TEST_LEAKAGE_PASS"] = bool(cross_dataset_rows or cross_dataset_selector_plans) and _no_selection_leakage([*cross_dataset_rows, *cross_dataset_selector_plans])

    flags["COVERAGE_REACHABILITY_TABLE_EMITTED"] = bool(coverage_rows)
    flags["COVERAGE_REACHABILITY_SANITY_PASS"] = bool(coverage_rows) and all(_blank_or_true(row.get("relation_direction_matches_official_relation_name")) and _blank_or_true(row.get("node_type_offsets_match_node_dat_counts")) for row in coverage_rows)
    flags["COVERAGE_DISTRIBUTIONAL_TABLE_EMITTED"] = bool(coverage_rows)
    flags["COVERAGE_DISTRIBUTIONAL_MECHANISM_READY"] = bool(coverage_rows) and all(
        str(row.get("per_class_venue_coverage", row.get("per_class_venue_coverage_json", ""))).strip()
        and str(row.get("author_degree_bucket_recovery", row.get("author_degree_bucket_coverage_json", ""))).strip()
        and _finite(row.get("venue_class_proxy_purity_trainval"))
        and _finite(row.get("paper_class_proxy_purity_trainval"))
        for row in coverage_rows
    )

    flags["ICDE_SUBMISSION_EVIDENCE_READY"] = all(
        flags[name]
        for name in (
            "OFFICIAL_MAIN_DBLP_APV12_READY",
            "OFFICIAL_MAIN_DBLP_APV16_READY",
            "BUDGETED_SELECTOR_HASH_AUDIT_PASS",
            "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
            "EXTERNAL_TP_BUDGET_FAIRNESS_PASS",
            "METAPATH_TENSOR_DUMP_READY",
            "CACHE_HASH_REAL_PASS",
            "FEATURE_ABLATION_TASK_METRICS_READY",
            "SYSTEM_COST_END_TO_END_READY",
            "CROSS_DATASET_ACM_TASK_RESULTS_READY",
            "CROSS_DATASET_IMDB_TASK_RESULTS_READY",
        )
    )
    return flags


def decision_status(flags: Mapping[str, Any]) -> str:
    if _bool(flags.get("ICDE_SUBMISSION_EVIDENCE_READY")):
        return "ICDE_SUBMISSION_EVIDENCE_READY"
    if _bool(flags.get("OFFICIAL_MAIN_BUDGETED_SELECTOR_READY")):
        return "GATE21_12_PARTIAL_EXECUTED_EVIDENCE"
    return "NOT_READY"


def _official_method_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(token in _method(row) and _official_row_ready(row) for row in rows)


def _official_row_ready(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("row_kind", "direct_task_result")) != "planner_plan"
        and _bool(row.get("schema_compatible", True))
        and _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and _bool(row.get("training_executed"))
        and not _bool(row.get("uses_weighted_superedges"))
        and not _bool(row.get("uses_synthetic_nodes"))
        and _bool(row.get("eligible_for_official_main_table"))
        and _finite(row.get("test_micro_f1", row.get("test_micro_f1_mean")))
        and _finite(row.get("test_macro_f1", row.get("test_macro_f1_mean")))
    )


def _selector_plan_ready(rows: Sequence[Mapping[str, Any]], budget: float, token: str) -> bool:
    return any(
        str(row.get("row_kind")) == "planner_plan"
        and abs((_float(row.get("requested_structural_budget")) or -1.0) - budget) <= 0.005
        and token in str(row.get("selected_canonical_method", ""))
        and _bool(row.get("eligible_for_planner_decision"))
        and not _bool(row.get("eligible_for_official_main_table"))
        and not _bool(row.get("official_hgb_exported"))
        and not _bool(row.get("training_executed"))
        and str(row.get("selected_edge_hash", "")).strip()
        for row in rows
    )


def _external_5x5_ready(rows: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(
            any(_method(row) == method and (_float(row.get("ready_run_count")) or 0.0) >= 25 for row in by_method)
            for method in REQUIRED_EXTERNAL_METHODS
        )
    return all(_external_method_budget_ready(rows, method, budget_type, budget) for method in REQUIRED_EXTERNAL_METHODS for budget_type, budget in REQUIRED_EXTERNAL_BUDGETS)


def _external_method_ready(rows: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]], method: str) -> bool:
    if by_method:
        return any(_method(row) == method and (_float(row.get("ready_run_count")) or 0.0) >= 25 for row in by_method)
    return any(_method(row) == method and _task_ready(row) for row in rows)


def _external_method_budget_ready(rows: Sequence[Mapping[str, Any]], method: str, budget_type: str, budget: float) -> bool:
    ready = [
        row
        for row in rows
        if _method(row) == method
        and str(row.get("budget_type", row.get("budget_family", ""))) == budget_type
        and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001
        and _task_ready(row)
    ]
    return len({(str(row.get("graph_seed")), str(row.get("training_seed"))) for row in ready}) >= 25


def _external_budget_fair(rows: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in rows if _task_ready(row)]
    if not ready:
        return False
    return all(_bool(row.get("budget_matched_within_tolerance", row.get("budget_match"))) for row in ready)


def _freehgc_standard_ready(rows: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(any(_round(row.get("ratio", row.get("reduction_rate"))) == ratio and (_float(row.get("success_count")) or 0.0) >= 5 and _finite(row.get("test_micro_f1_mean")) for row in by_method) for ratio in REQUIRED_FREEHGC_RATIOS)
    ready = [row for row in rows if _task_ready(row) and _bool(row.get("success"))]
    return REQUIRED_FREEHGC_RATIOS.issubset({_round(row.get("ratio", row.get("reduction_rate"))) for row in ready}) and len({str(row.get("seed")) for row in ready}) >= 5


def _freehgc_tp_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(token in str(row.get("variant", row.get("method", ""))).lower() and _task_ready(row) and _bool(row.get("official_hgb_exported")) for row in rows)


def _freehgc_hard_proof(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("failure_reason", row.get("hard_reason", ""))).strip()
    return (_bool(row.get("hard_failure", row.get("hard_incompatibility"))) or str(row.get("failure_type")) == "hard_incompatibility") and reason not in {"", "adapter_not_implemented", "not_exportable"}


def _feature_ablation_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        ("HeSF-RCS-APV12", "raw"),
        ("HeSF-RCS-APV12", "zero-paper-preserve-dim"),
        ("HeSF-RCS-APV12", "zero-term-preserve-dim"),
        ("HeSF-RCS-APV12", "zero-all-support-preserve-dim"),
    }
    seen = {(str(row.get("method")), str(row.get("feature_transform"))) for row in rows if _task_ready(row)}
    return required.issubset(seen)


def _adapter_ready(rows: Sequence[Mapping[str, Any]], base_token: str, adapter_token: str) -> bool:
    return any(base_token in str(row.get("base_method", row.get("base_graph_method", ""))) and adapter_token in _adapter_name(row) and _task_ready(row) and _bool(row.get("success")) for row in rows)


def _adapter_row_ok(row: Mapping[str, Any]) -> bool:
    if not _bool(row.get("success")):
        return _failed_adapter_clean(row)
    return all(_finite(row.get(field)) for field in ("static_inference_package_ratio", "transform_recipe_package_ratio", "reconstructable_package_ratio"))


def _failed_adapter_clean(row: Mapping[str, Any]) -> bool:
    if _bool(row.get("success")):
        return True
    return all(not _finite(row.get(field)) for field in ("static_inference_package_ratio", "transform_recipe_package_ratio", "reconstructable_package_ratio")) and bool(str(row.get("failure_type", "")).strip() and str(row.get("failure_reason", row.get("failure_message", ""))).strip())


def _system_cost_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in rows if _task_ready(row)]
    return bool(ready) and all(_positive(row.get(field)) for row in ready for field in ("official_sehgnn_preprocess_time_seconds", "training_time_seconds", "peak_cpu_rss_mb", "preprocessed_cache_bytes"))


def _system_field_ready(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    ready = [row for row in rows if _task_ready(row)]
    return bool(ready) and all(_positive(row.get(field)) for row in ready)


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    required = ("full", "export", "H6", "auto")
    methods = [str(row.get("method", "")) for row in rows if str(row.get("dataset", "")).upper() == dataset and _task_ready(row) and _finite(row.get("recovery_vs_native_full_micro"))]
    return all(any(token.lower() in method.lower() for method in methods) for token in required)


def _task_ready(row: Mapping[str, Any]) -> bool:
    return _bool(row.get("training_executed")) and _finite(row.get("test_micro_f1", row.get("test_micro_f1_mean"))) and _finite(row.get("test_macro_f1", row.get("test_macro_f1_mean")))


def _no_selection_leakage(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(not _bool(row.get("uses_test_metrics_for_selection", row.get("selection_uses_test_metrics"))) and not _bool(row.get("uses_test_labels_for_selection")) for row in rows)


def _real_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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
    return -1.0 if parsed is None else round(parsed, 3)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ready"}


def _blank_or_true(value: Any) -> bool:
    return value in {"", None} or _bool(value)


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("baseline_name", "")))


def _adapter_name(row: Mapping[str, Any]) -> str:
    return str(row.get("adapter_method", row.get("feature_adapter", row.get("adapter_name", ""))))
