from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.coverage_diagnostics import gate21_11_distributional_coverage_ready
from hesf_coarsen.eval.official.end_to_end_system_cost import gate21_11_system_cost_ready


REQUIRED_DECISION_FLAGS = (
    "OFFICIAL_MAIN_DBLP_READY",
    "EXTERNAL_TP_5X5_READY",
    "EXTERNAL_TP_SMOKE_ONLY",
    "EXTERNAL_TP_BUDGET_MATCH_READY",
    "COARSENING_HG_TP_READY",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_STANDARD_SINGLE_SEED_PARTIAL_READY",
    "FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON",
    "FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED",
    "FREEHGC_TP_SELECTION_TASK_READY",
    "FREEHGC_TP_SYNTHETIC_SUPPORT_TASK_READY",
    "FREEHGC_TP_HARD_INCOMPATIBILITY_PROVEN",
    "METAPATH_TENSOR_DUMP_READY",
    "CACHE_HASH_REAL_PASS",
    "APV12_APV16_CACHE_DIFF_PASS",
    "PTTP_CACHE_DIFF_PASS",
    "FEATURE_ABLATION_TASK_READY",
    "FEATURE_SHAPE_SAFE_PASS",
    "NO_LABEL_FEATS_ABLATION_READY",
    "FEATURE_HOPS_ABLATION_READY",
    "PAPER_FEATURE_REDUNDANCY_TEST_READY",
    "APV12_RP64_ADAPTER_READY",
    "APV16_RP64_ADAPTER_READY",
    "APV16_INT8_ADAPTER_READY",
    "ADAPTER_PACKAGE_ACCOUNTING_PASS",
    "PCA_REPRODUCIBLE_PACKAGE_COMPLETE",
    "FAILED_ADAPTER_PLACEHOLDERS_REMOVED",
    "SYSTEM_COST_END_TO_END_READY",
    "PREPROCESS_TIME_MEASURED_PASS",
    "TRAINING_TIME_MEASURED_PASS",
    "PEAK_MEMORY_MEASURED_PASS",
    "CACHE_BYTES_MEASURED_PASS",
    "STORAGE_ONLY_BASELINES_CONTEXTUALIZED",
    "CROSS_DATASET_ACM_TASK_READY",
    "CROSS_DATASET_IMDB_TASK_READY",
    "CROSS_DATASET_RECOVERY_TABLE_READY",
    "AUTO_SELECTOR_NO_DBLP_HARDCODE_PASS",
    "BUDGETED_PLANNER_DBLP_012_PASS",
    "BUDGETED_PLANNER_DBLP_016_PASS",
    "BUDGETED_PLANNER_NO_TEST_LEAKAGE_PASS",
    "BUDGETED_PLANNER_TRACE_READY",
    "COVERAGE_REACHABILITY_TABLE_EMITTED",
    "COVERAGE_REACHABILITY_SANITY_PASS",
    "COVERAGE_DISTRIBUTIONAL_TABLE_EMITTED",
    "COVERAGE_DISTRIBUTIONAL_MECHANISM_READY",
    "APV12_VS_APV16_FEEDBACK_COVERAGE_EXPLAINED",
    "PV_PRUNING_COVERAGE_EXPLAINED",
    "ICDE_SUBMISSION_EVIDENCE_READY",
)

REQUIRED_EXTERNAL_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP")
REQUIRED_STRUCTURAL_BUDGETS = {0.16, 0.30}
REQUIRED_FREEHGC_RATIOS = {0.012, 0.024, 0.048, 0.096, 0.120}


def gate21_11_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    budgeted_selector_rows: Sequence[Mapping[str, Any]] = (),
    channel_trace_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_runs: Sequence[Mapping[str, Any]] = (),
    external_tp_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_runs: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_by_method: Sequence[Mapping[str, Any]] = (),
    freehgc_env: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_audit: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    adapter_by_method: Sequence[Mapping[str, Any]] = (),
    system_cost_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
    coverage_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    flags = {name: False for name in REQUIRED_DECISION_FLAGS}
    flags["OFFICIAL_MAIN_DBLP_READY"] = _official_main_ready(official_rows)
    flags["EXTERNAL_TP_5X5_READY"] = _external_tp_5x5_ready(external_tp_runs, external_tp_by_method)
    flags["EXTERNAL_TP_SMOKE_ONLY"] = _external_smoke_only(external_tp_runs, external_tp_by_method)
    flags["EXTERNAL_TP_BUDGET_MATCH_READY"] = _external_budget_match_ready(external_tp_runs, external_tp_by_method)
    flags["COARSENING_HG_TP_READY"] = _method_task_ready(external_tp_runs, "Coarsening-HG-TP")

    flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"] = any(_bool(row.get("upstream_config_verified")) and _bool(row.get("required_files_present", True)) for row in freehgc_env)
    flags["FREEHGC_STANDARD_SINGLE_SEED_PARTIAL_READY"] = any(_ready_metric(row) and _bool(row.get("success")) and _bool(row.get("training_executed")) for row in freehgc_standard_runs)
    flags["FREEHGC_STANDARD_5SEED_READY"] = flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"] and _freehgc_standard_5seed_ready(freehgc_standard_runs, freehgc_standard_by_method)
    flags["FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON"] = any(not _bool(row.get("required_files_present", True)) and str(row.get("hard_failure_reason", row.get("failure_reason", ""))).strip() for row in freehgc_env)
    flags["FREEHGC_TP_SELECTION_TASK_READY"] = _freehgc_tp_task_ready(freehgc_tp_audit, "selection")
    flags["FREEHGC_TP_SYNTHETIC_SUPPORT_TASK_READY"] = _freehgc_tp_task_ready(freehgc_tp_audit, "synthetic")
    flags["FREEHGC_TP_HARD_INCOMPATIBILITY_PROVEN"] = any(_freehgc_tp_hard(row) for row in freehgc_tp_audit)

    flags["METAPATH_TENSOR_DUMP_READY"] = bool(metapath_rows) and all(_real_hash(row.get("feature_tensor_hash")) and _positive(row.get("feature_tensor_bytes")) for row in metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_rows) and all(_bool(row.get("assertion_pass")) and _real_hash(row.get("cache_hash", row.get("cache_file_hash"))) for row in cache_rows)
    flags["APV12_APV16_CACHE_DIFF_PASS"] = any(_bool(row.get("APV12_APV16_CACHE_DIFF_PASS", row.get("apv12_apv16_cache_diff_pass"))) for row in cache_rows)
    flags["PTTP_CACHE_DIFF_PASS"] = any(_bool(row.get("PTTP_CACHE_DIFF_PASS", row.get("pttp_cache_diff_pass"))) for row in cache_rows)

    flags["FEATURE_ABLATION_TASK_READY"] = _feature_ablation_ready(feature_ablation_rows)
    flags["FEATURE_SHAPE_SAFE_PASS"] = bool(feature_ablation_rows) and all(_blank_or_true(row.get("shape_safe_pass")) for row in feature_ablation_rows)
    flags["NO_LABEL_FEATS_ABLATION_READY"] = _ablation_setting_ready(feature_ablation_rows, "no_label_feats")
    flags["FEATURE_HOPS_ABLATION_READY"] = _ablation_setting_ready(feature_ablation_rows, "num_feature_hops_0")
    flags["PAPER_FEATURE_REDUNDANCY_TEST_READY"] = _paper_redundancy_ready(feature_ablation_rows)

    flags["APV12_RP64_ADAPTER_READY"] = _adapter_ready(adapter_rows, "APV12", "random_projection_dim64")
    flags["APV16_RP64_ADAPTER_READY"] = _adapter_ready(adapter_rows, "APV16", "random_projection_dim64")
    flags["APV16_INT8_ADAPTER_READY"] = _adapter_ready(adapter_rows, "APV16", "int8")
    flags["ADAPTER_PACKAGE_ACCOUNTING_PASS"] = bool(adapter_rows) and all(_adapter_accounting_row_ok(row) for row in adapter_rows)
    flags["PCA_REPRODUCIBLE_PACKAGE_COMPLETE"] = any("pca" in _adapter_name(row).lower() and _bool(row.get("pca_reproducible_package_complete", row.get("reconstructable_package_complete"))) for row in adapter_rows)
    flags["FAILED_ADAPTER_PLACEHOLDERS_REMOVED"] = bool(adapter_rows) and all(_failed_adapter_clean(row) for row in adapter_rows)

    flags["SYSTEM_COST_END_TO_END_READY"] = gate21_11_system_cost_ready(system_cost_rows)
    flags["PREPROCESS_TIME_MEASURED_PASS"] = _system_field_measured(system_cost_rows, "official_sehgnn_preprocess_time_seconds")
    flags["TRAINING_TIME_MEASURED_PASS"] = _system_field_measured(system_cost_rows, "training_time_seconds")
    flags["PEAK_MEMORY_MEASURED_PASS"] = _system_field_measured(system_cost_rows, "peak_cpu_rss_mb")
    flags["CACHE_BYTES_MEASURED_PASS"] = _system_field_measured(system_cost_rows, "preprocessed_cache_bytes")
    flags["STORAGE_ONLY_BASELINES_CONTEXTUALIZED"] = any(_bool(row.get("archive_only_compression")) for row in system_cost_rows)

    flags["CROSS_DATASET_ACM_TASK_READY"] = _cross_dataset_ready(cross_dataset_rows, "ACM")
    flags["CROSS_DATASET_IMDB_TASK_READY"] = _cross_dataset_ready(cross_dataset_rows, "IMDB")
    executed_cross_rows = [row for row in cross_dataset_rows if _bool(row.get("training_executed"))]
    flags["CROSS_DATASET_RECOVERY_TABLE_READY"] = bool(executed_cross_rows) and all(_finite(row.get("recovery_vs_native_full_micro")) for row in executed_cross_rows)
    flags["AUTO_SELECTOR_NO_DBLP_HARDCODE_PASS"] = bool(cross_dataset_rows) and all("AP" not in str(row.get("selected_channel_plan_human", "")) or str(row.get("dataset", "")).upper() == "DBLP" for row in cross_dataset_rows)

    flags["BUDGETED_PLANNER_DBLP_012_PASS"] = _budget_row_pass(budgeted_selector_rows, 0.12, "APV12")
    flags["BUDGETED_PLANNER_DBLP_016_PASS"] = _budget_row_pass(budgeted_selector_rows, 0.16, "APV16")
    flags["BUDGETED_PLANNER_NO_TEST_LEAKAGE_PASS"] = bool(budgeted_selector_rows or channel_trace_rows) and all(not _bool(row.get("uses_test_metrics_for_selection", row.get("uses_test_metric"))) and not _bool(row.get("uses_test_labels_for_selection")) for row in [*budgeted_selector_rows, *channel_trace_rows])
    flags["BUDGETED_PLANNER_TRACE_READY"] = bool(channel_trace_rows) and all(str(row.get("channel_name", "")).strip() for row in channel_trace_rows)

    flags["COVERAGE_REACHABILITY_TABLE_EMITTED"] = bool(coverage_rows)
    flags["COVERAGE_REACHABILITY_SANITY_PASS"] = bool(coverage_rows) and all(_blank_or_true(row.get("coverage_edge_count_matches_relation_retention")) and _blank_or_true(row.get("node_type_offsets_match_node_dat_counts")) for row in coverage_rows)
    flags["COVERAGE_DISTRIBUTIONAL_TABLE_EMITTED"] = bool(coverage_rows)
    flags["COVERAGE_DISTRIBUTIONAL_MECHANISM_READY"] = gate21_11_distributional_coverage_ready(coverage_rows)
    flags["APV12_VS_APV16_FEEDBACK_COVERAGE_EXPLAINED"] = flags["COVERAGE_DISTRIBUTIONAL_MECHANISM_READY"] and _coverage_has_methods(coverage_rows, ("APV12", "APV16"))
    flags["PV_PRUNING_COVERAGE_EXPLAINED"] = flags["COVERAGE_DISTRIBUTIONAL_MECHANISM_READY"] and _coverage_has_methods(coverage_rows, ("PV75", "PV50"))

    flags["ICDE_SUBMISSION_EVIDENCE_READY"] = all(
        flags[name]
        for name in (
            "OFFICIAL_MAIN_DBLP_READY",
            "EXTERNAL_TP_5X5_READY",
            "FREEHGC_STANDARD_5SEED_READY",
            "METAPATH_TENSOR_DUMP_READY",
            "CACHE_HASH_REAL_PASS",
            "FEATURE_ABLATION_TASK_READY",
            "APV12_RP64_ADAPTER_READY",
            "APV16_RP64_ADAPTER_READY",
            "SYSTEM_COST_END_TO_END_READY",
            "CROSS_DATASET_ACM_TASK_READY",
            "CROSS_DATASET_IMDB_TASK_READY",
            "BUDGETED_PLANNER_DBLP_012_PASS",
            "BUDGETED_PLANNER_DBLP_016_PASS",
            "BUDGETED_PLANNER_NO_TEST_LEAKAGE_PASS",
            "COVERAGE_DISTRIBUTIONAL_MECHANISM_READY",
            "FAILED_ADAPTER_PLACEHOLDERS_REMOVED",
        )
    )
    return flags


def decision_status(flags: Mapping[str, Any]) -> str:
    if _bool(flags.get("ICDE_SUBMISSION_EVIDENCE_READY")):
        return "ICDE_SUBMISSION_EVIDENCE_READY"
    if _bool(flags.get("OFFICIAL_MAIN_DBLP_READY")) and _bool(flags.get("BUDGETED_PLANNER_DBLP_016_PASS")):
        return "GATE21_11_PARTIAL_LOCKDOWN"
    return "NOT_READY"


def _official_main_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    has_full = any(("full" in _method(row).lower()) and _ready_metric(row) for row in rows)
    apv12 = any("APV12" in _method(row) and _official_row_ready(row) for row in rows)
    apv16 = any("APV16" in _method(row) and _official_row_ready(row) for row in rows)
    return has_full and apv12 and apv16


def _official_row_ready(row: Mapping[str, Any]) -> bool:
    return _ready_metric(row) and _bool(row.get("official_sehgnn_unmodified", True)) and _bool(row.get("eligible_for_official_main_table", True)) and _finite(row.get("structural_storage_ratio", row.get("actual_structural_storage_ratio")))


def _external_tp_5x5_ready(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(
            any(_method(row) == method and _float(row.get("ready_run_count")) is not None and float(row.get("ready_run_count")) >= 25 and (_bool(row.get("eligible_for_main_comparison")) or _bool(row.get("budget_infeasible"))) for row in by_method)
            for method in REQUIRED_EXTERNAL_METHODS
        )
    return all(_external_method_budget_ready(runs, method, budget) for method in REQUIRED_EXTERNAL_METHODS for budget in REQUIRED_STRUCTURAL_BUDGETS)


def _external_method_budget_ready(runs: Sequence[Mapping[str, Any]], method: str, budget: float) -> bool:
    ready = [row for row in runs if _method(row) == method and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001 and _ready_metric(row) and _bool(row.get("training_executed"))]
    cells = {(str(row.get("graph_seed")), str(row.get("training_seed"))) for row in ready}
    return len(cells) >= 25


def _external_smoke_only(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    counts = [_float(row.get("ready_run_count")) for row in by_method]
    if counts:
        return any(count == 1 for count in counts if count is not None)
    methods = {str(row.get("method", "")) for row in runs if _ready_metric(row)}
    return any(sum(1 for row in runs if _method(row) == method and _ready_metric(row)) == 1 for method in methods)


def _external_budget_match_ready(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in runs if _ready_metric(row) and _bool(row.get("training_executed"))]
    if not ready:
        return False
    return all(_bool(row.get("budget_matched_within_tolerance", row.get("budget_match_pass"))) for row in ready)


def _method_task_ready(rows: Sequence[Mapping[str, Any]], method: str) -> bool:
    return any(_method(row) == method and _ready_metric(row) and _bool(row.get("training_executed")) for row in rows)


def _freehgc_standard_5seed_ready(runs: Sequence[Mapping[str, Any]], by_method: Sequence[Mapping[str, Any]]) -> bool:
    if by_method:
        return all(any(_round(row.get("ratio")) == ratio and (_float(row.get("success_count")) or 0.0) >= (_float(row.get("expected_seed_count")) or 5.0) and _finite(row.get("test_micro_f1_mean")) for row in by_method) for ratio in REQUIRED_FREEHGC_RATIOS)
    ready = [row for row in runs if _ready_metric(row) and _bool(row.get("training_executed")) and _bool(row.get("success"))]
    return REQUIRED_FREEHGC_RATIOS.issubset({_round(row.get("ratio", row.get("reduction_rate"))) for row in ready}) and len({str(row.get("seed")) for row in ready}) >= 5


def _freehgc_tp_task_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(token in str(row.get("variant", row.get("freehgc_variant", ""))).lower() and _ready_metric(row) and _bool(row.get("training_executed")) and _bool(row.get("official_hgb_exported")) for row in rows)


def _freehgc_tp_hard(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("failure_reason", row.get("hard_reason", ""))).strip()
    failure_type = str(row.get("failure_type", "")).strip()
    return (_bool(row.get("hard_failure", row.get("hard_incompatibility"))) or failure_type == "hard_incompatibility") and reason not in {"", "adapter_not_implemented", "not_exportable"}


def _feature_ablation_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    methods = {"HeSF-RCS-APV12", "HeSF-RCS-APV16"}
    transforms = {"raw", "zero-paper-preserve-dim", "zero-term-preserve-dim", "zero-all-support-preserve-dim", "paper-random-projection64"}
    seen = {(str(row.get("method")), str(row.get("feature_transform"))) for row in rows if _ready_metric(row) and _bool(row.get("training_executed"))}
    return all((method, transform) in seen for method in methods for transform in transforms)


def _ablation_setting_ready(rows: Sequence[Mapping[str, Any]], setting: str) -> bool:
    return any(str(row.get("label_graph_setting")) == setting and _ready_metric(row) and _bool(row.get("training_executed")) for row in rows)


def _paper_redundancy_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(any(str(row.get("feature_transform")) == transform and _ready_metric(row) for row in rows) for transform in ("zero-paper-preserve-dim", "zero-term-preserve-dim"))


def _adapter_ready(rows: Sequence[Mapping[str, Any]], base_token: str, adapter_token: str) -> bool:
    return any(base_token in str(row.get("base_method", row.get("base_graph_method", ""))) and adapter_token in _adapter_name(row) and _ready_metric(row) and _bool(row.get("success")) for row in rows)


def _adapter_accounting_row_ok(row: Mapping[str, Any]) -> bool:
    if not _bool(row.get("success")):
        return _failed_adapter_clean(row)
    return all(_finite(row.get(field)) for field in ("static_inference_package_ratio", "transform_recipe_package_ratio", "reconstructable_package_ratio"))


def _failed_adapter_clean(row: Mapping[str, Any]) -> bool:
    if _bool(row.get("success")):
        return True
    return all(not _finite(row.get(field)) for field in ("static_inference_package_ratio", "transform_recipe_package_ratio", "reconstructable_package_ratio")) and bool(str(row.get("failure_type", "")).strip() and str(row.get("failure_reason", row.get("failure_message", ""))).strip())


def _system_field_measured(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    executed = [row for row in rows if _bool(row.get("training_executed"))]
    return bool(executed) and all(_positive(row.get(field)) for row in executed)


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    required = ("full", "export", "H6", "auto")
    methods = [str(row.get("method", "")) for row in rows if str(row.get("dataset", "")).upper() == dataset and _ready_metric(row) and _bool(row.get("training_executed"))]
    return all(any(token.lower() in method.lower() for method in methods) for token in required)


def _budget_row_pass(rows: Sequence[Mapping[str, Any]], budget: float, canonical_token: str) -> bool:
    return any(abs((_float(row.get("requested_structural_budget")) or -1.0) - budget) <= 0.005 and canonical_token in str(row.get("selected_canonical_method", "")) and not _bool(row.get("uses_test_metrics_for_selection")) for row in rows)


def _coverage_has_methods(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> bool:
    methods = " ".join(str(row.get("method", "")) for row in rows)
    return all(token in methods for token in tokens)


def _blank_or_true(value: Any) -> bool:
    return value in {"", None} or _bool(value)


def _ready_metric(row: Mapping[str, Any]) -> bool:
    return _finite(row.get("test_micro_f1", row.get("test_micro_f1_mean"))) and _finite(row.get("test_macro_f1", row.get("test_macro_f1_mean")))


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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("baseline_name", "")))


def _adapter_name(row: Mapping[str, Any]) -> str:
    return str(row.get("adapter_method", row.get("feature_adapter", row.get("adapter_name", ""))))
