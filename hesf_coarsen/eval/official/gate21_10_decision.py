from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REQUIRED_DECISION_FLAGS = (
    "OFFICIAL_MAIN_DBLP_APV12_PASS",
    "OFFICIAL_MAIN_DBLP_APV16_PASS",
    "AUTO_SELECTOR_DBLP_BUDGET12_ALIGNED",
    "AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED",
    "AUTO_SELECTOR_NO_TEST_LEAKAGE_PASS",
    "EXTERNAL_TP_SMOKE_TASK_RESULTS_READY",
    "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
    "EXTERNAL_TP_ALL_REQUIRED_METHODS_READY",
    "EXTERNAL_TP_BUDGET_MATCH_PASS",
    "EXTERNAL_TP_COARSENING_READY",
    "EXTERNAL_TP_FREEHGC_SELECTION_READY",
    "FREEHGC_STANDARD_SINGLE_SEED_PARTIAL_READY",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON",
    "FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED",
    "FREEHGC_STANDARD_SPLIT_VERIFIED",
    "FREEHGC_TP_SELECTION_READY",
    "FREEHGC_TP_SYNTHETIC_SUPPORT_READY",
    "FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY",
    "FREEHGC_TP_TASK_RESULTS_READY",
    "METAPATH_INTROSPECTION_PASS",
    "CACHE_HASH_REAL_PASS",
    "FEATURE_ABLATION_TASK_RESULTS_READY",
    "ADAPTER_APV12_RP64_READY",
    "ADAPTER_APV16_RP64_READY",
    "ADAPTER_BY_METHOD_RATIO_MERGE_PASS",
    "ADAPTER_PACKAGE_AUDIT_PASS",
    "ADAPTER_FAILED_ROWS_EXCLUDED_FROM_NUMERIC_SUMMARY",
    "APV12_RP64_ADAPTER_TASK_READY",
    "APV16_RP64_ADAPTER_TASK_READY",
    "PCA_REPRODUCIBLE_PACKAGE_COMPLETE",
    "STORAGE_DENOMINATOR_AUDIT_PASS",
    "SYSTEM_WORKLOAD_COST_READY",
    "COVERAGE_REACHABILITY_TABLE_PASS",
    "COVERAGE_SEMANTIC_DISTRIBUTIONAL_PASS",
    "CROSS_DATASET_ACM_TASK_RESULTS_READY",
    "CROSS_DATASET_IMDB_TASK_RESULTS_READY",
    "ICDE_EVIDENCE_READY",
)

REQUIRED_EXTERNAL_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP")
REQUIRED_FREEHGC_RATIOS = {0.012, 0.024, 0.048, 0.096, 0.120}


def gate21_10_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    auto_selector_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_env_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_rows: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    storage_denominator_rows: Sequence[Mapping[str, Any]] = (),
    system_cost_rows: Sequence[Mapping[str, Any]] = (),
    coverage_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    flags = {name: False for name in REQUIRED_DECISION_FLAGS}
    flags["OFFICIAL_MAIN_DBLP_APV12_PASS"] = any("APV12" in _method(row) and _ready_metric(row) for row in official_rows)
    flags["OFFICIAL_MAIN_DBLP_APV16_PASS"] = any("APV16" in _method(row) and _ready_metric(row) for row in official_rows)
    flags["AUTO_SELECTOR_DBLP_BUDGET12_ALIGNED"] = any(_budget_plan_aligned(row, 0.12, pa=0.0, vp=0.0) for row in auto_selector_rows)
    flags["AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED"] = any(_budget_plan_aligned(row, 0.16, pa=0.5, vp=0.5) for row in auto_selector_rows)
    flags["AUTO_SELECTOR_NO_TEST_LEAKAGE_PASS"] = bool(auto_selector_rows) and all(
        not _bool(row.get("selection_uses_test_metrics", row.get("uses_test_metrics_for_selection")))
        and not _bool(row.get("uses_test_labels_for_selection"))
        and not _bool(row.get("leakage_detected"))
        for row in auto_selector_rows
    )
    flags["EXTERNAL_TP_SMOKE_TASK_RESULTS_READY"] = any(_ready_metric(row) for row in external_tp_rows)
    flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] = _external_5x5_ready(external_tp_rows)
    flags["EXTERNAL_TP_ALL_REQUIRED_METHODS_READY"] = all(
        _external_method_5x5_ready(external_tp_rows, method) for method in REQUIRED_EXTERNAL_METHODS
    )
    flags["EXTERNAL_TP_BUDGET_MATCH_PASS"] = bool(external_tp_rows) and all(
        _bool(row.get("budget_match_pass", True)) and str(row.get("budget_match_status", "within_tolerance")) != "budget_infeasible"
        for row in external_tp_rows
        if _ready_metric(row)
    )
    flags["EXTERNAL_TP_COARSENING_READY"] = any("coarsening" in _method(row).lower() and _ready_metric(row) and _bool(row.get("training_executed")) for row in external_tp_rows)
    flags["EXTERNAL_TP_FREEHGC_SELECTION_READY"] = any("freehgc" in _method(row).lower() and "selection" in _method(row).lower() and _ready_metric(row) and _bool(row.get("training_executed")) for row in external_tp_rows)
    flags["FREEHGC_STANDARD_SINGLE_SEED_PARTIAL_READY"] = any(_ready_metric(row) and _bool(row.get("success", True)) for row in freehgc_standard_rows)
    flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"] = any(_bool(row.get("upstream_config_verified")) and _bool(row.get("required_files_present")) for row in freehgc_env_rows)
    flags["FREEHGC_STANDARD_SPLIT_VERIFIED"] = any(_bool(row.get("split_matches_hgb_official")) for row in freehgc_env_rows)
    flags["FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON"] = any(_freehgc_standard_hard_failure(row) for row in freehgc_env_rows)
    flags["FREEHGC_STANDARD_5SEED_READY"] = (
        flags["FREEHGC_STANDARD_UPSTREAM_CONFIG_VERIFIED"]
        and flags["FREEHGC_STANDARD_SPLIT_VERIFIED"]
        and _freehgc_standard_5seed_ready(freehgc_standard_rows)
    )
    flags["FREEHGC_TP_SELECTION_READY"] = any("selection" in str(row.get("freehgc_variant", row.get("method", ""))).lower() and _ready_metric(row) for row in freehgc_tp_rows)
    flags["FREEHGC_TP_SYNTHETIC_SUPPORT_READY"] = any("synthetic" in str(row.get("freehgc_variant", row.get("method", ""))).lower() and _ready_metric(row) for row in freehgc_tp_rows)
    flags["FREEHGC_TP_TASK_RESULTS_READY"] = flags["FREEHGC_TP_SELECTION_READY"] or flags["FREEHGC_TP_SYNTHETIC_SUPPORT_READY"]
    flags["FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY"] = any(_freehgc_hard_gap_ready(row) for row in freehgc_tp_rows)
    flags["METAPATH_INTROSPECTION_PASS"] = bool(metapath_rows) and all(_real_hash(row.get("feature_tensor_hash")) and _positive(row.get("feature_tensor_bytes")) and _bool(row.get("introspection_supported")) for row in metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_rows) and all(_bool(row.get("cache_hash_non_empty")) and _bool(row.get("assertion_pass")) for row in cache_rows)
    flags["FEATURE_ABLATION_TASK_RESULTS_READY"] = _feature_ablation_ready(feature_ablation_rows)
    flags["ADAPTER_APV12_RP64_READY"] = any("APV12" in _adapter_base(row) and "random_projection_dim64" in _adapter_name(row) and _ready_metric(row) for row in adapter_rows)
    flags["ADAPTER_APV16_RP64_READY"] = any("APV16" in _adapter_base(row) and "random_projection_dim64" in _adapter_name(row) and _ready_metric(row) for row in adapter_rows)
    flags["APV12_RP64_ADAPTER_TASK_READY"] = flags["ADAPTER_APV12_RP64_READY"]
    flags["APV16_RP64_ADAPTER_TASK_READY"] = flags["ADAPTER_APV16_RP64_READY"]
    flags["ADAPTER_BY_METHOD_RATIO_MERGE_PASS"] = bool(adapter_rows) and all(_finite_ratio(row.get("static_inference_package_ratio")) for row in adapter_rows if _bool(row.get("success", True)))
    flags["ADAPTER_PACKAGE_AUDIT_PASS"] = flags["ADAPTER_BY_METHOD_RATIO_MERGE_PASS"]
    flags["ADAPTER_FAILED_ROWS_EXCLUDED_FROM_NUMERIC_SUMMARY"] = all(not _bool(row.get("eligible_for_official_main_table")) for row in adapter_rows if _bool(row.get("uses_feature_adapter", True)))
    flags["PCA_REPRODUCIBLE_PACKAGE_COMPLETE"] = any("pca" in _adapter_name(row).lower() and _bool(row.get("pca_reproducible_package_complete")) for row in adapter_rows)
    flags["STORAGE_DENOMINATOR_AUDIT_PASS"] = bool(storage_denominator_rows) and all(_bool(row.get("ratio_field_name_consistent")) for row in storage_denominator_rows)
    executed_system_rows = [row for row in system_cost_rows if _bool(row.get("training_executed", False))]
    flags["SYSTEM_WORKLOAD_COST_READY"] = bool(executed_system_rows) and all(_system_cost_row_ready(row) for row in executed_system_rows)
    flags["COVERAGE_REACHABILITY_TABLE_PASS"] = bool(coverage_rows) and all(_bool(row.get("reachability_assertions_pass", True)) for row in coverage_rows)
    flags["COVERAGE_SEMANTIC_DISTRIBUTIONAL_PASS"] = bool(coverage_rows) and all(str(row.get("per_class_venue_coverage_json", "")).strip() for row in coverage_rows)
    flags["CROSS_DATASET_ACM_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "ACM")
    flags["CROSS_DATASET_IMDB_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows, "IMDB")
    flags["ICDE_EVIDENCE_READY"] = all(
        flags[name]
        for name in (
            "OFFICIAL_MAIN_DBLP_APV12_PASS",
            "OFFICIAL_MAIN_DBLP_APV16_PASS",
            "AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED",
            "AUTO_SELECTOR_NO_TEST_LEAKAGE_PASS",
            "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
            "EXTERNAL_TP_BUDGET_MATCH_PASS",
            "METAPATH_INTROSPECTION_PASS",
            "CACHE_HASH_REAL_PASS",
            "FEATURE_ABLATION_TASK_RESULTS_READY",
            "SYSTEM_WORKLOAD_COST_READY",
            "STORAGE_DENOMINATOR_AUDIT_PASS",
        )
    ) and (flags["FREEHGC_STANDARD_5SEED_READY"] or flags["FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON"]) and (
        flags["FREEHGC_TP_TASK_RESULTS_READY"] or flags["FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY"]
    )
    return flags


def decision_status(flags: Mapping[str, Any]) -> str:
    if _bool(flags.get("ICDE_EVIDENCE_READY")):
        return "ICDE_EVIDENCE_READY"
    if _bool(flags.get("AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED")) and (_bool(flags.get("FREEHGC_TP_TASK_RESULTS_READY")) or _bool(flags.get("FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY"))):
        return "GATE21_10_PARTIAL_READY"
    return "NOT_READY"


def _budget_plan_aligned(row: Mapping[str, Any], target: float, *, pa: float, vp: float) -> bool:
    budget = _float(row.get("budget_target"))
    return bool(
        budget is not None
        and abs(budget - target) <= 0.005
        and (_float(row.get("AP_keep")) or 0.0) >= 0.9
        and (_float(row.get("PV_keep")) or 0.0) >= 0.9
        and (_float(row.get("PA_keep")) or 0.0) == pa
        and (_float(row.get("VP_keep")) or 0.0) == vp
        and (_float(row.get("PT_keep")) or 0.0) <= 0.05
        and (_float(row.get("TP_keep")) or 0.0) <= 0.05
    )


def _external_5x5_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(_external_method_5x5_ready(rows, method) for method in REQUIRED_EXTERNAL_METHODS)


def _external_method_5x5_ready(rows: Sequence[Mapping[str, Any]], method: str) -> bool:
    ready = [row for row in rows if _method(row) == method and _ready_metric(row) and _bool(row.get("training_executed"))]
    return len({str(row.get("graph_seed", "")) for row in ready}) >= 5 and len({str(row.get("training_seed", "")) for row in ready}) >= 5


def _freehgc_standard_5seed_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in rows if _ready_metric(row) and _bool(row.get("success", True))]
    ratios = {_round_ratio(row.get("ratio", row.get("support_node_ratio", row.get("reduction_rate")))) for row in ready}
    seeds = {str(row.get("seed", row.get("training_seed", ""))) for row in ready}
    return REQUIRED_FREEHGC_RATIOS.issubset(ratios) and len({seed for seed in seeds if seed}) >= 5


def _freehgc_standard_hard_failure(row: Mapping[str, Any]) -> bool:
    if _bool(row.get("standard_condensation_supported", True)) and _bool(row.get("required_files_present", True)):
        return False
    reason = str(
        row.get(
            "hard_failure_reason",
            row.get("failure_reason", row.get("backbone_matches_or_reason", "")),
        )
    ).strip()
    required_files = row.get("required_files", "")
    return bool(reason or required_files)


def _freehgc_hard_gap_ready(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("hard_reason", "")).strip()
    artifact = str(
        row.get(
            "minimal_blocking_artifact",
            row.get("blocking_artifact", row.get("freehgc_source_file", "")),
        )
    ).strip()
    return _bool(row.get("hard_incompatibility")) and reason not in {"", "adapter_not_implemented", "not_exportable"} and bool(artifact or reason)


def _feature_ablation_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    methods = {"full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"}
    transforms = {"raw", "zero-paper-preserve-dim", "zero-term-preserve-dim", "zero-venue-preserve-dim", "zero-all-support-preserve-dim", "paper-only-preserve-original-dims", "term-only-preserve-original-dims", "paper-random-projection64", "paper-pca64"}
    settings = {"default", "no_label_feats", "num_feature_hops_0", "num_label_hops_0", "feature_only_mlp_adapter"}
    seen = {(str(row.get("method")), str(row.get("feature_transform")), str(row.get("label_graph_setting"))) for row in rows if _ready_metric(row) and _bool(row.get("training_executed"))}
    return all((method, transform, setting) in seen for method in methods for transform in transforms for setting in settings)


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    methods = {str(row.get("method", "")) for row in rows if str(row.get("dataset", "")).upper() == dataset and _ready_metric(row) and _bool(row.get("training_executed"))}
    required = {"full-native", "export-full", "H6-node30", "random-edge-relation-wise", "HeSF-RCS-auto-structural30", "HeSF-RCS-auto-structural20"}
    return required.issubset(methods)


def _system_cost_row_ready(row: Mapping[str, Any]) -> bool:
    return all(_positive(row.get(field)) for field in ("load_time_seconds", "official_sehgnn_preprocess_time_seconds", "training_time_seconds", "peak_cpu_rss_mb", "preprocessed_cache_bytes")) and _ready_metric({"test_micro_f1": row.get("task_micro_f1"), "test_macro_f1": row.get("task_macro_f1")})


def _ready_metric(row: Mapping[str, Any]) -> bool:
    metric_keys = (
        "test_micro_f1",
        "test_micro_f1_mean",
        "task_micro_f1",
        "test_macro_f1",
        "test_macro_f1_mean",
        "task_macro_f1",
    )
    present = [row.get(key) for key in metric_keys if row.get(key) not in {"", None}]
    return bool(present) and all(_finite(value) for value in present)


def _finite_ratio(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed >= 0 and parsed not in {10240.0, 1000000.0}


def _real_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _round_ratio(value: Any) -> float:
    parsed = _float(value)
    return -1.0 if parsed is None else round(parsed, 3)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("baseline_name", row.get("method_name", ""))))


def _adapter_base(row: Mapping[str, Any]) -> str:
    return str(row.get("base_method", row.get("base_graph_method", row.get("method", ""))))


def _adapter_name(row: Mapping[str, Any]) -> str:
    return str(row.get("adapter_method", row.get("feature_adapter", row.get("adapter_name", ""))))
