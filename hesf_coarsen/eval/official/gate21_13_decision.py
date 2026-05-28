from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


REQUIRED_DECISION_FLAGS = (
    "OFFICIAL_DBLP_APV12_PASS",
    "OFFICIAL_DBLP_APV16_PASS",
    "BUDGETED_SELECTOR_HASH_AUDIT_PASS",
    "APV16_DETERMINISTIC_PROOF_PASS",
    "EXTERNAL_TP_5X5_REQUIRED_READY",
    "EXTERNAL_TP_BUDGET_FAIRNESS_PASS",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_TP_SELECTION_READY",
    "FREEHGC_TP_SYNTHETIC_READY_OR_HARD_INCOMPATIBILITY_PROVEN",
    "METAPATH_TENSOR_DUMP_READY",
    "CACHE_HASH_REAL_PASS",
    "FEATURE_ABLATION_TASK_READY",
    "APV16_ADAPTER_READY",
    "SYSTEM_COST_END_TO_END_READY",
    "CROSS_DATASET_ACM_READY",
    "CROSS_DATASET_IMDB_READY",
    "ICDE_EVIDENCE_READY",
)

REQUIRED_EXTERNAL_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP")
REQUIRED_EXTERNAL_BUDGETS = {
    ("structural_storage_ratio", 0.12),
    ("structural_storage_ratio", 0.16),
    ("structural_storage_ratio", 0.20),
    ("structural_storage_ratio", 0.30),
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
}
REQUIRED_FREEHGC_RATIOS = {0.012, 0.024, 0.048, 0.096, 0.120}
REQUIRED_FEATURE_TRANSFORMS = {"raw", "zero-paper", "zero-term", "zero-all-support", "paper-RP64"}
REQUIRED_APV16_ADAPTERS = {"random_projection_dim64", "random_projection_dim128", "int8_per_feature", "fp16_features"}
REQUIRED_SYSTEM_METHOD_TOKENS = ("full", "APV12", "APV16", "gzip", "RP64")


def gate21_13_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    selector_hash_audit: Sequence[Mapping[str, Any]] = (),
    deterministic_proof_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_by_method_budget: Sequence[Mapping[str, Any]] = (),
    external_tp_budget_fairness: Sequence[Mapping[str, Any]] = (),
    freehgc_env_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_runs: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_by_ratio: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_runs: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_adapter_audit: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    system_cost_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, bool]:
    flags = {name: False for name in REQUIRED_DECISION_FLAGS}
    flags["OFFICIAL_DBLP_APV12_PASS"] = _official_ready(official_rows, "APV12")
    flags["OFFICIAL_DBLP_APV16_PASS"] = _official_ready(official_rows, "APV16")
    flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] = bool(selector_hash_audit) and all(_bool(row.get("selector_hash_audit_pass")) for row in selector_hash_audit)
    flags["APV16_DETERMINISTIC_PROOF_PASS"] = any("APV16" in _method(row) and _bool(row.get("deterministic_proof_pass")) and _int(row.get("actual_export_hash_unique_count")) == 1 for row in deterministic_proof_rows)
    flags["EXTERNAL_TP_5X5_REQUIRED_READY"] = _external_tp_5x5_ready(external_tp_rows, external_tp_by_method_budget)
    flags["EXTERNAL_TP_BUDGET_FAIRNESS_PASS"] = _external_budget_fair(external_tp_rows, external_tp_budget_fairness)
    flags["FREEHGC_STANDARD_5SEED_READY"] = _freehgc_standard_ready(freehgc_standard_runs, freehgc_standard_by_ratio) and any(_bool(row.get("upstream_config_verified")) for row in freehgc_env_rows)
    flags["FREEHGC_TP_SELECTION_READY"] = _freehgc_tp_ready(freehgc_tp_runs, "selection")
    flags["FREEHGC_TP_SYNTHETIC_READY_OR_HARD_INCOMPATIBILITY_PROVEN"] = _freehgc_tp_ready(freehgc_tp_runs, "synthetic") or _freehgc_hard_proof(freehgc_tp_runs, freehgc_tp_adapter_audit, freehgc_env_rows)
    flags["METAPATH_TENSOR_DUMP_READY"] = _metapath_ready(metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_rows) and all(_bool(row.get("assertion_pass")) and _real_hash(row.get("cache_file_hash", row.get("cache_hash"))) for row in cache_rows)
    flags["FEATURE_ABLATION_TASK_READY"] = _feature_ablation_ready(feature_ablation_rows)
    flags["APV16_ADAPTER_READY"] = _apv16_adapter_ready(adapter_rows)
    flags["SYSTEM_COST_END_TO_END_READY"] = _system_cost_ready(system_cost_rows)
    flags["CROSS_DATASET_ACM_READY"] = _cross_dataset_ready(cross_dataset_rows, "ACM")
    flags["CROSS_DATASET_IMDB_READY"] = _cross_dataset_ready(cross_dataset_rows, "IMDB")
    flags["ICDE_EVIDENCE_READY"] = (
        flags["OFFICIAL_DBLP_APV12_PASS"]
        and flags["OFFICIAL_DBLP_APV16_PASS"]
        and flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"]
        and flags["APV16_DETERMINISTIC_PROOF_PASS"]
        and flags["EXTERNAL_TP_5X5_REQUIRED_READY"]
        and flags["METAPATH_TENSOR_DUMP_READY"]
        and flags["CACHE_HASH_REAL_PASS"]
        and flags["SYSTEM_COST_END_TO_END_READY"]
        and (flags["CROSS_DATASET_ACM_READY"] or flags["CROSS_DATASET_IMDB_READY"])
    )
    return flags


def decision_status(flags: Mapping[str, Any]) -> str:
    if _bool(flags.get("ICDE_EVIDENCE_READY")):
        return "ICDE_EVIDENCE_READY"
    if _bool(flags.get("OFFICIAL_DBLP_APV12_PASS")) and _bool(flags.get("OFFICIAL_DBLP_APV16_PASS")) and _bool(flags.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS")):
        return "GATE21_13_PARTIAL_EXECUTED_EVIDENCE"
    return "NOT_READY"


def _official_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(
        token in _method(row)
        and str(row.get("row_kind", "direct_task_result")) != "planner_plan"
        and _bool(row.get("schema_compatible", True))
        and _bool(row.get("target_preserving", row.get("keeps_all_target_nodes", True)))
        and _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and not _bool(row.get("uses_adapter_loader", row.get("uses_feature_adapter", False)))
        and not _bool(row.get("uses_synthetic_nodes"))
        and not _bool(row.get("uses_weighted_superedges"))
        and _bool(row.get("eligible_for_official_main_table"))
        and _finite(row.get("test_micro_f1", row.get("test_micro_f1_mean", row.get("test_micro_mean"))))
        and _finite(row.get("test_macro_f1", row.get("test_macro_f1_mean", row.get("test_macro_mean"))))
        for row in rows
    )


def _external_tp_5x5_ready(rows: Sequence[Mapping[str, Any]], by_method_budget: Sequence[Mapping[str, Any]]) -> bool:
    if by_method_budget:
        return all(
            any(
                _method(row) == method
                and _budget_key(row) == budget_key
                and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001
                and (_float(row.get("success_count")) or _float(row.get("ready_run_count")) or 0.0) >= 25
                and _bool(row.get("budget_fairness_pass", row.get("budget_match_rate", False)))
                for row in by_method_budget
            )
            for method in REQUIRED_EXTERNAL_METHODS
            for budget_key, budget in REQUIRED_EXTERNAL_BUDGETS
        )
    return all(_external_cell_ready(rows, method, budget_key, budget) for method in REQUIRED_EXTERNAL_METHODS for budget_key, budget in REQUIRED_EXTERNAL_BUDGETS)


def _external_cell_ready(rows: Sequence[Mapping[str, Any]], method: str, budget_key: str, budget: float) -> bool:
    ready = [
        row
        for row in rows
        if _method(row) == method
        and _budget_key(row) == budget_key
        and abs((_float(row.get("requested_budget")) or -1.0) - budget) <= 0.001
        and _task_ready(row)
        and _bool(row.get("budget_matched_within_tolerance", row.get("budget_match_pass")))
    ]
    return len({(str(row.get("graph_seed")), str(row.get("training_seed"))) for row in ready}) >= 25


def _external_budget_fair(rows: Sequence[Mapping[str, Any]], fairness: Sequence[Mapping[str, Any]]) -> bool:
    if fairness:
        return all(_bool(row.get("budget_fairness_pass")) for row in fairness if (_float(row.get("success_count")) or 0.0) > 0)
    ready = [row for row in rows if _task_ready(row)]
    return bool(ready) and all(_bool(row.get("budget_matched_within_tolerance", row.get("budget_match_pass"))) for row in ready)


def _freehgc_standard_ready(runs: Sequence[Mapping[str, Any]], by_ratio: Sequence[Mapping[str, Any]]) -> bool:
    if by_ratio:
        return all(any(abs((_float(row.get("ratio", row.get("reduction_rate"))) or -1.0) - ratio) <= 0.0005 and (_float(row.get("success_count")) or 0.0) >= 5 and _finite(row.get("test_micro_f1_mean")) for row in by_ratio) for ratio in REQUIRED_FREEHGC_RATIOS)
    ready = [row for row in runs if _task_ready(row)]
    return REQUIRED_FREEHGC_RATIOS.issubset({_round(row.get("ratio", row.get("reduction_rate"))) for row in ready}) and len({str(row.get("seed")) for row in ready}) >= 5


def _freehgc_tp_ready(rows: Sequence[Mapping[str, Any]], token: str) -> bool:
    return any(token in _method(row).lower() and _task_ready(row) and _bool(row.get("official_hgb_exported")) for row in rows)


def _freehgc_hard_proof(*groups: Sequence[Mapping[str, Any]]) -> bool:
    for rows in groups:
        for row in rows:
            reason = str(row.get("failure_reason", row.get("hard_failure_reason", ""))).strip()
            if reason and (_bool(row.get("hard_failure", row.get("hard_incompatibility"))) or str(row.get("failure_type", "")) == "hard_incompatibility"):
                return True
    return False


def _metapath_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {"full", "APV12", "APV16"}
    return all(
        any(token.lower() in _method(row).lower() and _bool(row.get("real_tensor_dumped")) and _real_hash(row.get("feature_tensor_hash")) and _positive(row.get("feature_tensor_bytes")) for row in rows)
        for token in required
    )


def _feature_ablation_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        any(
            "APV12" in _method(row)
            and str(row.get("feature_transform", "")).lower() == transform.lower()
            and _task_ready(row)
            and _bool(row.get("shape_safe_pass", row.get("feature_shape_safe", True)))
            for row in rows
        )
        for transform in REQUIRED_FEATURE_TRANSFORMS
    )


def _apv16_adapter_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        any("APV16" in str(row.get("base_method", row.get("method", ""))) and adapter in str(row.get("adapter_method", row.get("adapter_variant", ""))) and _task_ready(row) for row in rows)
        for adapter in REQUIRED_APV16_ADAPTERS
    )


def _system_cost_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        any(token.lower() in _method(row).lower() and _task_ready(row) and _positive(row.get("official_preprocess_time_seconds", row.get("official_sehgnn_preprocess_time_seconds"))) and _positive(row.get("training_time_seconds")) and _positive(row.get("peak_cpu_rss_mb")) and _positive(row.get("preprocessed_cache_bytes")) for row in rows)
        for token in REQUIRED_SYSTEM_METHOD_TOKENS
    )


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]], dataset: str) -> bool:
    return any(str(row.get("dataset", "")).upper() == dataset and "HeSF-RCS-auto" in _method(row) and _task_ready(row) for row in rows)


def _task_ready(row: Mapping[str, Any]) -> bool:
    return (
        _bool(row.get("training_executed"))
        and _bool(row.get("success", True))
        and _finite(row.get("test_micro_f1", row.get("test_micro_f1_mean")))
        and _finite(row.get("test_macro_f1", row.get("test_macro_f1_mean")))
    )


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("base_method", "")))


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


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _round(value: Any) -> float:
    parsed = _float(value)
    return -1.0 if parsed is None else round(parsed, 3)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ready"}
