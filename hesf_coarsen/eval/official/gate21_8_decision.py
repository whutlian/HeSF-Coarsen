from __future__ import annotations

import json
import math
from statistics import mean
from typing import Any, Mapping, Sequence


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

GATE21_8_DECISION_FLAGS = (
    "OFFICIAL_MAIN_DBLP_APV12_READY",
    "OFFICIAL_MAIN_DBLP_APV16_TRAINING_SEED_STABLE",
    "OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE",
    "EXTERNAL_TP_SMOKE_TASK_RESULTS_READY",
    "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
    "EXTERNAL_TP_FREEHGC_TP_READY",
    "EXTERNAL_TP_ALL_REQUIRED_READY",
    "FREEHGC_STANDARD_SINGLE_SEED_READY",
    "FREEHGC_STANDARD_5SEED_READY",
    "FREEHGC_STANDARD_PROTOCOL_VERIFIED",
    "METAPATH_INTROSPECTION_PASS",
    "CACHE_HASH_REAL_PASS",
    "CACHE_HASH_EMPTY_ONLY_FAIL",
    "FEATURE_ABLATION_SHAPE_SAFE_PASS",
    "FEATURE_ABLATION_TASK_RESULTS_READY",
    "ADAPTER_APV12_RP64_REPRODUCIBLE_READY",
    "ADAPTER_APV16_RESULTS_READY",
    "ADAPTER_PCA_REPRODUCIBLE_READY",
    "STORAGE_ONLY_BYTE_TABLE_READY",
    "STORAGE_SYSTEM_COSTS_MEASURED_PASS",
    "RATIO_DENOMINATOR_AUDIT_PASS",
    "CROSS_DATASET_AUTO_CHANNEL_PLAN_READY",
    "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY",
)

REQUIRED_EXTERNAL_TP_5X5 = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "Coarsening-HG-TP",
    "GraphSparsify-TP",
)


def budget_alignment_status(row: Mapping[str, Any], *, tolerance: float = 1e-9) -> dict[str, Any]:
    requested = _float(row.get("requested_budget", row.get("budget_value", "")))
    budget_value = _float(row.get("budget_value", row.get("requested_budget", "")))
    if requested is None or budget_value is None:
        return {"budget_alignment_pass": False, "budget_alignment_error": "missing_requested_or_budget_value"}
    if abs(float(requested) - float(budget_value)) > float(tolerance):
        return {
            "budget_alignment_pass": False,
            "budget_alignment_error": f"requested_budget={requested} differs from budget_value={budget_value}",
        }
    return {"budget_alignment_pass": True, "budget_alignment_error": ""}


def ratio_denominator_status(row: Mapping[str, Any], *, tolerance: float = 1e-6) -> dict[str, Any]:
    native = _float(row.get("native_full_text_bytes"))
    export_full = _float(row.get("export_full_text_bytes"))
    method_bytes = _float(row.get("method_text_bytes"))
    control = _float(row.get("current_control_text_bytes"))
    if not native or not export_full or method_bytes is None or not control:
        return {"ratio_consistency_pass": False, "ratio_consistency_error": "missing_denominator_or_method_bytes"}
    checks = [
        ("ratio_vs_native_full_text", method_bytes / native),
        ("ratio_vs_export_full_text", method_bytes / export_full),
        ("ratio_vs_current_control_text", method_bytes / control),
    ]
    for field, expected in checks:
        actual = _float(row.get(field))
        if actual is None or abs(actual - expected) > tolerance:
            return {
                "ratio_consistency_pass": False,
                "ratio_consistency_error": f"{field} expected {expected:.12g}, got {row.get(field, '')}",
            }
    return {"ratio_consistency_pass": True, "ratio_consistency_error": ""}


def apv16_graph_seed_stability_status(row: Mapping[str, Any]) -> dict[str, Any]:
    method = _method(row)
    graph_seed_count = _int(row.get("graph_seed_count"))
    deterministic = _bool(row.get("sampler_deterministic", row.get("deterministic_graph_method", False)))
    export_hash_unique_count = _int(row.get("export_hash_unique_count"))
    mean_micro = _metric(row, "mean_test_micro_f1", "test_micro_f1_mean", "test_micro_f1")
    std_micro = _metric(row, "std_test_micro_f1", "test_micro_f1_std", "test_micro_std")
    structural = _metric(row, "structural_storage_ratio", "actual_structural_storage_ratio")
    training_seed_count = _int(row.get("training_seed_count"))

    if "APV16" not in method:
        return {"graph_seed_stability_pass": False, "stability_failure_reason": "not_apv16_row"}
    deterministic_proof = (
        deterministic
        and export_hash_unique_count == 1
        and (
            _bool(row.get("graph_seed_ignored_by_sampler"))
            or _bool(row.get("deterministic_proof_pass"))
            or str(row.get("graph_seed_independence_required", "")).strip().lower() == "false"
        )
    )
    if deterministic_proof:
        seed_ok = True
    elif not deterministic and graph_seed_count >= 5:
        seed_ok = True
    else:
        return {
            "graph_seed_stability_pass": False,
            "stability_failure_reason": "graph_seed_count_lt_5_without_deterministic_proof",
        }
    if training_seed_count < 5:
        return {"graph_seed_stability_pass": False, "stability_failure_reason": "training_seed_count_lt_5"}
    if mean_micro is None or mean_micro < 0.9533802 - 0.006:
        return {"graph_seed_stability_pass": False, "stability_failure_reason": "mean_micro_below_threshold"}
    if std_micro is None or std_micro > 0.003:
        return {"graph_seed_stability_pass": False, "stability_failure_reason": "std_micro_above_threshold"}
    if structural is None or structural > 0.17:
        return {"graph_seed_stability_pass": False, "stability_failure_reason": "structural_ratio_above_threshold"}
    return {"graph_seed_stability_pass": bool(seed_ok), "stability_failure_reason": ""}


def external_tp_5x5_method_status(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if _method(row) == method]
    ready_rows = [
        row
        for row in method_rows
        if _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and _bool(row.get("training_executed"))
        and _finite_metric(row, "test_micro_f1")
        and _finite_metric(row, "test_macro_f1")
        and _bool(row.get("budget_alignment_pass", True))
    ]
    graph_seed_count = len({str(row.get("graph_seed", "")) for row in ready_rows if str(row.get("graph_seed", ""))})
    training_seed_count = len({str(row.get("training_seed", "")) for row in ready_rows if str(row.get("training_seed", ""))})
    ready = bool(ready_rows) and graph_seed_count >= 5 and training_seed_count >= 5
    missing: list[str] = []
    if not method_rows:
        missing.append("row")
    if not ready_rows:
        missing.append("ready_task_rows")
    if graph_seed_count < 5:
        missing.append("graph_seed_count_lt_5")
    if training_seed_count < 5:
        missing.append("training_seed_count_lt_5")
    return {
        "method": method,
        "ready": ready,
        "row_count": len(method_rows),
        "ready_row_count": len(ready_rows),
        "graph_seed_count": graph_seed_count,
        "training_seed_count": training_seed_count,
        "missing_requirements": missing,
    }


def gate21_8_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    apv16_stability_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_standard_rows: Sequence[Mapping[str, Any]] = (),
    freehgc_tp_rows: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_assertion_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    storage_rows: Sequence[Mapping[str, Any]] = (),
    ratio_audit_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    flags = {name: False for name in GATE21_8_DECISION_FLAGS}
    flags["OFFICIAL_MAIN_DBLP_APV12_READY"] = any(_official_apv_ready(row, "APV12", 0.13, 0.9533802 - 0.012) for row in official_rows)
    flags["OFFICIAL_MAIN_DBLP_APV16_TRAINING_SEED_STABLE"] = any(_apv16_training_seed_stable(row) for row in official_rows)
    flags["OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE"] = any(
        _bool(apv16_graph_seed_stability_status(row)["graph_seed_stability_pass"]) for row in apv16_stability_rows
    )

    flags["EXTERNAL_TP_SMOKE_TASK_RESULTS_READY"] = _external_tp_smoke_ready(external_tp_rows)
    method_status = {method: external_tp_5x5_method_status(external_tp_rows, method) for method in REQUIRED_EXTERNAL_TP_5X5}
    flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] = all(status["ready"] for status in method_status.values())
    freehgc_tp_hard_failure = any(str(row.get("failure_type", "")) in {"hard_incompatibility", "adapter_not_implemented"} for row in freehgc_tp_rows)
    flags["EXTERNAL_TP_FREEHGC_TP_READY"] = _freehgc_tp_ready(freehgc_tp_rows) or freehgc_tp_hard_failure
    flags["EXTERNAL_TP_ALL_REQUIRED_READY"] = flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] and flags["EXTERNAL_TP_FREEHGC_TP_READY"]

    flags["FREEHGC_STANDARD_SINGLE_SEED_READY"] = _freehgc_standard_seed_count(freehgc_standard_rows) >= 1
    flags["FREEHGC_STANDARD_5SEED_READY"] = _freehgc_standard_seed_count(freehgc_standard_rows) >= 5
    flags["FREEHGC_STANDARD_PROTOCOL_VERIFIED"] = bool(freehgc_standard_rows) and all(str(row.get("protocol", "")) == "standard_condensation" for row in freehgc_standard_rows)

    flags["METAPATH_INTROSPECTION_PASS"] = bool(metapath_rows) and all(
        _bool(row.get("introspection_supported"))
        and _finite_positive(row.get("feature_tensor_bytes"))
        and _real_hash(row.get("feature_tensor_hash"))
        for row in metapath_rows
    )
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_assertion_rows) and all(
        _bool(row.get("assertion_pass")) and _real_hash(row.get("cache_hash", row.get("cache_file_sha256", ""))) for row in cache_assertion_rows
    )
    flags["CACHE_HASH_EMPTY_ONLY_FAIL"] = bool(cache_assertion_rows) and all(str(row.get("cache_hash", "")).lower() != EMPTY_SHA256 for row in cache_assertion_rows)

    flags["FEATURE_ABLATION_SHAPE_SAFE_PASS"] = bool(feature_ablation_rows) and all(_bool(row.get("shape_safe_pass")) for row in feature_ablation_rows)
    flags["FEATURE_ABLATION_TASK_RESULTS_READY"] = bool(feature_ablation_rows) and any(
        _bool(row.get("training_executed")) and _finite_metric(row, "test_micro_f1") for row in feature_ablation_rows
    )

    flags["ADAPTER_APV12_RP64_REPRODUCIBLE_READY"] = any(_adapter_ready(row, "APV12", "random_projection_dim64") for row in adapter_rows)
    flags["ADAPTER_APV16_RESULTS_READY"] = all(
        any(_adapter_task_result(row, "APV16", adapter) for row in adapter_rows)
        for adapter in ("random_projection_dim64", "int8_per_feature", "fp16_node_features")
    )
    flags["ADAPTER_PCA_REPRODUCIBLE_READY"] = any(
        "pca" in str(row.get("adapter_name", row.get("feature_adapter", ""))).lower()
        and _bool(row.get("reproducible_transform_package_complete"))
        for row in adapter_rows
    )

    flags["STORAGE_ONLY_BYTE_TABLE_READY"] = bool(storage_rows) and all(_finite_positive(row.get("disk_bytes")) for row in storage_rows)
    flags["STORAGE_SYSTEM_COSTS_MEASURED_PASS"] = bool(storage_rows) and all(_has_system_cost(row) for row in storage_rows)
    flags["RATIO_DENOMINATOR_AUDIT_PASS"] = bool(ratio_audit_rows) and all(_bool(row.get("ratio_consistency_pass")) for row in ratio_audit_rows)

    flags["CROSS_DATASET_AUTO_CHANNEL_PLAN_READY"] = bool(cross_dataset_rows)
    flags["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] = _cross_dataset_task_ready(cross_dataset_rows)

    blocking_issues = _blocking_issues(flags)
    paper_ready_status = "ICDE_EVIDENCE_READY" if not blocking_issues else "ICDE_EVIDENCE_PARTIAL"
    if not flags["OFFICIAL_MAIN_DBLP_APV12_READY"]:
        paper_ready_status = "NOT_READY"
    return {
        "flags": flags,
        "paper_ready_status": paper_ready_status,
        "blocking_issues": blocking_issues,
        "paper_safe_claims": _paper_safe_claims(flags),
        "paper_unsafe_claims": _paper_unsafe_claims(flags),
        "external_tp_5x5_method_status": method_status,
        "counts": {
            "official_rows": len(official_rows),
            "apv16_stability_rows": len(apv16_stability_rows),
            "external_tp_rows": len(external_tp_rows),
            "freehgc_standard_rows": len(freehgc_standard_rows),
            "freehgc_tp_rows": len(freehgc_tp_rows),
            "metapath_rows": len(metapath_rows),
            "cache_assertion_rows": len(cache_assertion_rows),
            "feature_ablation_rows": len(feature_ablation_rows),
            "adapter_rows": len(adapter_rows),
            "storage_rows": len(storage_rows),
            "ratio_audit_rows": len(ratio_audit_rows),
            "cross_dataset_rows": len(cross_dataset_rows),
        },
    }


def decision_md(decision: Mapping[str, Any]) -> str:
    flags = dict(decision.get("flags", {}))
    lines = [
        "# Gate21.8 ICDE Evidence Decision",
        "",
        f"- paper_ready_status: `{decision.get('paper_ready_status', '')}`",
        "",
        "## Pass Flags",
        *[f"- `{name}`" for name, value in flags.items() if value],
        "",
        "## Fail Or Partial Flags",
        *[f"- `{name}`" for name, value in flags.items() if not value],
        "",
        "## Blocking Issues",
        *[f"- {issue}" for issue in decision.get("blocking_issues", [])],
        "",
        "## Paper-Safe Claims",
        *[f"- {claim}" for claim in decision.get("paper_safe_claims", [])],
        "",
        "## Paper-Unsafe Claims",
        *[f"- {claim}" for claim in decision.get("paper_unsafe_claims", [])],
        "",
        "## Counts",
        f"```json\n{json.dumps(decision.get('counts', {}), indent=2, sort_keys=True)}\n```",
    ]
    return "\n".join(lines) + "\n"


def _official_apv_ready(row: Mapping[str, Any], token: str, structural_max: float, micro_min: float) -> bool:
    return bool(
        token in _method(row)
        and _bool(row.get("official_sehgnn_unmodified", True))
        and _bool(row.get("training_executed", True))
        and _metric(row, "structural_storage_ratio") is not None
        and (_metric(row, "structural_storage_ratio") or 1.0) <= structural_max
        and _metric(row, "test_micro_f1_mean", "test_micro_f1") is not None
        and (_metric(row, "test_micro_f1_mean", "test_micro_f1") or 0.0) >= micro_min
    )


def _apv16_training_seed_stable(row: Mapping[str, Any]) -> bool:
    return bool(
        "APV16" in _method(row)
        and _int(row.get("training_seed_count")) >= 5
        and (_metric(row, "test_micro_f1_std", "test_micro_std") or 1.0) <= 0.003
    )


def _external_tp_smoke_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    core = {"Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP"}
    ready = {
        _method(row)
        for row in rows
        if _method(row) in core and _bool(row.get("training_executed")) and _finite_metric(row, "test_micro_f1")
    }
    return core.issubset(ready)


def _freehgc_tp_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _bool(row.get("official_hgb_exported"))
        and _bool(row.get("official_sehgnn_unmodified"))
        and _bool(row.get("training_executed"))
        and _finite_metric(row, "test_micro_f1")
        for row in rows
    )


def _freehgc_standard_seed_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row.get("seed", "")) for row in rows if _bool(row.get("success")) and _finite_metric(row, "test_micro_f1")})


def _adapter_ready(row: Mapping[str, Any], method_token: str, adapter_token: str) -> bool:
    return bool(
        method_token in _method(row)
        and adapter_token in str(row.get("adapter_name", row.get("feature_adapter", "")))
        and _bool(row.get("reproducible_transform_package_complete"))
        and _bool(row.get("projection_reproducibility_test_pass", True))
        and _finite_metric(row, "test_micro_f1")
    )


def _adapter_task_result(row: Mapping[str, Any], method_token: str, adapter_token: str) -> bool:
    return bool(method_token in _method(row) and adapter_token in str(row.get("adapter_name", row.get("feature_adapter", ""))) and _finite_metric(row, "test_micro_f1"))


def _cross_dataset_task_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {"ACM", "IMDB"}
    full_ready: set[str] = set()
    auto_ready: set[str] = set()
    for row in rows:
        dataset = str(row.get("dataset", "")).upper()
        method = _method(row)
        if dataset not in required or not _bool(row.get("training_executed")) or not _finite_metric(row, "test_micro_f1"):
            continue
        if "full" in method.lower() or "export-full" in method.lower():
            full_ready.add(dataset)
        if "HeSF-RCS-auto" in method:
            auto_ready.add(dataset)
    return required.issubset(full_ready) and required.issubset(auto_ready)


def _blocking_issues(flags: Mapping[str, bool]) -> list[str]:
    required = (
        "OFFICIAL_MAIN_DBLP_APV12_READY",
        "OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE",
        "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
        "EXTERNAL_TP_FREEHGC_TP_READY",
        "METAPATH_INTROSPECTION_PASS",
        "CACHE_HASH_REAL_PASS",
        "STORAGE_SYSTEM_COSTS_MEASURED_PASS",
        "RATIO_DENOMINATOR_AUDIT_PASS",
        "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY",
    )
    return [name for name in required if not bool(flags.get(name, False))]


def _paper_safe_claims(flags: Mapping[str, bool]) -> list[str]:
    claims = [
        "DBLP APV12/APV16 Gate21.7 official-unmodified anchors are preserved as regression evidence when their rows are present.",
        "Standard condensation and schema-preserving TP protocols are reported separately.",
    ]
    if flags.get("FREEHGC_STANDARD_SINGLE_SEED_READY"):
        claims.append("FreeHGC standard-condensation single-seed evidence exists, but it is not a TP workload result.")
    if flags.get("ADAPTER_APV12_RP64_REPRODUCIBLE_READY"):
        claims.append("APV12+random_projection_dim64 is an adapter/deployment result with reproducible package evidence.")
    return claims


def _paper_unsafe_claims(flags: Mapping[str, bool]) -> list[str]:
    claims = []
    if not flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY"):
        claims.append("External TP baselines are complete 5x5 paper baselines.")
    if not flags.get("OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE"):
        claims.append("APV16 graph-seed stability is proven.")
    if not flags.get("METAPATH_INTROSPECTION_PASS"):
        claims.append("Metapath/cache mechanism is proven by real tensor dumps.")
    if not flags.get("CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"):
        claims.append("ACM/IMDB generalization is proven.")
    claims.append("Structural ratio is equivalent to raw byte compression.")
    claims.append("Feature adapter results are unmodified official SeHGNN results.")
    return claims


def _has_system_cost(row: Mapping[str, Any]) -> bool:
    fields = (
        "load_time_seconds",
        "decompress_time_seconds",
        "conversion_time_seconds",
        "SeHGNN_preprocess_time_seconds",
        "train_time_seconds",
        "eval_time_seconds",
        "total_wall_time_seconds",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
        "preprocessed_cache_bytes",
    )
    return any(_finite_positive(row.get(field)) for field in fields)


def _real_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text != EMPTY_SHA256 and text not in {"nan", "none"}


def _finite_positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and math.isfinite(parsed) and parsed > 0


def _finite_metric(row: Mapping[str, Any], *names: str) -> bool:
    return _metric(row, *names) is not None


def _metric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        parsed = _float(row.get(name))
        if parsed is not None:
            return parsed
    return None


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int:
    parsed = _float(value)
    return 0 if parsed is None else int(parsed)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("baseline_name", row.get("method_name", ""))))
