from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


REQUIRED_EXTERNAL_TP_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "Coarsening-HG-TP",
    "GraphSparsify-TP",
    "FreeHGC-TP",
)

CORE_EXTERNAL_TP_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

GATE21_7_DECISION_FLAGS = (
    "OFFICIAL_MAIN_APV12_PASS",
    "OFFICIAL_MAIN_APV16_PASS",
    "APV16_GRAPH_SEED_STABILITY_PASS",
    "EXTERNAL_TP_ARTIFACT_ESTIMATES_READY",
    "EXTERNAL_TP_TASK_RESULTS_READY",
    "EXTERNAL_TP_FREEHGC_READY",
    "EXTERNAL_TP_ALL_REQUIRED_READY",
    "STANDARD_CONDENSATION_PROTOCOL_CONFIGURED",
    "STANDARD_CONDENSATION_TASK_RESULTS_READY",
    "CROSS_DATASET_AUTO_CHANNEL_PLAN_READY",
    "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY",
    "COVERAGE_TABLE_EMITTED",
    "COVERAGE_SEMANTIC_VALIDATION_PASS",
    "METAPATH_INTROSPECTION_EMITTED",
    "METAPATH_INTROSPECTION_PASS",
    "CACHE_HASH_REAL_PASS",
    "FEATURE_ABLATION_TABLE_EMITTED",
    "FEATURE_ABLATION_SHAPE_SAFE_PASS",
    "FEATURE_ABLATION_LABEL_GRAPH_SETTINGS_READY",
    "ADAPTER_STATIC_PACKAGE_READY",
    "ADAPTER_REPRODUCIBLE_PACKAGE_READY",
    "ADAPTER_PACKAGE_ACCOUNTING_PASS",
    "STORAGE_ONLY_BYTES_READY",
    "STORAGE_ONLY_SYSTEM_COSTS_READY",
    "SYSTEM_RESOURCE_SCHEMA_READY",
    "SYSTEM_RESOURCE_MEASURED_PASS",
    "ICDE_READY_MINIMAL_PASS",
    "ICDE_READY_STRONG_PASS",
)


def external_tp_task_row_status(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return strict Gate21.7 task-result readiness for one external TP row."""

    missing: list[str] = []
    if _bool(row.get("missing_dependency", False)):
        missing.append("missing_dependency")
    if not _bool(row.get("official_hgb_exported", False)):
        missing.append("official_hgb_exported")
    if not _bool(row.get("training_executed", False)):
        missing.append("training_executed")
    if _int(row.get("success_count")) <= 0:
        missing.append("success_count")
    if not _finite_metric(row, "test_micro_f1", "test_micro_f1_mean", "test_micro_mean", "mean_test_micro_f1"):
        missing.append("test_micro_f1")
    if not _finite_metric(row, "test_macro_f1", "test_macro_f1_mean", "test_macro_mean", "mean_test_macro_f1"):
        missing.append("test_macro_f1")

    return {
        "method": _method_name(row),
        "ready": not missing,
        "missing_requirements": missing,
    }


def external_tp_main_eligible(row: Mapping[str, Any]) -> bool:
    """Gate an external TP row before it can enter the official TP main comparison."""

    task_status = external_tp_task_row_status(row)
    return bool(
        task_status["ready"]
        and _bool(row.get("official_sehgnn_unmodified", False))
        and _bool(row.get("schema_compatible", True))
        and _bool(row.get("keeps_all_target_nodes", True))
        and not _bool(row.get("used_test_data", False))
        and not _bool(row.get("uses_test_data_for_transform", False))
        and _bool(row.get("eligible_for_tp_main_comparison", True))
    )


def evaluate_external_tp_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_methods: Sequence[str] = REQUIRED_EXTERNAL_TP_METHODS,
) -> dict[str, Any]:
    """Evaluate strict external TP readiness from by-method or audit rows."""

    rows_by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_method.setdefault(_method_name(row), []).append(row)

    method_status: dict[str, dict[str, Any]] = {}
    for method in required_methods:
        method_rows = rows_by_method.get(str(method), [])
        if not method_rows:
            method_status[str(method)] = {
                "method": str(method),
                "ready": False,
                "missing_requirements": ["row"],
            }
            continue

        statuses = [external_tp_task_row_status(row) for row in method_rows]
        ready_status = next((status for status in statuses if status["ready"]), None)
        if ready_status is not None:
            method_status[str(method)] = ready_status
            continue

        missing = sorted({item for status in statuses for item in status["missing_requirements"]})
        method_status[str(method)] = {
            "method": str(method),
            "ready": False,
            "missing_requirements": missing,
        }

    freehgc_status = method_status.get("FreeHGC-TP")
    core_methods = [method for method in CORE_EXTERNAL_TP_METHODS if method in {str(value) for value in required_methods}]
    task_results_ready = bool(core_methods) and all(method_status[method]["ready"] for method in core_methods)
    freehgc_required = "FreeHGC-TP" in {str(method) for method in required_methods}
    freehgc_ready = bool(freehgc_status and freehgc_status["ready"])
    return {
        "EXTERNAL_TP_TASK_RESULTS_READY": task_results_ready,
        "EXTERNAL_TP_FREEHGC_READY": freehgc_ready,
        "EXTERNAL_TP_ALL_REQUIRED_READY": task_results_ready and (freehgc_ready or not freehgc_required),
        "required_methods": [str(method) for method in required_methods],
        "method_status": method_status,
    }


def gate21_7_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    adapter_rows: Sequence[Mapping[str, Any]] = (),
    external_tp_rows: Sequence[Mapping[str, Any]] = (),
    coverage_assertion_rows: Sequence[Mapping[str, Any]] = (),
    metapath_rows: Sequence[Mapping[str, Any]] = (),
    cache_hash_rows: Sequence[Mapping[str, Any]] = (),
    feature_ablation_rows: Sequence[Mapping[str, Any]] = (),
    storage_rows: Sequence[Mapping[str, Any]] = (),
    system_resource_rows: Sequence[Mapping[str, Any]] = (),
    cross_dataset_rows: Sequence[Mapping[str, Any]] = (),
    full_micro: float = 0.9533802,
    full_macro: float = 0.9498198,
) -> dict[str, Any]:
    """Build Gate21.7 decision flags with conservative false defaults."""

    flags = {name: False for name in GATE21_7_DECISION_FLAGS}
    flags["OFFICIAL_MAIN_APV12_PASS"] = any(_official_apv_pass(row, "APV12", 0.13, full_micro - 0.010, full_macro - 0.012) for row in official_rows)
    flags["OFFICIAL_MAIN_APV16_PASS"] = any(_official_apv_pass(row, "APV16", 0.17, 0.9480, full_macro - 0.006) for row in official_rows)
    flags["APV16_GRAPH_SEED_STABILITY_PASS"] = _apv16_stability_pass(official_rows)

    external = evaluate_external_tp_readiness(external_tp_rows)
    flags.update({name: bool(external[name]) for name in ("EXTERNAL_TP_TASK_RESULTS_READY", "EXTERNAL_TP_FREEHGC_READY", "EXTERNAL_TP_ALL_REQUIRED_READY")})
    flags["EXTERNAL_TP_ARTIFACT_ESTIMATES_READY"] = bool(external_tp_rows) and all(
        _bool(row.get("official_hgb_exported", False)) for row in external_tp_rows
    )

    flags["COVERAGE_TABLE_EMITTED"] = bool(coverage_assertion_rows)
    flags["COVERAGE_SEMANTIC_VALIDATION_PASS"] = bool(coverage_assertion_rows) and all(_bool(row.get("pass", row.get("assertion_pass", False))) for row in coverage_assertion_rows)
    flags["METAPATH_INTROSPECTION_EMITTED"] = bool(metapath_rows)
    flags["METAPATH_INTROSPECTION_PASS"] = bool(metapath_rows) and any(_bool(row.get("real_tensor_dumped", row.get("tensor_key_dumped", False))) for row in metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = _cache_hash_real_pass(cache_hash_rows)

    flags["FEATURE_ABLATION_TABLE_EMITTED"] = bool(feature_ablation_rows)
    flags["FEATURE_ABLATION_SHAPE_SAFE_PASS"] = bool(feature_ablation_rows) and all(_bool(row.get("shape_safe_pass", row.get("feature_ablation_shape_safe_pass", False))) for row in feature_ablation_rows)
    flags["FEATURE_ABLATION_LABEL_GRAPH_SETTINGS_READY"] = bool(feature_ablation_rows) and all(
        _bool(row.get("label_graph_setting_ready", True)) or str(row.get("failure_type", "")) == "unsupported_by_official_pipeline"
        for row in feature_ablation_rows
    )

    adapter_flags = _adapter_package_flags(adapter_rows)
    flags.update(adapter_flags)

    flags["STORAGE_ONLY_BYTES_READY"] = bool(storage_rows) and all(_int(row.get("disk_bytes")) > 0 for row in storage_rows)
    flags["STORAGE_ONLY_SYSTEM_COSTS_READY"] = bool(storage_rows) and all(
        _has_measured_system_cost(row) or bool(str(row.get("failure_message", "")).strip()) for row in storage_rows
    )
    flags["SYSTEM_RESOURCE_SCHEMA_READY"] = bool(system_resource_rows)
    flags["SYSTEM_RESOURCE_MEASURED_PASS"] = bool(system_resource_rows) and any(_has_measured_system_cost(row) for row in system_resource_rows)

    flags["CROSS_DATASET_AUTO_CHANNEL_PLAN_READY"] = any(_bool(row.get("plan_ready", False)) for row in cross_dataset_rows)
    flags["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] = _cross_dataset_task_results_ready(cross_dataset_rows)
    flags["STANDARD_CONDENSATION_PROTOCOL_CONFIGURED"] = False
    flags["STANDARD_CONDENSATION_TASK_RESULTS_READY"] = False

    flags["ICDE_READY_MINIMAL_PASS"] = bool(
        flags["OFFICIAL_MAIN_APV12_PASS"]
        and flags["OFFICIAL_MAIN_APV16_PASS"]
        and flags["EXTERNAL_TP_TASK_RESULTS_READY"]
        and flags["COVERAGE_SEMANTIC_VALIDATION_PASS"]
        and flags["FEATURE_ABLATION_SHAPE_SAFE_PASS"]
        and flags["ADAPTER_PACKAGE_ACCOUNTING_PASS"]
    )
    flags["ICDE_READY_STRONG_PASS"] = bool(
        flags["ICDE_READY_MINIMAL_PASS"]
        and flags["EXTERNAL_TP_FREEHGC_READY"]
        and flags["METAPATH_INTROSPECTION_PASS"]
        and flags["CACHE_HASH_REAL_PASS"]
        and flags["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"]
        and flags["STORAGE_ONLY_SYSTEM_COSTS_READY"]
        and flags["SYSTEM_RESOURCE_MEASURED_PASS"]
    )

    return {
        "native_full_micro": float(full_micro),
        "native_full_macro": float(full_macro),
        "flags": flags,
        "external_tp": external,
        "decisions": [name for name, value in flags.items() if value],
        "failures": [name for name, value in flags.items() if not value],
        "counts": {
            "official_rows": len(official_rows),
            "adapter_rows": len(adapter_rows),
            "external_tp_rows": len(external_tp_rows),
            "coverage_assertion_rows": len(coverage_assertion_rows),
            "metapath_rows": len(metapath_rows),
            "cache_hash_rows": len(cache_hash_rows),
            "feature_ablation_rows": len(feature_ablation_rows),
            "storage_rows": len(storage_rows),
            "system_resource_rows": len(system_resource_rows),
            "cross_dataset_rows": len(cross_dataset_rows),
        },
    }


def decision_md(decision: Mapping[str, Any]) -> str:
    flags = decision.get("flags", {})
    lines = [
        "# Gate21.7 ICDE-Ready Decision",
        "",
        f"- native_full_micro: `{decision.get('native_full_micro', '')}`",
        f"- native_full_macro: `{decision.get('native_full_macro', '')}`",
        "",
        "## Pass Flags",
        *[f"- `{name}`" for name, value in flags.items() if value],
        "",
        "## Fail Or Not Ready Flags",
        *[f"- `{name}`" for name, value in flags.items() if not value],
        "",
        "## Counts",
        f"```json\n{json.dumps(decision.get('counts', {}), indent=2, sort_keys=True)}\n```",
        "",
        "## Claim Boundaries",
        "- External TP READY requires HGB export, executed training, success_count > 0, and finite test metrics.",
        "- FreeHGC missing dependencies are failure rows, not READY rows.",
        "- Adapter rows are separated from official-unmodified main-table rows.",
        "- Static snapshot package accounting is separate from reproducible transform package accounting.",
    ]
    return "\n".join(lines) + "\n"


def _method_name(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("baseline_name", row.get("method_name", ""))))


def _official_apv_pass(row: Mapping[str, Any], token: str, structural_threshold: float, micro_threshold: float, macro_threshold: float) -> bool:
    method = _method_name(row)
    return bool(
        token in method
        and _official_main_eligible(row)
        and _le(_metric(row, "structural_storage_ratio", "semantic_structural_storage_ratio"), structural_threshold)
        and _ge(_metric(row, "test_micro_f1", "test_micro_f1_mean", "test_micro_mean", "mean_test_micro_f1"), micro_threshold)
        and _ge(_metric(row, "test_macro_f1", "test_macro_f1_mean", "test_macro_mean", "mean_test_macro_f1"), macro_threshold)
    )


def _official_main_eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        _bool(row.get("schema_compatible", True))
        and _bool(row.get("official_sehgnn_unmodified", False))
        and _bool(row.get("training_executed", True))
        and not _bool(row.get("uses_feature_adapter", False))
        and not _bool(row.get("uses_weighted_superedges", False))
        and not _bool(row.get("used_test_data", False))
        and not _bool(row.get("uses_test_data_for_transform", False))
        and _bool(row.get("eligible_for_official_main_table", row.get("eligible_for_main_decision", False)))
    )


def _apv16_stability_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if "APV16" not in _method_name(row):
            continue
        deterministic = _bool(row.get("deterministic_graph_method", False))
        graph_seed_count = _int(row.get("graph_seed_count"))
        graph_seed_ok = deterministic or graph_seed_count >= 5
        if (
            graph_seed_ok
            and _ge(_metric(row, "test_micro_f1_mean", "test_micro_mean", "mean_test_micro_f1", "test_micro_f1"), 0.9480)
            and _le(_metric(row, "test_micro_f1_std", "test_micro_std", "std_test_micro_f1"), 0.0030)
            and _le(_metric(row, "structural_storage_ratio", "semantic_structural_storage_ratio"), 0.17)
        ):
            return True
    return False


def _adapter_package_flags(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    if not rows:
        return {
            "ADAPTER_STATIC_PACKAGE_READY": False,
            "ADAPTER_REPRODUCIBLE_PACKAGE_READY": False,
            "ADAPTER_PACKAGE_ACCOUNTING_PASS": False,
        }

    by_adapter = {str(row.get("feature_adapter", row.get("adapter", row.get("method", "")))): row for row in rows}
    static_ready = all(
        _bool(row.get("static_snapshot_package_complete", False))
        for adapter, row in by_adapter.items()
        if "random_projection_dim64" in adapter or "int8" in adapter
    )
    static_ready = static_ready and any("random_projection_dim64" in adapter for adapter in by_adapter) and any("int8" in adapter for adapter in by_adapter)
    reproducible_ready = bool(rows) and all(
        _bool(row.get("reproducible_transform_package_complete", False)) for row in rows if _bool(row.get("eligible_for_adapter_table", True))
    )
    accounting_pass = bool(rows) and all(
        (
            _bool(row.get("static_snapshot_package_complete", False))
            and _bool(row.get("reproducible_transform_package_complete", False))
        )
        or bool(str(row.get("missing_reason", row.get("failure_message", ""))).strip())
        for row in rows
    )
    return {
        "ADAPTER_STATIC_PACKAGE_READY": static_ready,
        "ADAPTER_REPRODUCIBLE_PACKAGE_READY": reproducible_ready,
        "ADAPTER_PACKAGE_ACCOUNTING_PASS": accounting_pass,
    }


def _cache_hash_real_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        cache_hash = str(row.get("preprocess_cache_hash_after", row.get("cache_hash", ""))).strip().lower()
        if not cache_hash or cache_hash == EMPTY_SHA256:
            return False
        if not _bool(row.get("hash_changes_when_link_dat_changes", row.get("perturbation_hash_changed", False))):
            return False
    return True


def _cross_dataset_task_results_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready_datasets: set[str] = set()
    full_reference: set[str] = set()
    for row in rows:
        dataset = str(row.get("dataset", ""))
        method = _method_name(row)
        trained = _bool(row.get("training_executed", False))
        metrics_ready = _finite_metric(row, "test_micro_f1", "test_micro_f1_mean") and _finite_metric(row, "test_macro_f1", "test_macro_f1_mean")
        if dataset in {"ACM", "IMDB"} and trained and metrics_ready:
            if "full" in method.lower():
                full_reference.add(dataset)
            if "HeSF-RCS-auto" in method:
                ready_datasets.add(dataset)
    return {"ACM", "IMDB"}.issubset(ready_datasets) and {"ACM", "IMDB"}.issubset(full_reference)


def _has_measured_system_cost(row: Mapping[str, Any]) -> bool:
    fields = (
        "wall_time_seconds",
        "cpu_time_seconds",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
        "load_wall_time_seconds",
        "preprocess_wall_time_seconds",
        "train_wall_time_seconds",
    )
    return any((_metric(row, field) or 0.0) > 0.0 for field in fields)


def _finite_metric(row: Mapping[str, Any], *names: str) -> bool:
    value = _metric(row, *names)
    return value is not None and math.isfinite(value)


def _metric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    if value in {"", None}:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _le(value: float | None, threshold: float) -> bool:
    return value is not None and math.isfinite(value) and value <= threshold


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and math.isfinite(value) and value >= threshold
