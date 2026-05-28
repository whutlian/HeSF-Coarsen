from __future__ import annotations

import json
import math
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

REQUIRED_EXTERNAL_TP_5X5 = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
)

GATE21_9_DECISION_FLAGS = (
    "OFFICIAL_MAIN_DBLP_APV12_READY",
    "OFFICIAL_MAIN_DBLP_APV16_READY",
    "OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE",
    "AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS",
    "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
    "FREEHGC_TP_ADAPTER_IMPLEMENTED",
    "FREEHGC_TP_TASK_RESULTS_READY",
    "FREEHGC_TP_HARD_GAP_REPORTED",
    "FREEHGC_STANDARD_SINGLE_SEED_RESULTS_READY",
    "FREEHGC_STANDARD_5SEED_RESULTS_READY",
    "FREEHGC_UPSTREAM_ENV_VERIFIED",
    "FREEHGC_STANDARD_PROTOCOL_CONFIG_VERIFIED",
    "METAPATH_INTROSPECTION_PASS",
    "CACHE_HASH_REAL_PASS",
    "FEATURE_ABLATION_TASK_RESULTS_READY",
    "ADAPTER_APV12_RP64_READY",
    "ADAPTER_APV16_RP64_READY",
    "ADAPTER_PACKAGE_SEMANTICS_PASS",
    "STORAGE_BYTE_TABLE_READY",
    "STORAGE_WORKLOAD_COSTS_MEASURED_PASS",
    "RATIO_DENOMINATOR_AUDIT_V2_PASS",
    "CROSS_DATASET_AUTO_CHANNEL_PLAN_READY",
    "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY",
    "COVERAGE_REACHABILITY_PASS",
    "COVERAGE_SEMANTIC_DIAGNOSTICS_READY",
)

FREEHGC_TP_HARD_GAP_REASONS = {
    "synthetic_support_nodes_without_hgb_identity",
    "edge_provenance_missing",
    "relation_schema_not_preserved",
    "feature_table_incompatible",
    "label_test_identity_incompatible",
    "freehgc_output_not_exportable_to_official_hgb",
}

KEY_FEATURE_TRANSFORMS = {
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-random-projection64",
}


def apv16_graph_seed_status(row: Mapping[str, Any]) -> dict[str, Any]:
    deterministic = _bool(row.get("sampler_deterministic"))
    seed_ignored = _bool(row.get("graph_seed_ignored_by_design", row.get("graph_seed_ignored_by_sampler")))
    deterministic_hash_pass = _bool(row.get("deterministic_export_hash_unit_test_pass", row.get("deterministic_proof_pass")))
    empirical_count = _int(row.get("empirical_graph_seed_count", row.get("graph_seed_count")))
    empirical_pass = _bool(row.get("empirical_graph_seed_stability_pass", row.get("graph_seed_stability_pass")))
    if deterministic and seed_ignored and deterministic_hash_pass:
        mode = "deterministic_proof"
        ok = True
    elif empirical_count >= 5 and empirical_pass:
        mode = "empirical_5x5"
        ok = True
    else:
        mode = "not_validated"
        ok = False
    return {
        "sampler_deterministic": deterministic,
        "graph_seed_ignored_by_design": seed_ignored,
        "deterministic_export_hash_unit_test_pass": deterministic_hash_pass,
        "empirical_graph_seed_count": empirical_count,
        "empirical_graph_seed_stability_pass": empirical_pass,
        "apv16_stability_mode": mode,
        "graph_seed_stability_pass": ok,
    }


def external_tp_method_status(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if _method(row) == method]
    ready_rows = [row for row in method_rows if _ready_task_row(row, require_export=True, require_official=True)]
    graph_seeds = {str(row.get("graph_seed", "")) for row in ready_rows if str(row.get("graph_seed", "")).strip()}
    training_seeds = {str(row.get("training_seed", "")) for row in ready_rows if str(row.get("training_seed", "")).strip()}
    deterministic_proof = bool(ready_rows) and all(
        _bool(row.get("sampler_deterministic")) and _bool(row.get("deterministic_export_hash_unit_test_pass", row.get("deterministic_proof_pass")))
        for row in ready_rows
    )
    ready = len(training_seeds) >= 5 and (len(graph_seeds) >= 5 or deterministic_proof)
    missing: list[str] = []
    if not method_rows:
        missing.append("row")
    if not ready_rows:
        missing.append("ready_task_rows")
    if len(graph_seeds) < 5 and not deterministic_proof:
        missing.append("graph_seed_count_lt_5_without_deterministic_proof")
    if len(training_seeds) < 5:
        missing.append("training_seed_count_lt_5")
    return {
        "method": method,
        "row_count": len(method_rows),
        "ready_row_count": len(ready_rows),
        "graph_seed_count": len(graph_seeds),
        "training_seed_count": len(training_seeds),
        "deterministic_proof_pass": deterministic_proof,
        "ready_5x5_flag": ready,
        "ready": ready,
        "missing_requirements": missing,
        "test_micro_f1_mean": _mean_metric(ready_rows, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean"),
        "test_micro_f1_std": _std_metric(ready_rows, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean"),
        "test_macro_f1_mean": _mean_metric(ready_rows, "test_macro_f1", "test_macro_mean", "test_macro_f1_mean"),
        "structural_storage_ratio_mean": _mean_metric(ready_rows, "actual_structural_storage_ratio", "structural_storage_ratio"),
    }


def pca_reproducible_package_complete(row: Mapping[str, Any]) -> bool:
    adapter = str(row.get("feature_adapter", row.get("adapter_name", ""))).lower()
    if "pca" not in adapter:
        return False
    return bool(
        _positive(row.get("pca_basis_bytes"))
        and _positive(row.get("pca_mean_bytes"))
        and str(row.get("pca_fit_config", "")).strip()
        and str(row.get("pca_training_node_ids_hash", "")).strip()
    )


def gate21_9_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]] = (),
    auto_selector_rows: Sequence[Mapping[str, Any]] = (),
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
    coverage_rows: Sequence[Mapping[str, Any]] = (),
    coverage_assertion_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    flags = {name: False for name in GATE21_9_DECISION_FLAGS}

    flags["OFFICIAL_MAIN_DBLP_APV12_READY"] = any(_official_apv_ready(row, "APV12", 0.13, 0.941) for row in official_rows)
    flags["OFFICIAL_MAIN_DBLP_APV16_READY"] = any(_official_apv_ready(row, "APV16", 0.17, 0.944) for row in official_rows)
    flags["OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE"] = any(
        "APV16" in _method(row) and apv16_graph_seed_status(row)["graph_seed_stability_pass"] for row in official_rows
    )

    flags["AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS"] = any(_auto_selector_row_pass(row) for row in auto_selector_rows)

    external_status = {method: external_tp_method_status(external_tp_rows, method) for method in REQUIRED_EXTERNAL_TP_5X5}
    flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] = all(status["ready_5x5_flag"] for status in external_status.values())

    freehgc_tp_ready_rows = [row for row in freehgc_tp_rows if _ready_task_row(row, require_export=True, require_official=True)]
    flags["FREEHGC_TP_TASK_RESULTS_READY"] = bool(freehgc_tp_ready_rows)
    flags["FREEHGC_TP_ADAPTER_IMPLEMENTED"] = bool(freehgc_tp_ready_rows) or any(_bool(row.get("adapter_implemented")) for row in freehgc_tp_rows)
    flags["FREEHGC_TP_HARD_GAP_REPORTED"] = _freehgc_tp_hard_gap_reported(freehgc_tp_rows)

    flags["FREEHGC_STANDARD_SINGLE_SEED_RESULTS_READY"] = _freehgc_standard_seed_count(freehgc_standard_rows) >= 1
    flags["FREEHGC_STANDARD_5SEED_RESULTS_READY"] = _freehgc_standard_seed_count(freehgc_standard_rows) >= 5
    flags["FREEHGC_UPSTREAM_ENV_VERIFIED"] = any(_bool(row.get("upstream_env_verified", row.get("standard_condensation_supported"))) for row in freehgc_standard_rows)
    flags["FREEHGC_STANDARD_PROTOCOL_CONFIG_VERIFIED"] = bool(freehgc_standard_rows) and all(
        str(row.get("protocol", "standard_condensation")) == "standard_condensation" for row in freehgc_standard_rows
    )

    flags["METAPATH_INTROSPECTION_PASS"] = bool(metapath_rows) and all(_metapath_row_real(row) for row in metapath_rows)
    flags["CACHE_HASH_REAL_PASS"] = bool(cache_assertion_rows) and all(_cache_assertion_real(row) for row in cache_assertion_rows)
    flags["FEATURE_ABLATION_TASK_RESULTS_READY"] = _feature_ablation_ready(feature_ablation_rows)

    flags["ADAPTER_APV12_RP64_READY"] = any(_rp64_ready(row, "APV12") for row in adapter_rows)
    flags["ADAPTER_APV16_RP64_READY"] = any(_rp64_ready(row, "APV16") for row in adapter_rows)
    flags["ADAPTER_PACKAGE_SEMANTICS_PASS"] = bool(adapter_rows) and all(_adapter_package_semantics_row(row) for row in adapter_rows)

    flags["STORAGE_BYTE_TABLE_READY"] = bool(storage_rows) and all(_storage_byte_row(row) for row in storage_rows)
    flags["STORAGE_WORKLOAD_COSTS_MEASURED_PASS"] = bool(storage_rows) and all(_storage_workload_measured(row) for row in storage_rows)
    flags["RATIO_DENOMINATOR_AUDIT_V2_PASS"] = bool(ratio_audit_rows) and all(_bool(row.get("ratio_denominator_audit_v2_pass")) for row in ratio_audit_rows)

    flags["CROSS_DATASET_AUTO_CHANNEL_PLAN_READY"] = bool(cross_dataset_rows)
    flags["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] = _cross_dataset_ready(cross_dataset_rows)

    flags["COVERAGE_REACHABILITY_PASS"] = _coverage_reachability_pass(coverage_assertion_rows)
    flags["COVERAGE_SEMANTIC_DIAGNOSTICS_READY"] = _coverage_semantic_ready(coverage_rows, coverage_assertion_rows)

    blocking_issues = _blocking_issues(flags)
    if not blocking_issues:
        paper_ready_status = "ICDE_READY_CANDIDATE"
    elif (
        flags["AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS"]
        and flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"]
        and (flags["FREEHGC_TP_TASK_RESULTS_READY"] or flags["FREEHGC_TP_HARD_GAP_REPORTED"])
    ):
        paper_ready_status = "ICDE_EVIDENCE_PARTIAL_PLUS_EXTERNAL_BASELINES"
    elif flags["OFFICIAL_MAIN_DBLP_APV12_READY"] or flags["OFFICIAL_MAIN_DBLP_APV16_READY"] or flags["AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS"]:
        paper_ready_status = "ICDE_EVIDENCE_PARTIAL"
    else:
        paper_ready_status = "NOT_READY"

    return {
        "paper_ready_status": paper_ready_status,
        "flags": flags,
        "blocking_issues": blocking_issues,
        "paper_safe_claims": _paper_safe_claims(flags),
        "paper_unsafe_claims": _paper_unsafe_claims(flags),
        "method_status": {
            "official_apv12_ready_rows": sum(1 for row in official_rows if _official_apv_ready(row, "APV12", 0.13, 0.941)),
            "official_apv16_ready_rows": sum(1 for row in official_rows if _official_apv_ready(row, "APV16", 0.17, 0.944)),
            "auto_selector_rows": len(auto_selector_rows),
        },
        "external_baseline_status": external_status,
        "cross_dataset_status": _cross_dataset_status(cross_dataset_rows),
        "adapter_status": _adapter_status(adapter_rows),
        "mechanism_audit_status": {
            "metapath_rows": len(metapath_rows),
            "cache_assertion_rows": len(cache_assertion_rows),
            "coverage_rows": len(coverage_rows),
            "coverage_assertion_rows": len(coverage_assertion_rows),
        },
        "system_cost_status": {
            "storage_rows": len(storage_rows),
            "ratio_audit_rows": len(ratio_audit_rows),
            "workload_measured_rows": sum(1 for row in storage_rows if _storage_workload_measured(row)),
        },
        "counts": {
            "official_rows": len(official_rows),
            "auto_selector_rows": len(auto_selector_rows),
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
            "coverage_rows": len(coverage_rows),
            "coverage_assertion_rows": len(coverage_assertion_rows),
        },
    }


def decision_md(decision: Mapping[str, Any]) -> str:
    flags = dict(decision.get("flags", {}))
    lines = [
        "# Gate21.9 ICDE Evidence Decision",
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
        *[f"- `{issue}`" for issue in decision.get("blocking_issues", [])],
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
    method = _method(row)
    return bool(
        token in method
        and _bool(row.get("schema_compatible", True))
        and _bool(row.get("keeps_all_target_nodes", True))
        and _bool(row.get("official_hgb_exported", True))
        and _bool(row.get("official_sehgnn_unmodified", True))
        and _success_ok(row)
        and _metric(row, "structural_storage_ratio", "actual_structural_storage_ratio") is not None
        and (_metric(row, "structural_storage_ratio", "actual_structural_storage_ratio") or 1.0) <= structural_max
        and _metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") is not None
        and (_metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") or 0.0) >= micro_min
    )


def _auto_selector_row_pass(row: Mapping[str, Any]) -> bool:
    if _bool(row.get("AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS")):
        return True
    ap = _metric(row, "AP_keep", "AP_selected_keep_ratio")
    pv = _metric(row, "PV_keep", "PV_selected_keep_ratio")
    pt = _metric(row, "PT_keep", "PT_selected_keep_ratio")
    tp = _metric(row, "TP_keep", "TP_selected_keep_ratio")
    structural = _metric(row, "structural_storage_ratio", "actual_structural_storage_ratio")
    micro = _metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean")
    full_micro = _metric(row, "full_micro_f1", "full_test_micro_f1")
    return bool(
        (ap or 0.0) >= 0.90
        and (pv or 0.0) >= 0.90
        and (pt if pt is not None else 1.0) <= 0.05
        and (tp if tp is not None else 1.0) <= 0.05
        and structural is not None
        and structural <= 0.20
        and micro is not None
        and (full_micro is None or micro >= full_micro - 0.015)
    )


def _ready_task_row(row: Mapping[str, Any], *, require_export: bool, require_official: bool) -> bool:
    if not _success_ok(row):
        return False
    if require_export and not _bool(row.get("official_hgb_exported")):
        return False
    if require_official and not _bool(row.get("official_sehgnn_unmodified")):
        return False
    if not _bool(row.get("training_executed", True)):
        return False
    if str(row.get("failure_type", "")).strip():
        return False
    return _metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") is not None and _metric(
        row, "test_macro_f1", "test_macro_mean", "test_macro_f1_mean"
    ) is not None


def _success_ok(row: Mapping[str, Any]) -> bool:
    if "success" not in row:
        return True
    return _bool(row.get("success"))


def _freehgc_tp_hard_gap_reported(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if str(row.get("failure_type", "")).strip() != "hard_incompatibility":
            continue
        reason = str(row.get("hard_incompatibility_reason", "")).strip()
        message = str(row.get("failure_message", "")).strip()
        if reason in FREEHGC_TP_HARD_GAP_REASONS and message:
            return True
    return False


def _freehgc_standard_seed_count(rows: Sequence[Mapping[str, Any]]) -> int:
    ready = [row for row in rows if _success_ok(row) and _metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") is not None]
    return len({str(row.get("seed", row.get("training_seed", ""))) for row in ready if str(row.get("seed", row.get("training_seed", ""))).strip()})


def _metapath_row_real(row: Mapping[str, Any]) -> bool:
    return bool(
        _real_hash(row.get("feature_tensor_hash", row.get("tensor_sha256", "")))
        and _positive(row.get("feature_tensor_bytes", row.get("tensor_bytes", 0)))
        and str(row.get("metapath_key", row.get("tensor_key", "x"))).strip()
    )


def _cache_assertion_real(row: Mapping[str, Any]) -> bool:
    value = row.get("cache_file_hash", row.get("cache_hash", row.get("cache_file_sha256", "")))
    return _bool(row.get("assertion_pass")) and _real_hash(value)


def _feature_ablation_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    seen: dict[str, set[str]] = {"APV12": set(), "APV16": set()}
    for row in rows:
        method = _method(row)
        transform = _feature_transform(row)
        if transform not in KEY_FEATURE_TRANSFORMS or not _ready_task_row(row, require_export=False, require_official=False):
            continue
        for token in seen:
            if token in method:
                seen[token].add(transform)
    return all(KEY_FEATURE_TRANSFORMS.issubset(values) for values in seen.values())


def _feature_transform(row: Mapping[str, Any]) -> str:
    raw = str(row.get("feature_transform", row.get("feature_setting", ""))).strip()
    aliases = {
        "zero-paper": "zero-paper-preserve-dim",
        "zero-term": "zero-term-preserve-dim",
        "zero-all-support": "zero-all-support-preserve-dim",
        "paper-rp64": "paper-random-projection64",
        "random_projection_dim64": "paper-random-projection64",
    }
    return aliases.get(raw, raw)


def _rp64_ready(row: Mapping[str, Any], method_token: str) -> bool:
    method_text = f"{_method(row)} {row.get('base_graph_method', '')} {row.get('canonical_base_graph_method', '')}"
    adapter = str(row.get("feature_adapter", row.get("adapter_name", "")))
    return bool(
        method_token in method_text
        and "random_projection_dim64" in adapter
        and _metric(row, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") is not None
        and _adapter_package_semantics_row(row)
        and _rp_metadata_complete(row)
    )


def _adapter_package_semantics_row(row: Mapping[str, Any]) -> bool:
    return all(
        _metric(row, field) is not None
        for field in ("static_inference_package_ratio", "transform_recipe_package_ratio", "reconstructable_package_ratio")
    )


def _rp_metadata_complete(row: Mapping[str, Any]) -> bool:
    adapter = str(row.get("feature_adapter", row.get("adapter_name", "")))
    if "random_projection" not in adapter:
        return True
    required = (
        "projection_generator_name",
        "projection_generator_version",
        "projection_matrix_shape",
        "projection_matrix_dtype",
        "projection_distribution",
    )
    return all(str(row.get(field, "")).strip() for field in required)


def _storage_byte_row(row: Mapping[str, Any]) -> bool:
    return any(_positive(row.get(field)) for field in ("raw_hgb_text_bytes", "static_inference_package_bytes", "total_artifact_bytes", "disk_bytes"))


def _storage_workload_measured(row: Mapping[str, Any]) -> bool:
    required_any = (
        "official_sehgnn_preprocess_time_seconds",
        "preprocess_time_seconds",
        "training_time_seconds",
        "train_time_seconds",
        "eval_time_seconds",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
        "preprocessed_cache_bytes",
    )
    return all(
        any(_positive(row.get(field)) for field in group)
        for group in (
            ("load_time_seconds", "load_wall_time_seconds"),
            ("official_sehgnn_preprocess_time_seconds", "preprocess_time_seconds"),
            ("training_time_seconds", "train_time_seconds"),
            ("peak_cpu_rss_mb", "peak_cpu_memory_mb"),
            ("preprocessed_cache_bytes",),
        )
    ) and any(_positive(row.get(field)) for field in required_any)


def _cross_dataset_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    status = _cross_dataset_status(rows)
    return all(item["ready"] for item in status.values())


def _cross_dataset_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for dataset in ("ACM", "IMDB"):
        ready_methods: set[str] = set()
        for row in rows:
            if str(row.get("dataset", "")).upper() != dataset or not _ready_task_row(row, require_export=False, require_official=False):
                continue
            method = _method(row)
            if method == "full-native-SeHGNN":
                ready_methods.add("full-native-SeHGNN")
            if method == "export-full-SeHGNN":
                ready_methods.add("export-full-SeHGNN")
            if method == "H6-node30":
                ready_methods.add("H6-node30")
            if "HeSF-RCS-auto structural30" in method or "HeSF-RCS-auto structural20" in method:
                ready_methods.add("HeSF-RCS-auto")
        required = {"full-native-SeHGNN", "export-full-SeHGNN", "H6-node30", "HeSF-RCS-auto"}
        status[dataset] = {
            "ready": required.issubset(ready_methods),
            "ready_methods": sorted(ready_methods),
            "missing_methods": sorted(required - ready_methods),
        }
    return status


def _coverage_reachability_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    return all(_bool(row.get("assertion_pass")) for row in rows if "semantic" not in str(row.get("assertion", "")).lower())


def _coverage_semantic_ready(coverage_rows: Sequence[Mapping[str, Any]], assertion_rows: Sequence[Mapping[str, Any]]) -> bool:
    semantic_fields = (
        "per_class_venue_coverage",
        "paper_venue_entropy",
        "venue_class_proxy_purity_trainval",
        "paper_class_proxy_purity_trainval",
        "edge_jaccard_across_graph_seeds",
    )
    has_semantic_rows = bool(coverage_rows) and any(any(str(row.get(field, "")).strip() for field in semantic_fields) for row in coverage_rows)
    semantic_assertions = [row for row in assertion_rows if "semantic" in str(row.get("assertion", "")).lower()]
    return has_semantic_rows and bool(semantic_assertions) and all(_bool(row.get("assertion_pass")) for row in semantic_assertions)


def _adapter_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "apv12_rp64_rows": sum(1 for row in rows if _rp64_ready(row, "APV12")),
        "apv16_rp64_rows": sum(1 for row in rows if _rp64_ready(row, "APV16")),
        "pca_reproducible_rows": sum(1 for row in rows if pca_reproducible_package_complete(row)),
    }


def _blocking_issues(flags: Mapping[str, bool]) -> list[str]:
    mandatory = (
        "AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS",
        "EXTERNAL_TP_5X5_TASK_RESULTS_READY",
        "FREEHGC_STANDARD_5SEED_RESULTS_READY",
        "METAPATH_INTROSPECTION_PASS",
        "CACHE_HASH_REAL_PASS",
        "COVERAGE_SEMANTIC_DIAGNOSTICS_READY",
        "FEATURE_ABLATION_TASK_RESULTS_READY",
        "STORAGE_WORKLOAD_COSTS_MEASURED_PASS",
        "RATIO_DENOMINATOR_AUDIT_V2_PASS",
        "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY",
    )
    blockers = [name for name in mandatory if not flags.get(name, False)]
    if not (flags.get("FREEHGC_TP_TASK_RESULTS_READY") or flags.get("FREEHGC_TP_HARD_GAP_REPORTED")):
        blockers.append("FREEHGC_TP_TASK_RESULTS_READY_OR_HARD_GAP_REPORTED")
    return blockers


def _paper_safe_claims(flags: Mapping[str, bool]) -> list[str]:
    claims = [
        "Standard condensation and schema-preserving TP workload evidence are separated.",
        "Feature adapter rows are separated from official-unmodified SeHGNN rows.",
    ]
    if flags.get("AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS"):
        claims.append("DBLP validation-only auto selector recovers the AP/PV skeleton and suppresses PT/TP under the configured evidence.")
    if flags.get("FREEHGC_TP_HARD_GAP_REPORTED"):
        claims.append("FreeHGC-TP remains a hard compatibility gap with a concrete reason, not a completed TP result.")
    if flags.get("FREEHGC_STANDARD_SINGLE_SEED_RESULTS_READY"):
        claims.append("FreeHGC standard condensation has at least single-seed evidence and is not a TP workload baseline.")
    return claims


def _paper_unsafe_claims(flags: Mapping[str, bool]) -> list[str]:
    claims: list[str] = []
    if not flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY"):
        claims.append("External TP baselines are complete 5x5 paper baselines.")
    if not flags.get("FREEHGC_TP_TASK_RESULTS_READY"):
        claims.append("HeSF-RCS beats FreeHGC under the same TP protocol.")
    if not flags.get("CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"):
        claims.append("The auto selector generalizes to ACM/IMDB task results.")
    if not flags.get("METAPATH_INTROSPECTION_PASS") or not flags.get("CACHE_HASH_REAL_PASS"):
        claims.append("SeHGNN metapath/cache mechanism is proven by real tensor dumps.")
    claims.append("Structural storage ratio equals raw HGB byte compression.")
    claims.append("Feature adapter rows are unmodified official SeHGNN rows.")
    return claims


def _mean_metric(rows: Sequence[Mapping[str, Any]], *names: str) -> float | str:
    values = [_metric(row, *names) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if not finite else mean(finite)


def _std_metric(rows: Sequence[Mapping[str, Any]], *names: str) -> float | str:
    values = [_metric(row, *names) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if len(finite) < 2 else pstdev(finite)


def _metric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def _positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _real_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text not in {"nan", "none", "null", EMPTY_SHA256}


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
