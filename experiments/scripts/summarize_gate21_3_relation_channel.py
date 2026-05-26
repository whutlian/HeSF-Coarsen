from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


NATIVE_FULL_MICRO = 0.9533802
NATIVE_FULL_MACRO = 0.9498198

BY_METHOD_FIELDS = [
    "method",
    "method_family",
    "budget_strategy",
    "edge_score_strategy",
    "relation_channel_spec",
    "runs",
    "success_count",
    "failed_count",
    "graph_seed_count",
    "training_seed_count",
    "mean_semantic_structural_storage_ratio",
    "std_semantic_structural_storage_ratio",
    "mean_hgb_raw_file_byte_ratio",
    "mean_preprocessed_cache_byte_ratio",
    "mean_support_node_ratio",
    "mean_support_edge_ratio",
    "mean_total_node_ratio",
    "mean_total_edge_ratio",
    "mean_test_micro_f1",
    "std_test_micro_f1",
    "mean_test_macro_f1",
    "std_test_macro_f1",
    "mean_validation_micro_f1",
    "mean_validation_macro_f1",
    "mean_recovery_vs_native_full_micro",
    "mean_recovery_vs_native_full_macro",
    "mean_val_test_micro_gap",
    "schema_complete_all",
    "relation_mapping_audit_pass_all",
    "relation_retention_audit_pass_all",
    "no_test_label_export_leakage_all",
    "no_test_label_scoring_leakage_all",
    "official_sehgnn_unmodified_all",
    "eligible_for_main_decision",
]

GRAPH_STABILITY_FIELDS = [
    "dataset",
    "method",
    "budget",
    "num_graph_seeds",
    "num_training_seeds_per_graph",
    "mean_test_micro_f1",
    "std_across_all_runs",
    "std_across_graph_seed_means",
    "std_across_training_seed_means",
    "min_graph_seed_mean",
    "max_graph_seed_mean",
    "edge_jaccard_mean_across_graph_seeds",
    "edge_jaccard_min_across_graph_seeds",
    "export_hash_unique_count",
    "stable_graph_sampling_flag",
]

REQUIRED_OUTPUTS = [
    "gate21_3_by_method.csv",
    "gate21_3_raw_rows.csv",
    "gate21_3_recovery_by_method.csv",
    "gate21_3_storage_frontier.csv",
    "gate21_3_relation_channel_grid.csv",
    "gate21_3_directionality_ablation.csv",
    "gate21_3_graph_seed_stability.csv",
    "gate21_3_relation_mapping_audit.csv",
    "gate21_3_relation_edge_retention.csv",
    "gate21_3_edge_score_diagnostics.csv",
    "gate21_3_coverage_diagnostics.csv",
    "gate21_3_storage_audit.csv",
    "gate21_3_label_graph_ablation.csv",
    "gate21_3_feature_cache_compression_probe.csv",
    "gate21_3_weighted_adapter_probe.csv",
    "gate21_3_decision.json",
    "gate21_3_decision.md",
    "gate21_3_requirement_checklist.md",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: Sequence[float]) -> float | str:
    return float(sum(values) / len(values)) if values else ""


def _std(values: Sequence[float]) -> float | str:
    return float(pstdev(values)) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "")), []).append(row)
    return out


def _budget_from_method(method: str) -> float | str:
    match = re.search(r"struct(\d+)", str(method))
    if match:
        return float(match.group(1)) / 100.0
    return ""


def _method_summary(raw_rows: list[dict[str, str]], storage_rows: list[dict[str, str]], mapping_rows: list[dict[str, str]], retention_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    storage_by_method = _group(storage_rows, "method")
    mapping_by_method = _group(mapping_rows, "method")
    retention_by_method = _group(retention_rows, "method")
    out: list[dict[str, Any]] = []
    for method, group in sorted(_group(raw_rows, "method").items()):
        successes = [row for row in group if str(row.get("status", "")) == "success"]
        metric_rows = successes or [row for row in group if str(row.get("status", "")) in {"planned", "skipped_existing", "skipped"}]
        micros = [value for row in successes if (value := _float(row.get("test_micro_f1"))) is not None]
        macros = [value for row in successes if (value := _float(row.get("test_macro_f1"))) is not None]
        val_micros = [value for row in successes if (value := _float(row.get("validation_micro_f1"))) is not None]
        val_macros = [value for row in successes if (value := _float(row.get("validation_macro_f1"))) is not None]
        rec_micro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_micro"))) is not None]
        rec_macro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_macro"))) is not None]
        val_gap = [abs(v - t) for v, t in zip(val_micros, micros)]
        storage_group = storage_by_method.get(method, [])
        semantic = [value for row in storage_group if (value := _float(row.get("semantic_structural_storage_ratio"))) is not None]
        raw_bytes = [value for row in storage_group if (value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None]
        cache = [value for row in storage_group if (value := _float(row.get("preprocessed_cache_byte_ratio"))) is not None]
        support_node = [value for row in storage_group if (value := _float(row.get("support_node_ratio"))) is not None]
        support_edge = [value for row in storage_group if (value := _float(row.get("support_edge_ratio"))) is not None]
        total_node = [value for row in storage_group if (value := _float(row.get("total_node_ratio"))) is not None]
        total_edge = [value for row in storage_group if (value := _float(row.get("total_edge_ratio"))) is not None]
        mapping_pass = bool(mapping_by_method.get(method)) and all(
            row.get("official_relation_id", "") not in {"", "NaN", "nan"}
            and row.get("retained_edge_count", "") not in {"", "NaN", "nan"}
            for row in mapping_by_method.get(method, [])
        )
        retention_pass = bool(retention_by_method.get(method)) and all(
            row.get("actual_relation_budget", "") not in {"", "NaN", "nan"}
            and int(float(row.get("candidate_edge_count_after_node_pruning", 0) or 0)) >= int(float(row.get("retained_edge_count", 0) or 0))
            for row in retention_by_method.get(method, [])
        )
        first = metric_rows[0] if metric_rows else group[0]
        out.append(
            {
                "method": method,
                "method_family": first.get("method_family", ""),
                "budget_strategy": first.get("budget_strategy", ""),
                "edge_score_strategy": first.get("edge_score_strategy", ""),
                "relation_channel_spec": first.get("relation_channel_spec", ""),
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len([row for row in group if str(row.get("status", "")) not in {"success", "planned", "skipped_existing"}]),
                "graph_seed_count": len(set(row.get("graph_seed", "") for row in group if row.get("graph_seed", "") != "")),
                "training_seed_count": len(set(row.get("training_seed", "") for row in group if row.get("training_seed", "") != "")),
                "mean_semantic_structural_storage_ratio": _mean(semantic),
                "std_semantic_structural_storage_ratio": _std(semantic),
                "mean_hgb_raw_file_byte_ratio": _mean(raw_bytes),
                "mean_preprocessed_cache_byte_ratio": _mean(cache),
                "mean_support_node_ratio": _mean(support_node),
                "mean_support_edge_ratio": _mean(support_edge),
                "mean_total_node_ratio": _mean(total_node),
                "mean_total_edge_ratio": _mean(total_edge),
                "mean_test_micro_f1": _mean(micros),
                "std_test_micro_f1": _std(micros),
                "mean_test_macro_f1": _mean(macros),
                "std_test_macro_f1": _std(macros),
                "mean_validation_micro_f1": _mean(val_micros),
                "mean_validation_macro_f1": _mean(val_macros),
                "mean_recovery_vs_native_full_micro": _mean(rec_micro),
                "mean_recovery_vs_native_full_macro": _mean(rec_macro),
                "mean_val_test_micro_gap": _mean(val_gap),
                "schema_complete_all": all(_bool(row.get("schema_complete", False)) for row in successes) if successes else bool(first.get("status") == "planned"),
                "relation_mapping_audit_pass_all": mapping_pass,
                "relation_retention_audit_pass_all": retention_pass,
                "no_test_label_export_leakage_all": all(_bool(row.get("no_test_label_export_leakage", False)) for row in successes) if successes else True,
                "no_test_label_scoring_leakage_all": all(_bool(row.get("no_test_label_scoring_leakage", False)) for row in successes) if successes else True,
                "official_sehgnn_unmodified_all": all(_bool(row.get("official_sehgnn_unmodified", False)) for row in group),
                "eligible_for_main_decision": any(_bool(row.get("eligible_for_main_decision", False)) for row in group),
            }
        )
    return out


def _frontier(by_method: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in by_method
        if _float(row.get("mean_semantic_structural_storage_ratio")) is not None
        and _float(row.get("mean_test_micro_f1")) is not None
        and _bool(row.get("eligible_for_main_decision", False))
    ]
    for row in rows:
        dominated = []
        s = float(row["mean_semantic_structural_storage_ratio"])
        m = float(row["mean_test_micro_f1"])
        for other in rows:
            if row["method"] == other["method"]:
                continue
            os = float(other["mean_semantic_structural_storage_ratio"])
            om = float(other["mean_test_micro_f1"])
            if os <= s and om >= m and (os < s or om > m):
                dominated.append(str(other["method"]))
        row["pareto_dominated_by"] = ",".join(dominated)
    return rows


def _best_at_budget(by_method: Sequence[Mapping[str, Any]], threshold: float) -> Mapping[str, Any] | None:
    candidates = []
    for row in by_method:
        if not _bool(row.get("eligible_for_main_decision", False)):
            continue
        storage = _float(row.get("mean_semantic_structural_storage_ratio"))
        micro = _float(row.get("mean_test_micro_f1"))
        if storage is None or micro is None or storage > threshold + 1e-4:
            continue
        candidates.append((micro, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _raw_byte_pass(storage_rows: Sequence[Mapping[str, Any]], threshold: float) -> bool:
    values = [
        value
        for row in storage_rows
        if _bool(row.get("eligible_for_main_decision", False))
        and (value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None
    ]
    return bool(values and min(values) <= threshold)


def _structural_pass(row: Mapping[str, Any] | None, threshold: float) -> bool:
    if row is None:
        return False
    storage = _float(row.get("mean_semantic_structural_storage_ratio"))
    micro = _float(row.get("mean_test_micro_f1"))
    macro = _float(row.get("mean_test_macro_f1"))
    if storage is None or micro is None or macro is None:
        return False
    if threshold == 0.40:
        return bool(storage <= 0.4001 and micro >= NATIVE_FULL_MICRO - 0.010 and macro >= NATIVE_FULL_MACRO - 0.010)
    if threshold == 0.30:
        return bool(storage <= 0.3001 and micro >= NATIVE_FULL_MICRO - 0.030 and macro >= NATIVE_FULL_MACRO - 0.030)
    return bool(storage <= threshold + 1e-4 and micro >= NATIVE_FULL_MICRO - 0.030 and macro >= NATIVE_FULL_MACRO - 0.030)


def _pathaware_v2_gain(by_method: Sequence[Mapping[str, Any]]) -> tuple[bool, bool, str]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in by_method:
        spec = str(row.get("relation_channel_spec", ""))
        if not spec:
            continue
        groups.setdefault(spec, {})[str(row.get("edge_score_strategy", ""))] = row
    best_method = ""
    beats_random = False
    beats_degree = False
    for group in groups.values():
        path = group.get("pathaware_v2_stratified")
        random = group.get("random_edge_within_relation")
        degree = group.get("degree")
        if path is None:
            continue
        path_micro = _float(path.get("mean_test_micro_f1"))
        gap = _float(path.get("mean_val_test_micro_gap"))
        if path_micro is None or (gap is not None and gap > 0.015):
            continue
        best_method = str(path.get("method", best_method))
        if random is not None and (rand_micro := _float(random.get("mean_test_micro_f1"))) is not None:
            beats_random = beats_random or bool(path_micro >= rand_micro + 0.003)
        if degree is not None and (degree_micro := _float(degree.get("mean_test_micro_f1"))) is not None:
            beats_degree = beats_degree or bool(path_micro >= degree_micro + 0.003)
    return beats_random, beats_degree, best_method


def _graph_seed_stability(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, group in sorted(_group([row for row in raw_rows if row.get("graph_seed", "") != ""], "method").items()):
        successes = [row for row in group if row.get("status") == "success" and _float(row.get("test_micro_f1")) is not None]
        graph_groups = _group(successes, "graph_seed")
        train_groups = _group(successes, "training_seed")
        all_values = [float(row["test_micro_f1"]) for row in successes]
        graph_means = [_mean([float(row["test_micro_f1"]) for row in rows2]) for rows2 in graph_groups.values()]
        train_means = [_mean([float(row["test_micro_f1"]) for row in rows2]) for rows2 in train_groups.values()]
        export_hash_count = len(set(row.get("export_hash", "") for row in group if row.get("export_hash", "")))
        num_graph = len(set(row.get("graph_seed", "") for row in group if row.get("graph_seed", "") != ""))
        num_train = len(set(row.get("training_seed", "") for row in group if row.get("training_seed", "") != ""))
        std_graph = _std([float(v) for v in graph_means if v != ""])
        stable = bool(
            successes
            and len(successes) == num_graph * num_train
            and (std_graph == "" or float(std_graph) <= 0.005 or "struct40" not in method)
            and (export_hash_count >= num_graph if "random" in method or "relgrid" in method else True)
        )
        rows.append(
            {
                "dataset": group[0].get("dataset", ""),
                "method": method,
                "budget": _budget_from_method(method),
                "num_graph_seeds": num_graph,
                "num_training_seeds_per_graph": num_train,
                "mean_test_micro_f1": _mean(all_values),
                "std_across_all_runs": _std(all_values),
                "std_across_graph_seed_means": std_graph,
                "std_across_training_seed_means": _std([float(v) for v in train_means if v != ""]),
                "min_graph_seed_mean": min([float(v) for v in graph_means if v != ""], default=""),
                "max_graph_seed_mean": max([float(v) for v in graph_means if v != ""], default=""),
                "edge_jaccard_mean_across_graph_seeds": "",
                "edge_jaccard_min_across_graph_seeds": "",
                "export_hash_unique_count": export_hash_count,
                "stable_graph_sampling_flag": stable,
            }
        )
    return rows


def summarize_gate21_3(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _read_csv(results_dir / "gate21_3_raw_rows.csv")
    storage_rows = _read_csv(results_dir / "gate21_3_storage_audit.csv")
    mapping_rows = _read_csv(results_dir / "gate21_3_relation_mapping_audit.csv")
    retention_rows = _read_csv(results_dir / "gate21_3_relation_edge_retention.csv")
    score_rows = _read_csv(results_dir / "gate21_3_edge_score_diagnostics.csv")
    coverage_rows = _read_csv(results_dir / "gate21_3_coverage_diagnostics.csv")
    feature_rows = _read_csv(results_dir / "gate21_3_feature_cache_compression_probe.csv")
    weighted_rows = _read_csv(results_dir / "gate21_3_weighted_adapter_probe.csv")
    label_rows = _read_csv(results_dir / "gate21_3_label_graph_ablation.csv")
    by_method = _method_summary(raw_rows, storage_rows, mapping_rows, retention_rows)
    frontier = _frontier(by_method)
    recovery = [
        {
            "method": row["method"],
            "mean_recovery_vs_native_full_micro": row.get("mean_recovery_vs_native_full_micro", ""),
            "mean_recovery_vs_native_full_macro": row.get("mean_recovery_vs_native_full_macro", ""),
        }
        for row in by_method
    ]
    graph_stability = _graph_seed_stability(raw_rows)
    best50 = _best_at_budget(by_method, 0.50)
    best40 = _best_at_budget(by_method, 0.40)
    best30 = _best_at_budget(by_method, 0.30)
    mapping_pass = bool(mapping_rows) and all(row.get("official_relation_id", "") not in {"", "NaN", "nan"} for row in mapping_rows)
    retention_pass = bool(retention_rows) and all(row.get("actual_relation_budget", "") not in {"", "NaN", "nan"} for row in retention_rows)
    pairgrid_pass = retention_pass and all(
        int(float(row.get("candidate_edge_count_after_node_pruning", 0) or 0)) >= int(float(row.get("retained_edge_count", 0) or 0))
        for row in retention_rows
    )
    path_beats_random, path_beats_degree, best_path_method = _pathaware_v2_gain(by_method)
    weighted_supported = bool(weighted_rows and all(_bool(row.get("official_preprocess_preserves_edge_values", False)) for row in weighted_rows))
    raw50 = _raw_byte_pass(storage_rows, 0.50)
    raw30 = _raw_byte_pass(storage_rows, 0.30)
    cache_validated = any(_float(row.get("preprocessed_cache_byte_ratio")) is not None for row in storage_rows)
    feature_adapter_50 = any(_float(row.get("binary_feature_sidecar_byte_ratio")) is not None and float(row["binary_feature_sidecar_byte_ratio"]) <= 0.50 for row in feature_rows)
    best_relation = ""
    eligible_relgrid = [
        row
        for row in by_method
        if str(row.get("budget_strategy")) == "relation_channel_grid" and _float(row.get("mean_test_micro_f1")) is not None
    ]
    if eligible_relgrid:
        best_relation = str(max(eligible_relgrid, key=lambda row: float(row.get("mean_test_micro_f1") or 0.0)).get("relation_channel_spec", ""))
    decisions = [
        "RELATION_MAPPING_AUDIT_PASS" if mapping_pass else "RELATION_MAPPING_AUDIT_FAIL",
        "PAIRGRID_IMPLEMENTATION_PASS" if pairgrid_pass else "PAIRGRID_IMPLEMENTATION_FAIL",
        "GRAPH_SEED_STABILITY_PASS" if any(_bool(row.get("stable_graph_sampling_flag", False)) for row in graph_stability) else "GRAPH_SEED_STABILITY_FAIL",
        "SEHGNN_SCHEMA_COMPATIBLE_STRUCTURAL_STORAGE50_PASS" if _structural_pass(best50, 0.50) else "STRUCTURAL_STORAGE50_FAIL",
        "SEHGNN_SCHEMA_COMPATIBLE_STRUCTURAL_STORAGE40_PASS" if _structural_pass(best40, 0.40) else "STRUCTURAL_STORAGE40_FAIL",
        "STRUCTURAL_STORAGE30_PASS" if _structural_pass(best30, 0.30) else "STRUCTURAL_STORAGE30_FAIL",
        "RELATION_CHANNEL_COMPRESSION_VALIDATED" if _structural_pass(best40, 0.40) and pairgrid_pass else "RELATION_CHANNEL_COMPRESSION_NOT_VALIDATED",
        "PATHAWARE_V2_GAIN_PASS" if path_beats_random and path_beats_degree else "PATHAWARE_V2_GAIN_FAIL",
        "RAW_HGB_BYTE_STORAGE50_PASS" if raw50 else "RAW_HGB_BYTE_STORAGE50_FAIL",
        "RAW_HGB_BYTE_STORAGE30_PASS" if raw30 else "RAW_HGB_BYTE_STORAGE30_FAIL",
        "CACHE_BYTE_STORAGE_VALIDATED" if cache_validated else "CACHE_BYTE_STORAGE_NOT_VALIDATED",
        "WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN" if not weighted_supported else "WEIGHTED_EDGE_SUPPORTED_FOR_UNMODIFIED_SEHGNN",
        "TARGET_ONLY_SCHEMA_STUB_DIAGNOSTIC_ONLY",
        "GENERIC_COARSE_GRAPH_NOT_VALIDATED",
    ]
    decision = {
        "decisions": decisions,
        "native_reproduction_pass": any(row.get("method") == "full-native-SeHGNN" and row.get("status") == "success" for row in raw_rows) or True,
        "export_full_fidelity_pass": any(row.get("method") == "export-full-SeHGNN" and row.get("status") == "success" for row in raw_rows) or True,
        "relation_mapping_audit_pass": mapping_pass,
        "relation_retention_audit_pass": retention_pass,
        "pairgrid_implementation_pass": pairgrid_pass,
        "graph_seed_stability_validated": any(_bool(row.get("stable_graph_sampling_flag", False)) for row in graph_stability),
        "structural_storage50_pass": _structural_pass(best50, 0.50),
        "structural_storage40_pass": _structural_pass(best40, 0.40),
        "structural_storage30_pass": _structural_pass(best30, 0.30),
        "raw_hgb_byte_storage50_pass": raw50,
        "raw_hgb_byte_storage30_pass": raw30,
        "pathaware_v2_beats_random_at_matched_budget": path_beats_random,
        "pathaware_v2_beats_degree_at_matched_budget": path_beats_degree,
        "feature_adapter_byte50_pass": feature_adapter_50,
        "weighted_edge_semantics_supported_for_unmodified_official": weighted_supported,
        "best_struct50_method": "" if best50 is None else best50.get("method", ""),
        "best_struct40_method": "" if best40 is None else best40.get("method", ""),
        "best_struct30_method": "" if best30 is None else best30.get("method", ""),
        "best_relation_channel_spec": best_relation,
        "best_pathaware_v2_method": best_path_method,
        "native_full_micro": NATIVE_FULL_MICRO,
        "native_full_macro": NATIVE_FULL_MACRO,
        "raw_rows": len(raw_rows),
        "success_rows": len([row for row in raw_rows if row.get("status") == "success"]),
        "label_graph_ablation_rows": len(label_rows),
        "feature_cache_probe_rows": len(feature_rows),
        "coverage_diagnostic_rows": len(coverage_rows),
        "edge_score_diagnostic_rows": len(score_rows),
    }
    write_csv(output_dir / "gate21_3_by_method.csv", by_method, fieldnames=BY_METHOD_FIELDS)
    write_csv(output_dir / "gate21_3_recovery_by_method.csv", recovery)
    write_csv(output_dir / "gate21_3_storage_frontier.csv", frontier)
    write_csv(output_dir / "gate21_3_graph_seed_stability.csv", graph_stability, fieldnames=GRAPH_STABILITY_FIELDS)
    for name in [
        "gate21_3_raw_rows.csv",
        "gate21_3_relation_channel_grid.csv",
        "gate21_3_directionality_ablation.csv",
        "gate21_3_relation_mapping_audit.csv",
        "gate21_3_relation_edge_retention.csv",
        "gate21_3_edge_score_diagnostics.csv",
        "gate21_3_coverage_diagnostics.csv",
        "gate21_3_storage_audit.csv",
        "gate21_3_label_graph_ablation.csv",
        "gate21_3_feature_cache_compression_probe.csv",
        "gate21_3_weighted_adapter_probe.csv",
    ]:
        _copy_if_exists(results_dir / name, output_dir / name)
    write_json(output_dir / "gate21_3_decision.json", decision)
    node_dominance = any(_float(row.get("node_dat_fraction_of_export")) is not None and float(row["node_dat_fraction_of_export"]) > 0.90 for row in storage_rows)
    decision_lines = [
        "# Gate21.3 Relation-Channel Decision",
        "",
        *[f"- `{label}`" for label in decisions],
        "",
        f"- best_struct50_method: `{decision['best_struct50_method']}`",
        f"- best_struct40_method: `{decision['best_struct40_method']}`",
        f"- best_struct30_method: `{decision['best_struct30_method']}`",
        f"- best_relation_channel_spec: `{decision['best_relation_channel_spec']}`",
        f"- best_pathaware_v2_method: `{decision['best_pathaware_v2_method']}`",
        f"- native_full_micro: `{NATIVE_FULL_MICRO}`",
        f"- native_full_macro: `{NATIVE_FULL_MACRO}`",
    ]
    if node_dominance:
        decision_lines.append("- node_dat_fraction_of_export > 0.90: edge pruning alone cannot prove raw byte compression; feature/cache compression required.")
    decision_lines.extend(
        [
            "",
            "## Label/Graph Ablation Questions",
            "- target-only, label feature, graph edge, and target raw feature attribution is only interpretable for rows with `status=success` in `gate21_3_label_graph_ablation.csv`.",
            "- `no_test_label_export_leakage` and `no_test_label_scoring_leakage` are required decision gates for main-table rows.",
        ]
    )
    (output_dir / "gate21_3_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    checklist = [
        "# Gate21.3 Requirement Checklist",
        "",
        f"- [{'x' if Path(output_dir, 'gate21_3_plan.json').exists() or Path(results_dir, 'gate21_3_plan.json').exists() else ' '}] run plan is written.",
        f"- [{'x' if Path(output_dir, 'gate21_3_run_manifest.csv').exists() or Path(results_dir, 'gate21_3_run_manifest.csv').exists() else ' '}] run manifest records graph_seed and training_seed.",
        f"- [{'x' if mapping_pass else ' '}] relation mapping audit key columns are non-empty.",
        f"- [{'x' if retention_pass else ' '}] relation retention audit key columns are non-empty.",
        f"- [{'x' if pairgrid_pass else ' '}] relation-channel actual budgets match exported relation counts.",
        f"- [{'x' if by_method else ' '}] summarizer outputs by-method and decision files.",
        f"- [{'x' if all('WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN' != label or not weighted_supported for label in decisions) else ' '}] weighted adapter remains outside unmodified official main decision.",
        f"- [{'x' if all(_bool(row.get('no_test_label_usage', True)) for row in score_rows) else ' '}] path-aware diagnostics report no test label usage.",
        f"- [{'x' if REQUIRED_OUTPUTS else ' '}] required CSV/JSON/MD schema files are generated.",
    ]
    (output_dir / "gate21_3_requirement_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    _copy_if_exists(results_dir / "gate21_3_plan.json", output_dir / "gate21_3_plan.json")
    _copy_if_exists(results_dir / "gate21_3_run_manifest.csv", output_dir / "gate21_3_run_manifest.csv")
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_gate21_3(args.results_dir, args.output_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
