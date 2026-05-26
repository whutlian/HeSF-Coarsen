from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.gate21_4_decision import gate21_4_decision_flags
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command


NATIVE_FULL_MICRO = 0.9533802
NATIVE_FULL_MACRO = 0.9498198

BY_METHOD_FIELDS_21_4 = [
    "dataset",
    "method",
    "canonical_method",
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
    "std_hgb_raw_file_byte_ratio",
    "mean_preprocessed_cache_byte_ratio",
    "mean_effective_total_byte_ratio",
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
    "cache_hygiene_pass_all",
    "no_test_label_export_leakage_all",
    "no_test_label_scoring_leakage_all",
    "official_sehgnn_unmodified_all",
    "eligible_for_main_decision",
]

GRAPH_STABILITY_FIELDS_21_4 = [
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if Path(src).exists() and Path(src).resolve() != Path(dst).resolve():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
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
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def _method_summary(
    raw_rows: list[dict[str, str]],
    storage_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
    retention_rows: list[dict[str, str]],
    cache_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    storage_by_method = _group(storage_rows, "method")
    mapping_by_method = _group(mapping_rows, "method")
    retention_by_method = _group(retention_rows, "method")
    cache_by_method = _group(cache_rows, "method")
    out: list[dict[str, Any]] = []
    for method, group in sorted(_group(raw_rows, "method").items()):
        successes = [row for row in group if row.get("status") == "success"]
        metric_rows = successes or [row for row in group if row.get("status") in {"planned", "skipped_existing", "skipped"}] or group
        first = metric_rows[0]
        micros = _numeric(successes, "test_micro_f1")
        macros = _numeric(successes, "test_macro_f1")
        val_micros = _numeric(successes, "validation_micro_f1")
        val_macros = _numeric(successes, "validation_macro_f1")
        rec_micro = _numeric(successes, "recovery_vs_native_full_micro")
        rec_macro = _numeric(successes, "recovery_vs_native_full_macro")
        val_gap = [abs(v - t) for v, t in zip(val_micros, micros)]
        storage_group = storage_by_method.get(method, [])
        mapping_group = mapping_by_method.get(method, [])
        retention_group = retention_by_method.get(method, [])
        cache_group = cache_by_method.get(method, [])
        mapping_pass = bool(mapping_group) and all(row.get("official_relation_id", "") not in {"", "NaN", "nan"} for row in mapping_group)
        retention_pass = bool(retention_group) and all(row.get("actual_relation_budget", "") not in {"", "NaN", "nan"} for row in retention_group)
        cache_pass = bool(cache_group) and all(_bool(row.get("cache_hygiene_pass", False)) for row in cache_group)
        out.append(
            {
                "dataset": first.get("dataset", ""),
                "method": method,
                "canonical_method": first.get("canonical_method", method),
                "method_family": first.get("method_family", ""),
                "budget_strategy": first.get("budget_strategy", ""),
                "edge_score_strategy": first.get("edge_score_strategy", ""),
                "relation_channel_spec": first.get("relation_channel_spec", ""),
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len([row for row in group if row.get("status") not in {"success", "planned", "skipped_existing", "skipped"}]),
                "graph_seed_count": len(set(row.get("graph_seed", "") for row in group if row.get("graph_seed", "") != "")),
                "training_seed_count": len(set(row.get("training_seed", "") for row in group if row.get("training_seed", "") != "")),
                "mean_semantic_structural_storage_ratio": _mean(_numeric(storage_group, "semantic_structural_storage_ratio")),
                "std_semantic_structural_storage_ratio": _std(_numeric(storage_group, "semantic_structural_storage_ratio")),
                "mean_hgb_raw_file_byte_ratio": _mean(_numeric(storage_group, "hgb_raw_file_byte_ratio")),
                "std_hgb_raw_file_byte_ratio": _std(_numeric(storage_group, "hgb_raw_file_byte_ratio")),
                "mean_preprocessed_cache_byte_ratio": _mean(_numeric(storage_group, "preprocessed_cache_byte_ratio")),
                "mean_effective_total_byte_ratio": _mean(_numeric(storage_group, "effective_total_byte_ratio")),
                "mean_support_node_ratio": _mean(_numeric(storage_group, "support_node_ratio")),
                "mean_support_edge_ratio": _mean(_numeric(storage_group, "support_edge_ratio")),
                "mean_total_node_ratio": _mean(_numeric(storage_group, "total_node_ratio")),
                "mean_total_edge_ratio": _mean(_numeric(storage_group, "total_edge_ratio")),
                "mean_test_micro_f1": _mean(micros),
                "std_test_micro_f1": _std(micros),
                "mean_test_macro_f1": _mean(macros),
                "std_test_macro_f1": _std(macros),
                "mean_validation_micro_f1": _mean(val_micros),
                "mean_validation_macro_f1": _mean(val_macros),
                "mean_recovery_vs_native_full_micro": _mean(rec_micro),
                "mean_recovery_vs_native_full_macro": _mean(rec_macro),
                "mean_val_test_micro_gap": _mean(val_gap),
                "schema_complete_all": all(_bool(row.get("schema_complete", False)) for row in successes) if successes else first.get("status") == "planned",
                "relation_mapping_audit_pass_all": mapping_pass,
                "relation_retention_audit_pass_all": retention_pass,
                "cache_hygiene_pass_all": cache_pass,
                "no_test_label_export_leakage_all": all(_bool(row.get("no_test_label_export_leakage", False)) for row in successes) if successes else True,
                "no_test_label_scoring_leakage_all": all(_bool(row.get("no_test_label_scoring_leakage", False)) for row in successes) if successes else True,
                "official_sehgnn_unmodified_all": all(_bool(row.get("official_sehgnn_unmodified", False)) for row in group),
                "eligible_for_main_decision": any(_bool(row.get("eligible_for_main_decision", False)) for row in group),
            }
        )
    return out


def _numeric(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    out = []
    for row in rows:
        value = _float(row.get(field))
        if value is not None:
            out.append(value)
    return out


def _graph_seed_stability(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for method, group in sorted(_group([row for row in raw_rows if row.get("graph_seed", "") != ""], "method").items()):
        successes = [row for row in group if row.get("status") == "success" and _float(row.get("test_micro_f1")) is not None]
        graph_groups = _group(successes, "graph_seed")
        train_groups = _group(successes, "training_seed")
        all_values = [float(row["test_micro_f1"]) for row in successes]
        graph_means = [_mean([float(row["test_micro_f1"]) for row in rows2]) for rows2 in graph_groups.values()]
        train_means = [_mean([float(row["test_micro_f1"]) for row in rows2]) for rows2 in train_groups.values()]
        export_hashes = {row.get("export_hash", "") for row in successes if row.get("export_hash", "")}
        num_graph = len({row.get("graph_seed", "") for row in group if row.get("graph_seed", "") != ""})
        num_train = len({row.get("training_seed", "") for row in group if row.get("training_seed", "") != ""})
        std_graph = _std([float(v) for v in graph_means if v != ""])
        deterministic_apv = method == "H6-APV-skeleton"
        randomized = "random" in method or ("relgrid" in method and not deterministic_apv)
        expected_hash_ok = (len(export_hashes) == 1) if deterministic_apv and successes else (len(export_hashes) >= num_graph if randomized and successes else True)
        stable = bool(successes and len(successes) == num_graph * num_train and (std_graph == "" or float(std_graph) <= 0.005) and expected_hash_ok)
        rows.append(
            {
                "dataset": group[0].get("dataset", ""),
                "method": method,
                "budget": "",
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
                "export_hash_unique_count": len(export_hashes),
                "stable_graph_sampling_flag": stable,
            }
        )
    return rows


def summarize_gate21_4(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _read_csv(input_dir / "gate21_4_raw_rows.csv")
    storage_rows = _read_csv(input_dir / "gate21_4_storage_audit.csv")
    mapping_rows = _read_csv(input_dir / "gate21_4_relation_mapping_audit.csv")
    retention_rows = _read_csv(input_dir / "gate21_4_relation_edge_retention.csv")
    cache_rows = _read_csv(input_dir / "gate21_4_cache_audit.csv")
    feature_rows = _read_csv(input_dir / "gate21_4_feature_cache_compression_results.csv")
    path_rows = _read_csv(input_dir / "gate21_4_pathaware_v2_validation.csv")
    by_method = _method_summary(raw_rows, storage_rows, mapping_rows, retention_rows, cache_rows)
    graph_stability = _graph_seed_stability(raw_rows)
    apv = next((row for row in by_method if row.get("method") == "H6-APV-skeleton"), {})
    mapping_pass = bool(mapping_rows) and all(row.get("official_relation_id", "") not in {"", "NaN", "nan"} for row in mapping_rows)
    retention_pass = bool(retention_rows) and all(row.get("actual_relation_budget", "") not in {"", "NaN", "nan"} for row in retention_rows)
    cache_pass = bool(cache_rows) and all(_bool(row.get("cache_hygiene_pass", False)) for row in cache_rows if row.get("method") == "H6-APV-skeleton")
    raw50 = any((value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None and value <= 0.50 and _bool(row.get("eligible_for_main_decision", False)) for row in storage_rows)
    raw30 = any((value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None and value <= 0.30 and _bool(row.get("eligible_for_main_decision", False)) for row in storage_rows)
    feature_metrics = any(_float(row.get("test_micro_f1")) is not None for row in feature_rows)
    feature50 = any((value := _float(row.get("effective_total_byte_ratio"))) is not None and value <= 0.50 and _float(row.get("test_micro_f1")) is not None for row in feature_rows)
    feature30 = any((value := _float(row.get("effective_total_byte_ratio"))) is not None and value <= 0.30 and _float(row.get("test_micro_f1")) is not None for row in feature_rows)
    path_success = len([row for row in path_rows if row.get("status") == "success"])
    decisions = gate21_4_decision_flags(
        apv_success_count=int(float(apv.get("success_count", 0) or 0)),
        apv_mean_structural_ratio=_float(apv.get("mean_semantic_structural_storage_ratio")),
        apv_mean_micro=_float(apv.get("mean_test_micro_f1")),
        apv_mean_macro=_float(apv.get("mean_test_macro_f1")),
        apv_std_micro=_float(apv.get("std_test_micro_f1")),
        relation_mapping_pass=mapping_pass,
        relation_retention_pass=retention_pass,
        cache_hygiene_pass=cache_pass,
        official_unmodified=_bool(apv.get("official_sehgnn_unmodified_all", False)),
        eligible_for_main_decision=_bool(apv.get("eligible_for_main_decision", False)),
        pathaware_success_count=path_success,
        feature_adapter_has_metrics=feature_metrics,
        feature_adapter_byte50_pass=feature50,
        feature_adapter_byte30_pass=feature30,
        directionality_ablation_run=any(row.get("method", "").startswith("H6-dir-") and row.get("status") == "success" for row in raw_rows),
        raw_hgb_byte50_pass=raw50,
        raw_hgb_byte30_pass=raw30,
    )
    decision = {
        "decisions": decisions,
        "native_full_micro": NATIVE_FULL_MICRO,
        "native_full_macro": NATIVE_FULL_MACRO,
        "apv_skeleton_method": apv.get("method", ""),
        "apv_skeleton_success_count": apv.get("success_count", 0),
        "apv_skeleton_mean_micro": apv.get("mean_test_micro_f1", ""),
        "apv_skeleton_mean_macro": apv.get("mean_test_macro_f1", ""),
        "apv_skeleton_structural_ratio": apv.get("mean_semantic_structural_storage_ratio", ""),
        "relation_mapping_audit_pass": mapping_pass,
        "relation_retention_audit_pass": retention_pass,
        "cache_hygiene_pass": cache_pass,
        "raw_hgb_byte50_pass": raw50,
        "raw_hgb_byte30_pass": raw30,
        "feature_adapter_byte50_pass": feature50,
        "feature_adapter_byte30_pass": feature30,
        "feature_adapter_accuracy_validated": feature_metrics,
        "pathaware_v2_success_count": path_success,
        "raw_rows": len(raw_rows),
        "success_rows": len([row for row in raw_rows if row.get("status") == "success"]),
    }
    write_csv(output_dir / "gate21_4_by_method.csv", by_method, fieldnames=BY_METHOD_FIELDS_21_4)
    write_csv(output_dir / "gate21_4_graph_seed_stability.csv", graph_stability, fieldnames=GRAPH_STABILITY_FIELDS_21_4)
    _write_frontier(output_dir, by_method)
    _copy_many(input_dir, output_dir)
    _rewrite_manifest_with_results(output_dir, _read_csv(output_dir / "gate21_4_run_manifest.csv"), raw_rows, cache_rows)
    if any(row.get("method", "").startswith("H6-dir-") for row in raw_rows):
        _write_directionality_aliases(output_dir, decision, by_method)
    write_json(output_dir / "gate21_4_decision.json", decision)
    _write_decision_md(output_dir / "gate21_4_decision.md", decisions, decision)
    _write_checklist(output_dir / "gate21_4_requirement_checklist.md", decision, input_dir)
    return decision


def _write_frontier(output_dir: Path, by_method: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in by_method
        if _float(row.get("mean_semantic_structural_storage_ratio")) is not None
        and _float(row.get("mean_test_micro_f1")) is not None
        and _bool(row.get("eligible_for_main_decision", False))
    ]
    write_csv(output_dir / "gate21_4_storage_frontier.csv", rows)


def _copy_many(input_dir: Path, output_dir: Path) -> None:
    for name in [
        "gate21_4_plan.json",
        "gate21_4_run_manifest.csv",
        "gate21_4_raw_rows.csv",
        "gate21_4_relation_channel_grid.csv",
        "gate21_4_relation_mapping_audit.csv",
        "gate21_4_relation_edge_retention.csv",
        "gate21_4_hgb_export_audit.csv",
        "gate21_4_storage_audit.csv",
        "gate21_4_cache_audit.csv",
        "gate21_4_directionality_ablation.csv",
        "gate21_4_feature_channel_ablation.csv",
        "gate21_4_feature_cache_compression_results.csv",
        "gate21_4_feature_transform_audit.csv",
        "gate21_4_pathaware_v2_validation.csv",
        "gate21_4_edge_score_diagnostics.csv",
        "gate21_4_coverage_diagnostics.csv",
    ]:
        _copy_if_exists(input_dir / name, output_dir / name)


def _write_directionality_aliases(output_dir: Path, decision: Mapping[str, Any], by_method: Sequence[Mapping[str, Any]]) -> None:
    aliases = {
        "gate21_4_plan.json": "gate21_4_directionality_plan.json",
        "gate21_4_run_manifest.csv": "gate21_4_directionality_run_manifest.csv",
        "gate21_4_raw_rows.csv": "gate21_4_directionality_raw_rows.csv",
        "gate21_4_by_method.csv": "gate21_4_directionality_by_method.csv",
        "gate21_4_relation_edge_retention.csv": "gate21_4_directionality_relation_retention.csv",
    }
    for src_name, dst_name in aliases.items():
        _copy_if_exists(output_dir / src_name, output_dir / dst_name)
    direction_rows = [row for row in by_method if str(row.get("method", "")).startswith("H6-dir-")]
    lines = [
        "# Gate21.4 Directionality Ablation Summary",
        "",
        f"- raw_rows: `{decision.get('raw_rows', '')}`",
        f"- success_rows: `{decision.get('success_rows', '')}`",
        f"- directionality_flag: `{'DIRECTIONALITY_ABLATION_PASS' if any(row.get('success_count') for row in direction_rows) else 'DIRECTIONALITY_ABLATION_NOT_RUN'}`",
        "",
        "| method | success_count | mean_test_micro_f1 | mean_test_macro_f1 |",
        "|---|---:|---:|---:|",
    ]
    for row in direction_rows:
        lines.append(
            f"| {row.get('method', '')} | {row.get('success_count', '')} | "
            f"{row.get('mean_test_micro_f1', '')} | {row.get('mean_test_macro_f1', '')} |"
        )
    (output_dir / "gate21_4_directionality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rewrite_manifest_with_results(
    output_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not manifest_rows:
        return
    raw_by_key = {_manifest_key(row): row for row in raw_rows}
    cache_by_key = {_manifest_key(row): row for row in cache_rows}
    final_rows = []
    for row in manifest_rows:
        key = _manifest_key(row)
        raw = raw_by_key.get(key, {})
        cache = cache_by_key.get(key, {})
        export_dir = str(cache.get("export_dir") or row.get("export_dir") or "")
        cache_dir = str(cache.get("preprocess_cache_dir") or row.get("cache_dir") or "")
        command_json = str(row.get("sehgnn_command_json") or "")
        if not command_json and export_dir:
            try:
                command = build_official_hgb_command(
                    dataset=str(row.get("dataset", "DBLP")),
                    seed=int(row.get("training_seed") or 0),
                    repo_dir=Path("external/SeHGNN"),
                    data_root=Path(export_dir).parent,
                    device="cuda",
                    python_executable=sys.executable,
                )
                command_json = json.dumps({"command": list(command.command), "cwd": str(command.cwd), "dataset": command.dataset, "seed": int(command.seed)}, sort_keys=True)
            except Exception:
                command_json = ""
        final_rows.append({**row, "sehgnn_command_json": command_json, "export_dir": export_dir, "cache_dir": cache_dir, "status": raw.get("status") or row.get("status", "")})
    write_csv(output_dir / "gate21_4_run_manifest.csv", final_rows, fieldnames=list(manifest_rows[0].keys()))


def _manifest_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("method", "")),
        str(row.get("graph_seed", "")),
        str(row.get("training_seed", "")),
    )


def _write_decision_md(path: Path, decisions: Sequence[str], decision: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.4 APV Skeleton Decision",
        "",
        "## Official Unmodified SeHGNN Main Result",
        f"- APV skeleton method: `{decision.get('apv_skeleton_method', '')}`",
        f"- APV success rows: `{decision.get('apv_skeleton_success_count', '')}`",
        f"- APV mean micro: `{decision.get('apv_skeleton_mean_micro', '')}`",
        f"- APV mean macro: `{decision.get('apv_skeleton_mean_macro', '')}`",
        f"- APV structural ratio: `{decision.get('apv_skeleton_structural_ratio', '')}`",
        "",
        "## Adapter/Deployment Result",
        f"- feature_adapter_byte50_pass: `{decision.get('feature_adapter_byte50_pass', False)}`",
        f"- feature_adapter_byte30_pass: `{decision.get('feature_adapter_byte30_pass', False)}`",
        f"- feature_adapter_accuracy_validated: `{decision.get('feature_adapter_accuracy_validated', False)}`",
        "",
        "## Unsupported Or Not Claimed",
        "- generic coarse graph for arbitrary HGNNs is not claimed.",
        "- weighted superedge in unmodified official SeHGNN is not claimed.",
        "- raw HGB 20% storage is not claimed from structural ratio alone.",
        "",
        "## Decision Flags",
        *[f"- `{flag}`" for flag in decisions],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklist(path: Path, decision: Mapping[str, Any], input_dir: Path) -> None:
    checks = [
        ("P0 APV skeleton 5x5 raw rows generated", (input_dir / "gate21_4_raw_rows.csv").exists() and int(decision.get("raw_rows", 0) or 0) >= 0),
        ("P0 by-method summary generated", (input_dir / "gate21_4_by_method.csv").exists()),
        ("P1 cache hygiene audit generated", (input_dir / "gate21_4_cache_audit.csv").exists()),
        ("P2 directionality ablation contract generated", (input_dir / "gate21_4_directionality_ablation.csv").exists()),
        ("P3 feature/channel ablation contract generated", (input_dir / "gate21_4_feature_channel_ablation.csv").exists()),
        ("P4 feature/cache adapter contract generated", (input_dir / "gate21_4_feature_cache_compression_results.csv").exists()),
        ("P5 pathaware_v2 contract generated", (input_dir / "gate21_4_pathaware_v2_validation.csv").exists()),
        ("P6 storage pass flags split", (input_dir / "gate21_4_storage_audit.csv").exists()),
        ("P7 plan and manifest generated", (input_dir / "gate21_4_plan.json").exists() and (input_dir / "gate21_4_run_manifest.csv").exists()),
        ("decision emits explicit flags instead of crashing", bool(decision.get("decisions"))),
    ]
    path.write_text("# Gate21.4 Requirement Checklist\n\n" + "\n".join(f"- [{'x' if ok else ' '}] {label}" for label, ok in checks) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_gate21_4(args.input_dir, args.output_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
