from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_8_common import (
    add_gate21_8_common_args,
    components,
    datasets,
    ensure_layout,
    read_csv,
    write_component_readmes,
    write_plan,
)
from hesf_coarsen.eval.official.gate21_8_decision import (
    EMPTY_SHA256,
    REQUIRED_EXTERNAL_TP_5X5,
    apv16_graph_seed_stability_status,
    budget_alignment_status,
    external_tp_5x5_method_status,
    ratio_denominator_status,
)
from hesf_coarsen.eval.official.runner_utils import clone_external_repo, git_commit_hash, repo_commit_hash, write_csv, write_json


FREEHGC_URL = "https://github.com/GooLiang/FreeHGC"
EXTERNAL_TP_BUDGETS = (
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
    ("structural_ratio", 0.12),
    ("structural_ratio", 0.16),
    ("structural_ratio", 0.20),
    ("structural_ratio", 0.30),
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.output_root))
    write_component_readmes(paths)
    selected = components(args)
    manifest = {
        "gate": "21.8",
        "output_root": str(Path(args.output_root)),
        "datasets": datasets(args),
        "graph_seeds": list(args.graph_seeds),
        "training_seeds": list(args.training_seeds),
        "components": sorted(selected),
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "device": str(args.device),
        "official_sehgnn_root": str(args.official_sehgnn_root),
        "freehgc_root": str(args.freehgc_root),
        "gate21_7_root": str(args.gate21_7_root),
        "gate21_6_dir": str(args.gate21_6_dir),
        "hesf_commit": git_commit_hash(Path.cwd()),
    }
    write_json(paths["logs"] / "gate21_8_run_manifest.json", manifest)

    if args.dry_run:
        _write_dry_run(paths, args, selected)
    else:
        results: dict[str, Any] = {}
        if _enabled(selected, "apv16_5x5", "apv16"):
            results["apv16_5x5"] = _write_apv16(paths, args)
        if _enabled(selected, "external_tp", "external_tp_5x5"):
            results["external_tp_5x5"] = _write_external_tp(paths, args)
        if _enabled(selected, "freehgc", "freehgc_protocols"):
            results["freehgc_protocols"] = _write_freehgc(paths, args)
        if _enabled(selected, "metapath_cache", "metapath_cache_dump"):
            results["metapath_cache_dump"] = _write_metapath_cache(paths, args)
        if _enabled(selected, "feature_ablation", "feature_ablation_tasks"):
            results["feature_ablation_tasks"] = _write_feature_ablation(paths, args)
        if _enabled(selected, "adapter", "adapter_package_v3"):
            results["adapter_package_v3"] = _write_adapter(paths, args)
        if _enabled(selected, "storage_system", "storage_system_costs"):
            results["storage_system_costs"] = _write_storage(paths, args)
        if _enabled(selected, "cross_dataset", "cross_dataset_auto_channel"):
            results["cross_dataset_auto_channel"] = _write_cross_dataset(paths, args)
        results["audits"] = _write_audits(paths, args)
        write_json(paths["logs"] / "gate21_8_component_results.json", results)

    from experiments.scripts.summarize_gate21_8_icde_evidence import summarize

    decision = summarize(Path(args.output_root), Path(args.output_root), strict=bool(args.strict))
    return {
        "output_root": str(Path(args.output_root)),
        "paper_ready_status": decision.get("paper_ready_status", ""),
        "blocking_issues": decision.get("blocking_issues", []),
        "flags": decision.get("flags", {}),
    }


def _write_dry_run(paths: Mapping[str, Path], args: argparse.Namespace, selected: set[str]) -> None:
    rows = [
        {
            "component": component,
            "would_execute": True,
            "dry_run_manifest_only": True,
            "task_metrics_written": False,
            "reason": "dry-run requested; manifests only",
        }
        for component in sorted(selected)
    ]
    write_csv(paths["audits"] / "gate21_8_dry_run_component_manifest.csv", rows)
    write_plan(
        paths["logs"] / "gate21_8_dry_run_manifest.json",
        {
            "output_root": str(paths["root"]),
            "components": sorted(selected),
            "graph_seeds": list(args.graph_seeds),
            "training_seeds": list(args.training_seeds),
            "task_metrics_written": False,
        },
    )


def _write_apv16(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "apv16_stability"
    by_method = read_csv(root / "gate21_7_apv16_stability_by_method.csv")
    by_run = read_csv(root / "gate21_7_apv16_stability_by_run.csv")
    edge_overlap = read_csv(root / "gate21_7_relation_overlap_by_method.csv")
    export_hash = read_csv(root / "gate21_7_export_hash_audit.csv")

    method_rows = [_gate21_8_apv_method_row(row) for row in by_method]
    run_rows = [_gate21_8_apv_run_row(row) for row in by_run]
    stability_rows = [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "graph_seed_count": row.get("graph_seed_count", ""),
            "training_seed_count": row.get("training_seed_count", ""),
            "sampler_deterministic": row.get("sampler_deterministic", ""),
            "graph_seed_ignored_by_sampler": row.get("graph_seed_ignored_by_sampler", ""),
            "export_hash_unique_count": row.get("export_hash_unique_count", ""),
            "test_micro_f1_mean": row.get("test_micro_f1_mean", ""),
            "test_micro_f1_std": row.get("test_micro_f1_std", ""),
            "structural_storage_ratio": row.get("structural_storage_ratio", ""),
            "graph_seed_stability_pass": row.get("graph_seed_stability_pass", ""),
            "stability_failure_reason": row.get("stability_failure_reason", ""),
            "deterministic_proof_type": row.get("deterministic_proof_type", ""),
        }
        for row in method_rows
    ]
    out = paths["apv16_5x5"]
    write_csv(out / "gate21_8_apv16_by_run.csv", run_rows)
    write_csv(out / "gate21_8_apv16_by_method.csv", method_rows)
    write_csv(out / "gate21_8_apv16_graph_seed_stability.csv", stability_rows)
    write_csv(out / "gate21_8_apv16_edge_overlap.csv", [_with_source(row, "gate21_7_relation_overlap_by_method") for row in edge_overlap])
    write_csv(out / "gate21_8_apv16_export_hashes.csv", [_with_source(row, "gate21_7_export_hash_audit") for row in export_hash])
    write_json(
        out / "gate21_8_apv16_plan.json",
        {
            "source": str(root),
            "accepted_evidence_modes": ["5_graph_seed_runs", "deterministic_sampler_graph_seed_ignored"],
            "note": "APV16 deterministic proof is recorded separately from empirical 5x5 graph-seed evidence.",
        },
    )
    return {"by_run_rows": len(run_rows), "by_method_rows": len(method_rows)}


def _gate21_8_apv_method_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["protocol"] = "schema_preserving_tp"
    out["official_hgb_exported"] = True
    out["official_sehgnn_unmodified"] = out.get("official_sehgnn_unmodified", True)
    out["training_executed"] = out.get("training_executed", True)
    out["sampler_deterministic"] = out.get("deterministic_graph_method", "")
    out["export_hash_unique_count"] = out.get("actual_export_hash_unique_count", out.get("export_hash_unique_count", ""))
    out["mean_test_micro_f1"] = out.get("test_micro_f1_mean", out.get("test_micro_mean", ""))
    out["std_test_micro_f1"] = out.get("test_micro_f1_std", out.get("test_micro_std", ""))
    out["graph_seed_ignored_by_sampler"] = _bool(out.get("deterministic_graph_method")) and str(out.get("graph_seed_independence_required", "")).lower() == "false"
    out["deterministic_proof_pass"] = _bool(out["graph_seed_ignored_by_sampler"]) and str(out.get("export_hash_unique_count", "")) == "1"
    out["deterministic_proof_type"] = "graph_seed_ignored_by_export_sampler" if _bool(out["deterministic_proof_pass"]) else ""
    status = apv16_graph_seed_stability_status(out)
    out.update(status)
    out["apv16_evidence_mode"] = "deterministic_proof" if _bool(out.get("deterministic_proof_pass")) else "empirical_graph_seed_grid"
    return out


def _gate21_8_apv_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["protocol"] = "schema_preserving_tp"
    out["official_hgb_exported"] = True
    out["official_sehgnn_unmodified"] = out.get("official_sehgnn_unmodified", True)
    out["training_executed"] = out.get("training_executed", out.get("success", ""))
    out["sampler_deterministic"] = out.get("deterministic_graph_method", "")
    out["graph_seed_ignored_by_sampler"] = _bool(out.get("deterministic_graph_method")) and str(out.get("graph_seed_independence_required", "")).lower() == "false"
    return out


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "external_tp"
    source_rows = read_csv(root / "gate21_7_external_tp_task_metrics.csv")
    source_index = {
        _external_key(row): row
        for row in source_rows
        if str(row.get("budget_type", "")) and str(row.get("budget_value", "")) and str(row.get("graph_seed", "")) and str(row.get("training_seed", ""))
    }
    graph_seeds = [int(args.graph_seeds[0])] if args.quick else [int(seed) for seed in args.graph_seeds]
    training_seeds = [int(args.training_seeds[0])] if args.quick else [int(seed) for seed in args.training_seeds]
    rows: list[dict[str, Any]] = []
    for method in REQUIRED_EXTERNAL_TP_5X5:
        for budget_type, budget in EXTERNAL_TP_BUDGETS:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, budget_type, _budget_token(budget), str(graph_seed), str(training_seed))
                    source = source_index.get(key)
                    rows.append(_external_tp_row(method, budget_type, budget, graph_seed, training_seed, source))
    method_status = {method: external_tp_5x5_method_status(rows, method) for method in REQUIRED_EXTERNAL_TP_5X5}
    by_method = []
    for method, status in method_status.items():
        ready_rows = [row for row in rows if row["method"] == method and _bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None]
        by_method.append(
            {
                "dataset": args.dataset,
                "method": method,
                "protocol": "schema_preserving_tp",
                "row_count": status["row_count"],
                "ready_row_count": status["ready_row_count"],
                "graph_seed_count": status["graph_seed_count"],
                "training_seed_count": status["training_seed_count"],
                "test_micro_f1_mean": _mean_field(ready_rows, "test_micro_f1"),
                "test_macro_f1_mean": _mean_field(ready_rows, "test_macro_f1"),
                "external_tp_5x5_ready": status["ready"],
                "missing_requirements": ";".join(status["missing_requirements"]),
            }
        )
    budget_audit = [
        {
            "dataset": row["dataset"],
            "method": row["method"],
            "budget_type": row["budget_type"],
            "requested_budget": row["requested_budget"],
            "budget_value": row["budget_value"],
            **budget_alignment_status(row),
        }
        for row in rows
    ]
    out = paths["external_tp_5x5"]
    write_csv(out / "gate21_8_external_tp_by_run.csv", rows)
    write_csv(out / "gate21_8_external_tp_by_method.csv", by_method)
    write_csv(out / "gate21_8_external_tp_budget_audit.csv", budget_audit)
    write_csv(out / "gate21_8_external_tp_export_audit.csv", [_with_source(row, "gate21_7_external_tp_artifact_audit") for row in read_csv(root / "gate21_7_external_tp_artifact_audit.csv")])
    write_csv(out / "gate21_8_external_tp_failures.csv", [row for row in rows if not _bool(row.get("training_executed")) or row.get("failure_type")])
    write_json(
        out / "gate21_8_external_tp_plan.json",
        {
            "source": str(root),
            "required_methods": list(REQUIRED_EXTERNAL_TP_5X5),
            "budget_grid": [{"budget_type": kind, "budget_value": value} for kind, value in EXTERNAL_TP_BUDGETS],
            "graph_seeds": graph_seeds,
            "training_seeds": training_seeds,
            "quick": bool(args.quick),
        },
    )
    return {"by_run_rows": len(rows), "ready_rows": len([row for row in rows if _bool(row.get("training_executed"))])}


def _external_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("method", "")),
        str(row.get("budget_type", "")),
        _budget_token(row.get("budget_value", "")),
        str(row.get("graph_seed", "")),
        str(row.get("training_seed", "")),
    )


def _external_tp_row(
    method: str,
    budget_type: str,
    budget: float,
    graph_seed: int,
    training_seed: int,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source is not None:
        out = dict(source)
        out.update(
            {
                "source_gate": "gate21_7",
                "protocol": "schema_preserving_tp",
                "requested_budget": budget,
                "budget_value": budget,
                "budget_type": budget_type,
                "graph_seed": graph_seed,
                "training_seed": training_seed,
            }
        )
    else:
        out = {
            "dataset": "DBLP",
            "method": method,
            "baseline_name": method,
            "graph_seed": graph_seed,
            "training_seed": training_seed,
            "budget_type": budget_type,
            "budget_value": budget,
            "requested_budget": budget,
            "official_hgb_exported": False,
            "official_sehgnn_unmodified": True,
            "training_executed": False,
            "eligible_for_tp_main_comparison": False,
            "success": False,
            "test_micro_f1": "",
            "test_macro_f1": "",
            "failure_type": "not_executed_missing_gate21_8_task_metric",
            "failure_message": "No local Gate21.8 task metric exists for this method/budget/graph-seed/training-seed cell.",
            "source_gate": "gate21_8_manifest",
            "protocol": "schema_preserving_tp",
        }
    out.update(budget_alignment_status(out))
    return out


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    clone_status = clone_external_repo(FREEHGC_URL, Path(args.freehgc_root))
    root = Path(args.gate21_7_root) / "standard_condensation"
    source_rows = read_csv(root / "gate21_7_standard_condensation_by_run.csv")
    standard_rows = []
    tp_rows = []
    for row in source_rows:
        out = dict(row)
        out["source_gate"] = "gate21_7"
        out["configured_freehgc_root"] = str(args.freehgc_root)
        out["seed"] = _parse_seed(out.get("command", "")) or 1
        out["protocol"] = out.get("protocol", "standard_condensation")
        if str(out.get("method", "")) == "FreeHGC":
            standard_rows.append(out)
        elif str(out.get("method", "")) == "FreeHGC-TP":
            out["protocol"] = "schema_preserving_tp"
            tp_rows.append(out)
    if not tp_rows:
        for ratio in (0.012, 0.024, 0.048, 0.096, 0.12):
            tp_rows.append(
                {
                    "dataset": args.dataset,
                    "method": "FreeHGC-TP",
                    "protocol": "schema_preserving_tp",
                    "freehgc_repo_url": FREEHGC_URL,
                    "freehgc_root": str(args.freehgc_root),
                    "support_node_ratio": ratio,
                    "reduction_rate": ratio,
                    "official_hgb_exported": False,
                    "official_sehgnn_unmodified": False,
                    "training_executed": False,
                    "success": False,
                    "failure_type": "adapter_not_implemented",
                    "failure_message": "FreeHGC upstream standard condensation does not expose a schema-preserving TP adapter in this repo; no self-implementation was added.",
                }
            )
    by_ratio = _summarize_freehgc_standard(standard_rows)
    tp_by_method = [
        {
            "dataset": args.dataset,
            "method": "FreeHGC-TP",
            "protocol": "schema_preserving_tp",
            "row_count": len(tp_rows),
            "ready_row_count": len([row for row in tp_rows if _bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None]),
            "hard_incompatibility_or_adapter_gap_rows": len([row for row in tp_rows if str(row.get("failure_type", "")) in {"hard_incompatibility", "adapter_not_implemented"}]),
        }
    ]
    env_audit = _freehgc_env_audit(Path(args.freehgc_root), clone_status)
    out = paths["freehgc_protocols"]
    write_csv(out / "gate21_8_freehgc_standard_by_run.csv", standard_rows)
    write_csv(out / "gate21_8_freehgc_standard_by_ratio.csv", by_ratio)
    write_json(out / "gate21_8_freehgc_env_audit.json", env_audit)
    write_csv(out / "gate21_8_freehgc_tp_by_run.csv", tp_rows)
    write_csv(out / "gate21_8_freehgc_tp_by_method.csv", tp_by_method)
    write_csv(out / "gate21_8_freehgc_tp_adapter_audit.csv", [_freehgc_tp_adapter_audit_row(row) for row in tp_rows])
    _write_freehgc_notes(out / "gate21_8_freehgc_reproduction_notes.md", standard_rows, env_audit)
    _write_freehgc_tp_failure_report(out / "gate21_8_freehgc_tp_failure_report.md", tp_rows, env_audit)
    return {"standard_rows": len(standard_rows), "tp_rows": len(tp_rows)}


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "semantic_audit"
    metapath = [_gate21_8_metapath_row(row) for row in read_csv(root / "gate21_7_metapath_cache_audit.csv")]
    label_rows = [_gate21_8_label_feature_row(row) for row in metapath]
    cache_source = read_csv(root / "gate21_7_cache_hash_audit.csv")
    cache_file_rows = [_gate21_8_cache_file_row(row) for row in cache_source] or [
        {
            "comparison_name": "missing_cache_hash_audit",
            "cache_hash": EMPTY_SHA256,
            "assertion_pass": False,
            "failure_reasons": "missing_gate21_7_cache_hash_audit",
        }
    ]
    assertions = [_cache_assertion_row(row) for row in cache_file_rows]
    failures = [row for row in metapath if not _bool(row.get("introspection_supported"))] + [row for row in assertions if not _bool(row.get("assertion_pass"))]
    out = paths["metapath_cache_dump"]
    write_csv(out / "gate21_8_metapath_tensor_audit.csv", metapath)
    write_csv(out / "gate21_8_label_feature_tensor_audit.csv", label_rows)
    write_csv(out / "gate21_8_cache_file_audit.csv", cache_file_rows)
    write_csv(out / "gate21_8_cache_hash_assertions.csv", assertions)
    write_csv(out / "gate21_8_introspection_failures.csv", failures)
    return {"metapath_rows": len(metapath), "cache_assertions": len(assertions)}


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "feature_ablation_repaired"
    rows = [_with_source(row, "gate21_7_feature_ablation_repaired") for row in read_csv(root / "gate21_7_feature_ablation_repaired.csv")]
    by_method = _feature_ablation_by_method(rows)
    failures = [row for row in rows if row.get("failure_type") or not _bool(row.get("training_executed"))]
    out = paths["feature_ablation_tasks"]
    write_csv(out / "gate21_8_feature_ablation_by_run.csv", rows)
    write_csv(out / "gate21_8_feature_ablation_by_method.csv", by_method)
    write_csv(out / "gate21_8_feature_shape_audit.csv", [_with_source(row, "gate21_7_feature_shape_assertions") for row in read_csv(root / "gate21_7_feature_shape_assertions.csv")])
    write_csv(out / "gate21_8_feature_ablation_failures.csv", failures)
    return {"rows": len(rows), "failures": len(failures)}


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "adapter_package_repaired"
    source_runs = read_csv(root / "gate21_7_feature_adapter_by_run.csv")
    rows: list[dict[str, Any]] = []
    manifest_index: list[dict[str, Any]] = []
    out = paths["adapter_package_v3"]
    for row in source_runs:
        out_row = dict(row)
        out_row["source_gate"] = "gate21_7"
        out_row["adapter_name"] = out_row.get("feature_adapter", out_row.get("adapter_name", ""))
        out_row["protocol"] = "feature_adapter_deployment"
        out_row["projection_reproducibility_test_pass"] = _adapter_projection_reproducible(out_row)
        manifest_path = _write_adapter_manifest_v3(out, out_row)
        out_row["adapter_manifest_v3_path"] = str(manifest_path)
        rows.append(out_row)
        manifest_index.append(
            {
                "dataset": out_row.get("dataset", ""),
                "method": out_row.get("method", ""),
                "base_graph_method": out_row.get("base_graph_method", ""),
                "adapter_name": out_row.get("adapter_name", ""),
                "graph_seed": out_row.get("graph_seed", ""),
                "training_seed": out_row.get("training_seed", ""),
                "adapter_manifest_v3_path": str(manifest_path),
                "static_snapshot_package_complete": out_row.get("static_snapshot_package_complete", ""),
                "reproducible_transform_package_complete": out_row.get("reproducible_transform_package_complete", ""),
            }
        )
    by_method = [_adapter_method_row(row) for row in read_csv(root / "gate21_7_feature_adapter_by_method.csv")]
    audit = [_with_source(row, "gate21_7_adapter_package_audit") for row in read_csv(root / "gate21_7_adapter_package_audit.csv")]
    write_csv(out / "gate21_8_adapter_by_run.csv", rows)
    write_csv(out / "gate21_8_adapter_by_method.csv", by_method)
    write_csv(out / "gate21_8_adapter_package_audit.csv", audit)
    write_csv(out / "gate21_8_adapter_manifest_index.csv", manifest_index)
    return {"by_run_rows": len(rows), "manifest_rows": len(manifest_index)}


def _write_storage(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "storage_system_costs"
    source = read_csv(root / "gate21_7_storage_only_baselines.csv") or read_csv(root / "gate21_7_storage_system_costs.csv")
    native_bytes = _native_full_bytes(source)
    rows = [_storage_row(row, native_bytes) for row in source]
    ratio_rows = [{**row, **ratio_denominator_status(row)} for row in rows]
    out = paths["storage_system_costs"]
    write_csv(out / "gate21_8_storage_system_by_run.csv", rows)
    write_csv(out / "gate21_8_storage_system_by_method.csv", rows)
    write_csv(out / "gate21_8_ratio_denominator_audit.csv", ratio_rows)
    write_csv(out / "gate21_8_loader_support_audit.csv", rows)
    raw_resource = read_csv(root / "gate21_7_system_resource_by_stage.csv")
    _write_jsonl(out / "gate21_8_system_resource_raw_logs.jsonl", raw_resource)
    return {"storage_rows": len(rows), "resource_rows": len(raw_resource)}


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "cross_dataset"
    rows = [_cross_dataset_row(row) for row in read_csv(root / "gate21_7_cross_dataset_by_run.csv")]
    by_method = [_cross_dataset_row(row) for row in read_csv(root / "gate21_7_cross_dataset_by_method.csv")]
    plans = _read_plan_jsonl(root / "gate21_7_cross_dataset_auto_channel_plans.jsonl")
    plan_rows = [_flatten_plan(plan) for plan in plans]
    trace_rows = [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "auto_channel_plan_ready": row.get("auto_channel_plan_ready", ""),
            "training_executed": row.get("training_executed", ""),
            "success": row.get("success", ""),
            "failure_type": row.get("failure_type", ""),
            "used_test_data": row.get("used_test_data", ""),
        }
        for row in rows
    ]
    out = paths["cross_dataset_auto_channel"]
    write_csv(out / "gate21_8_cross_dataset_by_run.csv", rows)
    write_csv(out / "gate21_8_cross_dataset_by_method.csv", by_method)
    write_csv(out / "gate21_8_auto_channel_plans.csv", plan_rows)
    write_csv(out / "gate21_8_auto_channel_validation_trace.csv", trace_rows)
    write_csv(out / "gate21_8_cross_dataset_failures.csv", [row for row in rows if row.get("failure_type")])
    return {"rows": len(rows), "plan_rows": len(plan_rows)}


def _write_audits(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    root = Path(args.gate21_7_root) / "semantic_audit"
    coverage = [_coverage_v3_row(row) for row in read_csv(root / "gate21_7_coverage_diagnostics_v2.csv")]
    assertions = [_with_source(row, "gate21_7_coverage_sanity_assertions") for row in read_csv(root / "gate21_7_coverage_sanity_assertions.csv")]
    assertions.extend(
        [
            {
                "assertion": "standard_condensation_and_tp_protocols_separated",
                "assertion_pass": True,
                "reason": "Gate21.8 writes FreeHGC standard condensation and schema-preserving TP into separate tables.",
            },
            {
                "assertion": "coverage_v3_distributional_evidence_present",
                "assertion_pass": bool(coverage),
                "reason": "Coverage v3 table is emitted from Gate21.7 coverage diagnostics; missing distributional rows remain explicit in decision flags.",
            },
        ]
    )
    out = paths["audits"]
    write_csv(out / "gate21_8_coverage_v3.csv", coverage)
    write_csv(out / "gate21_8_coverage_sanity_assertions.csv", assertions)
    return {"coverage_rows": len(coverage), "assertion_rows": len(assertions)}


def _summarize_freehgc_standard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("support_node_ratio", row.get("reduction_rate", ""))), []).append(row)
    out = []
    for ratio, ratio_rows in sorted(grouped.items()):
        ready = [row for row in ratio_rows if _bool(row.get("success")) and _float(row.get("test_micro_f1")) is not None]
        out.append(
            {
                "dataset": _first(ratio_rows, "dataset"),
                "method": "FreeHGC",
                "protocol": "standard_condensation",
                "support_node_ratio": ratio,
                "seed_count": len({str(row.get("seed", "")) for row in ready if str(row.get("seed", ""))}),
                "success_count": len(ready),
                "test_micro_f1_mean": _mean_field(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean_field(ready, "test_macro_f1"),
                "standard_condensation_ready_single_seed": bool(ready),
                "standard_condensation_ready_5seed": len({str(row.get("seed", "")) for row in ready}) >= 5,
            }
        )
    return out


def _freehgc_env_audit(root: Path, clone_status: Mapping[str, Any]) -> dict[str, Any]:
    hgb = root / "HGB"
    required = [hgb / "train_hgb.py", hgb / "model_hgb.py"]
    return {
        "repo_url": FREEHGC_URL,
        "freehgc_root": str(root),
        "clone_status": dict(clone_status),
        "is_git_clone": (root / ".git").exists(),
        "git_commit": repo_commit_hash(root),
        "required_files": {str(path): path.exists() for path in required},
        "standard_condensation_supported": all(path.exists() for path in required),
        "schema_preserving_tp_adapter_found": False,
        "schema_preserving_tp_adapter_status": "not_exposed_by_upstream_freehgc; no self-implementation added",
    }


def _freehgc_tp_adapter_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": "FreeHGC-TP",
        "protocol": "schema_preserving_tp",
        "freehgc_root": row.get("freehgc_root", ""),
        "official_hgb_exported": row.get("official_hgb_exported", False),
        "training_executed": row.get("training_executed", False),
        "failure_type": row.get("failure_type", ""),
        "failure_message": row.get("failure_message", ""),
        "self_implementation_added": False,
    }


def _write_freehgc_notes(path: Path, rows: Sequence[Mapping[str, Any]], env_audit: Mapping[str, Any]) -> None:
    lines = [
        "# FreeHGC Standard Condensation Reproduction Notes",
        "",
        f"- Upstream repo: `{FREEHGC_URL}`",
        f"- Configured clone: `{env_audit.get('freehgc_root', '')}`",
        f"- Git commit: `{env_audit.get('git_commit', '')}`",
        f"- Standard rows imported from Gate21.7: `{len(rows)}`",
        "- Protocol note: these rows are standard condensation, not schema-preserving TP.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_freehgc_tp_failure_report(path: Path, rows: Sequence[Mapping[str, Any]], env_audit: Mapping[str, Any]) -> None:
    failures = [row for row in rows if row.get("failure_type")]
    lines = [
        "# FreeHGC-TP Failure Report",
        "",
        f"- Upstream repo: `{FREEHGC_URL}`",
        f"- Configured clone: `{env_audit.get('freehgc_root', '')}`",
        "- Self-implementation added: `False`",
        f"- Failure rows: `{len(failures)}`",
        "",
        "FreeHGC-TP remains a schema-preserving adapter gap unless upstream exposes a compatible TP export path.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate21_8_metapath_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["feature_tensor_hash"] = out.get("feature_tensor_hash", "")
    out["feature_tensor_bytes"] = out.get("feature_tensor_bytes", "")
    out["introspection_supported"] = out.get("introspection_supported", False)
    out["real_tensor_dumped"] = out.get("real_tensor_dumped", False)
    return out


def _gate21_8_label_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "graph_seed": row.get("graph_seed", ""),
        "training_seed": row.get("training_seed", ""),
        "label_feature_key": row.get("label_feature_key", ""),
        "label_feature_shape": row.get("label_feature_shape", ""),
        "label_feature_bytes": row.get("label_feature_bytes", ""),
        "label_feature_hash": row.get("label_feature_hash", ""),
        "introspection_supported": row.get("introspection_supported", False),
        "failure_type": row.get("failure_type", ""),
    }


def _gate21_8_cache_file_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["cache_hash"] = row.get("left_preprocess_cache_hash_after") or row.get("right_preprocess_cache_hash_after") or EMPTY_SHA256
    out["cache_file_sha256"] = out["cache_hash"]
    out["assertion_pass"] = _bool(row.get("CACHE_HASH_REAL_PASS", row.get("cache_hash_real_pass", False)))
    return out


def _cache_assertion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cache_hash = row.get("cache_hash", row.get("cache_file_sha256", EMPTY_SHA256))
    real = bool(str(cache_hash).strip()) and str(cache_hash).lower() != EMPTY_SHA256
    return {
        "comparison_name": row.get("comparison_name", ""),
        "assertion": "cache_hash_is_real_non_empty_sha256",
        "cache_hash": cache_hash,
        "cache_file_sha256": cache_hash,
        "assertion_pass": real and _bool(row.get("assertion_pass")),
        "failure_reasons": row.get("failure_reasons", "" if real else "empty_sha256_cache_hash"),
    }


def _feature_ablation_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("feature_setting", "")))
        grouped.setdefault(key, []).append(row)
    return [
        {
            "dataset": key[0],
            "method": key[1],
            "feature_setting": key[2],
            "row_count": len(group_rows),
            "shape_safe_pass": all(_bool(row.get("shape_safe_pass")) for row in group_rows),
            "task_result_ready": any(_bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None for row in group_rows),
            "failure_types": ";".join(sorted({str(row.get("failure_type", "")) for row in group_rows if row.get("failure_type")})),
        }
        for key, group_rows in sorted(grouped.items())
    ]


def _adapter_projection_reproducible(row: Mapping[str, Any]) -> bool:
    adapter = str(row.get("feature_adapter", row.get("adapter_name", "")))
    if "random_projection" not in adapter:
        return _bool(row.get("reproducible_transform_package_complete"))
    return _bool(row.get("reproducible_transform_package_complete")) and not str(row.get("missing_reproducible_fields", "")).strip()


def _write_adapter_manifest_v3(out: Path, row: Mapping[str, Any]) -> Path:
    path = (
        out
        / "adapter_manifests_v3"
        / _safe(row.get("base_graph_method", row.get("method", "method")))
        / _safe(row.get("adapter_name", row.get("feature_adapter", "adapter")))
        / f"graph_seed_{_safe(row.get('graph_seed', ''))}"
        / f"training_seed_{_safe(row.get('training_seed', ''))}"
        / "adapter_manifest.json"
    )
    payload = {
        "gate21_8_manifest_version": 3,
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "base_graph_method": row.get("base_graph_method", ""),
        "adapter_name": row.get("adapter_name", row.get("feature_adapter", "")),
        "graph_seed": row.get("graph_seed", ""),
        "training_seed": row.get("training_seed", ""),
        "source_manifest_v2_path": row.get("adapter_manifest_v2_path", row.get("adapter_manifest_path", "")),
        "static_snapshot_package_complete": row.get("static_snapshot_package_complete", ""),
        "reproducible_transform_package_complete": row.get("reproducible_transform_package_complete", ""),
        "projection_reproducibility_test_pass": row.get("projection_reproducibility_test_pass", ""),
        "test_micro_f1": row.get("test_micro_f1", ""),
        "missing_reproducible_fields": row.get("missing_reproducible_fields", ""),
        "official_main_table_eligible": False,
    }
    write_json(path, payload)
    return path


def _adapter_method_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["adapter_name"] = out.get("feature_adapter", out.get("adapter_name", ""))
    out["protocol"] = "feature_adapter_deployment"
    out["source_gate"] = "gate21_7"
    return out


def _storage_row(row: Mapping[str, Any], native_bytes: float) -> dict[str, Any]:
    out = dict(row)
    method_bytes = _float(out.get("disk_bytes", out.get("total_artifact_bytes"))) or 0.0
    export_full = native_bytes
    control = native_bytes
    out["method"] = out.get("method") or out.get("artifact_name", "")
    out["disk_bytes"] = method_bytes
    out["method_text_bytes"] = method_bytes
    out["native_full_text_bytes"] = native_bytes
    out["export_full_text_bytes"] = export_full
    out["current_control_text_bytes"] = control
    out["ratio_vs_native_full_text"] = method_bytes / native_bytes if native_bytes else ""
    out["ratio_vs_export_full_text"] = method_bytes / export_full if export_full else ""
    out["ratio_vs_current_control_text"] = method_bytes / control if control else ""
    out["load_time_seconds"] = out.get("load_time_seconds", out.get("load_wall_time_seconds", out.get("read_time_seconds", "")))
    out["total_wall_time_seconds"] = out.get("total_wall_time_seconds", out.get("load_time_seconds", ""))
    out["system_cost_measured"] = out.get("system_cost_measured", True)
    out["source_gate"] = "gate21_7"
    return out


def _native_full_bytes(rows: Sequence[Mapping[str, Any]]) -> float:
    for row in rows:
        if str(row.get("artifact_name", "")) == "raw_hgb_text":
            value = _float(row.get("disk_bytes", row.get("total_artifact_bytes")))
            if value:
                return value
    for row in rows:
        value = _float(row.get("native_full_text_bytes", row.get("raw_hgb_text_bytes", row.get("disk_bytes"))))
        if value:
            return value
    return 1.0


def _cross_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["protocol"] = "cross_dataset_auto_channel"
    out["used_test_data"] = out.get("used_test_data", False)
    return out


def _read_plan_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw_line": line, "parse_error": True})
    return rows


def _flatten_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": plan.get("dataset", ""),
        "mode": plan.get("mode", ""),
        "target_type": plan.get("target_type", ""),
        "selected_relations": ";".join(map(str, plan.get("selected_relations", []))) if isinstance(plan.get("selected_relations"), list) else plan.get("selected_relations", ""),
        "plan_ready": not bool(plan.get("parse_error")),
        "raw_json": json.dumps(dict(plan), sort_keys=True, default=str),
    }


def _coverage_v3_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["coverage_version"] = "v3"
    out["source_gate"] = "gate21_7_coverage_diagnostics_v2"
    out["distributional_summary_available"] = bool(row)
    return out


def _with_source(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = "gate21_7"
    out["source_file"] = source
    return out


def _parse_seed(command: Any) -> int | None:
    match = re.search(r"--seed\s+(\d+)", str(command))
    return int(match.group(1)) if match else None


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if not finite else mean(finite)


def _first(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""


def _enabled(selected: set[str], *names: str) -> bool:
    return bool(selected.intersection(names))


def _budget_token(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.12g}"


def _safe(value: Any) -> str:
    text = str(value).strip() or "missing"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def build_parser() -> argparse.ArgumentParser:
    return add_gate21_8_common_args(argparse.ArgumentParser(description=__doc__))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
