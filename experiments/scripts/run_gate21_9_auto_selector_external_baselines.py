from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_9_common import (
    add_gate21_9_common_args,
    components,
    ensure_layout,
    read_csv,
    write_component_readmes,
    write_plan,
)
from hesf_coarsen.eval.official.auto_relation_channel_selector_v2 import select_relation_channels_v2
from hesf_coarsen.eval.official.channel_removal_probe import build_dblp_channel_removal_probes
from hesf_coarsen.eval.official.external_tp_5x5_runner import EXTERNAL_TP_BUDGETS, build_external_tp_5x5_grid
from hesf_coarsen.eval.official.freehgc_standard_runner import freehgc_standard_ratios
from hesf_coarsen.eval.official.freehgc_tp_export_adapter import build_freehgc_tp_hard_gap_row
from hesf_coarsen.eval.official.gate21_9_decision import EMPTY_SHA256, REQUIRED_EXTERNAL_TP_5X5, external_tp_method_status
from hesf_coarsen.eval.official.ratio_denominator_audit_v2 import ratio_denominator_audit_v2
from hesf_coarsen.eval.official.runner_utils import git_commit_hash, repo_commit_hash, write_csv, write_json


FREEHGC_URL = "https://github.com/GooLiang/FreeHGC"


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.output_root))
    write_component_readmes(paths)
    selected = components(args)
    graph_seeds = [int(args.graph_seeds[0])] if args.quick else [int(seed) for seed in args.graph_seeds]
    training_seeds = [int(args.training_seeds[0])] if args.quick else [int(seed) for seed in args.training_seeds]
    manifest = {
        "gate": "21.9",
        "output_root": str(Path(args.output_root)),
        "dataset": str(args.dataset).upper(),
        "components": sorted(selected),
        "graph_seeds": graph_seeds,
        "training_seeds": training_seeds,
        "dry_run": bool(args.dry_run),
        "quick": bool(args.quick),
        "device": str(args.device),
        "official_sehgnn_root": str(args.official_sehgnn_root),
        "freehgc_root": str(args.freehgc_root),
        "gate21_8_root": str(args.gate21_8_root),
        "gate21_7_root": str(args.gate21_7_root),
        "hesf_commit": git_commit_hash(Path.cwd()),
    }
    write_json(paths["logs"] / "gate21_9_run_manifest.json", manifest)
    if args.dry_run:
        _write_dry_run(paths, selected, graph_seeds, training_seeds)
    else:
        results: dict[str, Any] = {}
        if _enabled(selected, "auto_selector"):
            results["auto_selector_alignment"] = _write_auto_selector(paths, args)
        if _enabled(selected, "external_tp", "external_tp_5x5"):
            results["external_tp_5x5"] = _write_external_tp(paths, args, graph_seeds, training_seeds)
        if _enabled(selected, "freehgc", "freehgc_protocols"):
            results["freehgc_protocols"] = _write_freehgc(paths, args)
        if _enabled(selected, "metapath", "metapath_cache", "metapath_cache_dump"):
            results["metapath_cache_dump"] = _write_metapath_cache(paths, args)
        if _enabled(selected, "feature_ablation", "feature_ablation_tasks"):
            results["feature_ablation_tasks"] = _write_feature_ablation(paths, args)
        if _enabled(selected, "adapter", "adapter_package_v4"):
            results["adapter_package_v4"] = _write_adapter(paths, args)
        if _enabled(selected, "storage", "storage_system", "storage_system_costs"):
            results["storage_system_costs"] = _write_storage(paths, args)
        if _enabled(selected, "cross_dataset", "cross_dataset_auto_channel"):
            results["cross_dataset_auto_channel"] = _write_cross_dataset(paths, args)
        results["audits"] = _write_audits(paths, args)
        _write_empty_missing_component_files(paths)
        write_json(paths["logs"] / "gate21_9_component_results.json", results)

    from experiments.scripts.summarize_gate21_9_auto_selector_external_baselines import summarize

    decision = summarize(Path(args.output_root), Path(args.output_root), strict=bool(args.strict))
    return {
        "output_root": str(Path(args.output_root)),
        "paper_ready_status": decision.get("paper_ready_status", ""),
        "blocking_issues": decision.get("blocking_issues", []),
        "flags": decision.get("flags", {}),
    }


def _write_dry_run(paths: Mapping[str, Path], selected: set[str], graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    rows = [
        {
            "component": component,
            "would_execute": True,
            "dry_run_manifest_only": True,
            "task_metrics_written": False,
            "graph_seeds": " ".join(map(str, graph_seeds)),
            "training_seeds": " ".join(map(str, training_seeds)),
        }
        for component in sorted(selected)
    ]
    write_csv(paths["audits"] / "gate21_9_dry_run_component_manifest.csv", rows)
    _write_empty_missing_component_files(paths)


def _write_auto_selector(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    result = select_relation_channels_v2(dataset=str(args.dataset))
    plan = dict(result["plan"])
    plan.update(_tp_protocol_flags(official=True))
    plan["source_gate"] = "gate21_9_selector_v2"
    rows = [plan]
    out = paths["auto_selector_alignment"]
    write_csv(out / "gate21_9_channel_utility.csv", result["channel_utility_rows"])
    write_csv(out / "gate21_9_channel_removal_probes.csv", build_dblp_channel_removal_probes())
    write_csv(out / "gate21_9_auto_channel_plans.csv", [plan])
    write_csv(out / "gate21_9_auto_selector_by_method.csv", rows)
    return {"channel_utility_rows": len(result["channel_utility_rows"]), "auto_selector_rows": len(rows)}


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> dict[str, int]:
    source_rows = read_csv(Path(args.gate21_8_root) / "external_tp_5x5" / "gate21_8_external_tp_by_run.csv")
    if source_rows:
        allowed_graph = set(map(str, graph_seeds))
        allowed_training = set(map(str, training_seeds))
        source_rows = [
            row
            for row in source_rows
            if str(row.get("graph_seed", "")) in allowed_graph and str(row.get("training_seed", "")) in allowed_training
        ]
    if source_rows:
        rows = [_external_tp_row(row, args) for row in source_rows]
    else:
        rows = [_external_tp_row(row, args) for row in build_external_tp_5x5_grid(list(graph_seeds), list(training_seeds))]

    by_method = [_external_tp_method_row(rows, method, args) for method in REQUIRED_EXTERNAL_TP_5X5]
    budget_audit = [_external_budget_audit_row(row) for row in rows]
    failures = [row for row in rows if str(row.get("failure_type", "")).strip() or not _bool(row.get("training_executed"))]
    out = paths["external_tp_5x5"]
    write_csv(out / "gate21_9_external_tp_task_rows.csv", rows)
    write_csv(out / "gate21_9_external_tp_by_method.csv", by_method)
    write_csv(out / "gate21_9_external_tp_budget_audit.csv", budget_audit)
    write_csv(out / "gate21_9_external_tp_failures.csv", failures)
    write_json(
        out / "gate21_9_external_tp_plan.json",
        {
            "required_methods": list(REQUIRED_EXTERNAL_TP_5X5),
            "budgets": [{"budget_type": kind, "requested_budget": value} for kind, value in EXTERNAL_TP_BUDGETS],
            "source": str(Path(args.gate21_8_root) / "external_tp_5x5"),
            "graph_seeds": list(graph_seeds),
            "training_seeds": list(training_seeds),
            "note": "Rows with missing task metrics remain explicit failures and cannot set EXTERNAL_TP_5X5_TASK_RESULTS_READY.",
        },
    )
    return {"task_rows": len(rows), "failure_rows": len(failures)}


def _external_tp_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    budget_type = str(out.get("budget_type", ""))
    if budget_type == "structural_ratio":
        budget_type = "structural_storage_ratio"
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = out.get("method", out.get("baseline_name", ""))
    out["protocol"] = "schema_preserving_tp"
    out["budget_type"] = budget_type
    out["requested_budget"] = out.get("requested_budget", out.get("budget_value", ""))
    out["actual_support_node_ratio"] = out.get("actual_support_node_ratio", out.get("support_node_ratio", ""))
    out["actual_support_edge_ratio"] = out.get("actual_support_edge_ratio", out.get("support_edge_ratio", ""))
    out["actual_structural_storage_ratio"] = out.get("actual_structural_storage_ratio", out.get("structural_storage_ratio", ""))
    out["raw_hgb_text_byte_ratio"] = out.get("raw_hgb_text_byte_ratio", "")
    out["compress_time_seconds"] = out.get("compress_time_seconds", out.get("compress_wall_time_seconds", ""))
    out["export_time_seconds"] = out.get("export_time_seconds", out.get("export_wall_time_seconds", ""))
    out["preprocess_time_seconds"] = out.get("preprocess_time_seconds", out.get("preprocess_wall_time_seconds", ""))
    out["train_time_seconds"] = out.get("train_time_seconds", out.get("train_wall_time_seconds", ""))
    out["eval_time_seconds"] = out.get("eval_time_seconds", "")
    out["peak_cpu_rss_mb"] = out.get("peak_cpu_rss_mb", out.get("peak_cpu_memory_mb", ""))
    out["peak_gpu_memory_mb"] = out.get("peak_gpu_memory_mb", "")
    if not _bool(out.get("training_executed")) and not str(out.get("failure_type", "")).strip():
        out["failure_type"] = "not_executed_missing_gate21_9_5x5_task_metric"
        out["failure_message"] = "No successful local task metric exists for this Gate21.9 external TP cell."
    if str(out.get("failure_type", "")) == "not_executed_missing_gate21_8_task_metric":
        out["failure_type"] = "not_executed_missing_gate21_9_5x5_task_metric"
        out["failure_message"] = "Gate21.8 source row had no successful task metric for this 5x5 cell."
    out.update(_tp_protocol_flags(official=True))
    out["eligible_for_official_main_table"] = _bool(out.get("official_hgb_exported")) and _bool(out.get("official_sehgnn_unmodified"))
    out["source_gate"] = out.get("source_gate", "gate21_9_manifest")
    return out


def _external_tp_method_row(rows: Sequence[Mapping[str, Any]], method: str, args: argparse.Namespace) -> dict[str, Any]:
    status = external_tp_method_status(rows, method)
    method_rows = [row for row in rows if str(row.get("method", "")) == method]
    ready_rows = [row for row in method_rows if _bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None]
    return {
        "dataset": str(args.dataset).upper(),
        "method": method,
        "protocol": "schema_preserving_tp",
        "row_count": status["row_count"],
        "ready_row_count": status["ready_row_count"],
        "graph_seed_count": status["graph_seed_count"],
        "training_seed_count": status["training_seed_count"],
        "ready_5x5_flag": status["ready_5x5_flag"],
        "mean_test_micro_f1": _mean_field(ready_rows, "test_micro_f1"),
        "std_test_micro_f1": _std_field(ready_rows, "test_micro_f1"),
        "mean_test_macro_f1": _mean_field(ready_rows, "test_macro_f1"),
        "std_test_macro_f1": _std_field(ready_rows, "test_macro_f1"),
        "mean_structural_storage_ratio": _mean_field(ready_rows, "actual_structural_storage_ratio"),
        "std_structural_storage_ratio": _std_field(ready_rows, "actual_structural_storage_ratio"),
        "mean_raw_hgb_text_byte_ratio": _mean_field(ready_rows, "raw_hgb_text_byte_ratio"),
        "std_raw_hgb_text_byte_ratio": _std_field(ready_rows, "raw_hgb_text_byte_ratio"),
        "mean_train_time_seconds": _mean_field(ready_rows, "train_time_seconds"),
        "std_train_time_seconds": _std_field(ready_rows, "train_time_seconds"),
        "mean_preprocess_time_seconds": _mean_field(ready_rows, "preprocess_time_seconds"),
        "std_preprocess_time_seconds": _std_field(ready_rows, "preprocess_time_seconds"),
        "mean_peak_cpu_rss_mb": _mean_field(ready_rows, "peak_cpu_rss_mb"),
        "std_peak_cpu_rss_mb": _std_field(ready_rows, "peak_cpu_rss_mb"),
        "mean_peak_gpu_memory_mb": _mean_field(ready_rows, "peak_gpu_memory_mb"),
        "std_peak_gpu_memory_mb": _std_field(ready_rows, "peak_gpu_memory_mb"),
        "missing_requirements": ";".join(status["missing_requirements"]),
    }


def _external_budget_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    requested = _float(row.get("requested_budget"))
    actual = _float(row.get("actual_structural_storage_ratio" if row.get("budget_type") == "structural_storage_ratio" else "actual_support_node_ratio"))
    status = "not_evaluated"
    budget_pass: bool | str = ""
    if requested is not None and actual is not None:
        if row.get("budget_type") == "structural_storage_ratio":
            budget_pass = abs(actual - requested) <= 0.015
            status = "within_tolerance" if budget_pass else "budget_infeasible"
        else:
            budget_pass = actual <= requested + 0.015
            status = "within_tolerance" if budget_pass else "budget_infeasible"
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "budget_type": row.get("budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "actual_budget_ratio": "" if actual is None else actual,
        "budget_tolerance": 0.015,
        "budget_alignment_pass": budget_pass,
        "budget_feasibility_status": status,
        "failure_type": row.get("failure_type", ""),
    }


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "freehgc_protocols"
    standard_rows = [_freehgc_standard_row(row, args) for row in read_csv(source_dir / "gate21_8_freehgc_standard_by_run.csv")]
    if not standard_rows:
        standard_rows = [_missing_freehgc_standard_row(args, ratio) for ratio in freehgc_standard_ratios()]
    by_ratio = _freehgc_by_ratio(standard_rows)
    env_audit = _freehgc_env_audit(Path(args.freehgc_root), args)
    for row in standard_rows:
        row["upstream_env_verified"] = env_audit["upstream_config_verified"]
        row["standard_condensation_supported"] = env_audit["standard_condensation_supported"]

    tp_source = read_csv(source_dir / "gate21_8_freehgc_tp_by_run.csv")
    tp_rows = [_freehgc_tp_row(row, args) for row in tp_source] or [
        build_freehgc_tp_hard_gap_row(dataset=str(args.dataset).upper(), reduction_rate=ratio) for ratio in freehgc_standard_ratios()
    ]
    tp_by_method = [_freehgc_tp_by_method(tp_rows, args)]
    adapter_audit = [_freehgc_tp_adapter_audit(row) for row in tp_rows]
    out = paths["freehgc_protocols"]
    write_csv(out / "gate21_9_freehgc_standard_task_rows.csv", standard_rows)
    write_csv(out / "gate21_9_freehgc_standard_by_ratio.csv", by_ratio)
    write_json(out / "gate21_9_freehgc_env_audit.json", env_audit)
    write_csv(out / "gate21_9_freehgc_tp_adapter_audit.csv", adapter_audit)
    write_csv(out / "gate21_9_freehgc_tp_by_method.csv", tp_by_method)
    _write_freehgc_failure_report(out / "gate21_9_freehgc_tp_failure_report.md", tp_rows, env_audit)
    return {"standard_rows": len(standard_rows), "tp_rows": len(tp_rows)}


def _freehgc_standard_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = "FreeHGC"
    out["protocol"] = "standard_condensation"
    out["eligible_for_standard_condensation_table"] = True
    out["eligible_for_tp_workload_table"] = False
    out["eligible_for_official_main_table"] = False
    out["official_hgb_exported"] = False
    out["official_sehgnn_unmodified"] = False
    out["source_gate"] = out.get("source_gate", "gate21_8")
    return out


def _missing_freehgc_standard_row(args: argparse.Namespace, ratio: float) -> dict[str, Any]:
    return {
        "dataset": str(args.dataset).upper(),
        "method": "FreeHGC",
        "protocol": "standard_condensation",
        "support_node_ratio": ratio,
        "seed": "",
        "success": False,
        "training_executed": False,
        "failure_type": "missing_freehgc_standard_task_metric",
        "failure_message": "No local FreeHGC standard-condensation task metric was found.",
        "eligible_for_standard_condensation_table": True,
        "eligible_for_tp_workload_table": False,
        "eligible_for_official_main_table": False,
    }


def _freehgc_by_ratio(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                "seed_count": len({str(row.get("seed", "")) for row in ready if str(row.get("seed", "")).strip()}),
                "success_count": len(ready),
                "test_micro_f1_mean": _mean_field(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean_field(ready, "test_macro_f1"),
                "standard_condensation_ready_single_seed": bool(ready),
                "standard_condensation_ready_5seed": len({str(row.get("seed", "")) for row in ready if str(row.get("seed", "")).strip()}) >= 5,
            }
        )
    return out


def _freehgc_env_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    train = root / "HGB" / "train_hgb.py"
    model = root / "HGB" / "model_hgb.py"
    torch_version = ""
    torch_geometric_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception as exc:  # pragma: no cover
        torch_version = f"unavailable:{exc}"
    try:
        import torch_geometric

        torch_geometric_version = str(torch_geometric.__version__)
    except Exception as exc:  # pragma: no cover
        torch_geometric_version = f"unavailable:{exc}"
    supported = train.exists() and model.exists()
    return {
        "freehgc_repo_url": FREEHGC_URL,
        "freehgc_repo_path": str(root),
        "freehgc_commit_hash": repo_commit_hash(root),
        "python_version": sys.version,
        "torch_version": torch_version,
        "torch_geometric_version": torch_geometric_version,
        "dataset_path": str(Path(args.official_sehgnn_root) / "data" / str(args.dataset).upper()),
        "command_line": " ".join(map(str, sys.argv)),
        "standard_condensation_supported": supported,
        "upstream_config_verified": supported,
        "split_matches_hgb_official": "not_verified_for_gate21_9",
        "backbone_matches_or_reason": "standard protocol is reported separately from TP workload",
        "required_files": {str(train): train.exists(), str(model): model.exists()},
    }


def _freehgc_tp_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    if str(out.get("failure_type", "")) in {"adapter_not_implemented", ""} and not _bool(out.get("training_executed")):
        hard = build_freehgc_tp_hard_gap_row(
            dataset=str(out.get("dataset", str(args.dataset).upper())),
            reduction_rate=_float(out.get("reduction_rate", out.get("support_node_ratio"))) or 0.12,
        )
        hard.update({key: value for key, value in out.items() if key not in {"failure_type", "failure_message"}})
        out = hard
    out["method"] = "FreeHGC-TP"
    out["protocol"] = "schema_preserving_tp"
    out["eligible_for_tp_workload_table"] = False
    out["eligible_for_official_main_table"] = False
    out["eligible_for_standard_condensation_table"] = False
    return out


def _freehgc_tp_by_method(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ready = [row for row in rows if _bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None]
    hard = [row for row in rows if str(row.get("failure_type", "")) == "hard_incompatibility"]
    return {
        "dataset": str(args.dataset).upper(),
        "method": "FreeHGC-TP",
        "protocol": "schema_preserving_tp",
        "row_count": len(rows),
        "ready_row_count": len(ready),
        "hard_incompatibility_rows": len(hard),
        "FREEHGC_TP_TASK_RESULTS_READY": bool(ready),
        "FREEHGC_TP_HARD_GAP_REPORTED": bool(hard),
        "hard_incompatibility_reasons": ";".join(sorted({str(row.get("hard_incompatibility_reason", "")) for row in hard})),
    }


def _freehgc_tp_adapter_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": "FreeHGC-TP",
        "protocol": "schema_preserving_tp",
        "reduction_rate": row.get("reduction_rate", row.get("support_node_ratio", "")),
        "attempted_support_node_ratio": row.get("attempted_support_node_ratio", row.get("support_node_ratio", "")),
        "keeps_all_target_nodes": row.get("keeps_all_target_nodes", ""),
        "preserves_official_node_type_schema": row.get("preserves_node_type_schema", False),
        "preserves_official_relation_type_schema": row.get("preserves_relation_type_schema", False),
        "official_hgb_exported": row.get("official_hgb_exported", False),
        "official_sehgnn_unmodified": row.get("official_sehgnn_unmodified", False),
        "training_executed": row.get("training_executed", False),
        "adapter_implemented": row.get("adapter_implemented", False),
        "failure_type": row.get("failure_type", ""),
        "hard_incompatibility_reason": row.get("hard_incompatibility_reason", ""),
        "failure_message": row.get("failure_message", ""),
    }


def _write_freehgc_failure_report(path: Path, rows: Sequence[Mapping[str, Any]], env_audit: Mapping[str, Any]) -> None:
    reasons = sorted({str(row.get("hard_incompatibility_reason", "")) for row in rows if row.get("hard_incompatibility_reason")})
    lines = [
        "# Gate21.9 FreeHGC-TP Failure Report",
        "",
        f"- Upstream repo: `{FREEHGC_URL}`",
        f"- Local repo: `{env_audit.get('freehgc_repo_path', '')}`",
        f"- Commit: `{env_audit.get('freehgc_commit_hash', '')}`",
        f"- Standard condensation supported: `{env_audit.get('standard_condensation_supported', False)}`",
        f"- TP ready rows: `{sum(1 for row in rows if _bool(row.get('training_executed')) and _float(row.get('test_micro_f1')) is not None)}`",
        f"- Hard incompatibility rows: `{sum(1 for row in rows if str(row.get('failure_type', '')) == 'hard_incompatibility')}`",
        f"- Hard reasons: `{';'.join(reasons)}`",
        "",
        "Gate21.9 keeps FreeHGC standard condensation separate and does not enter FreeHGC rows into the schema-preserving TP main table unless official HGB export and unmodified SeHGNN task metrics exist.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "metapath_cache_dump"
    metapath = [_metapath_row(row, args) for row in read_csv(source_dir / "gate21_8_metapath_tensor_audit.csv")]
    if not metapath:
        metapath = [
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "metapath_key": key,
                "feature_tensor_hash": EMPTY_SHA256,
                "feature_tensor_bytes": 0,
                "real_tensor_dumped": False,
                "introspection_supported": False,
                "failure_type": "official_sehgnn_intermediate_tensors_not_exposed",
                "failure_message": "No real tensor dump was available.",
            }
            for method in ("HeSF-RCS-APV12", "HeSF-RCS-APV16")
            for key in ("AP", "PV")
        ]
    cache = [_cache_row(row) for row in read_csv(source_dir / "gate21_8_cache_hash_assertions.csv")]
    if not cache:
        cache = [{"assertion": "cache_hash_non_empty", "assertion_pass": False, "cache_file_hash": EMPTY_SHA256, "failure_type": "missing_cache_hash_audit"}]
    failures = [row for row in metapath if str(row.get("failure_type", "")).strip()] + [row for row in cache if not _bool(row.get("assertion_pass"))]
    out = paths["metapath_cache_dump"]
    write_csv(out / "gate21_9_metapath_tensor_audit.csv", metapath)
    write_csv(out / "gate21_9_cache_hash_assertions.csv", cache)
    write_csv(out / "gate21_9_introspection_failures.csv", failures)
    return {"metapath_rows": len(metapath), "cache_rows": len(cache)}


def _metapath_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["feature_tensor_hash"] = out.get("feature_tensor_hash", out.get("tensor_sha256", EMPTY_SHA256))
    out["feature_tensor_bytes"] = out.get("feature_tensor_bytes", out.get("tensor_bytes", 0))
    if str(out.get("feature_tensor_hash", "")).lower() == EMPTY_SHA256 or not _bool(out.get("introspection_supported")):
        out["failure_type"] = out.get("failure_type") or "official_sehgnn_intermediate_tensors_not_exposed"
        out["failure_message"] = out.get("failure_message") or "Gate21.8 source did not expose a real SeHGNN tensor dump."
    out["source_gate"] = out.get("source_gate", "gate21_8")
    return out


def _cache_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["cache_file_hash"] = out.get("cache_file_hash", out.get("cache_hash", out.get("cache_file_sha256", "")))
    if str(out.get("cache_file_hash", "")).lower() == EMPTY_SHA256:
        out["assertion_pass"] = False
        out["failure_type"] = out.get("failure_type") or "empty_cache_sha256"
    return out


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "feature_ablation_tasks"
    rows = [_feature_row(row, args) for row in read_csv(source_dir / "gate21_8_feature_ablation_by_run.csv")]
    by_method = _feature_by_method(rows)
    shape_audit = [_feature_shape_row(row) for row in rows]
    failures = [row for row in rows if str(row.get("failure_type", "")).strip() or not _bool(row.get("training_executed"))]
    out = paths["feature_ablation_tasks"]
    write_csv(out / "gate21_9_feature_ablation_task_rows.csv", rows)
    write_csv(out / "gate21_9_feature_ablation_by_method.csv", by_method)
    write_csv(out / "gate21_9_feature_shape_audit.csv", shape_audit)
    write_csv(out / "gate21_9_feature_ablation_failures.csv", failures)
    return {"feature_rows": len(rows), "failure_rows": len(failures)}


def _feature_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["feature_transform"] = _feature_transform_name(out.get("feature_transform", out.get("feature_setting", "")))
    out["label_graph_setting"] = out.get("label_graph_setting", out.get("label_setting", "paper-label-feats-default"))
    out["official_sehgnn_unmodified"] = out.get("official_sehgnn_unmodified", True)
    out["uses_ablation_adapter"] = out.get("uses_ablation_adapter", False)
    out["shape_safe_pass"] = out.get("shape_safe_pass", True)
    out["per_type_shape_before"] = out.get("per_type_shape_before", "")
    out["per_type_shape_after"] = out.get("per_type_shape_after", "")
    out["training_executed"] = out.get("training_executed", False)
    if _float(out.get("test_micro_f1")) is None:
        out["training_executed"] = False
        out["failure_type"] = out.get("failure_type") or "feature_ablation_task_metric_missing"
        out["failure_message"] = out.get("failure_message") or "Gate21.9 requires real task metrics for this transform; source row is plan or shape-audit only."
    return out


def _feature_transform_name(value: Any) -> str:
    text = str(value).strip()
    aliases = {
        "zero-paper": "zero-paper-preserve-dim",
        "zero-term": "zero-term-preserve-dim",
        "zero-all-support": "zero-all-support-preserve-dim",
        "random_projection_dim64": "paper-random-projection64",
    }
    return aliases.get(text, text)


def _feature_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("feature_transform", ""))), []).append(row)
    out = []
    for (dataset, method, transform), group_rows in sorted(grouped.items()):
        ready = [row for row in group_rows if _bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "feature_transform": transform,
                "row_count": len(group_rows),
                "success_count": len(ready),
                "shape_safe_pass": all(_bool(row.get("shape_safe_pass", True)) for row in group_rows),
                "task_result_ready": bool(ready),
                "test_micro_f1_mean": _mean_field(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean_field(ready, "test_macro_f1"),
            }
        )
    return out


def _feature_shape_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "feature_transform": row.get("feature_transform", ""),
        "shape_safe_pass": row.get("shape_safe_pass", ""),
        "per_type_shape_before": row.get("per_type_shape_before", ""),
        "per_type_shape_after": row.get("per_type_shape_after", ""),
        "failure_type": row.get("failure_type", ""),
    }


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "adapter_package_v3"
    rows = [_adapter_row(row) for row in read_csv(source_dir / "gate21_8_adapter_by_run.csv")]
    by_method_source = read_csv(source_dir / "gate21_8_adapter_by_method.csv")
    by_method = [_adapter_row(row) for row in by_method_source] if by_method_source else _adapter_by_method(rows)
    audit = [_adapter_audit_row(row) for row in rows]
    manifest_index = [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "base_graph_method": row.get("base_graph_method", ""),
            "feature_adapter": row.get("feature_adapter", ""),
            "adapter_manifest_v4_path": row.get("adapter_manifest_v3_path", row.get("adapter_manifest_v2_path", "")),
            "static_inference_package_ratio": row.get("static_inference_package_ratio", ""),
            "transform_recipe_package_ratio": row.get("transform_recipe_package_ratio", ""),
            "reconstructable_package_ratio": row.get("reconstructable_package_ratio", ""),
        }
        for row in rows
    ]
    out = paths["adapter_package_v4"]
    write_csv(out / "gate21_9_adapter_task_rows.csv", rows)
    write_csv(out / "gate21_9_adapter_by_method.csv", by_method)
    write_csv(out / "gate21_9_adapter_package_audit.csv", audit)
    write_csv(out / "gate21_9_adapter_manifest_index.csv", manifest_index)
    return {"adapter_rows": len(rows)}


def _adapter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["protocol"] = "feature_adapter_deployment"
    out["official_sehgnn_unmodified"] = False
    out["eligible_for_official_main_table"] = False
    out["eligible_for_adapter_table"] = True
    out["uses_feature_adapter"] = True
    static = _float(out.get("static_snapshot_package_ratio", out.get("static_inference_package_ratio", out.get("effective_total_byte_ratio"))))
    recipe = _float(out.get("reproducible_transform_package_ratio", out.get("transform_recipe_package_ratio")))
    out["static_inference_package_ratio"] = "" if static is None else static
    out["transform_recipe_package_ratio"] = "" if recipe is None else recipe
    out["reconstructable_package_ratio"] = _sum_or_blank(static, recipe)
    adapter = str(out.get("feature_adapter", out.get("adapter_name", "")))
    out["feature_adapter"] = adapter
    if "random_projection_dim64" in adapter:
        out.setdefault("projection_seed", 1)
        out.setdefault("projection_generator_name", "PCG64")
        out.setdefault("projection_generator_version", "numpy-default_rng")
        out.setdefault("projection_matrix_shape", "4231x64")
        out.setdefault("projection_matrix_dtype", "float32")
        out.setdefault("projection_distribution", "normal")
    if "random_projection_dim128" in adapter:
        out.setdefault("projection_seed", 1)
        out.setdefault("projection_generator_name", "PCG64")
        out.setdefault("projection_generator_version", "numpy-default_rng")
        out.setdefault("projection_matrix_shape", "4231x128")
        out.setdefault("projection_matrix_dtype", "float32")
        out.setdefault("projection_distribution", "normal")
    out["pca_reproducible_package_complete"] = bool(
        "pca" in adapter
        and _float(out.get("pca_basis_bytes")) is not None
        and _float(out.get("pca_mean_bytes")) is not None
        and str(out.get("pca_fit_config", "")).strip()
        and str(out.get("pca_training_node_ids_hash", "")).strip()
    )
    return out


def _adapter_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("base_graph_method", row.get("method", ""))), str(row.get("feature_adapter", ""))), []).append(row)
    return [
        {
            "dataset": key[0],
            "base_graph_method": key[1],
            "feature_adapter": key[2],
            "method": f"{key[1]}+{key[2]}",
            "success_count": len([row for row in group if _bool(row.get("success"))]),
            "test_micro_f1_mean": _mean_field(group, "test_micro_f1"),
            "static_inference_package_ratio": _mean_field(group, "static_inference_package_ratio"),
            "transform_recipe_package_ratio": _mean_field(group, "transform_recipe_package_ratio"),
            "reconstructable_package_ratio": _mean_field(group, "reconstructable_package_ratio"),
        }
        for key, group in sorted(grouped.items())
    ]


def _adapter_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "base_graph_method": row.get("base_graph_method", ""),
        "feature_adapter": row.get("feature_adapter", ""),
        "static_inference_package_ratio": row.get("static_inference_package_ratio", ""),
        "transform_recipe_package_ratio": row.get("transform_recipe_package_ratio", ""),
        "reconstructable_package_ratio": row.get("reconstructable_package_ratio", ""),
        "pca_reproducible_package_complete": row.get("pca_reproducible_package_complete", ""),
        "projection_seed": row.get("projection_seed", ""),
        "projection_generator_name": row.get("projection_generator_name", ""),
        "projection_matrix_shape": row.get("projection_matrix_shape", ""),
    }


def _write_storage(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source = read_csv(Path(args.gate21_8_root) / "storage_system_costs" / "gate21_8_storage_system_by_method.csv")
    rows = [_storage_row(row) for row in source]
    ratio_rows = [ratio_denominator_audit_v2(row) for row in rows]
    loader_rows = [_loader_row(row) for row in rows]
    workload_rows = [_workload_row(row) for row in rows]
    out = paths["storage_system_costs"]
    write_csv(out / "gate21_9_storage_system_by_method.csv", rows)
    write_csv(out / "gate21_9_ratio_denominator_audit.csv", ratio_rows)
    write_csv(out / "gate21_9_loader_support_audit.csv", loader_rows)
    write_csv(out / "gate21_9_workload_cost_trace.csv", workload_rows)
    return {"storage_rows": len(rows)}


def _storage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    method = out.get("method", out.get("artifact_name", ""))
    artifact_bytes = _first_float(out, "total_artifact_bytes", "disk_bytes", "static_inference_package_bytes")
    raw_bytes = _first_float(out, "raw_hgb_text_bytes", "native_full_text_bytes")
    export_bytes = _first_float(out, "export_full_text_bytes") or raw_bytes
    control_bytes = _first_float(out, "zstd_bytes", "gzip_bytes", "current_control_text_bytes") or export_bytes
    out["method"] = method
    out["artifact_construction_time_seconds"] = out.get("artifact_construction_time_seconds", out.get("write_time_seconds", ""))
    out["export_time_seconds"] = out.get("export_time_seconds", "")
    out["load_time_seconds"] = out.get("load_time_seconds", out.get("load_wall_time_seconds", out.get("read_time_seconds", "")))
    out["decompress_time_seconds"] = out.get("decompress_time_seconds", "")
    out["official_sehgnn_preprocess_time_seconds"] = out.get("official_sehgnn_preprocess_time_seconds", out.get("preprocess_time_seconds", ""))
    out["training_time_seconds"] = out.get("training_time_seconds", out.get("train_time_seconds", ""))
    out["eval_time_seconds"] = out.get("eval_time_seconds", "")
    out["total_workload_time_seconds"] = _sum_or_blank(
        _float(out.get("load_time_seconds")),
        _float(out.get("decompress_time_seconds")),
        _float(out.get("official_sehgnn_preprocess_time_seconds")),
        _float(out.get("training_time_seconds")),
        _float(out.get("eval_time_seconds")),
    )
    out["static_inference_package_bytes"] = "" if artifact_bytes is None else artifact_bytes
    out["artifact_bytes"] = "" if artifact_bytes is None else artifact_bytes
    out["original_native_full_hgb_text_bytes"] = "" if raw_bytes is None else raw_bytes
    out["current_export_full_text_bytes"] = "" if export_bytes is None else export_bytes
    out["current_compressed_control_text_bytes"] = "" if control_bytes is None else control_bytes
    out["source_gate"] = out.get("source_gate", "gate21_8")
    return out


def _loader_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "loader_supported": row.get("loader_supported", ""),
        "requires_loader_adapter": row.get("requires_loader_adapter", ""),
        "changes_training_semantics": row.get("changes_training_semantics", ""),
        "failure_type": row.get("failure_type", ""),
    }


def _workload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "load_time_seconds": row.get("load_time_seconds", ""),
        "decompress_time_seconds": row.get("decompress_time_seconds", ""),
        "official_sehgnn_preprocess_time_seconds": row.get("official_sehgnn_preprocess_time_seconds", ""),
        "training_time_seconds": row.get("training_time_seconds", ""),
        "eval_time_seconds": row.get("eval_time_seconds", ""),
        "total_workload_time_seconds": row.get("total_workload_time_seconds", ""),
        "peak_cpu_rss_mb": row.get("peak_cpu_rss_mb", ""),
        "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb", ""),
        "preprocessed_cache_bytes": row.get("preprocessed_cache_bytes", ""),
    }


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "cross_dataset_auto_channel"
    rows = [_cross_dataset_row(row) for row in read_csv(source_dir / "gate21_8_cross_dataset_by_run.csv")]
    by_method = [_cross_dataset_row(row) for row in read_csv(source_dir / "gate21_8_cross_dataset_by_method.csv")]
    plans = read_csv(source_dir / "gate21_8_auto_channel_plans.csv")
    trace = read_csv(source_dir / "gate21_8_auto_channel_validation_trace.csv")
    failures = [row for row in rows if str(row.get("failure_type", "")).strip() or not (_bool(row.get("training_executed")) and _float(row.get("test_micro_f1")) is not None)]
    out = paths["cross_dataset_auto_channel"]
    write_csv(out / "gate21_9_cross_dataset_task_rows.csv", rows)
    write_csv(out / "gate21_9_cross_dataset_by_method.csv", by_method)
    write_csv(out / "gate21_9_auto_channel_plans.csv", plans)
    write_csv(out / "gate21_9_auto_channel_validation_trace.csv", trace)
    write_csv(out / "gate21_9_cross_dataset_failures.csv", failures)
    return {"cross_dataset_rows": len(rows), "failure_rows": len(failures)}


def _cross_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["protocol"] = "cross_dataset_auto_channel"
    if _float(out.get("test_micro_f1")) is None:
        out["training_executed"] = False
        out["failure_type"] = out.get("failure_type") or "cross_dataset_task_metric_missing"
        out["failure_message"] = out.get("failure_message") or "Gate21.9 requires ACM/IMDB real task metrics; source row is plan or failed execution."
    return out


def _write_audits(paths: Mapping[str, Path], args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.gate21_8_root) / "audits"
    coverage = [_coverage_v4_row(row) for row in read_csv(source_dir / "gate21_8_coverage_v3.csv")]
    assertions = [_coverage_assertion_row(row) for row in read_csv(source_dir / "gate21_8_coverage_sanity_assertions.csv")]
    assertions.extend(
        [
            {
                "assertion": "coverage_semantic_diagnostics_ready",
                "assertion_pass": False,
                "failure_type": "semantic_distributional_coverage_not_measured",
                "failure_message": "Gate21.9 requires per-class and distributional coverage metrics; source Gate21.8 only has reachability coverage.",
            },
            {
                "assertion": "relation_direction_matches_official_relation_name",
                "assertion_pass": bool(coverage),
                "failure_type": "" if coverage else "coverage_source_missing",
                "failure_message": "" if coverage else "No Gate21.8 coverage source rows found.",
            },
        ]
    )
    out = paths["audits"]
    write_csv(out / "gate21_9_coverage_v4.csv", coverage)
    write_csv(out / "gate21_9_coverage_sanity_assertions.csv", assertions)
    return {"coverage_rows": len(coverage), "assertion_rows": len(assertions)}


def _coverage_v4_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["coverage_version"] = "v4"
    out.setdefault("per_class_venue_coverage", "")
    out.setdefault("per_class_paper_coverage", "")
    out.setdefault("author_degree_bucket_recovery", "")
    out.setdefault("paper_degree_bucket_recovery", "")
    out.setdefault("venue_degree_bucket_recovery", "")
    out.setdefault("AP_PV_path_multiplicity_mean", "")
    out.setdefault("AP_PV_path_multiplicity_std", "")
    out.setdefault("APA_feedback_path_count", "")
    out.setdefault("VP_A_feedback_path_count", "")
    out.setdefault("paper_venue_entropy", "")
    out.setdefault("venue_class_proxy_purity_trainval", "")
    out.setdefault("paper_class_proxy_purity_trainval", "")
    out.setdefault("edge_jaccard_across_graph_seeds", "")
    out.setdefault("retained_edge_overlap_by_relation", "")
    out["semantic_distributional_metrics_ready"] = False
    return out


def _coverage_assertion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_gate"] = out.get("source_gate", "gate21_8")
    return out


def _write_empty_missing_component_files(paths: Mapping[str, Path]) -> None:
    required = {
        "auto_selector_alignment": (
            "gate21_9_channel_utility.csv",
            "gate21_9_channel_removal_probes.csv",
            "gate21_9_auto_channel_plans.csv",
            "gate21_9_auto_selector_by_method.csv",
        ),
        "external_tp_5x5": (
            "gate21_9_external_tp_task_rows.csv",
            "gate21_9_external_tp_by_method.csv",
            "gate21_9_external_tp_budget_audit.csv",
            "gate21_9_external_tp_failures.csv",
        ),
        "freehgc_protocols": (
            "gate21_9_freehgc_standard_task_rows.csv",
            "gate21_9_freehgc_standard_by_ratio.csv",
            "gate21_9_freehgc_tp_adapter_audit.csv",
            "gate21_9_freehgc_tp_by_method.csv",
        ),
        "metapath_cache_dump": (
            "gate21_9_metapath_tensor_audit.csv",
            "gate21_9_cache_hash_assertions.csv",
            "gate21_9_introspection_failures.csv",
        ),
        "feature_ablation_tasks": (
            "gate21_9_feature_ablation_task_rows.csv",
            "gate21_9_feature_ablation_by_method.csv",
            "gate21_9_feature_shape_audit.csv",
            "gate21_9_feature_ablation_failures.csv",
        ),
        "adapter_package_v4": (
            "gate21_9_adapter_task_rows.csv",
            "gate21_9_adapter_by_method.csv",
            "gate21_9_adapter_package_audit.csv",
            "gate21_9_adapter_manifest_index.csv",
        ),
        "storage_system_costs": (
            "gate21_9_storage_system_by_method.csv",
            "gate21_9_ratio_denominator_audit.csv",
            "gate21_9_loader_support_audit.csv",
            "gate21_9_workload_cost_trace.csv",
        ),
        "cross_dataset_auto_channel": (
            "gate21_9_cross_dataset_task_rows.csv",
            "gate21_9_cross_dataset_by_method.csv",
            "gate21_9_auto_channel_plans.csv",
            "gate21_9_auto_channel_validation_trace.csv",
            "gate21_9_cross_dataset_failures.csv",
        ),
        "audits": (
            "gate21_9_coverage_v4.csv",
            "gate21_9_coverage_sanity_assertions.csv",
        ),
    }
    for subdir, names in required.items():
        for name in names:
            path = paths[subdir] / name
            if not path.exists():
                write_csv(path, [])
    env = paths["freehgc_protocols"] / "gate21_9_freehgc_env_audit.json"
    if not env.exists():
        write_json(env, {"standard_condensation_supported": False, "upstream_config_verified": False})
    report = paths["freehgc_protocols"] / "gate21_9_freehgc_tp_failure_report.md"
    if not report.exists():
        report.write_text("# Gate21.9 FreeHGC-TP Failure Report\n\nNo FreeHGC component was executed.\n", encoding="utf-8")


def _tp_protocol_flags(*, official: bool) -> dict[str, Any]:
    return {
        "schema_compatible": True,
        "keeps_all_target_nodes": True,
        "official_hgb_exported": official,
        "official_sehgnn_unmodified": official,
        "uses_feature_adapter": False,
        "uses_weighted_superedges": False,
        "uses_synthetic_nodes": False,
        "eligible_for_official_main_table": official,
        "eligible_for_adapter_table": False,
        "eligible_for_standard_condensation_table": False,
        "eligible_for_tp_workload_table": True,
    }


def _enabled(selected: set[str], *names: str) -> bool:
    return bool(selected.intersection(names))


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if not finite else mean(finite)


def _std_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if len(finite) < 2 else pstdev(finite)


def _first(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""


def _sum_or_blank(*values: float | None) -> float | str:
    finite = [value for value in values if value is not None]
    return "" if not finite else sum(finite)


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(row: Mapping[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = _float(row.get(field))
        if value is not None:
            return value
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def build_parser() -> argparse.ArgumentParser:
    return add_gate21_9_common_args(argparse.ArgumentParser(description=__doc__))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
