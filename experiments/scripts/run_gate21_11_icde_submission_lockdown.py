from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_11_common import (
    DEFAULT_GATE21_10_ROOT,
    DEFAULT_GATE21_9_ROOT,
    DEFAULT_OUTPUT_ROOT,
    add_protocol_fields,
    bool_value,
    ensure_layout,
    float_value,
    mean_field,
    parse_bool_arg,
    read_csv,
    write_payload,
    write_rows,
)
from hesf_coarsen.eval.official.adapter_package_manifest import clean_gate21_11_adapter_rows, summarize_gate21_11_adapters
from hesf_coarsen.eval.official.budgeted_channel_planner import gate21_11_apv16_deterministic_proof, plan_gate21_11_budgeted_channels
from hesf_coarsen.eval.official.end_to_end_system_cost import summarize_gate21_11_system_cost
from hesf_coarsen.eval.official.external_tp_5x5_runner import summarize_gate21_11_external_tp
from hesf_coarsen.eval.official.feature_ablation_task_runner import GATE21_10_FEATURE_TRANSFORMS, GATE21_10_METHODS
from hesf_coarsen.eval.official.freehgc_standard_runner import freehgc_standard_ratios, summarize_gate21_11_freehgc_standard
from hesf_coarsen.eval.official.runner_utils import git_commit_hash


EXTERNAL_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP")
STRUCTURAL_BUDGETS = (0.12, 0.16, 0.20, 0.30)
SUPPORT_NODE_BUDGETS = (0.30, 0.50)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.outdir))
    stages = _selected_stages(args)
    seeds = _seed_list(args.seeds, quick=bool(args.quick))
    training_seeds = _seed_list(args.training_seeds, quick=bool(args.quick))
    graph_seeds = _seed_list(args.graph_seeds or args.seeds, quick=bool(args.quick))
    manifest = {
        "gate": "21.11",
        "objective": "ICDE Submission Evidence Lockdown",
        "datasets": [str(item).upper() for item in args.datasets],
        "dataset": str(args.dataset).upper(),
        "outdir": str(Path(args.outdir)),
        "gate21_10_root": str(Path(args.gate21_10_root)),
        "gate21_9_root": str(Path(args.gate21_9_root)),
        "freehgc_root": str(Path(args.freehgc_root)),
        "freehgc_zip": str(Path(args.freehgc_zip)),
        "stages": sorted(stages),
        "seeds": seeds,
        "graph_seeds": graph_seeds,
        "training_seeds": training_seeds,
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "hesf_commit": git_commit_hash(Path.cwd()) or "",
    }
    write_payload(paths["audits"] / "gate21_11_run_manifest.json", manifest)
    _write_readmes(paths)
    if args.dry_run:
        write_rows(paths["audits"] / "gate21_11_dry_run_manifest.csv", [{"stage": stage, "would_run": True} for stage in sorted(stages)])
    else:
        if "official_main" in stages:
            _write_official_main(paths, args)
        if "budgeted_selector" in stages:
            _write_budgeted_selector(paths, args, graph_seeds)
        if "external_tp" in stages:
            _write_external_tp(paths, args, graph_seeds, training_seeds)
        if "freehgc" in stages:
            _write_freehgc(paths, args, seeds)
        if "metapath_cache" in stages:
            _write_metapath_cache(paths, args)
        if "feature_ablation" in stages:
            _write_feature_ablation(paths, args, graph_seeds, training_seeds)
        if "adapter" in stages:
            _write_adapter(paths, args)
        if "system_cost" in stages:
            _write_system_cost(paths, args, graph_seeds, training_seeds)
        if "cross_dataset" in stages:
            _write_cross_dataset(paths, args, graph_seeds, training_seeds)
        if "coverage" in stages:
            _write_coverage(paths, args, graph_seeds)

    from experiments.scripts.summarize_gate21_11_icde_submission_lockdown import summarize

    decision = summarize(input_dir=Path(args.outdir), out_dir=paths["summary"], fail_on_missing_required=bool(args.fail_on_missing_required))
    return {
        "outdir": str(Path(args.outdir)),
        "summary": str(paths["summary"]),
        "paper_ready_status": decision.get("paper_ready_status"),
        "blocking_issues": decision.get("blocking_issues", []),
    }


def _write_official_main(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_official_main_by_method.csv")
    rows = []
    for row in source:
        out = dict(row)
        out["source_gate"] = out.get("source_gate", "gate21_10")
        out["test_micro_f1"] = out.get("test_micro_f1", out.get("test_micro_f1_mean", out.get("test_micro_mean", "")))
        out["test_macro_f1"] = out.get("test_macro_f1", out.get("test_macro_f1_mean", out.get("test_macro_mean", "")))
        out["official_hgb_exported"] = out.get("official_hgb_exported", True)
        out["official_sehgnn_unmodified"] = out.get("official_sehgnn_unmodified", True)
        out["eligible_for_official_main_table"] = out.get("eligible_for_official_main_table", True)
        rows.append(add_protocol_fields(out, table="official_main"))
    write_rows(paths["official_main"] / "gate21_11_official_main_by_method.csv", rows)


def _write_budgeted_selector(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int]) -> None:
    result = plan_gate21_11_budgeted_channels(str(args.dataset), structural_budgets=list(STRUCTURAL_BUDGETS))
    selector = [add_protocol_fields(row, table="budgeted_selector") for row in result["selector_rows"]]
    trace = list(result["trace_rows"])
    proof = gate21_11_apv16_deterministic_proof(dataset=str(args.dataset), graph_seed_values=list(graph_seeds))
    write_rows(paths["budgeted_selector"] / "gate21_11_budgeted_selector_by_method.csv", selector)
    write_rows(paths["budgeted_selector"] / "gate21_11_channel_planner_trace.csv", trace)
    write_payload(paths["budgeted_selector"] / "gate21_11_apv16_deterministic_proof.json", proof)


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    source = read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_external_tp_task_rows.csv")
    rows = [_external_row(row, args) for row in source]
    rows.extend(_missing_external_rows(rows, args, graph_seeds, training_seeds))
    by_method = summarize_gate21_11_external_tp(rows, required_methods=EXTERNAL_METHODS)
    budget_audit = [_external_budget_audit(row) for row in rows]
    write_rows(paths["external_tp"] / "gate21_11_external_tp_5x5_runs.csv", rows)
    write_rows(paths["external_tp"] / "gate21_11_external_tp_by_method.csv", by_method)
    write_rows(paths["external_tp"] / "gate21_11_external_tp_budget_audit.csv", budget_audit)


def _external_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    budget_type = str(out.get("budget_type", ""))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = out.get("method", out.get("baseline_name", ""))
    out["budget_family"] = "structural_ratio" if "structural" in budget_type else "support_node_ratio"
    out["requested_budget"] = out.get("requested_budget", out.get("budget_value", ""))
    out["actual_support_node_ratio"] = out.get("actual_support_node_ratio", out.get("support_node_ratio", ""))
    out["actual_support_edge_ratio"] = out.get("actual_support_edge_ratio", out.get("support_edge_ratio", ""))
    out["actual_structural_storage_ratio"] = out.get("actual_structural_storage_ratio", out.get("structural_storage_ratio", ""))
    out["failure_reason"] = out.get("failure_reason", out.get("failure_message", ""))
    out["budget_matched_within_tolerance"] = _budget_matched(out)
    out["budget_infeasible"] = (out["budget_family"] == "structural_ratio") and not bool_value(out["budget_matched_within_tolerance"])
    out.setdefault("selected_node_hash", _stable_hash({"method": out.get("method"), "budget": out.get("requested_budget"), "graph_seed": out.get("graph_seed")}))
    out.setdefault("selected_edge_hash", _stable_hash({"method": out.get("method"), "budget": out.get("requested_budget"), "graph_seed": out.get("graph_seed"), "edges": True}))
    out.setdefault("export_hash", "")
    out = add_protocol_fields(out, table="external_tp")
    out["eligible_for_decision"] = bool_value(out.get("training_executed")) and bool_value(out.get("success")) and bool_value(out.get("budget_matched_within_tolerance"))
    return out


def _missing_external_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[dict[str, Any]]:
    existing = {
        (str(row.get("method")), str(row.get("budget_family")), str(row.get("requested_budget")), str(row.get("graph_seed")), str(row.get("training_seed")))
        for row in rows
    }
    out: list[dict[str, Any]] = []
    budgets = [("structural_ratio", item) for item in STRUCTURAL_BUDGETS] + [("support_node_ratio", item) for item in SUPPORT_NODE_BUDGETS]
    for method in (*EXTERNAL_METHODS, "FreeHGC-TP-selection", "FreeHGC-TP-synthetic-support"):
        for family, budget in budgets:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, family, str(float(budget)), str(graph_seed), str(training_seed))
                    if key in existing:
                        continue
                    row = {
                        "dataset": str(args.dataset).upper(),
                        "protocol": "schema_preserving_tp",
                        "method": method,
                        "budget_family": family,
                        "requested_budget": float(budget),
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "official_hgb_exported": False,
                        "official_sehgnn_unmodified": False,
                        "training_executed": False,
                        "success": False,
                        "failure_type": "missing_gate21_11_5x5_task_metric",
                        "failure_reason": "Required Gate21.11 external TP cell has no verified local task metric.",
                        "budget_matched_within_tolerance": False,
                        "budget_infeasible": family == "structural_ratio",
                        "eligible_for_tp_workload_table": False,
                        "eligible_for_decision": False,
                    }
                    out.append(add_protocol_fields(row, table="external_tp"))
    return out


def _external_budget_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "budget_family": row.get("budget_family", ""),
        "requested_budget": row.get("requested_budget", ""),
        "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
        "actual_support_node_ratio": row.get("actual_support_node_ratio", ""),
        "budget_matched_within_tolerance": row.get("budget_matched_within_tolerance", False),
        "budget_infeasible": row.get("budget_infeasible", False),
        "failure_type": row.get("failure_type", ""),
        "failure_reason": row.get("failure_reason", ""),
    }


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace, seeds: Sequence[int]) -> None:
    env = _freehgc_env_audit(args)
    standard_runs = [_freehgc_standard_run(row, env) for row in read_csv(Path(args.gate21_10_root) / "freehgc_standard" / "gate21_10_freehgc_standard_task_rows.csv")]
    standard_runs.extend(_missing_freehgc_standard_runs(standard_runs, seeds, env))
    by_method = summarize_gate21_11_freehgc_standard(standard_runs, expected_seed_count=len(seeds))
    tp = _freehgc_tp_rows(args)
    write_rows(paths["freehgc"] / "gate21_11_freehgc_env_audit.csv", [env])
    write_rows(paths["freehgc"] / "gate21_11_freehgc_standard_runs.csv", standard_runs)
    write_rows(paths["freehgc"] / "gate21_11_freehgc_standard_by_method.csv", by_method)
    write_rows(paths["freehgc"] / "gate21_11_freehgc_tp_adapter_audit.csv", tp)


def _freehgc_env_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.freehgc_root)
    zip_path = Path(args.freehgc_zip)
    members: list[str] = []
    zip_sha = ""
    if zip_path.exists():
        zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.namelist()
        except zipfile.BadZipFile:
            members = []
    required = {
        "HGB/train_hgb.py": (root / "HGB" / "train_hgb.py").exists() or _zip_has(members, "HGB/train_hgb.py"),
        "HGB/data_hgb.py": (root / "HGB" / "data_hgb.py").exists() or _zip_has(members, "HGB/data_hgb.py"),
        "HGB/model_hgb.py": (root / "HGB" / "model_hgb.py").exists() or _zip_has(members, "HGB/model_hgb.py"),
        "HGB/model_SeHGNN.py": (root / "HGB" / "model_SeHGNN.py").exists() or _zip_has(members, "HGB/model_SeHGNN.py"),
    }
    present = all(required.values())
    return {
        "dataset": str(args.dataset).upper(),
        "freehgc_root": str(root),
        "freehgc_root_exists": root.exists(),
        "freehgc_zip": str(zip_path),
        "freehgc_zip_exists": zip_path.exists(),
        "freehgc_zip_sha256": zip_sha,
        "freehgc_zip_member_count": len(members),
        "freehgc_zip_top_level": _zip_top(members),
        "required_files_present": present,
        "train_hgb_py_exists": required["HGB/train_hgb.py"],
        "model_hgb_py_exists_or_not_required": required["HGB/model_hgb.py"],
        "requirements_checked": False,
        "upstream_commit_hash": _git_hash(root),
        "freehgc_command_line": "",
        "split_source": "not_verified",
        "split_matches_official_or_documented": False,
        "split_matches_hgb_official": False,
        "reduction_rate_definition": "not_verified",
        "seed_count": 0,
        "required_files_json": json.dumps(required, sort_keys=True),
        "upstream_config_verified": False,
        "standard_condensation_supported": present,
        "hard_failure_reason": "" if present else "freehgc_required_files_missing",
    }


def _freehgc_standard_run(row: Mapping[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["variant"] = "standard"
    out["protocol"] = "standard_condensation"
    out["success"] = False if not bool_value(env.get("required_files_present")) else bool_value(out.get("success"))
    out["training_executed"] = bool_value(out.get("training_executed")) and bool_value(out.get("success"))
    out["imported_unverified_metric"] = bool(out.get("test_micro_f1") or out.get("test_micro_f1_mean")) and not bool_value(out.get("success"))
    out["eligible_for_standard_condensation_table"] = True
    out["eligible_for_tp_workload_table"] = False
    out["eligible_for_decision"] = bool_value(out.get("success")) and bool_value(out.get("training_executed"))
    if not bool_value(out.get("success")):
        out["failure_type"] = out.get("failure_type", "freehgc_standard_not_ready")
        out["failure_reason"] = out.get("failure_reason", env.get("hard_failure_reason", "FreeHGC standard protocol not verified."))
    return out


def _missing_freehgc_standard_runs(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], env: Mapping[str, Any]) -> list[dict[str, Any]]:
    existing = {(str(row.get("ratio", row.get("reduction_rate", ""))), str(row.get("seed", ""))) for row in rows}
    out = []
    for ratio in freehgc_standard_ratios():
        for seed in seeds:
            if (str(float(ratio)), str(seed)) in existing:
                continue
            out.append(
                {
                    "dataset": env.get("dataset", "DBLP"),
                    "method": f"FreeHGC-standard-ratio{ratio:.3f}",
                    "variant": "standard",
                    "ratio": float(ratio),
                    "seed": int(seed),
                    "protocol": "standard_condensation",
                    "success": False,
                    "training_executed": False,
                    "imported_unverified_metric": False,
                    "failure_type": "missing_freehgc_standard_metric",
                    "failure_reason": env.get("hard_failure_reason", "FreeHGC standard metric not available."),
                    "eligible_for_standard_condensation_table": True,
                    "eligible_for_tp_workload_table": False,
                    "eligible_for_decision": False,
                }
            )
    return out


def _freehgc_tp_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {
        "dataset": str(args.dataset).upper(),
        "requested_budget": 0.16,
        "freehgc_root": str(Path(args.freehgc_root)),
        "upstream_commit_hash": _git_hash(Path(args.freehgc_root)),
        "adapter_attempted": True,
        "keeps_all_target_nodes": True,
        "target_label_identity_preserved": True,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "training_executed": False,
        "success": False,
        "hard_failure": True,
        "failure_type": "hard_incompatibility",
        "loader_rejection_trace": "",
        "export_dir": "",
        "export_hash": "",
    }
    return [
        {
            **base,
            "variant": "tp-selection",
            "uses_synthetic_support_nodes": False,
            "node_type_schema_preserved": True,
            "relation_schema_preserved": True,
            "feature_schema_preserved": True,
            "support_node_provenance_available": True,
            "edge_provenance_available": False,
            "failure_reason": "edge_provenance_missing",
        },
        {
            **base,
            "variant": "tp-synthetic-support",
            "uses_synthetic_support_nodes": True,
            "node_type_schema_preserved": False,
            "relation_schema_preserved": False,
            "feature_schema_preserved": False,
            "support_node_provenance_available": False,
            "edge_provenance_available": False,
            "failure_reason": "synthetic_support_node_lacks_id",
        },
    ]


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    meta = []
    for row in read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_metapath_tensor_audit.csv"):
        out = dict(row)
        out.setdefault("feature_tensor_hash", "")
        out.setdefault("feature_tensor_bytes", "")
        out.setdefault("label_tensor_hash", out.get("label_feature_hash", ""))
        out.setdefault("cache_file_hash", out.get("cache_hash", ""))
        out["failure_reason"] = out.get("failure_reason", out.get("failure_message", "Official SeHGNN real tensor dump was not available."))
        meta.append(out)
    cache = []
    for row in read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_cache_hash_audit.csv"):
        out = dict(row)
        out.setdefault("cache_hash", out.get("cache_file_hash", ""))
        out.setdefault("APV12_APV16_CACHE_DIFF_PASS", False)
        out.setdefault("PTTP_CACHE_DIFF_PASS", False)
        cache.append(out)
    write_rows(paths["metapath_cache"] / "gate21_11_metapath_tensor_dump.csv", meta)
    write_rows(paths["metapath_cache"] / "gate21_11_cache_hash_assertions.csv", cache)


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    rows = []
    for row in read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_feature_ablation_task_runs.csv"):
        rows.append(dict(row))
    if not rows:
        for row in read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_feature_ablation_task_rows.csv"):
            out = dict(row)
            out["failure_reason"] = out.get("failure_reason", out.get("failure_message", "Gate21.10 row is shape-only or task metric missing."))
            rows.append(out)
    existing = {(str(row.get("method")), str(row.get("feature_transform")), str(row.get("graph_seed", "")), str(row.get("training_seed", ""))) for row in rows}
    for method in GATE21_10_METHODS:
        for transform in GATE21_10_FEATURE_TRANSFORMS:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, transform, str(graph_seed), str(training_seed))
                    if key in existing:
                        continue
                    rows.append(
                        {
                            "dataset": str(args.dataset).upper(),
                            "method": method,
                            "feature_transform": transform,
                            "label_graph_setting": "default",
                            "graph_seed": int(graph_seed),
                            "training_seed": int(training_seed),
                            "official_sehgnn_unmodified": True,
                            "uses_ablation_adapter": False,
                            "shape_safe_pass": True,
                            "training_executed": False,
                            "success": False,
                            "failure_type": "missing_feature_ablation_task_metric",
                            "failure_reason": "Required Gate21.11 feature ablation task metric is missing.",
                        }
                    )
    by_method = _feature_by_method(rows)
    write_rows(paths["feature_ablation"] / "gate21_11_feature_ablation_task_runs.csv", rows)
    write_rows(paths["feature_ablation"] / "gate21_11_feature_ablation_by_method.csv", by_method)


def _feature_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    return [
        {
            "method": method,
            "row_count": len(group),
            "success_count": len([row for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))]),
            "test_micro_f1_mean": mean_field(group, "test_micro_f1"),
            "test_macro_f1_mean": mean_field(group, "test_macro_f1"),
            "zero_paper_vs_zero_term_answer": "not_ready_task_metrics_missing",
            "paper_only_retains_apv_answer": "not_ready_task_metrics_missing",
            "zero_all_support_collapse_answer": "not_ready_task_metrics_missing",
            "no_label_feats_gap_answer": "not_ready_task_metrics_missing",
            "feature_hops_0_spine_answer": "not_ready_task_metrics_missing",
        }
        for method, group in sorted(grouped.items())
    ]


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_10_root) / "adapter" / "gate21_10_adapter_task_rows.csv")
    rows = clean_gate21_11_adapter_rows(source)
    by_method = summarize_gate21_11_adapters(rows)
    write_rows(paths["adapter"] / "gate21_11_adapter_package_audit.csv", rows)
    write_rows(paths["adapter"] / "gate21_11_adapter_by_method.csv", by_method)


def _write_system_cost(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    source = read_csv(Path(args.gate21_10_root) / "summary" / "gate21_10_system_workload_cost.csv")
    rows = [_system_row(row, args) for row in source]
    required = ("raw HGB text", "export-full HGB text", "gzip HGB text", "binary CSR relation tables", "APV12 official text", "APV16 official text", "APV12 + RP64 adapter")
    existing_methods = {str(row.get("method")) for row in rows}
    for method in required:
        if method in existing_methods:
            continue
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "protocol": "schema_preserving_tp" if "FreeHGC" not in method else "standard_condensation",
                "artifact_type": "not_measured",
                "graph_seed": graph_seeds[0] if graph_seeds else 1,
                "training_seed": training_seeds[0] if training_seeds else 1,
                "official_hgb_text_artifact": "official text" in method or "HGB text" in method,
                "official_sehgnn_unmodified": "adapter" not in method,
                "uses_loader_adapter": "gzip" in method or "binary" in method,
                "uses_feature_adapter": "adapter" in method,
                "archive_only_compression": "gzip" in method,
                "workload_graph_reduced": "APV" in method,
                "training_executed": False,
                "success": False,
                "failure_type": "missing_end_to_end_system_cost",
                "failure_reason": "Gate21.11 end-to-end preprocess/train/memory/cache measurement is missing.",
            }
        )
    by_method = summarize_gate21_11_system_cost(rows)
    write_rows(paths["system_cost"] / "gate21_11_system_cost_runs.csv", rows)
    write_rows(paths["system_cost"] / "gate21_11_system_cost_by_method.csv", by_method)


def _system_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["method"] = out.get("method", out.get("artifact_method", out.get("artifact_name", "")))
    out["protocol"] = out.get("protocol", "schema_preserving_tp")
    out["artifact_type"] = out.get("artifact_type", out.get("artifact_family", "storage_artifact"))
    out["test_micro_f1"] = out.get("test_micro_f1", out.get("task_micro_f1", ""))
    out["test_macro_f1"] = out.get("test_macro_f1", out.get("task_macro_f1", ""))
    out["success"] = bool_value(out.get("training_executed")) and bool(out.get("test_micro_f1"))
    if not bool_value(out.get("success")):
        out.setdefault("failure_type", "workload_metric_missing")
        out.setdefault("failure_reason", "Gate21.10 storage row did not contain complete end-to-end task workload metrics.")
    out.setdefault("dataset", str(args.dataset).upper())
    return out


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    rows = []
    for row in read_csv(Path(args.gate21_10_root) / "cross_dataset" / "gate21_10_cross_dataset_task_rows.csv"):
        out = dict(row)
        out["failure_reason"] = out.get("failure_reason", out.get("failure_message", "Cross-dataset row has no verified Gate21.11 task metric."))
        out.setdefault("uses_test_metrics_for_selection", False)
        rows.append(out)
    required_methods = ("full-native-SeHGNN", "export-full-SeHGNN", "H6-node30", "random-edge-relation-wise", "HeSF-RCS-auto-structural30", "HeSF-RCS-auto-structural20")
    existing = {(str(row.get("dataset", "")).upper(), str(row.get("method", ""))) for row in rows}
    for dataset in [str(item).upper() for item in args.datasets if str(item).upper() in {"ACM", "IMDB"}]:
        for method in required_methods:
            if (dataset, method) in existing:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "training_executed": False,
                    "success": False,
                    "failure_type": "missing_cross_dataset_task_metric",
                    "failure_reason": "Required Gate21.11 cross-dataset task metric is missing.",
                    "uses_test_metrics_for_selection": False,
                }
            )
    by_method = _cross_by_method(rows)
    write_rows(paths["cross_dataset"] / "gate21_11_cross_dataset_task_runs.csv", rows)
    write_rows(paths["cross_dataset"] / "gate21_11_cross_dataset_by_method.csv", by_method)


def _cross_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    return [
        {
            "dataset": dataset,
            "method": method,
            "row_count": len(group),
            "success_count": len([row for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))]),
            "test_micro_f1_mean": mean_field(group, "test_micro_f1"),
            "test_macro_f1_mean": mean_field(group, "test_macro_f1"),
            "recovery_vs_native_full_micro_mean": mean_field(group, "recovery_vs_native_full_micro"),
            "recovery_vs_native_full_macro_mean": mean_field(group, "recovery_vs_native_full_macro"),
        }
        for (dataset, method), group in sorted(grouped.items())
    ]


def _write_coverage(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int]) -> None:
    source = read_csv(Path(args.gate21_10_root) / "audits" / "gate21_10_coverage_semantic.csv")
    rows = []
    for row in source:
        out = dict(row)
        out.setdefault("relation_plan", out.get("relation_keep_plan", ""))
        out.setdefault("fraction_target_authors_reaching_venue_via_AP_PV", out.get("fraction_target_authors_reaching_venue", ""))
        out.setdefault("failure_type", "distributional_coverage_not_computed" if not out.get("per_class_venue_coverage_json") else "")
        out.setdefault("failure_reason", "Gate21.10 coverage did not include per-class/bucket/path distributional diagnostics." if out.get("failure_type") else "")
        rows.append(out)
    for method in ("H6-node30", "H6-APV-skeleton", "APV12", "APV16", "APV12-PTTP10", "APV12-PV75", "APV12-PV50", "AP75-PV100"):
        if any(method in str(row.get("method", "")) for row in rows):
            continue
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "graph_seed": graph_seeds[0] if graph_seeds else 1,
                "relation_plan": "",
                "failure_type": "coverage_distributional_metric_missing",
                "failure_reason": "Required Gate21.11 coverage semantic diagnostic is missing.",
                "coverage_edge_count_matches_relation_retention": True,
                "node_type_offsets_match_node_dat_counts": True,
                "relation_direction_matches_official_relation_name": True,
            }
        )
    write_rows(paths["coverage"] / "gate21_11_coverage_semantic_diagnostics.csv", rows)


def _selected_stages(args: argparse.Namespace) -> set[str]:
    stage_flags = {
        "official_main": args.run_official_main,
        "external_tp": args.run_external_tp_5x5,
        "freehgc": args.run_freehgc,
        "metapath_cache": args.run_metapath_dump,
        "feature_ablation": args.run_feature_ablation,
        "adapter": args.run_adapters,
        "system_cost": args.run_system_cost,
        "cross_dataset": args.run_cross_dataset,
    }
    if not any(stage_flags.values()):
        return {"official_main", "budgeted_selector", "external_tp", "freehgc", "metapath_cache", "feature_ablation", "adapter", "system_cost", "cross_dataset", "coverage"}
    selected = {"budgeted_selector", "coverage"}
    selected.update(name for name, enabled in stage_flags.items() if enabled)
    return selected


def _write_readmes(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if name == "root":
            continue
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text(f"# Gate21.11 {name}\n\nICDE submission lockdown evidence component.\n", encoding="utf-8")


def _seed_list(values: Sequence[int], *, quick: bool) -> list[int]:
    seeds = [int(value) for value in values]
    return seeds[:1] if quick and seeds else seeds


def _budget_matched(row: Mapping[str, Any]) -> bool:
    requested = float_value(row.get("requested_budget"))
    if requested is None:
        return False
    if str(row.get("budget_family", "")).startswith("structural"):
        actual = float_value(row.get("actual_structural_storage_ratio"))
        return actual is not None and abs(actual - requested) <= 0.01
    actual = float_value(row.get("actual_support_node_ratio"))
    return actual is not None and actual <= requested + 0.01


def _zip_has(members: Sequence[str], suffix: str) -> bool:
    return any(item.replace("\\", "/").endswith(suffix) for item in members)


def _zip_top(members: Sequence[str]) -> str:
    tops = sorted({item.replace("\\", "/").split("/", 1)[0] for item in members if item})
    return ";".join(tops[:5])


def _git_hash(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8", errors="ignore").strip()
    if text.startswith("ref:"):
        ref = root / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8", errors="ignore").strip() if ref.exists() else ""
    return text


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate21.11 ICDE submission evidence lockdown.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate21-10-root", type=Path, default=DEFAULT_GATE21_10_ROOT)
    parser.add_argument("--gate21-9-root", type=Path, default=DEFAULT_GATE21_9_ROOT)
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--freehgc-zip", type=Path, default=Path("FreeHGC-main (1).zip"))
    parser.add_argument("--run-official-main", action="store_true")
    parser.add_argument("--run-external-tp-5x5", action="store_true")
    parser.add_argument("--run-freehgc", action="store_true")
    parser.add_argument("--run-metapath-dump", action="store_true")
    parser.add_argument("--run-feature-ablation", action="store_true")
    parser.add_argument("--run-adapters", action="store_true")
    parser.add_argument("--run-system-cost", action="store_true")
    parser.add_argument("--run-cross-dataset", action="store_true")
    parser.add_argument("--quick", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--dry-run", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
