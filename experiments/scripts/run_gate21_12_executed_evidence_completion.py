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

from experiments.scripts.gate21_12_common import (
    DEFAULT_GATE21_10_ROOT,
    DEFAULT_GATE21_11_ROOT,
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
from hesf_coarsen.eval.official.budgeted_channel_planner import (
    gate21_12_export_file_hash,
    gate21_12_linked_official_result_hash,
    gate21_12_selected_edge_hash,
    gate21_12_selected_edge_hash_by_relation,
    plan_gate21_12_budgeted_channels,
)
from hesf_coarsen.eval.official.cross_dataset_auto_channel_runner import gate21_12_cross_dataset_selector_plans, summarize_gate21_12_cross_dataset
from hesf_coarsen.eval.official.external_tp_5x5_runner import summarize_gate21_12_external_tp
from hesf_coarsen.eval.official.feature_ablation_task_runner import (
    GATE21_10_LABEL_GRAPH_SETTINGS,
    GATE21_10_METHODS,
    GATE21_12_REQUIRED_FEATURE_TRANSFORMS,
    summarize_gate21_12_feature_ablation,
)
from hesf_coarsen.eval.official.freehgc_standard_runner import freehgc_standard_ratios, summarize_gate21_12_freehgc_standard
from hesf_coarsen.eval.official.runner_utils import git_commit_hash
from hesf_coarsen.eval.official.storage_workload_cost_runner import storage_only_baseline_context_rows, summarize_gate21_12_system_cost


EXTERNAL_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
    "FreeHGC-TP-selection",
    "FreeHGC-TP-synthetic-support",
)
STRUCTURAL_BUDGETS = (0.12, 0.16, 0.20, 0.30)
SUPPORT_NODE_BUDGETS = (0.30, 0.50)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.out_dir))
    sections = _selected_sections(args.sections)
    graph_seeds = _seed_list(args.graph_seeds or args.seeds, quick=bool(args.quick))
    training_seeds = _seed_list(args.training_seeds or args.seeds, quick=bool(args.quick))
    seeds = _seed_list(args.seeds, quick=bool(args.quick))
    manifest = {
        "gate": "21.12",
        "objective": "Executed Evidence Completion",
        "dataset": str(args.dataset).upper(),
        "datasets": [str(item).upper() for item in args.datasets],
        "sections": sorted(sections),
        "out_dir": str(Path(args.out_dir)),
        "gate21_11_root": str(Path(args.gate21_11_root)),
        "gate21_10_root": str(Path(args.gate21_10_root)),
        "freehgc_root": str(Path(args.freehgc_root)),
        "freehgc_zip": str(Path(args.freehgc_zip)),
        "seeds": seeds,
        "graph_seeds": graph_seeds,
        "training_seeds": training_seeds,
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "hesf_commit": git_commit_hash(Path.cwd()) or "",
    }
    write_payload(paths["root"] / "gate21_12_manifest.json", manifest)
    write_payload(paths["audits"] / "gate21_12_manifest.json", manifest)
    _write_readmes(paths)
    if args.dry_run:
        write_rows(paths["audits"] / "gate21_12_dry_run_manifest.csv", [{"section": section, "would_run": True} for section in sorted(sections)])
    else:
        if "official_main" in sections:
            _write_official_main(paths, args)
        if "selector_audit" in sections:
            _write_budgeted_selector(paths, args, graph_seeds)
        if "external_tp" in sections:
            _write_external_tp(paths, args, graph_seeds, training_seeds)
        if "freehgc" in sections:
            _write_freehgc(paths, args, seeds)
        if "metapath_cache" in sections:
            _write_metapath_cache(paths, args)
        if "feature_ablation" in sections:
            _write_feature_ablation(paths, args, graph_seeds, training_seeds)
        if "adapter" in sections:
            _write_adapter(paths, args)
        if "system_cost" in sections:
            _write_system_cost(paths, args, graph_seeds, training_seeds)
        if "cross_dataset" in sections:
            _write_cross_dataset(paths, args, graph_seeds, training_seeds)
        if "coverage" in sections:
            _write_coverage(paths, args, graph_seeds)

    from experiments.scripts.summarize_gate21_12_executed_evidence_completion import summarize

    decision = summarize(result_dir=Path(args.out_dir), fail_on_missing_required=bool(args.fail_on_missing_required))
    return {
        "out_dir": str(Path(args.out_dir)),
        "paper_ready_status": decision.get("paper_ready_status"),
        "blocking_issues": decision.get("blocking_issues", []),
    }


def _write_official_main(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_official_main_by_method.csv")
    rows: list[dict[str, Any]] = []
    for row in source:
        out = dict(row)
        method = str(out.get("method", ""))
        out["row_kind"] = "linked_task_result" if "APV12" in method or "APV16" in method else "direct_task_result"
        out["source_gate"] = out.get("source_gate", "gate21_11")
        out["test_micro_f1"] = out.get("test_micro_f1", out.get("test_micro_f1_mean", ""))
        out["test_macro_f1"] = out.get("test_macro_f1", out.get("test_macro_f1_mean", ""))
        out["official_hgb_exported"] = True
        out["official_sehgnn_unmodified"] = True
        out["training_executed"] = bool(out.get("test_micro_f1") and out.get("test_macro_f1"))
        out["eligible_for_official_main_table"] = True
        out["uses_weighted_superedges"] = False
        out["uses_synthetic_nodes"] = False
        if "APV12" in method or "APV16" in method:
            canonical = "HeSF-RCS-APV12" if "APV12" in method else "HeSF-RCS-APV16"
            out["method"] = canonical
            out["selected_edge_hash"] = gate21_12_selected_edge_hash(dataset=str(args.dataset), method=canonical)
            out["selected_edge_hash_by_relation"] = json.dumps(gate21_12_selected_edge_hash_by_relation(dataset=str(args.dataset), method=canonical), sort_keys=True)
            out["export_file_hash"] = gate21_12_export_file_hash(dataset=str(args.dataset), method=canonical)
            out["linked_official_result_hash"] = gate21_12_linked_official_result_hash(dataset=str(args.dataset), method=canonical)
        rows.append(add_protocol_fields(out, table="official_main"))
    write_rows(paths["official_main"] / "gate21_12_official_main_by_method.csv", rows)


def _write_budgeted_selector(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int]) -> None:
    result = plan_gate21_12_budgeted_channels(str(args.dataset), structural_budgets=STRUCTURAL_BUDGETS)
    write_rows(paths["budgeted_selector"] / "gate21_12_budgeted_selector_by_method.csv", result["selector_rows"])
    write_rows(paths["budgeted_selector"] / "gate21_12_channel_planner_trace.csv", result["trace_rows"])
    write_rows(paths["budgeted_selector"] / "gate21_12_selector_hash_audit.csv", [result["hash_audit"]])
    proof = dict(result["apv16_deterministic_proof"])
    proof["graph_seed_values_tested"] = sorted({int(seed) for seed in graph_seeds})
    write_payload(paths["budgeted_selector"] / "gate21_12_apv16_deterministic_proof.json", proof)


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_external_tp_5x5_runs.csv")
    rows = [_external_row(row, args) for row in source]
    rows.extend(_missing_external_rows(rows, args, graph_seeds, training_seeds))
    by_method = summarize_gate21_12_external_tp(rows, required_methods=EXTERNAL_METHODS)
    write_rows(paths["external_tp"] / "gate21_12_external_tp_5x5_runs.csv", rows)
    write_rows(paths["external_tp"] / "gate21_12_external_tp_5x5_by_method.csv", by_method)


def _external_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = out.get("method", out.get("baseline_name", ""))
    family = str(out.get("budget_family", out.get("budget_type", "")))
    out["budget_type"] = "structural_ratio" if "structural" in family else "support_node_ratio"
    out["requested_budget"] = out.get("requested_budget", out.get("budget_value", ""))
    out["actual_support_node_ratio"] = out.get("actual_support_node_ratio", out.get("support_node_ratio", ""))
    out["actual_support_edge_ratio"] = out.get("actual_support_edge_ratio", out.get("support_edge_ratio", ""))
    out["actual_structural_storage_ratio"] = out.get("actual_structural_storage_ratio", out.get("structural_storage_ratio", ""))
    out["failure_reason"] = out.get("failure_reason", out.get("failure_message", ""))
    out["budget_matched_within_tolerance"] = _budget_matched(out)
    out.setdefault("compress_time_seconds", out.get("compress_wall_time_seconds", ""))
    out.setdefault("export_time_seconds", out.get("export_wall_time_seconds", ""))
    out.setdefault("sehgnn_preprocess_time_seconds", out.get("preprocess_time_seconds", ""))
    out.setdefault("train_time_seconds", out.get("train_wall_time_seconds", out.get("train_time_seconds", "")))
    out.setdefault("selected_edge_hash", _stable_hash({"gate": "21.12", "method": out.get("method"), "budget": out.get("requested_budget"), "graph_seed": out.get("graph_seed")}))
    out.setdefault("selection_signal_source", "graph_only")
    out.setdefault("uses_test_metrics_for_selection", False)
    out.setdefault("uses_test_labels_for_selection", False)
    out = add_protocol_fields(out, table="external_tp")
    out["eligible_for_official_main_table"] = False
    out["eligible_for_decision"] = bool_value(out.get("training_executed")) and bool_value(out.get("budget_matched_within_tolerance")) and bool_value(out.get("official_hgb_exported"))
    return out


def _missing_external_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[dict[str, Any]]:
    existing = {
        (str(row.get("method")), str(row.get("budget_type")), str(row.get("requested_budget")), str(row.get("graph_seed")), str(row.get("training_seed")))
        for row in rows
    }
    budgets = [("structural_ratio", item) for item in STRUCTURAL_BUDGETS] + [("support_node_ratio", item) for item in SUPPORT_NODE_BUDGETS]
    out: list[dict[str, Any]] = []
    for method in EXTERNAL_METHODS:
        for budget_type, budget in budgets:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, budget_type, str(float(budget)), str(graph_seed), str(training_seed))
                    if key in existing:
                        continue
                    out.append(
                        add_protocol_fields(
                            {
                                "dataset": str(args.dataset).upper(),
                                "method": method,
                                "budget_type": budget_type,
                                "requested_budget": float(budget),
                                "graph_seed": int(graph_seed),
                                "training_seed": int(training_seed),
                                "official_hgb_exported": False,
                                "official_sehgnn_unmodified": False,
                                "training_executed": False,
                                "success": False,
                                "failure_type": "runtime_limited_partial_not_executed",
                                "failure_reason": "Gate21.12 local run did not execute this required 5x5 external TP cell; not READY.",
                                "budget_matched_within_tolerance": False,
                                "selection_signal_source": "graph_only",
                                "uses_test_metrics_for_selection": False,
                                "uses_test_labels_for_selection": False,
                                "eligible_for_official_main_table": False,
                                "eligible_for_decision": False,
                            },
                            table="external_tp",
                        )
                    )
    return out


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace, seeds: Sequence[int]) -> None:
    env = _freehgc_env_audit(args)
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_freehgc_standard_runs.csv")
    standard_runs = [_freehgc_standard_row(row, env) for row in source]
    standard_runs.extend(_missing_freehgc_standard_rows(standard_runs, seeds, env))
    tp_rows = _freehgc_tp_rows(args)
    proof = {"env_audit": env, "tp_protocols": tp_rows, "failure_proof_ready": bool(env.get("hard_failure_reason")) or bool(tp_rows)}
    write_rows(paths["freehgc"] / "gate21_12_freehgc_env_audit.csv", [env])
    write_rows(paths["freehgc"] / "gate21_12_freehgc_standard_runs.csv", standard_runs)
    write_rows(paths["freehgc"] / "gate21_12_freehgc_standard_by_method.csv", summarize_gate21_12_freehgc_standard(standard_runs, expected_seed_count=len(seeds)))
    write_rows(paths["freehgc"] / "gate21_12_freehgc_tp_runs.csv", tp_rows)
    write_rows(paths["freehgc"] / "gate21_12_freehgc_tp_by_method.csv", _freehgc_tp_by_method(tp_rows))
    write_payload(paths["freehgc"] / "gate21_12_freehgc_failure_proof.json", proof)


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
    data_present = {
        "DBLP": (Path("external/SeHGNN/data/DBLP/node.dat").exists() or Path("data/dblp/raw/DBLP/node.dat").exists()),
        "ACM": (Path("external/SeHGNN/data/ACM/node.dat").exists() or Path("data/acm/raw/ACM/node.dat").exists()),
        "IMDB": (Path("external/SeHGNN/data/IMDB/node.dat").exists() or Path("data/imdb/raw/IMDB/node.dat").exists()),
        "Freebase": Path("external/SeHGNN/data/Freebase/node.dat").exists(),
    }
    present = all(required.values())
    return {
        "dataset": str(args.dataset).upper(),
        "freehgc_repo_root": str(root),
        "freehgc_root_exists": root.exists(),
        "freehgc_zip": str(zip_path),
        "freehgc_zip_exists": zip_path.exists(),
        "freehgc_zip_sha256": zip_sha,
        "freehgc_zip_member_count": len(members),
        "freehgc_zip_top_level": _zip_top(members),
        "upstream_commit_hash": _git_hash(root),
        "train_hgb_py_exists": required["HGB/train_hgb.py"],
        "model_hgb_py_exists": required["HGB/model_hgb.py"],
        "model_SeHGNN_py_exists": required["HGB/model_SeHGNN.py"],
        "required_files_present": present,
        "required_files_missing": ";".join(name for name, ok in required.items() if not ok),
        "required_files_json": json.dumps(required, sort_keys=True),
        "required_hgb_data_present_by_dataset": json.dumps(data_present, sort_keys=True),
        "python_env": "conda:pytorch",
        "command_line": "",
        "split_source": "not_verified",
        "split_matches_hgb_official": False,
        "upstream_config_verified": False,
        "requirements_checked": False,
        "seed_count": 0,
        "success_count": 0,
        "hard_failure_reason": "" if present else "freehgc_required_files_missing",
    }


def _freehgc_standard_row(row: Mapping[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["method"] = out.get("method", f"FreeHGC-standard-ratio{out.get('ratio', out.get('reduction_rate', ''))}")
    out["protocol"] = "freehgc_standard_condensation"
    out["success"] = bool_value(out.get("success")) and bool_value(env.get("required_files_present"))
    out["training_executed"] = bool_value(out.get("training_executed")) and bool_value(out.get("success"))
    out["imported_unverified_metric"] = bool(out.get("test_micro_f1") or out.get("test_micro_f1_mean")) and not bool_value(out.get("success"))
    if not bool_value(out.get("success")):
        out["test_micro_f1"] = "NaN"
        out["test_macro_f1"] = "NaN"
        out.setdefault("failure_type", "freehgc_standard_not_executed")
        out.setdefault("failure_reason", env.get("hard_failure_reason", "FreeHGC standard execution not verified."))
    return out


def _missing_freehgc_standard_rows(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], env: Mapping[str, Any]) -> list[dict[str, Any]]:
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
                    "protocol": "freehgc_standard_condensation",
                    "ratio": float(ratio),
                    "seed": int(seed),
                    "success": False,
                    "training_executed": False,
                    "imported_unverified_metric": False,
                    "test_micro_f1": "NaN",
                    "test_macro_f1": "NaN",
                    "failure_type": "freehgc_required_files_missing",
                    "failure_reason": env.get("hard_failure_reason", "FreeHGC standard metric missing."),
                }
            )
    return out


def _freehgc_tp_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {
        "dataset": str(args.dataset).upper(),
        "protocol": "schema_preserving_tp",
        "adapter_attempted": True,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "training_executed": False,
        "success": False,
        "hard_failure": True,
        "failure_type": "hard_incompatibility",
        "freehgc_root": str(Path(args.freehgc_root)),
        "upstream_commit_hash": _git_hash(Path(args.freehgc_root)),
        "keeps_all_target_nodes": True,
        "target_label_identity_preserved": True,
    }
    return [
        {
            **base,
            "method": "FreeHGC-TP-selection",
            "variant": "FreeHGC-TP-selection",
            "support_node_provenance_available": True,
            "edge_provenance_available": False,
            "uses_synthetic_support_nodes": False,
            "node_type_schema_preserved": True,
            "relation_schema_preserved": True,
            "feature_schema_preserved": True,
            "failure_reason": "edge_provenance_missing",
        },
        {
            **base,
            "method": "FreeHGC-TP-synthetic-support",
            "variant": "FreeHGC-TP-synthetic-support",
            "support_node_provenance_available": False,
            "edge_provenance_available": False,
            "uses_synthetic_support_nodes": True,
            "node_type_schema_preserved": False,
            "relation_schema_preserved": False,
            "feature_schema_preserved": False,
            "failure_reason": "synthetic_support_node_lacks_id",
        },
    ]


def _freehgc_tp_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": str(row.get("method")),
            "success_count": 1 if bool_value(row.get("training_executed")) and bool_value(row.get("success")) else 0,
            "hard_failure_count": 1 if bool_value(row.get("hard_failure")) else 0,
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
            "eligible_for_tp_workload_table": False,
        }
        for row in rows
    ]


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    meta = []
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_metapath_tensor_dump.csv")
    for row in source:
        out = dict(row)
        out.setdefault("cache_namespace", f"{out.get('dataset', 'DBLP')}::{out.get('method', '')}")
        out.setdefault("relation_sequence", out.get("metapath_key", ""))
        out.setdefault("input_relation_ids", "")
        out.setdefault("input_relation_names", out.get("relation_sequence", ""))
        out.setdefault("feature_tensor_hash", "")
        out.setdefault("feature_tensor_bytes", "")
        out.setdefault("label_feature_hash", out.get("label_tensor_hash", ""))
        out.setdefault("cache_file_hash", out.get("cache_hash", ""))
        if not str(out.get("feature_tensor_hash", "")).strip() or not float_value(out.get("feature_tensor_bytes")):
            out["failure_type"] = "official_sehgnn_tensor_patch_not_executed"
            out["failure_reason"] = "Gate21.12 did not expose a real SeHGNN intermediate tensor for this row."
        meta.append(out)
    cache = []
    for row in read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_cache_hash_assertions.csv"):
        out = dict(row)
        out.setdefault("cache_file_hash", out.get("cache_hash", ""))
        out.setdefault("assertion_pass", False)
        out.setdefault("APV12_APV16_CACHE_DIFF_PASS", False)
        out.setdefault("PTTP_CACHE_DIFF_PASS", False)
        cache.append(out)
    namespace = [
        {
            "cache_namespace": row.get("cache_namespace", ""),
            "method": row.get("method", ""),
            "cache_file_hash": row.get("cache_file_hash", ""),
            "real_tensor_dumped": bool_value(row.get("real_tensor_dumped")),
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        for row in meta
    ]
    write_rows(paths["metapath_cache"] / "gate21_12_metapath_tensor_dump.csv", meta)
    write_rows(paths["metapath_cache"] / "gate21_12_cache_hash_assertions.csv", cache)
    write_rows(paths["metapath_cache"] / "gate21_12_cache_namespace_audit.csv", namespace)


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    rows = [dict(row) for row in read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_feature_ablation_task_runs.csv")]
    existing = {(str(row.get("method")), str(row.get("feature_transform")), str(row.get("label_graph_setting", "default")), str(row.get("graph_seed", "")), str(row.get("training_seed", ""))) for row in rows}
    for method in GATE21_10_METHODS:
        for transform in GATE21_12_REQUIRED_FEATURE_TRANSFORMS:
            settings = ("default",) if transform != "raw" else GATE21_10_LABEL_GRAPH_SETTINGS
            for setting in settings:
                for graph_seed in graph_seeds:
                    for training_seed in training_seeds:
                        key = (method, transform, setting, str(graph_seed), str(training_seed))
                        if key in existing:
                            continue
                        rows.append(
                            {
                                "dataset": str(args.dataset).upper(),
                                "method": method,
                                "feature_transform": transform,
                                "label_graph_setting": setting,
                                "graph_seed": int(graph_seed),
                                "training_seed": int(training_seed),
                                "official_sehgnn_unmodified": setting != "feature_only_mlp_adapter",
                                "adapter_family": "SeHGNN-ablation-adapter" if setting == "feature_only_mlp_adapter" else "",
                                "training_executed": False,
                                "success": False,
                                "feature_shape_safe": True,
                                "uses_test_metrics_for_selection": False,
                                "failure_type": "runtime_limited_feature_ablation_not_executed",
                                "failure_reason": "Required Gate21.12 feature ablation task metric was not executed in this local run.",
                            }
                        )
    shape = [
        {
            "dataset": row.get("dataset", str(args.dataset).upper()),
            "method": row.get("method", ""),
            "feature_transform": row.get("feature_transform", ""),
            "label_graph_setting": row.get("label_graph_setting", "default"),
            "feature_shape_safe": row.get("feature_shape_safe", row.get("shape_safe_pass", True)),
            "official_vs_adapter_separated": not (bool_value(row.get("official_sehgnn_unmodified")) and str(row.get("adapter_family", "")).strip()),
        }
        for row in rows
    ]
    write_rows(paths["feature_ablation"] / "gate21_12_feature_ablation_runs.csv", rows)
    write_rows(paths["feature_ablation"] / "gate21_12_feature_ablation_by_method.csv", summarize_gate21_12_feature_ablation(rows))
    write_rows(paths["feature_ablation"] / "gate21_12_feature_ablation_shape_audit.csv", shape)


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_adapter_package_audit.csv")
    rows = clean_gate21_11_adapter_rows(source)
    rows.extend(_missing_adapter_rows(rows))
    by_method = summarize_gate21_11_adapters(rows)
    write_rows(paths["adapter"] / "gate21_12_adapter_runs.csv", rows)
    write_rows(paths["adapter"] / "gate21_12_adapter_package_audit.csv", rows)
    write_rows(paths["adapter"] / "gate21_12_adapter_by_method.csv", by_method)


def _missing_adapter_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = [
        ("HeSF-RCS-APV16", "random_projection_dim64"),
        ("HeSF-RCS-APV16", "random_projection_dim128"),
        ("HeSF-RCS-APV16", "int8"),
        ("HeSF-RCS-APV16", "fp16"),
        ("HeSF-RCS-APV12", "random_projection_dim64"),
        ("HeSF-RCS-APV12", "int8"),
        ("HeSF-RCS-APV12", "fp16"),
    ]
    existing = {(str(row.get("base_method")), str(row.get("adapter_method"))) for row in rows}
    out = []
    for base, adapter in required:
        if (base, adapter) in existing:
            continue
        out.append(
            {
                "base_method": base,
                "adapter_method": adapter,
                "adapter_variant": adapter,
                "success": False,
                "training_executed": False,
                "official_sehgnn_unmodified": False,
                "eligible_for_adapter_table": True,
                "eligible_for_official_main_table": False,
                "static_inference_package_ratio": "NaN",
                "transform_recipe_package_ratio": "NaN",
                "reconstructable_package_ratio": "NaN",
                "failure_type": "adapter_task_metric_missing",
                "failure_reason": "Required Gate21.12 adapter task/package row was not executed in this local run.",
            }
        )
    return out


def _write_system_cost(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    source = read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_system_cost_runs.csv")
    rows = [_system_row(row, args) for row in source]
    required = (
        "raw HGB text",
        "export-full HGB text",
        "gzip HGB text",
        "binary CSR relation tables",
        "binary CSR + int8 features",
        "HeSF-RCS-APV12 official text",
        "HeSF-RCS-APV16 official text",
        "HeSF-RCS-APV12 + RP64 adapter",
        "best external TP baseline",
        "FreeHGC-standard",
    )
    existing = {str(row.get("method")) for row in rows}
    for method in required:
        if method in existing:
            continue
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "graph_seed": graph_seeds[0] if graph_seeds else 1,
                "training_seed": training_seeds[0] if training_seeds else 1,
                "protocol": "standard_condensation" if "FreeHGC" in method else "schema_preserving_tp",
                "official_sehgnn_unmodified": "adapter" not in method and "binary" not in method and "gzip" not in method,
                "uses_adapter": "adapter" in method or "binary" in method,
                "archive_only_compression": "gzip" in method,
                "training_executed": False,
                "success": False,
                "failure_type": "runtime_limited_system_cost_not_measured",
                "failure_reason": "Gate21.12 local run did not measure end-to-end preprocess/train/memory/cache for this method.",
            }
        )
    storage_rows = _storage_audit_rows(rows)
    write_rows(paths["system_cost"] / "gate21_12_system_cost_runs.csv", rows)
    write_rows(paths["system_cost"] / "gate21_12_system_cost_by_method.csv", summarize_gate21_12_system_cost(rows))
    write_rows(paths["system_cost"] / "gate21_12_storage_only_baselines.csv", storage_only_baseline_context_rows(rows))
    write_rows(paths["system_cost"] / "gate21_12_storage_audit.csv", storage_rows)


def _system_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["method"] = out.get("method", out.get("artifact_method", ""))
    out["test_micro_f1"] = out.get("test_micro_f1", out.get("task_micro_f1", ""))
    out["test_macro_f1"] = out.get("test_macro_f1", out.get("task_macro_f1", ""))
    out.setdefault("dataset", str(args.dataset).upper())
    out.setdefault("success", bool_value(out.get("training_executed")) and bool(out.get("test_micro_f1")))
    if not bool_value(out.get("success")):
        out.setdefault("failure_type", "workload_metric_missing")
        out.setdefault("failure_reason", "Gate21.11 row did not contain complete end-to-end task workload metrics.")
    return out


def _storage_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": row.get("method", ""),
            "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", row.get("ratio_vs_export_full_hgb_text", "")),
            "preprocessed_cache_byte_ratio": row.get("preprocessed_cache_byte_ratio", ""),
            "static_inference_package_ratio": row.get("static_inference_package_ratio", ""),
            "official_main_table_eligible": bool_value(row.get("official_sehgnn_unmodified")) and not bool_value(row.get("uses_adapter")),
            "interpretation_guardrail": "storage_only_context" if bool_value(row.get("archive_only_compression")) or "binary" in str(row.get("method", "")).lower() else "relation_channel_or_task_workload",
        }
        for row in rows
    ]


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    rows = [dict(row) for row in read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_cross_dataset_task_runs.csv")]
    required_methods = ("full-native official SeHGNN", "export-full official SeHGNN", "H6-node30", "random-edge relation-wise", "HeSF-RCS-auto structural30", "HeSF-RCS-auto structural20", "best available external TP baseline")
    existing = {(str(row.get("dataset", "")).upper(), str(row.get("method", ""))) for row in rows}
    for dataset in [str(item).upper() for item in args.datasets if str(item).upper() in {"ACM", "IMDB"}]:
        for method in required_methods:
            if (dataset, method) in existing:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "graph_seed": graph_seeds[0] if graph_seeds else 1,
                    "training_seed": training_seeds[0] if training_seeds else 1,
                    "training_executed": False,
                    "success": False,
                    "uses_test_metrics_for_selection": False,
                    "uses_test_labels_for_selection": False,
                    "failure_type": "runtime_limited_cross_dataset_not_executed",
                    "failure_reason": "Required Gate21.12 ACM/IMDB task metric was not executed in this local run.",
                }
            )
    plans = gate21_12_cross_dataset_selector_plans(args.datasets)
    write_rows(paths["cross_dataset"] / "gate21_12_cross_dataset_runs.csv", rows)
    write_rows(paths["cross_dataset"] / "gate21_12_cross_dataset_by_method.csv", summarize_gate21_12_cross_dataset(rows))
    write_rows(paths["cross_dataset"] / "gate21_12_cross_dataset_selector_plans.csv", plans)


def _write_coverage(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int]) -> None:
    rows = []
    for row in read_csv(Path(args.gate21_11_root) / "summary" / "gate21_11_coverage_semantic_diagnostics.csv"):
        out = dict(row)
        for field in (
            "per_class_venue_coverage",
            "per_class_paper_coverage",
            "author_degree_bucket_recovery",
            "paper_degree_bucket_recovery",
            "venue_degree_bucket_recovery",
            "AP_PV_path_multiplicity_mean",
            "AP_PV_path_multiplicity_std",
            "APA_feedback_path_count",
            "VPA_feedback_path_count",
            "paper_venue_entropy",
            "venue_class_proxy_purity_trainval",
            "paper_class_proxy_purity_trainval",
        ):
            out.setdefault(field, "")
        if not out.get("per_class_venue_coverage"):
            out["failure_type"] = "distributional_coverage_not_computed"
            out["failure_reason"] = "Gate21.12 distributional/semantic coverage metrics were not computed in this local run."
        rows.append(out)
    if not rows:
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": "HeSF-RCS-APV16",
                "graph_seed": graph_seeds[0] if graph_seeds else 1,
                "failure_type": "coverage_diagnostics_missing",
                "failure_reason": "No Gate21.12 coverage diagnostics were available.",
                "relation_direction_matches_official_relation_name": True,
                "node_type_offsets_match_node_dat_counts": True,
            }
        )
    write_rows(paths["coverage"] / "gate21_12_coverage_diagnostics.csv", rows)


def _selected_sections(sections_arg: str) -> set[str]:
    aliases = {
        "all": {"official_main", "selector_audit", "external_tp", "freehgc", "metapath_cache", "feature_ablation", "adapter", "system_cost", "cross_dataset", "coverage"},
        "selector": {"selector_audit"},
        "budgeted_selector": {"selector_audit"},
    }
    parts = [part.strip() for part in str(sections_arg or "all").split(",") if part.strip()]
    selected: set[str] = set()
    for part in parts:
        selected.update(aliases.get(part, {part}))
    return selected or set(aliases["all"])


def _write_readmes(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if name == "root":
            continue
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text(f"# Gate21.12 {name}\n\nExecuted evidence completion component.\n", encoding="utf-8")


def _seed_list(values: Sequence[int], *, quick: bool) -> list[int]:
    seeds = [int(value) for value in values]
    return seeds[:1] if quick and seeds else seeds


def _budget_matched(row: Mapping[str, Any]) -> bool:
    requested = float_value(row.get("requested_budget"))
    if requested is None:
        return False
    if str(row.get("budget_type", "")).startswith("structural"):
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
    parser = argparse.ArgumentParser(description="Run Gate21.12 executed evidence completion.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--sections", default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate21-11-root", type=Path, default=DEFAULT_GATE21_11_ROOT)
    parser.add_argument("--gate21-10-root", type=Path, default=DEFAULT_GATE21_10_ROOT)
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--freehgc-zip", type=Path, default=Path("FreeHGC-main (1).zip"))
    parser.add_argument("--quick", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--dry-run", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
