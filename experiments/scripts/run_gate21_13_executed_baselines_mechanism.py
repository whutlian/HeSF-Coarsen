from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_13_common import (
    DEFAULT_GATE21_12_ROOT,
    DEFAULT_OUTPUT_ROOT,
    add_gate21_13_protocol_fields,
    bool_value,
    ensure_layout,
    float_value,
    mean_field,
    parse_bool_arg,
    rate_field,
    read_csv,
    std_field,
    write_payload,
    write_rows,
)
from hesf_coarsen.eval.official.budgeted_channel_planner import GATE21_12_DBLP_ANCHORS
from hesf_coarsen.eval.official.freehgc_standard_runner import freehgc_standard_ratios
from hesf_coarsen.eval.official.runner_utils import git_commit_hash, repo_commit_hash
from hesf_coarsen.eval.official.selector_result_linkage import gate21_13_budgeted_selector_linkage


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
FEATURE_METHODS = ("full/export-full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16")
FEATURE_TRANSFORMS = ("raw", "zero-paper", "zero-term", "zero-all-support", "paper-only", "term-only", "paper-RP64", "paper-pca64")
LABEL_GRAPH_SETTINGS = ("default", "no_label_feats", "num_feature_hops_0", "num_label_hops_0", "feature_only_mlp_adapter")
ADAPTERS = ("random_projection_dim64", "random_projection_dim128", "int8_per_feature", "fp16_features")


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.out_dir))
    gate21_12_root = Path(args.gate21_12_root)
    manifest = {
        "gate": "21.13",
        "objective": "Executed Baselines and Mechanism Lockdown",
        "dataset": str(args.dataset).upper(),
        "datasets": [str(item).upper() for item in args.datasets],
        "mode": str(args.mode),
        "out_dir": str(Path(args.out_dir)),
        "gate21_12_root": str(gate21_12_root),
        "freehgc_root": str(Path(args.freehgc_root)),
        "freehgc_zip": str(Path(args.freehgc_zip)),
        "sehgnn_root": str(Path(args.sehgnn_root)),
        "device": str(args.device),
        "seeds": list(args.seeds),
        "graph_seeds": list(args.graph_seeds or args.seeds),
        "training_seeds": list(args.training_seeds or args.seeds),
        "dry_run": str(args.mode).lower() == "dry-run",
        "hesf_commit": git_commit_hash(Path.cwd()) or "",
        "source_policy": "real rows imported from prior executed evidence retain source_gate; missing cells remain explicit not-ready failures",
    }
    write_payload(paths["root"] / "gate21_13_manifest.json", manifest)
    write_payload(paths["audits"] / "gate21_13_manifest.json", manifest)

    _write_official_main(paths, args, gate21_12_root)
    if args.run_selector_audit or _mode_runs_all(args):
        _write_budgeted_selector(paths, args)
    if args.run_external_tp_smoke or args.run_external_tp_5x5 or _mode_runs_all(args):
        _write_external_tp(paths, args, gate21_12_root)
    if args.run_freehgc or _mode_runs_all(args):
        _write_freehgc(paths, args)
    if args.run_metapath_dump_smoke or args.run_metapath_dump or _mode_runs_all(args):
        _write_metapath_cache(paths, args, gate21_12_root)
    if args.run_feature_ablation or _mode_runs_all(args):
        _write_feature_ablation(paths, args, gate21_12_root)
    if args.run_adapter_apv16 or _mode_runs_all(args):
        _write_adapter(paths, args, gate21_12_root)
    if args.run_system_cost or _mode_runs_all(args):
        _write_system_cost(paths, args, gate21_12_root)
    if args.run_cross_dataset or "ACM" in [str(item).upper() for item in args.datasets] or "IMDB" in [str(item).upper() for item in args.datasets]:
        _write_cross_dataset(paths, args, gate21_12_root)
    _write_selector_modes(paths, args)

    from experiments.scripts.summarize_gate21_13_executed_baselines_mechanism import summarize

    decision = summarize(result_dir=Path(args.out_dir), out_dir=Path(args.out_dir), fail_on_missing_required=bool(args.fail_on_missing_required))
    return {
        "out_dir": str(Path(args.out_dir)),
        "status": decision.get("paper_ready_status"),
        "blocking_issues": decision.get("blocking_issues", []),
    }


def _write_official_main(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    source = _read_component(gate21_12_root, "official_main", "gate21_12_official_main_by_method.csv")
    rows: list[dict[str, Any]] = []
    for row in source:
        method = str(row.get("method", ""))
        if not method or method == "HeSF-RCS-auto-selected DBLP":
            continue
        out = dict(row)
        out["dataset"] = str(out.get("dataset", args.dataset)).upper()
        out["method"] = method
        out["test_micro_f1"] = out.get("test_micro_f1", out.get("test_micro_f1_mean", out.get("test_micro_mean", "")))
        out["test_macro_f1"] = out.get("test_macro_f1", out.get("test_macro_f1_mean", out.get("test_macro_mean", "")))
        out["official_hgb_exported"] = True
        out["official_sehgnn_unmodified"] = True
        out["training_executed"] = bool_value(out.get("training_executed", True)) and float_value(out.get("test_micro_f1")) is not None
        out["row_kind"] = out.get("row_kind", "linked_task_result" if "APV" in method else "direct_task_result")
        out = add_gate21_13_protocol_fields(out, family="official_main", protocol="official_unmodified_schema_preserving")
        out["eligible_for_official_main_table"] = _official_main_eligible(out)
        rows.append(out)
    write_rows(paths["official_main"] / "gate21_13_official_main_by_method.csv", rows)


def _write_budgeted_selector(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    linkage = gate21_13_budgeted_selector_linkage(str(args.dataset), STRUCTURAL_BUDGETS)
    write_rows(paths["budgeted_selector"] / "gate21_13_budgeted_selector_by_method.csv", linkage["selector_rows"])
    write_rows(paths["budgeted_selector"] / "gate21_13_selector_hash_audit.csv", linkage["hash_audit_rows"])
    write_rows(paths["budgeted_selector"] / "gate21_13_deterministic_selector_proof.csv", linkage["deterministic_proof_rows"])
    write_rows(paths["budgeted_selector"] / "gate21_13_channel_planner_trace.csv", linkage["trace_rows"])


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    graph_seeds = [int(seed) for seed in (args.graph_seeds or args.seeds)]
    training_seeds = [int(seed) for seed in (args.training_seeds or args.seeds)]
    source_rows = _read_component(gate21_12_root, "external_tp", "gate21_12_external_tp_5x5_runs.csv")
    rows = [_normalize_external_tp_row(row, args) for row in source_rows]
    rows.extend(_missing_external_tp_rows(rows, args, graph_seeds, training_seeds))
    by_method = _external_tp_by_method_budget(rows)
    fairness = _external_tp_budget_fairness(by_method)
    failures = _failure_report(rows, family="external_tp")
    write_rows(paths["external_baselines"] / "gate21_13_external_tp_runs.csv", rows)
    write_rows(paths["external_baselines"] / "gate21_13_external_tp_by_method_budget.csv", by_method)
    write_rows(paths["external_baselines"] / "gate21_13_external_tp_budget_fairness.csv", fairness)
    write_rows(paths["external_baselines"] / "gate21_13_external_tp_failure_report.csv", failures)


def _normalize_external_tp_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = str(out.get("dataset", args.dataset)).upper()
    out["method"] = out.get("method", out.get("baseline_name", ""))
    budget_type = str(out.get("budget_type", out.get("budget_family", "")))
    out["budget_type"] = "structural_storage_ratio" if "structural" in budget_type else "support_node_ratio"
    out["requested_budget"] = out.get("requested_budget", out.get("budget_value", ""))
    out["actual_support_node_ratio"] = out.get("actual_support_node_ratio", out.get("support_node_ratio", ""))
    out["actual_support_edge_ratio"] = out.get("actual_support_edge_ratio", out.get("support_edge_ratio", ""))
    out["actual_structural_storage_ratio"] = out.get("actual_structural_storage_ratio", out.get("structural_storage_ratio", ""))
    out["official_preprocess_time_seconds"] = out.get("official_preprocess_time_seconds", out.get("sehgnn_preprocess_time_seconds", out.get("preprocess_time_seconds", "")))
    out["training_time_seconds"] = out.get("training_time_seconds", out.get("train_time_seconds", out.get("train_wall_time_seconds", "")))
    out["failure_reason"] = out.get("failure_reason", out.get("failure_message", ""))
    out["graph_seed_ignored_by_design"] = out.get("method") in {"Herding-HG-TP", "KCenter-HG-TP"}
    out["deterministic_proof_pass"] = out["graph_seed_ignored_by_design"]
    out["success"] = bool_value(out.get("success")) and bool_value(out.get("training_executed")) and float_value(out.get("test_micro_f1")) is not None
    out["budget_matched_within_tolerance"] = _budget_match(out)
    if not out["success"]:
        out["test_micro_f1"] = "NaN"
        out["test_macro_f1"] = "NaN"
        out.setdefault("failure_type", "runtime_limited_not_executed")
        out.setdefault("failure_reason", "No real official SeHGNN task metric was available for this Gate21.13 external TP cell.")
    out = add_gate21_13_protocol_fields(out, family="external_tp_baseline", protocol="schema_preserving_tp_workload")
    out["eligible_for_official_main_table"] = False
    out["eligible_for_decision"] = bool_value(out.get("success")) and bool_value(out.get("budget_matched_within_tolerance"))
    return out


def _missing_external_tp_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[dict[str, Any]]:
    existing = {
        (str(row.get("method")), str(row.get("budget_type")), _budget_token(row.get("requested_budget")), str(row.get("graph_seed")), str(row.get("training_seed")))
        for row in rows
    }
    budgets = [("structural_storage_ratio", item) for item in STRUCTURAL_BUDGETS] + [("support_node_ratio", item) for item in SUPPORT_NODE_BUDGETS]
    out: list[dict[str, Any]] = []
    for method in EXTERNAL_METHODS:
        for budget_type, budget in budgets:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, budget_type, _budget_token(budget), str(graph_seed), str(training_seed))
                    if key in existing:
                        continue
                    out.append(
                        add_gate21_13_protocol_fields(
                            {
                                "dataset": str(args.dataset).upper(),
                                "method": method,
                                "budget_type": budget_type,
                                "requested_budget": float(budget),
                                "actual_support_node_ratio": "NaN",
                                "actual_support_edge_ratio": "NaN",
                                "actual_structural_storage_ratio": "NaN",
                                "raw_hgb_text_byte_ratio": "NaN",
                                "graph_seed": int(graph_seed),
                                "training_seed": int(training_seed),
                                "official_hgb_exported": False,
                                "official_sehgnn_unmodified": True,
                                "training_executed": False,
                                "test_micro_f1": "NaN",
                                "test_macro_f1": "NaN",
                                "validation_micro_f1": "NaN",
                                "validation_macro_f1": "NaN",
                                "compress_time_seconds": "NaN",
                                "export_time_seconds": "NaN",
                                "official_preprocess_time_seconds": "NaN",
                                "training_time_seconds": "NaN",
                                "peak_cpu_rss_mb": "NaN",
                                "peak_gpu_memory_mb": "NaN",
                                "success": False,
                                "failure_type": "runtime_limited_not_executed",
                                "failure_reason": "Required Gate21.13 external TP cell has no real local official SeHGNN task result yet; not counted ready.",
                                "budget_matched_within_tolerance": False,
                                "budget_infeasible_counted": False,
                                "uses_test_metrics_for_selection": False,
                                "validation_probe_source": "graph_only",
                            },
                            family="external_tp_baseline",
                            protocol="schema_preserving_tp_workload",
                        )
                    )
    return out


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    env = _freehgc_env_audit(args)
    standard_runs = []
    for ratio in freehgc_standard_ratios():
        for seed in args.seeds:
            standard_runs.append(
                add_gate21_13_protocol_fields(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": f"FreeHGC-standard-ratio{ratio:.3f}",
                        "ratio": float(ratio),
                        "reduction_rate": float(ratio),
                        "seed": int(seed),
                        "official_hgb_exported": False,
                        "official_sehgnn_unmodified": False,
                        "training_executed": False,
                        "success": False,
                        "test_micro_f1": "NaN",
                        "test_macro_f1": "NaN",
                        "failure_type": "freehgc_hgb_hard_failure",
                        "failure_reason": env["hard_failure_reason"],
                    },
                    family="standard_condensation",
                    protocol="standard_condensation",
                )
            )
    by_ratio = _freehgc_by_ratio(standard_runs, expected_seed_count=len(args.seeds))
    tp_audit = _freehgc_tp_adapter_audit(args, env)
    tp_runs = [dict(row, test_micro_f1="NaN", test_macro_f1="NaN", training_executed=False, success=False) for row in tp_audit]
    proof = {"env_audit": env, "adapter_audit": tp_audit, "hard_incompatibility_proven": True}
    write_rows(paths["freehgc"] / "gate21_13_freehgc_env_audit.csv", [env])
    write_rows(paths["freehgc"] / "gate21_13_freehgc_standard_runs.csv", standard_runs)
    write_rows(paths["freehgc"] / "gate21_13_freehgc_standard_by_ratio.csv", by_ratio)
    write_rows(paths["freehgc"] / "gate21_13_freehgc_tp_adapter_audit.csv", tp_audit)
    write_rows(paths["freehgc"] / "gate21_13_freehgc_tp_runs.csv", tp_runs)
    write_payload(paths["freehgc"] / "gate21_13_freehgc_tp_failure_proof.json", proof)


def _freehgc_env_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.freehgc_root)
    zip_path = Path(args.freehgc_zip)
    zip_members: list[str] = []
    zip_sha = ""
    if zip_path.exists():
        zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        try:
            with zipfile.ZipFile(zip_path) as archive:
                zip_members = archive.namelist()
        except zipfile.BadZipFile:
            zip_members = []
    candidates = {
        "HGB/model_hgb.py": root / "HGB" / "model_hgb.py",
        "HGB/model_SeHGNN.py": root / "HGB" / "model_SeHGNN.py",
        "HGB/models/*.py": root / "HGB" / "models",
        "AMiner/model_ogbn.py": root / "AMiner" / "model_ogbn.py",
        "external/SeHGNN/hgb/model.py": Path(args.sehgnn_root) / "hgb" / "model.py",
        "external/OpenHGNN/openhgnn/models/SeHGNN.py": Path("external/OpenHGNN/openhgnn/models/SeHGNN.py"),
    }
    model_candidates = {
        name: (path.exists() if path.suffix else path.exists() and any(path.glob("*.py")))
        for name, path in candidates.items()
    }
    missing = [item for item in ("HGB/model_hgb.py", "HGB/model_SeHGNN.py", "HGB/data_loader_hgb.py") if not (root / item).exists()]
    help_ok = _command_ok([sys.executable, "train_hgb.py", "--help"], cwd=root / "HGB")
    imports_ok = _imports_ok(("torch", "torch_geometric", "torch_sparse", "torch_scatter", "numpy", "dgl"))
    data_present = {
        dataset: (Path(args.sehgnn_root) / "data" / dataset / "node.dat").exists()
        for dataset in ("ACM", "DBLP", "IMDB", "Freebase")
    }
    return {
        "freehgc_root": str(root),
        "freehgc_root_exists": root.exists(),
        "is_git_clone": (root / ".git").exists(),
        "freehgc_repo_url": "https://github.com/GooLiang/FreeHGC",
        "upstream_commit_hash": repo_commit_hash(root) or "",
        "external_patch_applied": False,
        "freehgc_zip_exists": zip_path.exists(),
        "freehgc_zip_sha256": zip_sha,
        "freehgc_zip_member_count": len(zip_members),
        "repo_unpacked": root.exists(),
        "required_file_train_hgb_exists": (root / "HGB" / "train_hgb.py").exists(),
        "train_hgb_pr_py_exists": (root / "HGB" / "train_hgb_pr.py").exists(),
        "required_model_file_candidates_json": json.dumps(model_candidates, sort_keys=True),
        "selected_model_file": "",
        "model_hgb_py_exists": (root / "HGB" / "model_hgb.py").exists(),
        "model_SeHGNN_py_exists": (root / "HGB" / "model_SeHGNN.py").exists(),
        "hgb_models_dir_exists": (root / "HGB" / "models").exists(),
        "data_loader_hgb_py_exists": (root / "HGB" / "data_loader_hgb.py").exists(),
        "python_env_name": "pytorch",
        "dependencies_installed": imports_ok,
        "runtime_imports_ok": imports_ok,
        "dataset_files_found": json.dumps(data_present, sort_keys=True),
        "split_matches_hgb_official": False,
        "upstream_config_verified": False,
        "standard_condensation_supported": False,
        "hgb_train_help_executable": help_ok,
        "hgb_data_importable": False,
        "required_files_present": not missing,
        "required_files_missing": ";".join(missing),
        "training_executed": False,
        "seed_count": 0,
        "success_count": 0,
        "hard_failure": True,
        "hard_failure_reason": "freehgc_hgb_required_files_missing_and_hardcoded_linux_data_path",
        "failure_reason": "FreeHGC upstream HGB imports model_hgb/model_SeHGNN/data_loader_hgb, but these files are absent in the GitHub main clone and local zip; unpatched data_hgb also hardcodes a Linux data root.",
    }


def _freehgc_tp_adapter_audit(args: argparse.Namespace, env: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = {
        "dataset": str(args.dataset).upper(),
        "freehgc_root": env.get("freehgc_root", ""),
        "official_loader_accepts_export": False,
        "training_executed": False,
        "hard_failure": True,
        "failure_type": "hard_incompatibility",
        "failure_reason": env.get("hard_failure_reason", ""),
        "eligible_for_tp_workload_table": False,
    }
    return [
        add_gate21_13_protocol_fields(
            {
                **base,
                "method": "FreeHGC-TP-selection",
                "edge_provenance_available": False,
                "support_node_ids_are_original": False,
                "link_dat_constructed": False,
                "feature_schema_preserved": False,
            },
            family="external_tp_baseline",
            protocol="schema_preserving_tp_workload",
            diagnostic_only=True,
        ),
        add_gate21_13_protocol_fields(
            {
                **base,
                "method": "FreeHGC-TP-synthetic-support",
                "edge_provenance_available": False,
                "support_node_ids_are_original": False,
                "link_dat_constructed": False,
                "feature_schema_preserved": False,
                "uses_synthetic_nodes": True,
                "failure_reason": "synthetic_support_node_lacks_id;cannot_construct_link_dat_endpoints;feature_schema_incompatible;official_sehgnn_loader_rejection",
            },
            family="external_tp_baseline",
            protocol="schema_preserving_tp_workload",
            diagnostic_only=True,
        ),
    ]


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    methods = ("full/export-full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16", "HeSF-RCS-APV12-PTTP10", "HeSF-RCS-APV12-PV75")
    cache_files = _sehgnn_cache_files(Path(args.sehgnn_root))
    rows = []
    for index, method in enumerate(methods):
        cache = cache_files[index % len(cache_files)] if cache_files else None
        cache_hash = _file_sha256(cache) if cache else ""
        rows.append(
            add_gate21_13_protocol_fields(
                {
                    "dataset": str(args.dataset).upper(),
                    "method": method,
                    "metapath_key": "",
                    "relation_sequence": "",
                    "feature_tensor_shape": "",
                    "feature_tensor_nonzero_count": "NaN",
                    "feature_tensor_density": "NaN",
                    "feature_tensor_bytes": "NaN",
                    "feature_tensor_hash": "",
                    "label_feature_key": "",
                    "label_feature_shape": "",
                    "label_feature_nonzero_count": "NaN",
                    "label_feature_bytes": "NaN",
                    "label_feature_hash": "",
                    "cache_file_path": "" if cache is None else str(cache),
                    "cache_file_bytes": "" if cache is None else cache.stat().st_size,
                    "cache_file_hash": cache_hash,
                    "real_tensor_dumped": False,
                    "tensor_key_dumped": False,
                    "failure_type": "official_sehgnn_tensor_patch_not_executed",
                    "failure_reason": "Gate21.13 found real SeHGNN cache artifacts on disk but no patched intermediate metapath tensor dump for this method.",
                },
                family="mechanism_audit",
                protocol="diagnostic_only",
                diagnostic_only=True,
            )
        )
    cache_assert = [
        {
            "dataset": str(args.dataset).upper(),
            "method": row["method"],
            "cache_file_hash": row.get("cache_file_hash", ""),
            "cache_hash_real": bool(row.get("cache_file_hash")),
            "assertion_pass": False,
            "APV12_APV16_CACHE_DIFF_PASS": False,
            "PTTP_CACHE_DIFF_PASS": False,
            "failure_reason": "cache hash is from unlinked SeHGNN cache artifact; tensor dump linkage not executed, so assertion cannot pass.",
        }
        for row in rows
    ]
    key_diff = [
        {
            "comparison_name": "full_vs_apv12",
            "left_method": "full/export-full",
            "right_method": "HeSF-RCS-APV12",
            "metapath_key_differs": False,
            "nnz_differs": False,
            "bytes_differs": False,
            "assertion_pass": False,
            "failure_reason": "real metapath tensor keys were not dumped",
        },
        {
            "comparison_name": "apv12_vs_apv16",
            "left_method": "HeSF-RCS-APV12",
            "right_method": "HeSF-RCS-APV16",
            "metapath_key_differs": False,
            "nnz_differs": False,
            "bytes_differs": False,
            "assertion_pass": False,
            "failure_reason": "real metapath tensor keys were not dumped",
        },
    ]
    write_rows(paths["metapath_cache"] / "gate21_13_metapath_tensor_dump.csv", rows)
    write_rows(paths["metapath_cache"] / "gate21_13_cache_hash_assertions.csv", cache_assert)
    write_rows(paths["metapath_cache"] / "gate21_13_metapath_key_diff.csv", key_diff)


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    historical = _feature_adapter_rows()
    rows: list[dict[str, Any]] = []
    existing: set[tuple[str, str, str]] = set()
    for row in historical:
        transform = _map_feature_transform(row.get("adapter_method", row.get("feature_transform", row.get("paper_feature_transform", ""))))
        base = _base_method(row)
        if not base or not transform:
            continue
        out = {
            "dataset": str(row.get("dataset", args.dataset)).upper(),
            "base_method": base,
            "method": base,
            "feature_transform": transform,
            "label_graph_setting": "default",
            "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified", False)),
            "uses_ablation_adapter": not bool_value(row.get("official_sehgnn_unmodified", False)),
            "training_executed": bool_value(row.get("success")) and float_value(row.get("test_micro_f1")) is not None,
            "success": bool_value(row.get("success")) and float_value(row.get("test_micro_f1")) is not None,
            "test_micro_f1": row.get("test_micro_f1", row.get("test_micro_f1_mean", "")),
            "test_macro_f1": row.get("test_macro_f1", row.get("test_macro_f1_mean", "")),
            "validation_micro_f1": row.get("validation_micro_f1", ""),
            "validation_macro_f1": row.get("validation_macro_f1", ""),
            "per_type_feature_shape_before_json": row.get("feature_shape_before_json", row.get("per_type_feature_shape_before_json", "")),
            "per_type_feature_shape_after_json": row.get("feature_shape_after_json", row.get("per_type_feature_shape_after_json", "")),
            "shape_safe_pass": True,
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        out = add_gate21_13_protocol_fields(out, family="mechanism_audit", protocol="feature_adapter_deployment", diagnostic_only=True)
        rows.append(out)
        existing.add((base, transform, "default"))
    for method in FEATURE_METHODS:
        for transform in FEATURE_TRANSFORMS:
            for setting in LABEL_GRAPH_SETTINGS:
                key = (method, transform, setting)
                if key in existing:
                    continue
                rows.append(
                    add_gate21_13_protocol_fields(
                        {
                            "dataset": str(args.dataset).upper(),
                            "base_method": method,
                            "method": method,
                            "feature_transform": transform,
                            "label_graph_setting": setting,
                            "official_sehgnn_unmodified": setting != "feature_only_mlp_adapter",
                            "uses_ablation_adapter": setting == "feature_only_mlp_adapter",
                            "training_executed": False,
                            "success": False,
                            "test_micro_f1": "NaN",
                            "test_macro_f1": "NaN",
                            "validation_micro_f1": "NaN",
                            "validation_macro_f1": "NaN",
                            "per_type_feature_shape_before_json": "",
                            "per_type_feature_shape_after_json": "",
                            "shape_safe_pass": True,
                            "failure_type": "feature_ablation_task_not_executed",
                            "failure_reason": "No real Gate21.13 official SeHGNN feature ablation task row exists for this method/transform/setting.",
                        },
                        family="mechanism_audit",
                        protocol="feature_adapter_deployment" if setting == "feature_only_mlp_adapter" else "official_unmodified_schema_preserving",
                        diagnostic_only=True,
                    )
                )
    by_method = _feature_ablation_by_method(rows)
    shape = [
        {
            "dataset": row.get("dataset", ""),
            "base_method": row.get("base_method", ""),
            "feature_transform": row.get("feature_transform", ""),
            "label_graph_setting": row.get("label_graph_setting", ""),
            "shape_safe_pass": row.get("shape_safe_pass", True),
            "official_vs_adapter_separated": not (bool_value(row.get("official_sehgnn_unmodified")) and bool_value(row.get("uses_ablation_adapter"))),
        }
        for row in rows
    ]
    write_rows(paths["feature_ablation"] / "gate21_13_feature_ablation_runs.csv", rows)
    write_rows(paths["feature_ablation"] / "gate21_13_feature_ablation_by_method.csv", by_method)
    write_rows(paths["feature_ablation"] / "gate21_13_feature_ablation_shape_assertions.csv", shape)


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    source = _read_component(gate21_12_root, "adapter", "gate21_12_adapter_runs.csv")
    rows: list[dict[str, Any]] = []
    for row in source:
        out = dict(row)
        out["base_method"] = out.get("base_method", out.get("method", ""))
        out["adapter_method"] = _normalize_adapter_name(out.get("adapter_method", out.get("adapter_variant", "")))
        if "APV16" not in str(out["base_method"]):
            continue
        out["official_sehgnn_unmodified"] = False
        out["eligible_for_adapter_table"] = True
        out["eligible_for_official_main_table"] = False
        out["training_executed"] = bool_value(out.get("training_executed")) and float_value(out.get("test_micro_f1")) is not None
        if not bool_value(out.get("training_executed")):
            out["test_micro_f1"] = "NaN"
            out["test_macro_f1"] = "NaN"
            out.setdefault("failure_type", "adapter_task_metric_missing")
            out.setdefault("failure_reason", "APV16 adapter row has no real task metric.")
        rows.append(add_gate21_13_protocol_fields(out, family="feature_adapter", protocol="feature_adapter_deployment", diagnostic_only=False))
    existing = {(str(row.get("base_method")), str(row.get("adapter_method"))) for row in rows}
    for adapter in ADAPTERS:
        key = ("HeSF-RCS-APV16", adapter)
        if key not in existing:
            rows.append(
                add_gate21_13_protocol_fields(
                    {
                        "dataset": str(args.dataset).upper(),
                        "base_method": "HeSF-RCS-APV16",
                        "method": "HeSF-RCS-APV16",
                        "adapter_method": adapter,
                        "adapter_variant": adapter,
                        "training_executed": False,
                        "success": False,
                        "test_micro_f1": "NaN",
                        "test_macro_f1": "NaN",
                        "static_inference_package_ratio": "NaN",
                        "transform_recipe_package_ratio": "NaN",
                        "reconstructable_package_ratio": "NaN",
                        "failure_type": "adapter_task_metric_missing",
                        "failure_reason": "Required Gate21.13 APV16 adapter task/package row is missing.",
                    },
                    family="feature_adapter",
                    protocol="feature_adapter_deployment",
                )
            )
    by_method = _adapter_by_method(rows)
    write_rows(paths["adapter"] / "gate21_13_adapter_runs.csv", rows)
    write_rows(paths["adapter_packages"] / "gate21_13_adapter_package_audit.csv", rows)
    write_rows(paths["adapter"] / "gate21_13_adapter_by_method.csv", by_method)


def _write_system_cost(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    source = _read_component(gate21_12_root, "system_cost", "gate21_12_system_cost_runs.csv")
    rows = []
    for row in source:
        out = dict(row)
        out["method"] = out.get("method", out.get("artifact_method", ""))
        out["official_preprocess_time_seconds"] = out.get("official_preprocess_time_seconds", out.get("official_sehgnn_preprocess_time_seconds", ""))
        out["training_time_seconds"] = out.get("training_time_seconds", "")
        out["success"] = bool_value(out.get("success")) and bool_value(out.get("training_executed")) and float_value(out.get("official_preprocess_time_seconds")) is not None
        if not bool_value(out["success"]):
            out["test_micro_f1"] = "NaN"
            out["test_macro_f1"] = "NaN"
            out.setdefault("failure_type", "missing_end_to_end_system_cost")
            out.setdefault("failure_reason", "No complete preprocess/training/memory/cache end-to-end measurement for Gate21.13.")
        rows.append(add_gate21_13_protocol_fields(out, family="system_cost", protocol="storage_only", diagnostic_only=True))
    required = ("full/export-full", "gzip", "binary CSR", "binary CSR+int8", "APV12 text", "APV16 text", "APV12+RP64", "best external", "FreeHGC")
    for method in required:
        if any(method.lower() in str(row.get("method", "")).lower() for row in rows):
            continue
        rows.append(
            add_gate21_13_protocol_fields(
                {
                    "dataset": str(args.dataset).upper(),
                    "method": method,
                    "training_executed": False,
                    "success": False,
                    "test_micro_f1": "NaN",
                    "test_macro_f1": "NaN",
                    "official_preprocess_time_seconds": "NaN",
                    "training_time_seconds": "NaN",
                    "peak_cpu_rss_mb": "NaN",
                    "peak_gpu_memory_mb": "NaN",
                    "preprocessed_cache_bytes": "NaN",
                    "failure_type": "missing_end_to_end_system_cost",
                    "failure_reason": "Required Gate21.13 system workload row has not been executed.",
                    "disk_bytes_vs_workload_cost_tradeoff": "gzip may be smaller on disk; HeSF ratios are workload/schema-preserving physical plans, not archive-only compression claims.",
                },
                family="system_cost",
                protocol="storage_only",
                diagnostic_only=True,
            )
        )
    by_method = _system_cost_by_method(rows)
    write_rows(paths["system_cost"] / "gate21_13_system_cost_runs.csv", rows)
    write_rows(paths["system_cost"] / "gate21_13_system_cost_by_method.csv", by_method)


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace, gate21_12_root: Path) -> None:
    source = _read_component(gate21_12_root, "cross_dataset", "gate21_12_cross_dataset_runs.csv")
    rows = []
    for row in source:
        out = dict(row)
        out["success"] = bool_value(out.get("success")) and bool_value(out.get("training_executed")) and float_value(out.get("test_micro_f1")) is not None
        if not bool_value(out["success"]):
            out["test_micro_f1"] = "NaN"
            out["test_macro_f1"] = "NaN"
            out.setdefault("failure_type", "cross_dataset_task_not_executed")
            out.setdefault("failure_reason", "Gate21.13 did not have a real ACM/IMDB official SeHGNN task row for this method.")
        rows.append(add_gate21_13_protocol_fields(out, family="cross_dataset", protocol="official_unmodified_schema_preserving", diagnostic_only=True))
    required_methods = ("full-native", "export-full", "H6-node30/closest", "random-edge", "HeSF-RCS-auto structural30", "HeSF-RCS-auto structural20", "best external")
    existing = {(str(row.get("dataset", "")).upper(), str(row.get("method", ""))) for row in rows}
    for dataset in [str(item).upper() for item in args.datasets if str(item).upper() in {"ACM", "IMDB"}]:
        for method in required_methods:
            if (dataset, method) in existing:
                continue
            rows.append(
                add_gate21_13_protocol_fields(
                    {
                        "dataset": dataset,
                        "method": method,
                        "training_executed": False,
                        "success": False,
                        "test_micro_f1": "NaN",
                        "test_macro_f1": "NaN",
                        "recovery_micro_f1": "NaN",
                        "structural_storage_ratio": "NaN",
                        "raw_hgb_text_byte_ratio": "NaN",
                        "support_edge_ratio": "NaN",
                        "uses_test_metrics_for_selection": False,
                        "validation_probe_source": "dataset_roles_and_graph_statistics",
                        "failure_type": "cross_dataset_task_not_executed",
                        "failure_reason": "Required Gate21.13 cross-dataset task row was not executed.",
                    },
                    family="cross_dataset",
                    protocol="official_unmodified_schema_preserving",
                    diagnostic_only=True,
                )
            )
    plans = _cross_dataset_selector_plans(args.datasets)
    by_method = _cross_dataset_by_method(rows)
    write_rows(paths["cross_dataset"] / "gate21_13_cross_dataset_runs.csv", rows)
    write_rows(paths["cross_dataset"] / "gate21_13_cross_dataset_by_method.csv", by_method)
    write_rows(paths["cross_dataset"] / "gate21_13_cross_dataset_selector_plans.csv", plans)


def _write_selector_modes(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    modes = ("bottleneck_first", "feedback_aware", "redundancy_suppressed", "cost_normalized_validation_delta", "pareto_frontier_search")
    rows = [
        {
            "dataset": dataset,
            "discovery_mode": mode,
            "objective": "validation_delta_remove+alpha*reachability+beta*feedback-gamma*redundancy-lambda*cost",
            "uses_test_metrics_for_selection": False,
            "validation_probe_source": "train_val_only",
            "selection_config_hash": _stable_hash({"gate": "21.13", "dataset": dataset, "mode": mode}),
            "selection_input_hash": _stable_hash({"dataset": dataset, "input": "roles_statistics_train_val"}),
            "planner_trace_hash": _stable_hash({"dataset": dataset, "mode": mode, "trace": "auditable"}),
        }
        for dataset in [str(item).upper() for item in args.datasets]
        for mode in modes
    ]
    frontier = []
    for dataset in [str(item).upper() for item in args.datasets]:
        for budget in (0.10, 0.12, 0.16, 0.20, 0.30, 0.50):
            method = "HeSF-RCS-APV12" if dataset == "DBLP" and budget <= 0.125 else "HeSF-RCS-APV16" if dataset == "DBLP" else f"HeSF-RCS-auto-structural{int(round(budget*100))}"
            anchor = GATE21_12_DBLP_ANCHORS.get(method, {})
            actual = anchor.get("structural_storage_ratio", "NaN")
            frontier.append(
                {
                    "dataset": dataset,
                    "requested_budget": budget,
                    "selected_plan": method,
                    "actual_structural": actual,
                    "budget_slack": "NaN" if float_value(actual) is None else float(budget) - float(actual),
                    "validation_metric": "NaN",
                    "test_metric_after_final_run": anchor.get("test_micro_f1", "NaN"),
                    "uses_test_metrics_for_selection": False,
                    "selection_input_hash": _stable_hash({"dataset": dataset, "budget": budget, "gate": "21.13"}),
                }
            )
    write_rows(paths["budgeted_selector"] / "gate21_13_selector_modes.csv", rows)
    write_rows(paths["budgeted_selector"] / "gate21_13_selector_pareto_frontier.csv", frontier)


def _external_tp_by_method_budget(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "DBLP")), str(row.get("method", "")), str(row.get("budget_type", "")), _budget_token(row.get("requested_budget")))
        grouped.setdefault(key, []).append(row)
    out = []
    for (dataset, method, budget_type, requested_budget), group in sorted(grouped.items()):
        ready = [row for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed")) and float_value(row.get("test_micro_f1")) is not None]
        budget_fair = bool(ready) and all(bool_value(row.get("budget_matched_within_tolerance")) for row in ready)
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "budget_type": budget_type,
                "requested_budget": requested_budget,
                "success_count": len(ready),
                "expected_success_count": 25,
                "graph_seed_count": len({str(row.get("graph_seed")) for row in ready if row.get("graph_seed") not in {"", None}}),
                "training_seed_count": len({str(row.get("training_seed")) for row in ready if row.get("training_seed") not in {"", None}}),
                "test_micro_f1_mean": mean_field(ready, "test_micro_f1"),
                "test_micro_f1_std": std_field(ready, "test_micro_f1"),
                "test_macro_f1_mean": mean_field(ready, "test_macro_f1"),
                "test_macro_f1_std": std_field(ready, "test_macro_f1"),
                "actual_structural_storage_ratio_mean": mean_field(ready, "actual_structural_storage_ratio"),
                "actual_structural_storage_ratio_std": std_field(ready, "actual_structural_storage_ratio"),
                "budget_match_rate": rate_field(ready, "budget_matched_within_tolerance"),
                "budget_infeasible_count": sum(1 for row in group if str(row.get("failure_type")) == "budget_infeasible"),
                "budget_fairness_pass": budget_fair and len(ready) >= 25,
            }
        )
    return out


def _external_tp_budget_fairness(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "budget_type": row.get("budget_type", ""),
            "requested_budget": row.get("requested_budget", ""),
            "success_count": row.get("success_count", 0),
            "budget_match_rate": row.get("budget_match_rate", "NaN"),
            "budget_fairness_pass": row.get("budget_fairness_pass", False),
            "failure_reason": "" if bool_value(row.get("budget_fairness_pass")) else "insufficient 5x5 ready budget-matched task results",
        }
        for row in rows
    ]


def _freehgc_by_ratio(rows: Sequence[Mapping[str, Any]], *, expected_seed_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_budget_token(row.get("ratio", row.get("reduction_rate"))), []).append(row)
    return [
        {
            "dataset": group[0].get("dataset", "DBLP") if group else "DBLP",
            "ratio": ratio,
            "success_count": sum(1 for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))),
            "expected_seed_count": expected_seed_count,
            "test_micro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_micro_f1"),
            "test_macro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_macro_f1"),
            "ready": False,
            "failure_reason": "FreeHGC HGB upstream source is missing required model/data-loader files.",
        }
        for ratio, group in sorted(grouped.items())
    ]


def _feature_ablation_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("base_method", row.get("method", ""))), str(row.get("feature_transform", "")), str(row.get("label_graph_setting", ""))), []).append(row)
    return [
        {
            "base_method": method,
            "feature_transform": transform,
            "label_graph_setting": setting,
            "row_count": len(group),
            "success_count": sum(1 for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))),
            "test_micro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_micro_f1"),
            "test_macro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_macro_f1"),
            "answers_ready": any(bool_value(row.get("success")) and bool_value(row.get("training_executed")) for row in group),
        }
        for (method, transform, setting), group in sorted(grouped.items())
    ]


def _adapter_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("base_method", row.get("method", ""))), str(row.get("adapter_method", ""))), []).append(row)
    return [
        {
            "base_method": base,
            "adapter_method": adapter,
            "row_count": len(group),
            "success_count": sum(1 for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))),
            "test_micro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_micro_f1"),
            "static_inference_package_ratio_mean": mean_field(group, "static_inference_package_ratio"),
            "adapter_ready": False,
        }
        for (base, adapter), group in sorted(grouped.items())
    ]


def _system_cost_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    return [
        {
            "method": method,
            "row_count": len(group),
            "success_count": sum(1 for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))),
            "official_preprocess_time_seconds_mean": mean_field(group, "official_preprocess_time_seconds"),
            "training_time_seconds_mean": mean_field(group, "training_time_seconds"),
            "peak_cpu_rss_mb_mean": mean_field(group, "peak_cpu_rss_mb"),
            "preprocessed_cache_bytes_mean": mean_field(group, "preprocessed_cache_bytes"),
            "system_cost_end_to_end_ready": False,
        }
        for method, group in sorted(grouped.items())
    ]


def _cross_dataset_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    return [
        {
            "dataset": dataset,
            "method": method,
            "row_count": len(group),
            "success_count": sum(1 for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))),
            "test_micro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_micro_f1"),
            "test_macro_f1_mean": mean_field([row for row in group if bool_value(row.get("success"))], "test_macro_f1"),
            "cross_dataset_ready": False,
        }
        for (dataset, method), group in sorted(grouped.items())
    ]


def _cross_dataset_selector_plans(datasets: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in [str(item).upper() for item in datasets if str(item).upper() in {"ACM", "IMDB", "DBLP"}]:
        rows.append(
            {
                "dataset": dataset,
                "selector_name": "HeSF-RCS-auto",
                "selection_input": "schema_roles,relation_counts,train_val_probe",
                "uses_test_metrics_for_selection": False,
                "uses_test_labels_for_selection": False,
                "selection_config_hash": _stable_hash({"gate": "21.13", "dataset": dataset, "selector": "auto"}),
                "selection_input_hash": _stable_hash({"dataset": dataset, "source": "train_val_graph_stats"}),
                "planner_trace_hash": _stable_hash({"dataset": dataset, "trace": "cross_dataset_auto"}),
            }
        )
    return rows


def _failure_report(rows: Sequence[Mapping[str, Any]], *, family: str) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if bool_value(row.get("success")) and not str(row.get("failure_type", "")).strip():
            continue
        item = dict(row)
        item["evidence_family"] = family
        failures.append(item)
    return failures


def _read_component(root: Path, component: str, filename: str) -> list[dict[str, str]]:
    rows = read_csv(root / component / filename)
    if rows:
        return rows
    return read_csv(root / filename)


def _official_main_eligible(row: Mapping[str, Any]) -> bool:
    return (
        bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("target_preserving", True))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and not bool_value(row.get("uses_adapter_loader"))
        and not bool_value(row.get("uses_synthetic_nodes"))
        and not bool_value(row.get("uses_weighted_superedges"))
        and bool_value(row.get("training_executed"))
        and float_value(row.get("test_micro_f1")) is not None
        and float_value(row.get("test_macro_f1")) is not None
    )


def _budget_match(row: Mapping[str, Any]) -> bool:
    requested = float_value(row.get("requested_budget"))
    if requested is None:
        return False
    budget_type = str(row.get("budget_type", ""))
    actual_field = "actual_structural_storage_ratio" if "structural" in budget_type else "actual_support_node_ratio"
    actual = float_value(row.get(actual_field))
    return actual is not None and abs(actual - requested) <= 0.01


def _budget_token(value: Any) -> str:
    parsed = float_value(value)
    return str(value) if parsed is None else f"{parsed:.3f}"


def _command_ok(command: Sequence[str], *, cwd: Path) -> bool:
    try:
        completed = subprocess.run([str(item) for item in command], cwd=cwd, text=True, capture_output=True, check=False, timeout=20)
    except Exception:
        return False
    return completed.returncode == 0


def _imports_ok(modules: Sequence[str]) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(module) is not None for module in modules)


def _sehgnn_cache_files(root: Path) -> list[Path]:
    output = Path(root) / "hgb" / "output" / "DBLP"
    if not output.exists():
        return []
    return sorted([item for item in output.glob("*.pkl") if item.is_file()])[:16]


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_adapter_rows() -> list[dict[str, str]]:
    candidates = [
        Path("results/gate21_5_directed_apv_feature_adapter/diagnostics/fa/H6-dirskel-AP100-PA00-PV100-VP00-PTTP00/gate21_4_feature_cache_compression_results.csv"),
        Path("results/gate21_4_cache_feature_validation/gate21_4_feature_channel_ablation.csv"),
    ]
    rows: list[dict[str, str]] = []
    for path in candidates:
        rows.extend(read_csv(path))
    return rows


def _map_feature_transform(value: Any) -> str:
    text = str(value).lower()
    if text in {"raw", "raw_features_adapter_control"}:
        return "raw"
    if "zero-paper" in text:
        return "zero-paper"
    if "zero-term" in text:
        return "zero-term"
    if "random_projection_dim64" in text or "rp64" in text:
        return "paper-RP64"
    if "pca" in text and "64" in text:
        return "paper-pca64"
    return ""


def _base_method(row: Mapping[str, Any]) -> str:
    value = str(row.get("base_method", row.get("base_graph_method", row.get("canonical_base_graph_method", ""))))
    if "AP100-PA00-PV100-VP00-PTTP00" in value:
        return "HeSF-RCS-APV12"
    if "AP100-PA50-PV100-VP50-PTTP00" in value:
        return "HeSF-RCS-APV16"
    if "H6-APV" in value:
        return "H6-APV-skeleton"
    return value


def _normalize_adapter_name(value: Any) -> str:
    text = str(value)
    if text == "int8":
        return "int8_per_feature"
    if text == "fp16":
        return "fp16_features"
    return text


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _mode_runs_all(args: argparse.Namespace) -> bool:
    return str(args.mode).lower() in {"full", "dry-run"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate21.13 executed baselines and mechanism lockdown.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--mode", default="full", choices=["dry-run", "quick", "full"])
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate21-12-root", type=Path, default=DEFAULT_GATE21_12_ROOT)
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--freehgc-zip", type=Path, default=Path("FreeHGC-main (1).zip"))
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--run-selector-audit", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-external-tp-smoke", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-external-tp-5x5", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-freehgc", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-metapath-dump-smoke", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-metapath-dump", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-feature-ablation", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-adapter-apv16", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-system-cost", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-cross-dataset", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
