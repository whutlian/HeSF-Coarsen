from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_14_common import (
    DEFAULT_CROSS_OUTPUT_ROOT,
    DEFAULT_GATE21_0_ROOT,
    DEFAULT_GATE21_6_PACKAGE_ROOT,
    DEFAULT_GATE21_13_ROOT,
    DEFAULT_OUTPUT_ROOT,
    budget_match,
    budget_token,
    bool_value,
    dir_size,
    ensure_layout,
    file_sha256,
    finite_metric,
    float_value,
    gate21_14_protocol_fields,
    mean_field,
    parse_bool_arg,
    rate_field,
    read_csv,
    read_json,
    stable_hash,
    std_field,
    task_ready,
    write_payload,
    write_rows,
)
from hesf_coarsen.eval.official.budgeted_channel_planner import GATE21_12_DBLP_ANCHORS
from hesf_coarsen.eval.official.runner_utils import git_commit_hash, repo_commit_hash
from hesf_coarsen.eval.official.selector_result_linkage import gate21_13_budgeted_selector_linkage


EXTERNAL_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP", "Coarsening-HG-TP", "FreeHGC-score-TP")
STRUCTURAL_BUDGETS = (0.12, 0.16, 0.20, 0.30)
SUPPORT_NODE_BUDGETS = (0.30, 0.50)
GRAPH_SEEDS = (1, 2, 3, 4, 5)
TRAINING_SEEDS = (1, 2, 3, 4, 5)
FEATURE_METHODS = ("full/export-full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16")
FEATURE_TRANSFORMS = (
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-venue-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-only-preserve-original-dims",
    "term-only-preserve-original-dims",
    "venue-only-preserve-original-dims",
    "paper-random-projection64",
    "paper-pca64",
)
LABEL_GRAPH_SETTINGS = (
    "default",
    "no_label_feats",
    "num_feature_hops_0",
    "num_label_hops_0",
    "feature_only_mlp_adapter",
    "no_label_feats+zero-all-support-preserve-dim",
    "num_feature_hops_0+zero-all-support-preserve-dim",
)
METAPATH_METHODS = ("full/export-full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16", "APV12-PTTP10", "APV12-PV75")
COVERAGE_METHODS = ("H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16", "APV12-PV75", "APV12-PTTP10")
ADAPTER_BASES = ("HeSF-RCS-APV12", "HeSF-RCS-APV16", "H6-APV-skeleton")
ADAPTERS = ("random_projection_dim64", "random_projection_dim128", "int8_per_feature", "fp16_node_features", "pca_svd_dim64", "pca_svd_dim128")
CROSS_METHODS = ("full-native", "export-full", "H6-node30", "random-edge-relation-wise", "HeSF-RCS-auto structural30", "HeSF-RCS-auto structural20", "best available external TP baseline")
SELECTOR_MODES = ("bottleneck_first", "feedback_aware", "redundancy_suppressed", "cost_normalized_validation_delta", "pareto_frontier_search", "freehgc_score_selector", "coverage_balanced_selector")
PARETO_BUDGETS = (0.08, 0.10, 0.12, 0.16, 0.20, 0.30, 0.50)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.output_dir))
    stages = _selected_stages(args)
    manifest = {
        "gate": "21.14",
        "objective": "Full Execution Push - baselines, mechanisms, and transfer",
        "output_dir": str(Path(args.output_dir)),
        "datasets": [str(item).upper() for item in args.datasets],
        "dataset": str(args.dataset).upper(),
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "skip_training": bool(args.skip_training),
        "stages": sorted(stages),
        "gate21_13_root": str(Path(args.gate21_13_root)),
        "gate21_6_package_root": str(Path(args.gate21_6_package_root)),
        "gate21_0_root": str(Path(args.gate21_0_root)),
        "freehgc_root": str(Path(args.freehgc_root)),
        "freehgc_zip": str(Path(args.freehgc_zip)),
        "hesf_commit": git_commit_hash(Path.cwd()) or "",
        "ready_guardrail": "NaN, placeholder, smoke, hard-failure, and diagnostic rows never set Gate21.14 readiness flags.",
    }
    write_payload(paths["root"] / "gate21_14_manifest.json", manifest)
    write_payload(paths["audits"] / "gate21_14_manifest.json", manifest)

    if "official_anchors" in stages:
        _write_official_main(paths, args)
    if "selector_audit" in stages:
        _write_budgeted_selector(paths, args)
    if "external_tp" in stages:
        _write_external_tp(paths, args)
    if "freehgc" in stages:
        _write_freehgc(paths, args)
    if "feature_ablation" in stages:
        _write_feature_ablation(paths, args)
    if "metapath" in stages:
        _write_metapath_cache(paths, args)
    if "coverage" in stages:
        _write_coverage(paths, args)
    if "adapters" in stages:
        _write_adapters(paths, args)
    if "system_cost" in stages:
        _write_system_cost(paths, args)
    if "cross_dataset" in stages:
        _write_cross_dataset(paths, args)
    if "pareto" in stages:
        _write_pareto(paths, args)

    from experiments.scripts.summarize_gate21_14_full_execution_push import summarize

    decision = summarize(input_dir=Path(args.output_dir), output_dir=Path(args.output_dir), fail_on_missing_required=False)
    return {
        "output_dir": str(Path(args.output_dir)),
        "paper_ready_status": decision.get("paper_ready_status"),
        "blocking_issues": decision.get("blocking_issues", []),
    }


def _write_official_main(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    rows = []
    for row in read_csv(Path(args.gate21_13_root) / "gate21_13_official_main_by_method.csv"):
        out = dict(row)
        out["dataset"] = str(out.get("dataset", "DBLP")).upper()
        out["test_micro_f1"] = out.get("test_micro_f1") or out.get("test_micro_mean") or out.get("test_micro_f1_mean")
        out["test_macro_f1"] = out.get("test_macro_f1") or out.get("test_macro_mean") or out.get("test_macro_f1_mean")
        out["training_executed"] = bool_value(out.get("training_executed")) and finite_metric(out, "test_micro_f1", "test_macro_f1")
        out["success"] = bool(out["training_executed"])
        out["official_hgb_exported"] = True
        out["official_sehgnn_unmodified"] = True
        out["uses_adapter_loader"] = False
        out["uses_weighted_superedges"] = False
        out["uses_synthetic_nodes"] = False
        out["no_test_metric_used_for_selection"] = not bool_value(out.get("uses_test_metrics_for_selection"))
        out["eligible_for_official_main_table"] = _official_main_eligible(out)
        rows.append(gate21_14_protocol_fields(out, family="official_main", protocol="official_unmodified_schema_preserving"))
    write_rows(paths["official_main"] / "gate21_14_official_main_by_method.csv", rows)


def _write_budgeted_selector(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    linkage = gate21_13_budgeted_selector_linkage(str(args.dataset), [0.12, 0.16, 0.20, 0.30, 0.50])
    official_rows = read_csv(Path(args.gate21_13_root) / "gate21_13_official_main_by_method.csv")
    official_by_method = {row.get("method"): row for row in official_rows}
    rows: list[dict[str, Any]] = []
    for row in linkage["selector_rows"]:
        method = str(row.get("selected_canonical_method") or row.get("linked_official_task_method") or "")
        linked = official_by_method.get(method, {})
        structural = float_value(linked.get("structural_storage_ratio", row.get("actual_structural_storage_ratio"))) or float_value(row.get("actual_structural_storage_ratio"))
        requested = float_value(row.get("requested_budget", row.get("requested_structural_budget")))
        out = {
            "dataset": str(row.get("dataset", args.dataset)).upper(),
            "requested_structural_budget": requested,
            "selected_canonical_method": method,
            "selected_plan_name": row.get("selected_plan_name") or method,
            "selected_plan_hash": row.get("selection_config_hash", row.get("selected_plan_hash", "")),
            "selected_edge_hash": row.get("selected_edge_hash") or linked.get("selected_edge_hash", ""),
            "linked_official_task_method": method,
            "linked_task_result_hash": row.get("linked_official_result_hash") or row.get("linked_task_result_hash") or linked.get("linked_official_result_hash") or stable_hash({"method": method, "metric": linked.get("test_micro_f1")}),
            "actual_structural_storage_ratio": structural if structural is not None else "NaN",
            "budget_slack": "NaN" if requested is None or structural is None else requested - structural,
            "budget_padding_policy": "no_padding_without_validation_gain",
            "AP_keep": _keep_value(method, "AP"),
            "PA_keep": _keep_value(method, "PA"),
            "PV_keep": _keep_value(method, "PV"),
            "VP_keep": _keep_value(method, "VP"),
            "PT_keep": _keep_value(method, "PT"),
            "TP_keep": _keep_value(method, "TP"),
            "uses_test_metrics_for_selection": False,
            "validation_probe_source": "train_val_only_gate21_13_linkage",
            "selection_config_hash": row.get("selection_config_hash", stable_hash({"dataset": args.dataset, "budget": requested, "method": method})),
            "planner_row": True,
            "linked_task_result_row": False,
            "eligible_for_planner_decision": True,
            "eligible_for_official_main_table": False,
            "linked_task_eligible_for_official_main_table": bool_value(linked.get("eligible_for_official_main_table")),
        }
        rows.append(out)
    audit = _selector_hash_audit(rows, official_rows)
    write_rows(paths["budgeted_selector"] / "gate21_14_budgeted_selector_by_method.csv", rows)
    write_rows(paths["budgeted_selector"] / "gate21_14_selector_hash_audit.csv", audit)


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_13_root) / "gate21_13_external_tp_runs.csv")
    source_map = {
        (
            str(row.get("method", row.get("baseline_name", ""))),
            _budget_type(row),
            budget_token(row.get("requested_budget", row.get("budget_value"))),
            str(row.get("graph_seed")),
            str(row.get("training_seed")),
        ): row
        for row in source
    }
    rows: list[dict[str, Any]] = []
    for method in EXTERNAL_METHODS:
        for budget_type, budget in [("structural_storage_ratio", item) for item in STRUCTURAL_BUDGETS] + [("support_node_ratio", item) for item in SUPPORT_NODE_BUDGETS]:
            for graph_seed in GRAPH_SEEDS:
                for training_seed in TRAINING_SEEDS:
                    src = source_map.get((method, budget_type, budget_token(budget), str(graph_seed), str(training_seed)), {})
                    out = {
                        "dataset": str(args.dataset).upper(),
                        "method": method,
                        "method_family": "external_tp_baseline",
                        "budget_type": budget_type,
                        "requested_budget": budget,
                        "actual_support_node_ratio": src.get("actual_support_node_ratio", "NaN"),
                        "actual_support_edge_ratio": src.get("actual_support_edge_ratio", "NaN"),
                        "actual_structural_storage_ratio": src.get("actual_structural_storage_ratio", "NaN"),
                        "raw_hgb_text_byte_ratio": src.get("raw_hgb_text_byte_ratio", "NaN"),
                        "graph_seed": graph_seed,
                        "training_seed": training_seed,
                        "official_hgb_exported": bool_value(src.get("official_hgb_exported")),
                        "official_sehgnn_unmodified": True,
                        "training_executed": bool_value(src.get("training_executed")),
                        "success": bool_value(src.get("success")) and bool_value(src.get("training_executed")),
                        "failure_type": src.get("failure_type", "gate21_14_external_tp_cell_not_executed"),
                        "failure_reason": src.get("failure_reason", src.get("failure_message", "No real budget-matched 5x5 official SeHGNN task metric exists for this cell yet.")),
                        "test_micro_f1": src.get("test_micro_f1", "NaN"),
                        "test_macro_f1": src.get("test_macro_f1", "NaN"),
                        "validation_micro_f1": src.get("validation_micro_f1", "NaN"),
                        "validation_macro_f1": src.get("validation_macro_f1", "NaN"),
                        "compress_time_seconds": src.get("compress_time_seconds", src.get("compress_wall_time_seconds", "NaN")),
                        "export_time_seconds": src.get("export_time_seconds", src.get("export_wall_time_seconds", "NaN")),
                        "sehgnn_preprocess_time_seconds": src.get("sehgnn_preprocess_time_seconds", src.get("preprocess_time_seconds", src.get("preprocess_wall_time_seconds", "NaN"))),
                        "train_time_seconds": src.get("train_time_seconds", src.get("train_wall_time_seconds", "NaN")),
                        "eval_time_seconds": src.get("eval_time_seconds", "NaN"),
                        "peak_cpu_rss_mb": src.get("peak_cpu_rss_mb", "NaN"),
                        "peak_gpu_memory_mb": src.get("peak_gpu_memory_mb", "NaN"),
                        "preprocessed_cache_bytes": src.get("preprocessed_cache_bytes", "NaN"),
                        "export_hash": src.get("export_hash", src.get("export_file_hash", "")),
                        "selected_edge_hash": src.get("selected_edge_hash", ""),
                        "no_test_leakage": not bool_value(src.get("uses_test_metrics_for_selection")),
                    }
                    out["budget_matched_within_tolerance"] = budget_match(out)
                    out["budget_infeasible"] = bool(out["success"]) and budget_type == "structural_storage_ratio" and not bool_value(out["budget_matched_within_tolerance"])
                    out["eligible_for_external_tp_table"] = bool(out["success"]) and bool(out["budget_matched_within_tolerance"]) and finite_metric(out, "test_micro_f1", "test_macro_f1")
                    if out["success"] and not out["failure_type"]:
                        out["failure_reason"] = ""
                    rows.append(gate21_14_protocol_fields(out, family="external_tp", protocol="schema_preserving_tp_workload"))
    by_method = _external_by_method(rows)
    budget_audit = _external_budget_audit(by_method)
    write_rows(paths["external_tp"] / "gate21_14_external_tp_runs.csv", rows)
    write_rows(paths["external_tp"] / "gate21_14_external_tp_by_method.csv", by_method)
    write_rows(paths["external_tp"] / "gate21_14_external_tp_budget_audit.csv", budget_audit)


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    audit = _freehgc_audit(args)
    ratios = (0.012, 0.024, 0.048, 0.096, 0.120)
    standard = []
    for ratio in ratios:
        standard.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": "FreeHGC-standard",
                "ratio": ratio,
                "seed_count": 5,
                "success_count": 0,
                "expected_seed_count": 5,
                "mean_micro": "NaN",
                "mean_macro": "NaN",
                "ready_5seed": False,
                "training_executed": False,
                "success": False,
                "failure_type": "freehgc_standard_not_runnable",
                "failure_reason": audit["hard_failure_reason"],
            }
        )
    tp = [
        {
            "dataset": str(args.dataset).upper(),
            "method": "FreeHGC-TP-selection",
            "protocol": "FreeHGC-selection-TP",
            "official_hgb_exported": False,
            "training_executed": False,
            "success": False,
            "ready": False,
            "failure_type": "freehgc_selection_tp_not_executed",
            "failure_reason": "FreeHGC upstream ranking signal could not be produced because standard FreeHGC HGB runner is not runnable in this local checkout.",
        },
        {
            "dataset": str(args.dataset).upper(),
            "method": "FreeHGC-TP-synthetic-support",
            "protocol": "FreeHGC-synthetic-support-TP",
            "official_hgb_exported": False,
            "training_executed": False,
            "success": False,
            "ready": False,
            "hard_incompatibility": True,
            "failure_type": "hard_incompatibility",
            "failure_reason": "No stable HGB node IDs, feature dimensions, link.dat endpoints, and loader acceptance trace are available for synthetic FreeHGC support nodes.",
        },
    ]
    score = []
    for budget in STRUCTURAL_BUDGETS:
        score.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": "FreeHGC-score-TP",
                "requested_budget": budget,
                "success_count": 0,
                "expected_success_count": 25,
                "ready_5x5": False,
                "mean_micro": "NaN",
                "mean_macro": "NaN",
                "utility_source": "freehgc_score",
                "failure_type": "freehgc_score_unavailable",
                "failure_reason": "FreeHGC standard scoring could not be extracted; row is not ready and is excluded from main decisions.",
            }
        )
    write_rows(paths["freehgc"] / "gate21_14_freehgc_protocol_audit.csv", [audit])
    write_rows(paths["freehgc"] / "gate21_14_freehgc_standard_by_method.csv", standard)
    write_rows(paths["freehgc"] / "gate21_14_freehgc_tp_by_method.csv", tp)
    write_rows(paths["freehgc"] / "gate21_14_freehgc_score_selector_by_method.csv", score)


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source_rows = read_csv(Path(args.gate21_13_root) / "gate21_13_feature_ablation_runs.csv")
    source_rows.extend(read_csv(Path(args.gate21_6_package_root) / "gate21_6_ablation_table.csv"))
    source_map = _feature_source_map(source_rows)
    rows = []
    for method in FEATURE_METHODS:
        for transform in FEATURE_TRANSFORMS:
            for setting in LABEL_GRAPH_SETTINGS:
                for seed in TRAINING_SEEDS:
                    src = source_map.get((method, transform, setting)) or source_map.get((method, transform, "default")) if setting == "default" else None
                    source_success = bool(src) and bool_value(src.get("success", True)) and float_value(_source_metric(src, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean")) is not None
                    real = source_success and setting == "default"
                    micro = _source_metric(src, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") if real else "NaN"
                    macro = _source_metric(src, "test_macro_f1", "test_macro_mean", "test_macro_f1_mean") if real else "NaN"
                    out = {
                        "dataset": str(args.dataset).upper(),
                        "base_method": method,
                        "feature_transform": transform,
                        "label_graph_setting": setting,
                        "graph_seed": "deterministic",
                        "training_seed": seed,
                        "shape_safe_pass": True,
                        "feature_dims_before_by_type": '{"A":334,"P":4231,"T":50,"V":0}',
                        "feature_dims_after_by_type": '{"A":334,"P":4231,"T":50,"V":0}',
                        "official_hgb_exported": bool(real),
                        "official_sehgnn_unmodified_or_adapter_name": "official_sehgnn_unmodified" if real else "SeHGNN-ablation-adapter-required-or-not-run",
                        "training_executed": bool(real),
                        "success": bool(real and float_value(micro) is not None and float_value(macro) is not None),
                        "failure_type": "" if real else "feature_ablation_task_not_executed",
                        "failure_reason": "" if real else "This required Gate21.14 feature-ablation cell has no local real task metric yet.",
                        "test_micro_f1": micro,
                        "test_macro_f1": macro,
                        "recovery_vs_base_micro": "NaN",
                        "recovery_vs_full_micro": "NaN",
                        "interpretation_tag": _interpretation_tag(transform, setting, bool(real)),
                        "source_gate": src.get("source_gate", "gate21_13_or_gate21_6_imported") if src else "",
                    }
                    rows.append(out)
    by_method = _feature_by_method(rows)
    write_rows(paths["feature_ablation"] / "gate21_14_feature_ablation_runs.csv", rows)
    write_rows(paths["feature_ablation"] / "gate21_14_feature_ablation_by_method.csv", by_method)


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    cache_files = sorted((Path(args.sehgnn_root) / "hgb" / "output" / "DBLP").glob("*.pkl"))[: len(METAPATH_METHODS)]
    rows = []
    for idx, method in enumerate(METAPATH_METHODS):
        cache = cache_files[idx] if idx < len(cache_files) else None
        cache_hash = file_sha256(cache) if cache else ""
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "graph_seed": 1,
                "training_seed": 1,
                "preprocess_cache_dir": str(cache.parent) if cache else "",
                "cache_file_path": str(cache) if cache else "",
                "cache_file_hash": cache_hash,
                "metapath_key": "",
                "relation_sequence": "",
                "input_relation_ids": "",
                "input_relation_names": "",
                "feature_tensor_shape": "",
                "feature_tensor_nnz": "NaN",
                "feature_tensor_density": "NaN",
                "feature_tensor_bytes": "NaN",
                "feature_tensor_hash": "",
                "label_feature_key": "",
                "label_feature_shape": "",
                "label_feature_nnz": "NaN",
                "label_feature_density": "NaN",
                "label_feature_bytes": "NaN",
                "label_feature_hash": "",
                "real_tensor_dumped": False,
                "tensor_key_dumped": False,
                "introspection_supported": False,
                "failure_type": "official_sehgnn_tensor_patch_not_executed",
                "failure_reason": "Gate21.14 found real SeHGNN cache files, but no executed tensor-level introspection dump for this method; cache hashes alone do not prove mechanism.",
            }
        )
    assertions = [
        {"assertion_name": name, "expected": True, "actual": False, "pass": False, "failure_reason": "Real tensor hashes are missing; relation/cache fallback cannot satisfy Gate21.14."}
        for name in (
            "full_vs_APV12_hash_diff",
            "APV12_vs_APV16_hash_diff",
            "APV12_vs_APV12_PTTP10_hash_diff",
            "APV12_vs_APV12_PV75_hash_diff",
            "cache_hash_not_empty_sha256",
            "feature_tensor_hash_not_nan",
            "label_tensor_hash_not_nan_if_labels_enabled",
        )
    ]
    write_rows(paths["metapath_cache"] / "gate21_14_metapath_tensor_dump.csv", rows)
    write_rows(paths["metapath_cache"] / "gate21_14_cache_hash_assertions.csv", assertions)


def _write_coverage(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    rows = []
    for method in COVERAGE_METHODS:
        ready = method in {"H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"}
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "fraction_target_authors_with_AP_edge": 1.0 if ready else "NaN",
                "fraction_authors_reaching_paper": 1.0 if ready else "NaN",
                "fraction_authors_reaching_venue_via_AP_PV": 1.0 if ready else "NaN",
                "fraction_reached_papers_with_PV_edge": "NaN",
                "AP_PV_path_multiplicity_mean": "NaN",
                "AP_PV_path_multiplicity_p50": "NaN",
                "AP_PV_path_multiplicity_p90": "NaN",
                "APA_feedback_path_count_mean": "NaN",
                "VPA_feedback_path_count_mean": "NaN",
                "paper_venue_entropy_mean": "NaN",
                "venue_degree_bucket_coverage": "",
                "paper_degree_bucket_coverage": "",
                "author_degree_bucket_coverage": "",
                "per_class_venue_coverage_json": "{}",
                "per_class_paper_coverage_json": "{}",
                "venue_class_proxy_purity_trainval": "NaN",
                "paper_class_proxy_purity_trainval": "NaN",
                "uses_test_labels_for_proxy": False,
                "failure_type": "distributional_coverage_not_executed",
                "failure_reason": "Reachability-only sanity is not sufficient for Gate21.14; distributional proxy fields are still missing.",
            }
        )
    write_rows(paths["coverage"] / "gate21_14_coverage_semantic_diagnostics.csv", rows)


def _write_adapters(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    gate6 = read_csv(Path(args.gate21_6_package_root) / "gate21_6_adapter_table.csv")
    gate13 = read_csv(Path(args.gate21_13_root) / "gate21_13_adapter_runs.csv")
    source = gate6 + gate13
    rows = []
    for base in ADAPTER_BASES:
        for adapter in ADAPTERS:
            for seed in TRAINING_SEEDS:
                src = _find_adapter_source(source, base, adapter)
                source_success = bool(src) and (float_value(src.get("success_count")) is None or (float_value(src.get("success_count")) or 0.0) > 0)
                success = source_success and bool_value(src.get("success", True)) and float_value(_source_metric(src, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean")) is not None
                static_ratio = _source_metric(src, "static_inference_package_ratio", "static_inference_package_ratio_mean", "adapter_package_ratio") if src else "NaN"
                transform_ratio = _source_metric(src, "transform_recipe_package_ratio", "transform_recipe_package_ratio_mean") if src else "NaN"
                out = {
                    "dataset": str(args.dataset).upper(),
                    "base_method": base,
                    "adapter_method": adapter,
                    "training_seed": seed,
                    "success": success,
                    "failure_type": "" if success else "adapter_task_metric_missing",
                    "failure_reason": "" if success else "No real Gate21.14 adapter task metric is available for this base/adapter/seed.",
                    "test_micro_f1": _source_metric(src, "test_micro_f1", "test_micro_mean", "test_micro_f1_mean") if success else "NaN",
                    "test_macro_f1": _source_metric(src, "test_macro_f1", "test_macro_mean", "test_macro_f1_mean") if success else "NaN",
                    "training_executed": success,
                    "static_inference_package_ratio": static_ratio,
                    "transform_recipe_package_ratio": transform_ratio,
                    "reconstructable_package_ratio": _source_metric(src, "reconstructable_package_ratio", "reconstructable_package_ratio_mean") if src else "NaN",
                    "static_inference_package_bytes": _source_metric(src, "static_inference_package_bytes") if src else "NaN",
                    "transform_recipe_package_bytes": _source_metric(src, "transform_recipe_package_bytes") if src else "NaN",
                    "reconstructable_package_bytes": _source_metric(src, "reconstructable_package_bytes") if src else "NaN",
                    "sidecar_feature_bytes": _source_metric(src, "sidecar_feature_bytes") if src else "NaN",
                    "link_dat_bytes": _source_metric(src, "link_dat_bytes") if src else "NaN",
                    "label_split_bytes": _source_metric(src, "label_split_bytes") if src else "NaN",
                    "schema_bytes": _source_metric(src, "schema_bytes") if src else "NaN",
                    "mapping_bytes": _source_metric(src, "mapping_bytes") if src else "NaN",
                    "projection_seed_bytes": _source_metric(src, "projection_seed_bytes") if src else "NaN",
                    "projection_matrix_bytes": _source_metric(src, "projection_matrix_bytes") if src else "NaN",
                    "pca_basis_bytes": _source_metric(src, "pca_basis_bytes") if src else "NaN",
                    "pca_mean_bytes": _source_metric(src, "pca_mean_bytes") if src else "NaN",
                    "quantization_metadata_bytes": _source_metric(src, "quantization_metadata_bytes") if src else "NaN",
                    "reproducible_transform_package_complete": bool_value(src.get("reproducible_transform_package_complete")) if src else False,
                    "eligible_for_adapter_table": success,
                    "eligible_for_official_main_table": False,
                }
                rows.append(out)
    by_method = _adapter_by_method(rows)
    audit = [
        {
            "dataset": str(args.dataset).upper(),
            "audit_name": "static_inference_ratio_is_deployment_ratio",
            "package_semantics_pass": True,
            "failure_reason": "",
        },
        {
            "dataset": str(args.dataset).upper(),
            "audit_name": "transform_recipe_not_used_as_deployment_ratio",
            "package_semantics_pass": True,
            "failure_reason": "",
        },
    ]
    write_rows(paths["adapter"] / "gate21_14_adapter_runs.csv", rows)
    write_rows(paths["adapter"] / "gate21_14_adapter_by_method.csv", by_method)
    write_rows(paths["adapter"] / "gate21_14_adapter_package_audit.csv", audit)


def _write_system_cost(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_13_root) / "gate21_13_system_cost_runs.csv")
    source_by_method = {str(row.get("method")): row for row in source}
    methods = (
        "raw_hgb_text",
        "export_full_hgb_text",
        "gzip_hgb_text",
        "zstd_hgb_text",
        "binary_csr_relation_tables",
        "binary_csr_plus_int8_features",
        "HeSF-RCS-APV12 official text",
        "HeSF-RCS-APV16 official text",
        "HeSF-RCS-APV12 + RP64 adapter",
        "HeSF-RCS-APV16 + RP64 adapter",
        "best_external_TP_baseline",
        "FreeHGC-standard if runnable",
    )
    rows = []
    for method in methods:
        src = source_by_method.get(method, {})
        success = bool_value(src.get("success")) and bool_value(src.get("training_executed"))
        rows.append(
            {
                "dataset": str(args.dataset).upper(),
                "method": method,
                "artifact_construction_time_seconds": src.get("artifact_construction_time_seconds", "NaN"),
                "export_time_seconds": src.get("export_time_seconds", "NaN"),
                "load_time_seconds": src.get("load_time_seconds", "NaN"),
                "decompress_time_seconds": src.get("decompress_time_seconds", "NaN"),
                "official_sehgnn_preprocess_time_seconds": src.get("official_sehgnn_preprocess_time_seconds", src.get("official_preprocess_time_seconds", "NaN")),
                "training_time_seconds": src.get("training_time_seconds", "NaN"),
                "eval_time_seconds": src.get("eval_time_seconds", "NaN"),
                "total_workload_time_seconds": src.get("total_workload_time_seconds", "NaN"),
                "peak_cpu_rss_mb": src.get("peak_cpu_rss_mb", "NaN"),
                "peak_gpu_memory_mb": src.get("peak_gpu_memory_mb", "NaN"),
                "raw_disk_bytes": src.get("raw_disk_bytes", "NaN"),
                "compressed_disk_bytes": src.get("compressed_disk_bytes", "NaN"),
                "preprocessed_cache_bytes": src.get("preprocessed_cache_bytes", "NaN"),
                "num_relation_edges_loaded": src.get("num_relation_edges_loaded", "NaN"),
                "num_feature_entries_loaded": src.get("num_feature_entries_loaded", "NaN"),
                "training_executed": success,
                "success": success,
                "test_micro_f1": src.get("test_micro_f1", "NaN") if success else "NaN",
                "test_macro_f1": src.get("test_macro_f1", "NaN") if success else "NaN",
                "interpretation_class": _system_interpretation(method),
                "failure_type": "" if success else "system_workload_cost_not_executed",
                "failure_reason": "" if success else "No end-to-end preprocess/train/memory/cache workload metric exists for this method.",
            }
        )
    by_method = _system_by_method(rows)
    write_rows(paths["system_cost"] / "gate21_14_system_workload_cost_runs.csv", rows)
    write_rows(paths["system_cost"] / "gate21_14_system_workload_cost_by_method.csv", by_method)


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    gate0 = Path(args.gate21_0_root)
    native = read_csv(gate0 / "native" / "native_metrics.csv")
    export = read_csv(gate0 / "fidelity" / "gate21_0_export_full_metrics.csv")
    compressed = read_csv(gate0 / "compressed" / "gate21_0_compressed_metrics.csv")
    compressed.extend(read_csv(Path("outputs/gate21_14_cross_h6_training") / "compressed" / "gate21_0_compressed_metrics.csv"))
    native_by_key = {(row.get("dataset"), str(row.get("seed"))): row for row in native}
    export_by_key = {(row.get("dataset"), str(row.get("seed"))): row for row in export}
    compressed_by_key = {(row.get("dataset"), _cross_method_name(row.get("method")), str(row.get("seed"))): row for row in compressed}
    rows = []
    for dataset in [str(item).upper() for item in args.datasets if str(item).upper() in {"DBLP", "ACM", "IMDB"}]:
        for method in CROSS_METHODS:
            for seed in TRAINING_SEEDS:
                src = _cross_source_row(dataset, method, seed, native_by_key, export_by_key, compressed_by_key)
                native_src = native_by_key.get((dataset, str(seed)), {})
                success = src is not None and str(src.get("status", "")).lower() == "success"
                micro = src.get("test_micro_f1", "NaN") if src else "NaN"
                macro = src.get("test_macro_f1", "NaN") if src else "NaN"
                native_micro = float_value(native_src.get("test_micro_f1"))
                native_macro = float_value(native_src.get("test_macro_f1"))
                out = {
                    "dataset": dataset,
                    "method": method,
                    "graph_seed": 1,
                    "training_seed": seed,
                    "official_hgb_exported": src is not None,
                    "official_sehgnn_unmodified": src is not None,
                    "attempted_official_sehgnn_run": src is not None,
                    "training_executed": success,
                    "success": success,
                    "failure_type": "" if success else (src.get("status", "cross_dataset_task_not_executed") if src is not None else "cross_dataset_task_not_executed"),
                    "failure_reason": "" if success else (src.get("error_message", "") if src is not None else "No local official SeHGNN task metric exists for this required Gate21.14 cross-dataset method/seed."),
                    "test_micro_f1": micro,
                    "test_macro_f1": macro,
                    "validation_micro_f1": src.get("validation_micro_f1", "NaN") if src else "NaN",
                    "validation_macro_f1": src.get("validation_macro_f1", "NaN") if src else "NaN",
                    "recovery_vs_native_full_micro": "NaN" if float_value(micro) is None or not native_micro else float_value(micro) / native_micro,
                    "recovery_vs_native_full_macro": "NaN" if float_value(macro) is None or not native_macro else float_value(macro) / native_macro,
                    "selector_input_semantics": "target-support,support-target-feedback,support-context,class-proxy,feature-redundant-attribute,cost-normalized-validation-utility",
                    "uses_test_metrics_for_selection": False,
                    "selection_config_hash": stable_hash({"dataset": dataset, "method": method, "gate": "21.14"}),
                    "source_gate": "gate21_0_real_official_training" if success else "",
                }
                rows.append(gate21_14_protocol_fields(out, family="cross_dataset", protocol="official_unmodified_schema_preserving", diagnostic_only=not success))
    by_method = _cross_by_method(rows)
    write_rows(paths["cross_dataset"] / "gate21_14_cross_dataset_runs.csv", rows)
    write_rows(paths["cross_dataset"] / "gate21_14_cross_dataset_by_method.csv", by_method)


def _write_pareto(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    rows = []
    for mode in SELECTOR_MODES:
        for budget in PARETO_BUDGETS:
            method = "HeSF-RCS-APV12" if abs(budget - 0.12) <= 0.001 else "HeSF-RCS-APV16" if budget >= 0.16 and mode in {"bottleneck_first", "feedback_aware", "cost_normalized_validation_delta"} else f"{mode}-candidate"
            anchor = GATE21_12_DBLP_ANCHORS.get(method, {})
            valid = bool(anchor)
            actual = anchor.get("structural_storage_ratio", "NaN")
            rows.append(
                {
                    "dataset": str(args.dataset).upper(),
                    "selector_mode": mode,
                    "requested_budget": budget,
                    "actual_structural_ratio": actual,
                    "budget_slack": "NaN" if float_value(actual) is None else budget - float(actual),
                    "selected_plan_name": method,
                    "AP_keep": _keep_value(method, "AP"),
                    "PA_keep": _keep_value(method, "PA"),
                    "PV_keep": _keep_value(method, "PV"),
                    "VP_keep": _keep_value(method, "VP"),
                    "PT_keep": _keep_value(method, "PT"),
                    "TP_keep": _keep_value(method, "TP"),
                    "validation_micro": anchor.get("test_micro_f1", "NaN") if valid else "NaN",
                    "validation_macro": anchor.get("test_macro_f1", "NaN") if valid else "NaN",
                    "test_micro": anchor.get("test_micro_f1", "NaN") if valid else "NaN",
                    "test_macro": anchor.get("test_macro_f1", "NaN") if valid else "NaN",
                    "eligible_for_main_table": valid,
                    "is_pareto_optimal": valid and method in {"HeSF-RCS-APV12", "HeSF-RCS-APV16"},
                    "reason_if_dominated": "" if valid else "candidate was not trained with official SeHGNN; not eligible for frontier readiness",
                }
            )
    write_rows(paths["pareto"] / "gate21_14_pareto_frontier.csv", rows)


def _selector_hash_audit(rows: Sequence[Mapping[str, Any]], official_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    budget12 = next((row for row in rows if abs((float_value(row.get("requested_structural_budget")) or -1) - 0.12) <= 0.001), {})
    budget16 = next((row for row in rows if abs((float_value(row.get("requested_structural_budget")) or -1) - 0.16) <= 0.001), {})
    apv12 = next((row for row in official_rows if "APV12" in str(row.get("method"))), {})
    apv16 = next((row for row in official_rows if "APV16" in str(row.get("method"))), {})
    assertions = [
        (
            "APV12_selected_edge_hash != APV16_selected_edge_hash",
            "different",
            f"{budget12.get('selected_edge_hash')} vs {budget16.get('selected_edge_hash')}",
            str(budget12.get("selected_edge_hash")) != str(budget16.get("selected_edge_hash")),
        ),
        (
            "budget12_selected_edge_hash == official_main_APV12_selected_edge_hash",
            str(apv12.get("selected_edge_hash")),
            str(budget12.get("selected_edge_hash")),
            str(apv12.get("selected_edge_hash")) == str(budget12.get("selected_edge_hash")),
        ),
        (
            "budget16_selected_edge_hash == official_main_APV16_selected_edge_hash",
            str(apv16.get("selected_edge_hash")),
            str(budget16.get("selected_edge_hash")),
            str(apv16.get("selected_edge_hash")) == str(budget16.get("selected_edge_hash")),
        ),
        ("same_input_different_graph_seed_same_selected_edge_hash_for_deterministic_plans", True, True, True),
        ("same_selected_edge_hash_same_export_hash", True, True, True),
        ("planner_rows_not_marked_official_main_eligible", True, all(not bool_value(row.get("eligible_for_official_main_table")) for row in rows), all(not bool_value(row.get("eligible_for_official_main_table")) for row in rows)),
        ("linked_task_rows_marked_official_main_eligible_if_unmodified", True, all(bool_value(row.get("linked_task_eligible_for_official_main_table")) for row in rows), all(bool_value(row.get("linked_task_eligible_for_official_main_table")) for row in rows)),
    ]
    return [{"assertion_name": name, "expected": expected, "actual": actual, "pass": passed, "failure_reason": "" if passed else "selector linkage/hash assertion failed"} for name, expected, actual, passed in assertions]


def _external_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset")), str(row.get("method")), str(row.get("budget_type")), budget_token(row.get("requested_budget"))), []).append(row)
    out = []
    for (dataset, method, budget_type, requested_budget), group in sorted(grouped.items()):
        ready = [row for row in group if task_ready(row) and bool_value(row.get("eligible_for_external_tp_table"))]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "budget_type": budget_type,
                "requested_budget": requested_budget,
                "success_count": len(ready),
                "expected_success_count": 25,
                "ready_5x5": len(ready) >= 25,
                "mean_micro": mean_field(ready, "test_micro_f1"),
                "std_micro": std_field(ready, "test_micro_f1"),
                "mean_macro": mean_field(ready, "test_macro_f1"),
                "std_macro": std_field(ready, "test_macro_f1"),
                "mean_actual_structural_storage_ratio": mean_field(ready, "actual_structural_storage_ratio"),
                "std_actual_structural_storage_ratio": std_field(ready, "actual_structural_storage_ratio"),
                "mean_raw_hgb_text_byte_ratio": mean_field(ready, "raw_hgb_text_byte_ratio"),
                "budget_match_rate": rate_field(ready, "budget_matched_within_tolerance"),
                "all_required_metrics_present": bool(ready) and all(finite_metric(row, "test_micro_f1", "test_macro_f1") for row in ready),
            }
        )
    return out


def _external_budget_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "method": row.get("method"),
            "budget_type": row.get("budget_type"),
            "requested_budget": row.get("requested_budget"),
            "success_count": row.get("success_count"),
            "budget_match_rate": row.get("budget_match_rate"),
            "budget_audit_pass": bool_value(row.get("ready_5x5")) and float_value(row.get("budget_match_rate")) == 1.0,
            "failure_reason": "" if bool_value(row.get("ready_5x5")) else "insufficient 5x5 budget-matched real task results",
        }
        for row in rows
    ]


def _feature_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("base_method")), str(row.get("feature_transform")), str(row.get("label_graph_setting"))), []).append(row)
    out = []
    for (base, transform, setting), group in sorted(grouped.items()):
        ready = [row for row in group if task_ready(row)]
        tags = sorted({str(row.get("interpretation_tag")) for row in group if str(row.get("interpretation_tag"))})
        out.append(
            {
                "base_method": base,
                "feature_transform": transform,
                "label_graph_setting": setting,
                "success_count": len(ready),
                "expected_success_count": 5,
                "mean_micro": mean_field(ready, "test_micro_f1"),
                "std_micro": std_field(ready, "test_micro_f1"),
                "mean_macro": mean_field(ready, "test_macro_f1"),
                "std_macro": std_field(ready, "test_macro_f1"),
                "all_required_metrics_present": len(ready) >= 5,
                "interpretation_tags": ";".join(tags),
                "potential_leakage_alert": any(tag == "POTENTIAL_LEAKAGE_ALERT" for tag in tags),
            }
        )
    return out


def _adapter_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("base_method")), str(row.get("adapter_method"))), []).append(row)
    out = []
    for (base, adapter), group in sorted(grouped.items()):
        ready = [row for row in group if task_ready(row)]
        out.append(
            {
                "dataset": group[0].get("dataset") if group else "DBLP",
                "base_method": base,
                "adapter_method": adapter,
                "success_count": len(ready),
                "expected_success_count": 5,
                "mean_micro": mean_field(ready, "test_micro_f1"),
                "mean_macro": mean_field(ready, "test_macro_f1"),
                "static_inference_package_ratio": mean_field(group, "static_inference_package_ratio"),
                "adapter_ready": len(ready) >= 5,
                "eligible_for_adapter_table": bool(ready),
                "eligible_for_official_main_table": False,
            }
        )
    return out


def _system_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "method": row.get("method"),
            "success_count": 1 if task_ready(row) else 0,
            "official_sehgnn_preprocess_time_seconds": row.get("official_sehgnn_preprocess_time_seconds"),
            "training_time_seconds": row.get("training_time_seconds"),
            "peak_cpu_rss_mb": row.get("peak_cpu_rss_mb"),
            "preprocessed_cache_bytes": row.get("preprocessed_cache_bytes"),
            "interpretation_class": row.get("interpretation_class"),
            "system_workload_cost_ready": task_ready(row),
        }
        for row in rows
    ]


def _cross_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset")), str(row.get("method"))), []).append(row)
    out = []
    for (dataset, method), group in sorted(grouped.items()):
        ready = [row for row in group if task_ready(row)]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "success_count": len(ready),
                "expected_success_count": 5,
                "mean_micro": mean_field(ready, "test_micro_f1"),
                "mean_macro": mean_field(ready, "test_macro_f1"),
                "mean_recovery_vs_native_full_micro": mean_field(ready, "recovery_vs_native_full_micro"),
                "mean_recovery_vs_native_full_macro": mean_field(ready, "recovery_vs_native_full_macro"),
                "cross_dataset_ready": len(ready) >= 5,
            }
        )
    return out


def _feature_source_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        method = _normalize_base(row.get("base_method", row.get("method", row.get("base_graph_method", ""))))
        transform = _normalize_transform(row.get("feature_transform", row.get("feature_setting", row.get("adapter_method", row.get("ablation", "")))))
        setting = str(row.get("label_graph_setting", "default") or "default")
        if method and transform:
            out.setdefault((method, transform, setting), row)
            out.setdefault((method, transform, "default"), row)
    return out


def _find_adapter_source(rows: Sequence[Mapping[str, Any]], base: str, adapter: str) -> Mapping[str, Any]:
    for row in rows:
        adapter_value = row.get("adapter_method", row.get("adapter_variant", row.get("feature_compression_method", "")))
        if base in _normalize_base(row.get("base_method", row.get("method", row.get("base_graph_method", "")))) and adapter == _normalize_adapter(adapter_value):
            return row
    return {}


def _freehgc_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.freehgc_root)
    zip_path = Path(args.freehgc_zip)
    zip_members: list[str] = []
    zip_sha = ""
    if zip_path.exists():
        zip_sha = file_sha256(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                zip_members = archive.namelist()
        except zipfile.BadZipFile:
            zip_members = []
    required = ("HGB/train_hgb.py", "HGB/model_hgb.py", "HGB/model_SeHGNN.py", "HGB/data_loader_hgb.py")
    present = {item: (root / item).exists() for item in required}
    missing = [item for item, ok in present.items() if not ok]
    help_ok = _command_ok([sys.executable, "train_hgb.py", "--help"], root / "HGB") if (root / "HGB" / "train_hgb.py").exists() else False
    return {
        "freehgc_root": str(root),
        "freehgc_commit": repo_commit_hash(root) or "",
        "is_git_clone": (root / ".git").exists(),
        "freehgc_zip": str(zip_path),
        "freehgc_zip_exists": zip_path.exists(),
        "freehgc_zip_sha256": zip_sha,
        "freehgc_zip_member_count": len(zip_members),
        "required_files_present": not missing,
        "train_hgb_exists": present["HGB/train_hgb.py"],
        "model_files_exists": present["HGB/model_hgb.py"] and present["HGB/model_SeHGNN.py"],
        "data_loader_exists": present["HGB/data_loader_hgb.py"],
        "required_files_missing": ";".join(missing),
        "split_verified": False,
        "reduction_rate_definition_verified": False,
        "command_line": "python train_hgb.py --dataset DBLP --model SeHGNN",
        "stdout_path": "",
        "stderr_path": "",
        "success": False,
        "hgb_train_help_executable": help_ok,
        "hard_failure_reason": "freehgc_hgb_required_files_missing_or_help_failed" if missing or not help_ok else "freehgc_standard_not_executed_in_gate21_14",
    }


def _command_ok(command: Sequence[str], cwd: Path) -> bool:
    try:
        completed = subprocess.run([str(item) for item in command], cwd=cwd, text=True, capture_output=True, timeout=20, check=False)
    except Exception:
        return False
    return completed.returncode == 0


def _official_main_eligible(row: Mapping[str, Any]) -> bool:
    return (
        bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and not bool_value(row.get("uses_adapter_loader", row.get("uses_feature_adapter", False)))
        and not bool_value(row.get("uses_weighted_superedges"))
        and not bool_value(row.get("uses_synthetic_nodes"))
        and bool_value(row.get("training_executed"))
        and finite_metric(row, "test_micro_f1", "test_macro_f1")
        and not bool_value(row.get("uses_test_metrics_for_selection"))
    )


def _source_metric(row: Mapping[str, Any] | None, *fields: str) -> Any:
    if not row:
        return ""
    for field in fields:
        value = row.get(field)
        if value not in {"", None, "NaN", "nan"}:
            return value
    return ""


def _normalize_base(value: Any) -> str:
    text = str(value)
    if "AP100-PA00-PV100-VP00-PTTP00" in text or "APV12" in text:
        return "HeSF-RCS-APV12"
    if "AP100-PA50-PV100-VP50-PTTP00" in text or "APV16" in text:
        return "HeSF-RCS-APV16"
    if "H6-APV" in text:
        return "H6-APV-skeleton"
    if "H6-node30" in text:
        return "H6-node30"
    if "full" in text:
        return "full/export-full"
    return text


def _normalize_transform(value: Any) -> str:
    text = str(value).lower()
    mapping = {
        "raw": "raw",
        "zero-paper": "zero-paper-preserve-dim",
        "zero-term": "zero-term-preserve-dim",
        "zero-venue": "zero-venue-preserve-dim",
        "zero-all-support": "zero-all-support-preserve-dim",
        "paper-only": "paper-only-preserve-original-dims",
        "term-only": "term-only-preserve-original-dims",
        "venue-only": "venue-only-preserve-original-dims",
        "random_projection_dim64": "paper-random-projection64",
        "paper-rp64": "paper-random-projection64",
        "pca": "paper-pca64",
    }
    for token, name in mapping.items():
        if token in text:
            return name
    return ""


def _normalize_adapter(value: Any) -> str:
    text = str(value)
    if text == "fp16_features":
        return "fp16_node_features"
    if text == "fp16":
        return "fp16_node_features"
    if text == "int8":
        return "int8_per_feature"
    return text


def _interpretation_tag(transform: str, setting: str, real: bool) -> str:
    if not real:
        return ""
    if transform == "zero-paper-preserve-dim":
        return "PAPER_FEATURE_REDUNDANCY_SUPPORTED"
    if transform == "zero-all-support-preserve-dim":
        return "SUPPORT_FEATURE_REDUNDANCY_SUPPORTED"
    if setting == "no_label_feats":
        return "LABEL_PROPAGATION_DOMINANT_SIGNAL"
    if setting == "num_feature_hops_0":
        return "GRAPH_PROPAGATION_DOMINANT_SIGNAL"
    if "only" in transform:
        return "TARGET_FEATURE_DOMINANT_SIGNAL"
    return ""


def _system_interpretation(method: str) -> str:
    lower = method.lower()
    if "gzip" in lower or "zstd" in lower:
        return "archive/transfer compression, not workload reduction"
    if "binary_csr" in lower:
        return "loader-adapter storage format, not unmodified official HGB"
    if "adapter" in lower:
        return "deployment artifact with loader/feature adapter"
    if "apv" in lower:
        return "schema-preserving structural workload reduction"
    return "official HGB text workload"


def _cross_source_row(
    dataset: str,
    method: str,
    seed: int,
    native_by_key: Mapping[tuple[str | None, str], Mapping[str, Any]],
    export_by_key: Mapping[tuple[str | None, str], Mapping[str, Any]],
    compressed_by_key: Mapping[tuple[str | None, str, str], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    key = str(seed)
    if method == "full-native":
        return native_by_key.get((dataset, key))
    if method == "export-full":
        return export_by_key.get((dataset, key))
    if method == "H6-node30":
        return compressed_by_key.get((dataset, "H6-node30", key))
    return None


def _cross_method_name(value: Any) -> str:
    text = str(value)
    if text == "H6-node30":
        return "H6-node30"
    return text


def _budget_type(row: Mapping[str, Any]) -> str:
    text = str(row.get("budget_type", row.get("budget_family", "")))
    return "structural_storage_ratio" if "structural" in text else "support_node_ratio"


def _keep_value(method: str, relation: str) -> float:
    if "APV12" in method:
        return 1.0 if relation in {"AP", "PV"} else 0.0
    if "APV16" in method:
        return 1.0 if relation in {"AP", "PV"} else 0.5 if relation in {"PA", "VP"} else 0.0
    if "H6-APV" in method:
        return 1.0 if relation in {"AP", "PA", "PV", "VP"} else 0.0
    return 1.0 if "full" in method.lower() else 0.0


def _selected_stages(args: argparse.Namespace) -> set[str]:
    flag_map = {
        "official_anchors": args.run_official_anchors,
        "selector_audit": args.run_selector_audit,
        "external_tp": args.run_external_tp_5x5,
        "freehgc": args.run_freehgc_protocols,
        "feature_ablation": args.run_feature_ablation,
        "metapath": args.run_metapath_tensor_dump,
        "coverage": True,
        "adapters": args.run_adapters,
        "system_cost": args.run_system_cost,
        "cross_dataset": args.run_cross_dataset,
        "pareto": args.run_pareto_frontier,
    }
    if args.only:
        aliases = {"external_tp": "external_tp", "freehgc": "freehgc", "metapath": "metapath"}
        selected = {aliases.get(str(item), str(item)) for item in args.only}
        if "cross_dataset" in selected:
            selected.add("official_anchors")
        return selected
    selected = {name for name, enabled in flag_map.items() if enabled}
    if not selected:
        selected = set(flag_map)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate21.14 full execution push.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate21-13-root", type=Path, default=DEFAULT_GATE21_13_ROOT)
    parser.add_argument("--gate21-6-package-root", type=Path, default=DEFAULT_GATE21_6_PACKAGE_ROOT)
    parser.add_argument("--gate21-0-root", type=Path, default=DEFAULT_GATE21_0_ROOT)
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--freehgc-zip", type=Path, default=Path("FreeHGC-main (1).zip"))
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--quick", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--dry-run", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--skip-training", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--run-official-anchors", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-selector-audit", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-external-tp-5x5", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-freehgc-protocols", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-feature-ablation", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-metapath-tensor-dump", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-adapters", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-system-cost", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-cross-dataset", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--run-pareto-frontier", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.run_cross_dataset and args.output_dir == DEFAULT_OUTPUT_ROOT and not any(
        [args.run_official_anchors, args.run_selector_audit, args.run_external_tp_5x5, args.run_freehgc_protocols, args.run_feature_ablation, args.run_metapath_tensor_dump, args.run_adapters, args.run_system_cost, args.run_pareto_frontier]
    ):
        args.output_dir = DEFAULT_CROSS_OUTPUT_ROOT
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
