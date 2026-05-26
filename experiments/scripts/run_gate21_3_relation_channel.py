from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate21_1_sehgnn_schema_edge_budget import _compressed_labels_and_splits, _float, _native_labels, _storage_fields, _unweighted_graph
from experiments.scripts.summarize_gate21_3_relation_channel import summarize_gate21_3
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.coverage_sampler import CoverageSampler, sample_random_edge_indices
from hesf_coarsen.eval.official.edge_pruning_baselines import edge_budget_for_storage, semantic_storage_ratio
from hesf_coarsen.eval.official.feature_cache_compression_probe import FEATURE_CACHE_PROBE_FIELDS, planned_feature_cache_probe_rows
from hesf_coarsen.eval.official.path_aware_edge_scorer_v2 import EDGE_SCORE_V2_DIAGNOSTIC_FIELDS, PathAwareV2Scorer
from hesf_coarsen.eval.official.relation_budget_allocator import (
    RelationBudgetAllocation,
    RelationBudgetAllocator,
    RelationStats,
    allocate_relation_channel_spec,
    parse_relation_channel_spec,
)
from hesf_coarsen.eval.official.relation_schema import (
    build_relation_keys,
    official_relation_name_for_source,
    parse_link_relation_counts,
    relation_pair_name,
    validate_hgb_relation_order,
)
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.schema_stable_pruning import build_target_only_schema_stub_graph
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
from hesf_coarsen.eval.official.sehgnn_native_runner import NativeCommand, build_official_hgb_command, run_native_command
from hesf_coarsen.eval.official.storage_audit import STORAGE_AUDIT_FIELDS, audit_hgb_directory
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, validate_schema


TARGET_TYPE_BY_DATASET = {"DBLP": "A", "ACM": "P", "IMDB": "M"}
BASELINE_BY_SELECTOR = {"H6": "H6-no-spec-support-only", "flatten": "flatten-sum-support-only"}
DEFAULT_GATE21_1_ROOT = Path("outputs/gate21_1_sehgnn_schema_edge_budget")
DEFAULT_NATIVE_MICRO = 0.9533802
DEFAULT_NATIVE_MACRO = 0.9498198

RAW_FIELDS = [
    "dataset",
    "method",
    "method_family",
    "budget_strategy",
    "edge_score_strategy",
    "relation_channel_spec",
    "graph_seed",
    "training_seed",
    "export_hash",
    "run_id",
    "success",
    "status",
    "failed_reason",
    "traceback_path",
    "model_name",
    "schema_compatible",
    "uses_weighted_superedges",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "no_test_label_export_leakage",
    "no_test_label_scoring_leakage",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "total_node_ratio",
    "total_edge_ratio",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "test_accuracy_if_single_label",
    "native_full_test_micro_f1",
    "native_full_test_macro_f1",
    "export_full_test_micro_f1",
    "export_full_test_macro_f1",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "mapping_bijective",
    "split_disjoint",
    "relation_order_matches_official",
    "node_type_order_matches_official",
    "schema_complete",
    "stdout_path",
    "stderr_path",
]

RUN_MANIFEST_FIELDS = [
    "run_id",
    "dataset",
    "method",
    "preset",
    "method_family",
    "budget_strategy",
    "edge_score_strategy",
    "relation_channel_spec",
    "graph_seed",
    "training_seed",
    "export_dir",
    "run_dir",
    "expected_outputs",
    "eligible_for_main_decision",
    "planned_only",
]

RELATION_MAPPING_FIELDS_21_3 = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "source_relation_id",
    "source_relation_name",
    "source_src_type",
    "source_dst_type",
    "official_relation_id",
    "official_relation_name",
    "official_src_type",
    "official_dst_type",
    "relation_pair_name",
    "reciprocal_official_relation_id",
    "reciprocal_official_relation_name",
    "source_edge_count",
    "original_full_edge_count",
    "candidate_edge_count_after_node_pruning",
    "retained_edge_count",
    "retention_vs_candidate",
    "retention_vs_full",
    "requested_relation_budget",
    "actual_relation_budget",
    "reciprocal_count_consistent",
    "min_edges_constraint_active",
    "relation_dropped_flag",
]

RELATION_RETENTION_FIELDS_21_3 = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "budget_strategy",
    "edge_score_strategy",
    "official_relation_id",
    "official_relation_name",
    "relation_pair_name",
    "original_full_edge_count",
    "candidate_edge_count_after_node_pruning",
    "retained_edge_count",
    "retention_vs_candidate",
    "retention_vs_full",
    "requested_relation_budget",
    "actual_relation_budget",
    "source_relation_id",
    "source_relation_name",
    "source_src_type",
    "source_dst_type",
    "reciprocal_official_relation_id",
    "reciprocal_official_relation_name",
    "reciprocal_count_consistent",
    "min_edges_constraint_active",
    "relation_dropped_flag",
]

RELATION_GRID_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "relation_channel_spec",
    "AP_retention",
    "PA_retention",
    "PT_retention",
    "TP_retention",
    "PV_retention",
    "VP_retention",
    "AP_edge_count",
    "PA_edge_count",
    "PT_edge_count",
    "TP_edge_count",
    "PV_edge_count",
    "VP_edge_count",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "support_edge_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
]

DIRECTIONALITY_FIELDS = [
    "dataset",
    "method",
    "relation_pair_name",
    "forward_relation_name",
    "reverse_relation_name",
    "forward_retention",
    "reverse_retention",
    "forward_edge_count",
    "reverse_edge_count",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "relation_mapping_audit_pass",
]

COVERAGE_DIAGNOSTIC_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "relation_name",
    "candidate_source_node_count",
    "candidate_destination_node_count",
    "retained_source_node_count",
    "retained_destination_node_count",
    "source_coverage_ratio",
    "destination_coverage_ratio",
    "target_author_reachability_before",
    "target_author_reachability_after",
    "paper_coverage_ratio",
    "venue_coverage_ratio",
    "term_coverage_ratio",
    "max_endpoint_retained_degree",
    "p95_endpoint_retained_degree",
    "hub_cap_active_count",
    "orphan_rescue_count",
    "edge_gini_before",
    "edge_gini_after",
]

LABEL_GRAPH_ABLATION_FIELDS_21_3 = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "ablation_name",
    "label_feats_enabled",
    "num_label_hops",
    "num_feature_hops",
    "graph_edges_enabled",
    "feature_only_mode",
    "success",
    "status",
    "failed_reason",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_vs_default_method_micro",
    "recovery_vs_full_native_micro",
    "sehgnn_args_json",
    "notes",
]

WEIGHTED_ADAPTER_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "official_preprocess_accepts_edge_values",
    "official_preprocess_preserves_edge_values",
    "weighted_values_used_in_message_passing",
    "weighted_superedge_main_table_allowed",
    "adapter_name",
    "test_micro_f1",
    "test_macro_f1",
    "eligible_for_main_decision",
]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    preset: str
    method_family: str
    budget_strategy: str
    edge_score_strategy: str
    relation_channel_spec: str = ""
    storage_budget: float | None = None
    selector: str = "H6"
    eligible_for_main_decision: bool = True
    planned_only: bool = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _method_safe(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(method))


def _relation_stats(graph: HeteroGraph, *, min_edges_per_relation: int) -> list[RelationStats]:
    stats: list[RelationStats] = []
    for relation_id, rel in sorted(graph.relations.items()):
        spec = graph.relation_specs[int(relation_id)]
        official_name = official_relation_name_for_source("DBLP", int(spec.src_type), int(spec.dst_type))
        stats.append(
            RelationStats(
                relation_id=int(relation_id),
                relation_name=str(official_name),
                relation_pair_name=relation_pair_name(str(official_name)),
                src_type=str(spec.src_type),
                dst_type=str(spec.dst_type),
                full_edge_count=int(rel.num_edges),
                candidate_edge_count=int(rel.num_edges),
                min_edges=min(int(min_edges_per_relation), int(rel.num_edges)),
            )
        )
    return stats


def _support_graph(original: HeteroGraph, selector: str, *, graph_seed: int, candidate_k: int) -> tuple[HeteroGraph, np.ndarray]:
    graph, assignment, _diag = run_support_baseline(
        original,
        baseline=BASELINE_BY_SELECTOR[str(selector)],
        ratio=0.30,
        seed=int(graph_seed),
        candidate_k=int(candidate_k),
    )
    return _unweighted_graph(graph), np.asarray(assignment, dtype=np.int64)


def _method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    preset = str(args.preset if args.preset != "auto" else args.relation_grid_preset)
    specs: list[MethodSpec] = []
    best = str(args.best_relation_spec or "APPA100-PVVP100-PTTP30")
    if preset == "quick" or _bool_arg(args.quick):
        specs.append(MethodSpec("H6-relgrid-APPA100-PVVP100-PTTP30", "quick", "schema_compatible_subgraph", "relation_channel_grid", "random_edge_within_relation", "APPA100-PVVP100-PTTP30"))
    elif preset == "core":
        for token in ["00", "10", "20", "30", "40", "50"]:
            spec = f"APPA100-PVVP100-PTTP{token}"
            specs.append(MethodSpec(f"H6-relgrid-{spec}", preset, "schema_compatible_subgraph", "relation_channel_grid", "random_edge_within_relation", spec))
        for token in ["00", "25", "50", "75", "100"]:
            spec = f"APPA100-PTTP30-PVVP{token}"
            specs.append(MethodSpec(f"H6-relgrid-{spec}", preset, "schema_compatible_subgraph", "relation_channel_grid", "random_edge_within_relation", spec))
        for token in ["50", "75", "90", "100"]:
            spec = f"APPA{token}-PVVP100-PTTP20"
            specs.append(MethodSpec(f"H6-relgrid-{spec}", preset, "schema_compatible_subgraph", "relation_channel_grid", "random_edge_within_relation", spec))
        for budget in [0.50, 0.40, 0.35, 0.30]:
            token = int(round(budget * 100))
            specs.append(MethodSpec(f"H6-struct{token}-proportional-current", preset, "schema_compatible_subgraph", "proportional", "current_heuristic", storage_budget=budget))
        for budget in [0.50, 0.40, 0.30]:
            token = int(round(budget * 100))
            specs.append(MethodSpec(f"H6-struct{token}-degree-relwise", preset, "schema_compatible_subgraph", "proportional", "degree", storage_budget=budget))
            specs.append(MethodSpec(f"H6-struct{token}-random-edge-relwise", preset, "schema_compatible_subgraph", "proportional", "random_edge_within_relation", storage_budget=budget))
    elif preset == "directionality":
        for spec in [
            "AP100-PA00-PTTP30-PVVP100",
            "AP00-PA100-PTTP30-PVVP100",
            "AP100-PA50-PTTP30-PVVP100",
            "AP50-PA100-PTTP30-PVVP100",
            "APPA100-PT100-TP00-PVVP100",
            "APPA100-PT00-TP100-PVVP100",
            "APPA100-PT50-TP20-PVVP100",
            "APPA100-PT20-TP50-PVVP100",
            "APPA100-PTTP30-PV100-VP00",
            "APPA100-PTTP30-PV00-VP100",
            "APPA100-PTTP30-PV100-VP50",
            "APPA100-PTTP30-PV50-VP100",
        ]:
            specs.append(MethodSpec(f"H6-dir-{spec}", preset, "schema_compatible_subgraph", "relation_channel_grid", "random_edge_within_relation", spec))
    elif preset == "pathaware_v2":
        for budget in [0.40, 0.30]:
            token = int(round(budget * 100))
            for suffix, strategy in [
                ("random", "random_edge_within_relation"),
                ("degree", "degree"),
                ("pathaware-v2-topk-diagnostic", "pathaware_v2_topk_diagnostic"),
                ("pathaware-v2-stratified", "pathaware_v2_stratified"),
            ]:
                specs.append(MethodSpec(f"H6-struct{token}-relgrid-best-{suffix}", preset, "schema_compatible_subgraph", "relation_channel_grid", strategy, best, storage_budget=budget))
    elif preset == "label_graph_ablation":
        for method in ["H6-node30", "H6-struct40-best-relation-channel", "H6-struct30-best-relation-channel", "target-only-schema-stub", "full-native-SeHGNN"]:
            specs.append(MethodSpec(method, preset, "label_graph_ablation", "ablation", "none", best, eligible_for_main_decision=False))
    elif preset == "feature_cache_probe":
        specs.append(MethodSpec(str(args.base_method or "H6-struct40-best"), preset, "feature_cache_adapter_probe", "adapter_probe", "none", best, eligible_for_main_decision=False))
    else:
        raise ValueError(f"unsupported Gate21.3 preset: {preset}")
    if args.methods:
        allowed = set(str(v) for v in args.methods)
        specs = [spec for spec in specs if any(token in spec.method or token == spec.budget_strategy or token == spec.edge_score_strategy for token in allowed)]
    if args.max_runs is not None:
        specs = specs[: int(args.max_runs)]
    return specs


def _manifest_rows(args: argparse.Namespace, specs: Sequence[MethodSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = str(args.dataset).upper()
    out = Path(args.output_dir)
    for spec in specs:
        for graph_seed in args.graph_seeds:
            for training_seed in args.training_seeds:
                run_id = _run_id(dataset, spec.method, int(graph_seed), int(training_seed))
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "method": spec.method,
                        "preset": spec.preset,
                        "method_family": spec.method_family,
                        "budget_strategy": spec.budget_strategy,
                        "edge_score_strategy": spec.edge_score_strategy,
                        "relation_channel_spec": spec.relation_channel_spec,
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "export_dir": str(out / "exports" / dataset / f"graph_seed_{int(graph_seed)}" / _method_safe(spec.method)),
                        "run_dir": str(out / "runs" / dataset / f"graph_seed_{int(graph_seed)}" / f"training_seed_{int(training_seed)}" / _method_safe(spec.method)),
                        "expected_outputs": "stdout,stderr,raw_row,storage,row_audits",
                        "eligible_for_main_decision": spec.eligible_for_main_decision,
                        "planned_only": bool(_bool_arg(args.dry_run) or _bool_arg(args.plan_only) or spec.planned_only),
                    }
                )
    return rows


def _run_id(dataset: str, method: str, graph_seed: int, training_seed: int) -> str:
    token = f"{dataset}|{method}|g{int(graph_seed)}|t{int(training_seed)}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def _reference_rows(root: Path, dataset: str) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    native: dict[int, Mapping[str, Any]] = {}
    export: dict[int, Mapping[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in _read_csv(Path(root) / "gate21_1_raw_rows.csv"):
        if str(row.get("dataset", "")).upper() != dataset:
            continue
        method = str(row.get("method", ""))
        if method not in {"full-native-SeHGNN", "export-full-SeHGNN"}:
            continue
        seed = int(row.get("seed", 0) or 0)
        if method == "full-native-SeHGNN":
            native[seed] = row
        else:
            export[seed] = row
        out.append(_reference_to_raw(row, method))
    return native, export, out


def _reference_to_raw(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    seed = int(row.get("seed", 0) or 0)
    status = str(row.get("status", ""))
    return {
        "dataset": row.get("dataset", ""),
        "method": method,
        "method_family": "reference_full",
        "budget_strategy": "reference",
        "edge_score_strategy": "reference",
        "relation_channel_spec": "",
        "graph_seed": "",
        "training_seed": seed,
        "export_hash": "",
        "run_id": _run_id(str(row.get("dataset", "")), method, 0, seed),
        "success": status == "success",
        "status": status,
        "failed_reason": "",
        "model_name": "official-SeHGNN",
        "schema_compatible": True,
        "uses_weighted_superedges": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": False,
        "no_test_label_export_leakage": True,
        "no_test_label_scoring_leakage": True,
        "validation_micro_f1": row.get("validation_micro_f1", ""),
        "validation_macro_f1": row.get("validation_macro_f1", ""),
        "test_micro_f1": row.get("test_micro_f1", ""),
        "test_macro_f1": row.get("test_macro_f1", ""),
        "test_accuracy_if_single_label": row.get("test_accuracy_if_single_label", row.get("test_accuracy", "")),
        "native_full_test_micro_f1": row.get("test_micro_f1", ""),
        "native_full_test_macro_f1": row.get("test_macro_f1", ""),
        "export_full_test_micro_f1": row.get("test_micro_f1", ""),
        "export_full_test_macro_f1": row.get("test_macro_f1", ""),
        "mapping_bijective": True,
        "split_disjoint": True,
        "relation_order_matches_official": True,
        "node_type_order_matches_official": True,
        "schema_complete": True,
        "stdout_path": row.get("stdout_path", ""),
        "stderr_path": row.get("stderr_path", ""),
    }


def _raw_row(
    *,
    dataset: str,
    spec: MethodSpec,
    graph_seed: int,
    training_seed: int,
    run_row: Mapping[str, Any],
    storage_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    native: Mapping[str, Any] | None,
    export_full: Mapping[str, Any] | None,
    traceback_path: str = "",
) -> dict[str, Any]:
    native = native or {}
    export_full = export_full or {}
    status = str(run_row.get("status", ""))
    test_micro = _float(run_row.get("test_micro_f1"))
    test_macro = _float(run_row.get("test_macro_f1"))
    native_micro = _float(native.get("test_micro_f1")) or DEFAULT_NATIVE_MICRO
    native_macro = _float(native.get("test_macro_f1")) or DEFAULT_NATIVE_MACRO
    export_micro = _float(export_full.get("test_micro_f1")) or native_micro
    export_macro = _float(export_full.get("test_macro_f1")) or native_macro
    return {
        "dataset": dataset,
        "method": spec.method,
        "method_family": spec.method_family,
        "budget_strategy": spec.budget_strategy,
        "edge_score_strategy": spec.edge_score_strategy,
        "relation_channel_spec": spec.relation_channel_spec,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "export_hash": manifest.get("file_list_hash", manifest.get("export_hash", "")),
        "run_id": _run_id(dataset, spec.method, int(graph_seed), int(training_seed)),
        "success": status == "success",
        "status": status,
        "failed_reason": "" if status == "success" else str(run_row.get("error_message", run_row.get("failed_reason", status))),
        "traceback_path": traceback_path,
        "model_name": "official-SeHGNN",
        "schema_compatible": spec.method_family != "weighted_adapter_probe",
        "uses_weighted_superedges": False,
        "official_sehgnn_unmodified": spec.method_family not in {"feature_cache_adapter_probe", "weighted_adapter_probe"},
        "eligible_for_main_decision": spec.eligible_for_main_decision,
        "no_test_label_export_leakage": manifest.get("no_test_label_export_leakage", True),
        "no_test_label_scoring_leakage": True,
        "semantic_structural_storage_ratio": storage_row.get("semantic_structural_storage_ratio", ""),
        "hgb_raw_file_byte_ratio": storage_row.get("hgb_raw_file_byte_ratio", ""),
        "preprocessed_cache_byte_ratio": storage_row.get("preprocessed_cache_byte_ratio", ""),
        "support_node_ratio": storage_row.get("support_node_ratio", ""),
        "support_edge_ratio": storage_row.get("support_edge_ratio", ""),
        "total_node_ratio": storage_row.get("total_node_ratio", ""),
        "total_edge_ratio": storage_row.get("total_edge_ratio", ""),
        "validation_micro_f1": run_row.get("validation_micro_f1", ""),
        "validation_macro_f1": run_row.get("validation_macro_f1", ""),
        "test_micro_f1": "" if test_micro is None else test_micro,
        "test_macro_f1": "" if test_macro is None else test_macro,
        "test_accuracy_if_single_label": run_row.get("test_accuracy_if_single_label", ""),
        "native_full_test_micro_f1": native_micro,
        "native_full_test_macro_f1": native_macro,
        "export_full_test_micro_f1": export_micro,
        "export_full_test_macro_f1": export_macro,
        "recovery_vs_native_full_micro": "" if test_micro is None else test_micro / native_micro,
        "recovery_vs_native_full_macro": "" if test_macro is None else test_macro / native_macro,
        "mapping_bijective": manifest.get("mapping_bijective", True),
        "split_disjoint": manifest.get("split_disjoint", True),
        "relation_order_matches_official": manifest.get("relation_order_matches_official", ""),
        "node_type_order_matches_official": manifest.get("node_type_order_matches_official", ""),
        "schema_complete": manifest.get("can_load_with_official_data_loader", manifest.get("schema_complete", True)),
        "stdout_path": run_row.get("stdout_path", ""),
        "stderr_path": run_row.get("stderr_path", ""),
    }


def _empty_outputs(out: Path) -> None:
    write_csv(out / "gate21_3_raw_rows.csv", [], fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_3_by_method.csv", [])
    write_csv(out / "gate21_3_recovery_by_method.csv", [])
    write_csv(out / "gate21_3_storage_frontier.csv", [])
    write_csv(out / "gate21_3_relation_channel_grid.csv", [], fieldnames=RELATION_GRID_FIELDS)
    write_csv(out / "gate21_3_directionality_ablation.csv", [], fieldnames=DIRECTIONALITY_FIELDS)
    write_csv(out / "gate21_3_graph_seed_stability.csv", [])
    write_csv(out / "gate21_3_relation_mapping_audit.csv", [], fieldnames=RELATION_MAPPING_FIELDS_21_3)
    write_csv(out / "gate21_3_relation_edge_retention.csv", [], fieldnames=RELATION_RETENTION_FIELDS_21_3)
    write_csv(out / "gate21_3_edge_score_diagnostics.csv", [], fieldnames=EDGE_SCORE_V2_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_3_coverage_diagnostics.csv", [], fieldnames=COVERAGE_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_3_storage_audit.csv", [], fieldnames=[*STORAGE_AUDIT_FIELDS, "eligible_for_main_decision"])
    write_csv(out / "gate21_3_label_graph_ablation.csv", [], fieldnames=LABEL_GRAPH_ABLATION_FIELDS_21_3)
    write_csv(out / "gate21_3_feature_cache_compression_probe.csv", [], fieldnames=FEATURE_CACHE_PROBE_FIELDS)
    write_csv(out / "gate21_3_weighted_adapter_probe.csv", [], fieldnames=WEIGHTED_ADAPTER_FIELDS)


def _mock_relation_rows(dataset: str, specs: Sequence[MethodSpec], graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_counts = {"AP": 10, "PA": 10, "PT": 20, "PV": 4, "TP": 20, "VP": 4}
    official_ids = {"AP": 0, "PA": 1, "PT": 2, "PV": 3, "TP": 4, "VP": 5}
    rows: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    for spec in specs:
        parsed = parse_relation_channel_spec(spec.relation_channel_spec or "APPA100-PVVP100-PTTP30") if spec.budget_strategy == "relation_channel_grid" else None
        for graph_seed in graph_seeds:
            for training_seed in training_seeds:
                for name, official_id in official_ids.items():
                    candidate = base_counts[name]
                    requested = int(round(candidate * (parsed.retention_by_relation[name] if parsed is not None else 0.5)))
                    retained = max(1, requested)
                    reciprocal = {"AP": "PA", "PA": "AP", "PT": "TP", "TP": "PT", "PV": "VP", "VP": "PV"}[name]
                    row = {
                        "dataset": dataset,
                        "method": spec.method,
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "source_relation_id": int(official_id),
                        "source_relation_name": name,
                        "source_src_type": name[0],
                        "source_dst_type": name[1],
                        "official_relation_id": int(official_id),
                        "official_relation_name": name,
                        "official_src_type": name[0],
                        "official_dst_type": name[1],
                        "relation_pair_name": relation_pair_name(name),
                        "reciprocal_official_relation_id": official_ids[reciprocal],
                        "reciprocal_official_relation_name": reciprocal,
                        "source_edge_count": candidate,
                        "original_full_edge_count": candidate,
                        "candidate_edge_count_after_node_pruning": candidate,
                        "retained_edge_count": retained,
                        "retention_vs_candidate": retained / candidate,
                        "retention_vs_full": retained / candidate,
                        "requested_relation_budget": requested,
                        "actual_relation_budget": retained,
                        "reciprocal_count_consistent": True,
                        "min_edges_constraint_active": retained != requested,
                        "relation_dropped_flag": False,
                    }
                    rows.append(row)
                    retention.append({key: row.get(key, "") for key in RELATION_RETENTION_FIELDS_21_3} | {"budget_strategy": spec.budget_strategy, "edge_score_strategy": spec.edge_score_strategy})
    return rows, retention


def _write_dry_run(args: argparse.Namespace, specs: list[MethodSpec]) -> dict[str, Any]:
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    manifest_rows = _manifest_rows(args, specs)
    raw_rows = [
        {
            "dataset": str(args.dataset).upper(),
            "method": row["method"],
            "method_family": row["method_family"],
            "budget_strategy": row["budget_strategy"],
            "edge_score_strategy": row["edge_score_strategy"],
            "relation_channel_spec": row["relation_channel_spec"],
            "graph_seed": row["graph_seed"],
            "training_seed": row["training_seed"],
            "export_hash": "",
            "run_id": row["run_id"],
            "success": False,
            "status": "planned",
            "failed_reason": "",
            "model_name": "official-SeHGNN",
            "schema_compatible": True,
            "uses_weighted_superedges": False,
            "official_sehgnn_unmodified": row["method_family"] not in {"feature_cache_adapter_probe", "weighted_adapter_probe"},
            "eligible_for_main_decision": row["eligible_for_main_decision"],
            "no_test_label_export_leakage": True,
            "no_test_label_scoring_leakage": True,
        }
        for row in manifest_rows
    ]
    mapping_rows, retention_rows = _mock_relation_rows(str(args.dataset).upper(), specs, args.graph_seeds, args.training_seeds)
    write_csv(out / "gate21_3_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_3_run_manifest.csv", manifest_rows, fieldnames=RUN_MANIFEST_FIELDS)
    write_csv(out / "gate21_3_relation_mapping_audit.csv", mapping_rows, fieldnames=RELATION_MAPPING_FIELDS_21_3)
    write_csv(out / "gate21_3_relation_edge_retention.csv", retention_rows, fieldnames=RELATION_RETENTION_FIELDS_21_3)
    write_json(
        out / "gate21_3_plan.json",
        {
            "dataset": str(args.dataset).upper(),
            "preset": str(args.preset),
            "graph_seeds": [int(v) for v in args.graph_seeds],
            "training_seeds": [int(v) for v in args.training_seeds],
            "methods": [spec.method for spec in specs],
            "dry_run": True,
            "plan_only": bool(_bool_arg(args.plan_only)),
        },
    )
    summary = summarize_gate21_3(out, out)
    return {"dry_run": True, "planned_runs": len(raw_rows), "summary": summary}


def _degree_scores(rel: RelationAdj, graph: HeteroGraph) -> np.ndarray:
    src_degree = np.bincount(rel.src, minlength=graph.num_nodes).astype(np.float64)
    dst_degree = np.bincount(rel.dst, minlength=graph.num_nodes).astype(np.float64)
    return 1.0 / np.sqrt(np.maximum(src_degree[rel.src] * dst_degree[rel.dst], 1.0))


def _current_scores(rel: RelationAdj, graph: HeteroGraph, target_type: int) -> np.ndarray:
    src_degree = np.bincount(rel.src, minlength=graph.num_nodes).astype(np.float64)
    dst_degree = np.bincount(rel.dst, minlength=graph.num_nodes).astype(np.float64)
    target_bonus = ((graph.node_type[rel.src] == int(target_type)) | (graph.node_type[rel.dst] == int(target_type))).astype(np.float64)
    return target_bonus + 0.25 / np.maximum(src_degree[rel.src], 1.0) + 0.25 / np.maximum(dst_degree[rel.dst], 1.0)


def _topk(scores: np.ndarray, budget: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    budget = max(0, min(int(budget), int(scores.size)))
    if budget >= scores.size:
        return np.arange(scores.size, dtype=np.int64)
    return np.sort(np.argsort(-scores, kind="mergesort")[:budget].astype(np.int64))


def _required_edge_indices(graph: HeteroGraph) -> dict[int, set[int]]:
    required: dict[int, set[int]] = {int(rid): set() for rid in graph.relations}
    for type_id in sorted(set(int(v) for v in graph.node_type.tolist())):
        type_nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if type_nodes.size == 0:
            continue
        max_node = int(type_nodes[-1])
        for relation_id, rel in sorted(graph.relations.items()):
            hits = np.flatnonzero((rel.src == max_node) | (rel.dst == max_node)).astype(np.int64)
            if hits.size:
                required[int(relation_id)].add(int(hits[0]))
                break
    return required


def _merge_required_edges(keep: np.ndarray, required: set[int], *, budget: int, edge_count: int) -> np.ndarray:
    budget = max(0, min(int(budget), int(edge_count)))
    required_valid = [idx for idx in sorted(required) if 0 <= int(idx) < int(edge_count)]
    selected: list[int] = []
    selected_set: set[int] = set()
    for idx in required_valid:
        if len(selected) >= budget:
            break
        selected.append(int(idx))
        selected_set.add(int(idx))
    for idx in np.asarray(keep, dtype=np.int64).tolist():
        if len(selected) >= budget:
            break
        if int(idx) in selected_set:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
    return np.asarray(sorted(selected), dtype=np.int64)


def _prune_graph(
    *,
    graph: HeteroGraph,
    original: HeteroGraph,
    dataset: str,
    spec: MethodSpec,
    graph_seed: int,
    target_type: int,
    train_idx: np.ndarray,
    labels: np.ndarray,
    min_edges_per_relation: int,
) -> tuple[HeteroGraph, dict[int, int], dict[int, int], dict[int, int], dict[int, bool], list[dict[str, Any]], list[dict[str, Any]]]:
    stats = _relation_stats(graph, min_edges_per_relation=min_edges_per_relation)
    if spec.budget_strategy == "relation_channel_grid":
        allocations = allocate_relation_channel_spec(stats, parse_relation_channel_spec(spec.relation_channel_spec, sampling_strategy=spec.edge_score_strategy), min_edges_per_relation=min_edges_per_relation)
    else:
        total_budget = edge_budget_for_storage(original, graph, float(spec.storage_budget or 1.0))
        allocations = RelationBudgetAllocator().allocate(
            relation_stats=stats,
            total_edge_budget=total_budget,
            strategy="proportional",
            min_edges_per_relation=min_edges_per_relation,
            seed=int(graph_seed),
        )
    budget_by_relation = {int(row.relation_id): int(row.actual_edges) for row in allocations}
    requested_by_relation = {int(row.relation_id): int(row.requested_edges) for row in allocations}
    min_active = {int(row.relation_id): bool(row.min_edges_constraint_active) for row in allocations}
    scorer = PathAwareV2Scorer()
    sampler = CoverageSampler(hub_cap=32)
    required_by_relation = _required_edge_indices(graph)
    relations: dict[int, RelationAdj] = {}
    score_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for relation_id, rel in sorted(graph.relations.items()):
        rid = int(relation_id)
        spec_rel = graph.relation_specs[rid]
        official_relation_name = official_relation_name_for_source(dataset, int(spec_rel.src_type), int(spec_rel.dst_type))
        budget = budget_by_relation.get(rid, rel.num_edges)
        if spec.edge_score_strategy == "random_edge_within_relation":
            keep = sample_random_edge_indices(edge_count=rel.num_edges, budget=budget, graph_seed=graph_seed, relation_id=rid)
        elif spec.edge_score_strategy == "degree":
            keep = _topk(_degree_scores(rel, graph), budget)
        elif spec.edge_score_strategy == "current_heuristic":
            keep = _topk(_current_scores(rel, graph, target_type), budget)
        elif spec.edge_score_strategy in {"pathaware_v2_topk_diagnostic", "pathaware_v2_stratified"}:
            scores, diag = scorer.score_relation(
                dataset=dataset,
                method=spec.method,
                graph_seed=graph_seed,
                relation_id=rid,
                relation_name=str(official_relation_name),
                graph=graph,
                train_idx=train_idx,
                val_idx=np.array([], dtype=np.int64),
                labels=labels,
            )
            score_rows.append(diag)
            if spec.edge_score_strategy == "pathaware_v2_stratified":
                keep, coverage = sampler.select(src=rel.src, dst=rel.dst, scores=scores, budget=budget, graph_seed=graph_seed, relation_id=rid, min_edges=min_edges_per_relation)
                coverage_rows.append({"dataset": dataset, "method": spec.method, "graph_seed": int(graph_seed), "relation_name": str(official_relation_name), **coverage})
            else:
                keep = _topk(scores, budget)
        else:
            raise ValueError(f"unsupported edge_score_strategy: {spec.edge_score_strategy}")
        keep = _merge_required_edges(keep, required_by_relation.get(rid, set()), budget=budget, edge_count=rel.num_edges)
        relations[rid] = RelationAdj(rel.src[keep].copy(), rel.dst[keep].copy(), rel.weight[keep].copy(), rel.src_type, rel.dst_type, rid)
    pruned = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=relations,
        relation_specs=graph.relation_specs,
        features=None if graph.features is None else {int(k): v.copy() for k, v in graph.features.items()},
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
    )
    validate_schema(pruned)
    return (
        pruned,
        {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()},
        {int(rid): int(rel.num_edges) for rid, rel in pruned.relations.items()},
        requested_by_relation,
        min_active,
        score_rows,
        coverage_rows,
    )


def _relation_audit_rows(
    *,
    graph: HeteroGraph,
    original: HeteroGraph,
    dataset: str,
    spec: MethodSpec,
    graph_seed: int,
    training_seed: int,
    candidate_counts: Mapping[int, int],
    retained_counts: Mapping[int, int],
    requested_budgets: Mapping[int, int],
    min_active: Mapping[int, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = build_relation_keys(graph, dataset=dataset)
    name_to_source = {key.official_relation_name: int(key.source_relation_id if key.source_relation_id is not None else -1) for key in keys}
    retained_by_name = {
        official_relation_name_for_source(dataset, int(graph.relation_specs[int(rid)].src_type), int(graph.relation_specs[int(rid)].dst_type)): int(count)
        for rid, count in retained_counts.items()
    }
    mapping_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    for key in keys:
        source_id = int(key.source_relation_id if key.source_relation_id is not None else -1)
        source_count = int(graph.relations[source_id].num_edges) if source_id in graph.relations else int(retained_by_name.get(key.official_relation_name, 0))
        original_count = int(original.relations[source_id].num_edges) if source_id in original.relations else source_count
        candidate = int(candidate_counts.get(source_id, source_count))
        retained = int(retained_counts.get(source_id, source_count))
        requested = int(requested_budgets.get(source_id, retained))
        reciprocal_retained = retained_by_name.get(str(key.reciprocal_official_relation_name), retained)
        row = {
            "dataset": dataset,
            "method": spec.method,
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "source_relation_id": source_id if source_id >= 0 else -1,
            "source_relation_name": key.source_relation_name or key.official_relation_name,
            "source_src_type": key.source_src_type or key.official_src_type,
            "source_dst_type": key.source_dst_type or key.official_dst_type,
            "official_relation_id": int(key.official_relation_id),
            "official_relation_name": key.official_relation_name,
            "official_src_type": key.official_src_type,
            "official_dst_type": key.official_dst_type,
            "relation_pair_name": key.relation_pair_name,
            "reciprocal_official_relation_id": key.reciprocal_official_relation_id,
            "reciprocal_official_relation_name": key.reciprocal_official_relation_name,
            "source_edge_count": source_count,
            "original_full_edge_count": original_count,
            "candidate_edge_count_after_node_pruning": candidate,
            "retained_edge_count": retained,
            "retention_vs_candidate": float(retained / max(candidate, 1)),
            "retention_vs_full": float(retained / max(original_count, 1)),
            "requested_relation_budget": requested,
            "actual_relation_budget": retained,
            "reciprocal_count_consistent": bool(retained == int(reciprocal_retained)),
            "min_edges_constraint_active": bool(min_active.get(source_id, False)),
            "relation_dropped_flag": bool(retained == 0),
        }
        mapping_rows.append({field: row.get(field, "") for field in RELATION_MAPPING_FIELDS_21_3})
        retention_rows.append({field: row.get(field, "") for field in RELATION_RETENTION_FIELDS_21_3} | {"budget_strategy": spec.budget_strategy, "edge_score_strategy": spec.edge_score_strategy})
    return mapping_rows, retention_rows


def _assert_relation_consistency(
    *,
    export_dir: Path,
    export_audit: Mapping[str, Any],
    retention_rows: Sequence[Mapping[str, Any]],
    spec: MethodSpec,
) -> None:
    link_counts = parse_link_relation_counts(Path(export_dir))
    retained_by_official = {str(int(row["official_relation_id"])): int(row["retained_edge_count"]) for row in retention_rows}
    if sum(retained_by_official.values()) != sum(link_counts.values()):
        raise AssertionError("sum(retained_edge_count) does not match link.dat line count")
    if retained_by_official != link_counts:
        raise AssertionError(f"retained_edge_count_by_relation does not match link.dat: {retained_by_official} != {link_counts}")
    report = validate_hgb_relation_order(dataset=str(export_audit.get("dataset", "DBLP")), dataset_dir=export_dir, hgb_export_edge_counts=export_audit.get("edge_count_by_relation"))
    if not report["link_dat_relation_counts_match_export_audit"]:
        raise AssertionError("link.dat relation counts differ from hgb_export_audit edge_count_by_relation")
    for row in retention_rows:
        if int(row["candidate_edge_count_after_node_pruning"]) < int(row["retained_edge_count"]):
            raise AssertionError("candidate_edge_count_after_node_pruning < retained_edge_count")
        if int(row["original_full_edge_count"]) < int(row["candidate_edge_count_after_node_pruning"]):
            raise AssertionError("original_full_edge_count < candidate_edge_count_after_node_pruning")
        if int(row["retained_edge_count"]) == 0 and spec.method_family != "schema_stub_diagnostic":
            raise AssertionError("official relation disappeared in a non-diagnostic method")


def _relation_grid_row(
    *,
    dataset: str,
    spec: MethodSpec,
    graph_seed: int,
    training_seed: int,
    raw_row: Mapping[str, Any],
    retention_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_name = {str(row["official_relation_name"]): row for row in retention_rows}
    parsed = parse_relation_channel_spec(spec.relation_channel_spec or "APPA100-PVVP100-PTTP30")
    row = {
        "dataset": dataset,
        "method": spec.method,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "relation_channel_spec": spec.relation_channel_spec,
        "semantic_structural_storage_ratio": raw_row.get("semantic_structural_storage_ratio", ""),
        "hgb_raw_file_byte_ratio": raw_row.get("hgb_raw_file_byte_ratio", ""),
        "support_edge_ratio": raw_row.get("support_edge_ratio", ""),
        "test_micro_f1": raw_row.get("test_micro_f1", ""),
        "test_macro_f1": raw_row.get("test_macro_f1", ""),
        "validation_micro_f1": raw_row.get("validation_micro_f1", ""),
        "validation_macro_f1": raw_row.get("validation_macro_f1", ""),
        "recovery_vs_native_full_micro": raw_row.get("recovery_vs_native_full_micro", ""),
        "recovery_vs_native_full_macro": raw_row.get("recovery_vs_native_full_macro", ""),
    }
    for name in ["AP", "PA", "PT", "TP", "PV", "VP"]:
        row[f"{name}_retention"] = parsed.retention_by_relation.get(name, "")
        row[f"{name}_edge_count"] = by_name.get(name, {}).get("retained_edge_count", "")
    return {field: row.get(field, "") for field in RELATION_GRID_FIELDS}


def _directionality_rows(dataset: str, spec: MethodSpec, raw_row: Mapping[str, Any], retention_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not spec.method.startswith("H6-dir-"):
        return []
    parsed = parse_relation_channel_spec(spec.relation_channel_spec)
    by_name = {str(row["official_relation_name"]): row for row in retention_rows}
    rows = []
    for pair, forward, reverse in [("AP_PA", "AP", "PA"), ("PT_TP", "PT", "TP"), ("PV_VP", "PV", "VP")]:
        rows.append(
            {
                "dataset": dataset,
                "method": spec.method,
                "relation_pair_name": pair,
                "forward_relation_name": forward,
                "reverse_relation_name": reverse,
                "forward_retention": parsed.retention_by_relation[forward],
                "reverse_retention": parsed.retention_by_relation[reverse],
                "forward_edge_count": by_name.get(forward, {}).get("retained_edge_count", ""),
                "reverse_edge_count": by_name.get(reverse, {}).get("retained_edge_count", ""),
                "test_micro_f1": raw_row.get("test_micro_f1", ""),
                "test_macro_f1": raw_row.get("test_macro_f1", ""),
                "validation_micro_f1": raw_row.get("validation_micro_f1", ""),
                "validation_macro_f1": raw_row.get("validation_macro_f1", ""),
                "semantic_structural_storage_ratio": raw_row.get("semantic_structural_storage_ratio", ""),
                "hgb_raw_file_byte_ratio": raw_row.get("hgb_raw_file_byte_ratio", ""),
                "relation_mapping_audit_pass": True,
            }
        )
    return rows


def _write_weighted_probe(out: Path, dataset: str) -> None:
    write_csv(
        out / "gate21_3_weighted_adapter_probe.csv",
        [
            {
                "dataset": dataset,
                "method": "H6-struct30-weighted-superedge-adapter",
                "graph_seed": "",
                "training_seed": "",
                "official_preprocess_accepts_edge_values": False,
                "official_preprocess_preserves_edge_values": False,
                "weighted_values_used_in_message_passing": False,
                "weighted_superedge_main_table_allowed": False,
                "adapter_name": "SeHGNN-weighted-adapter",
                "test_micro_f1": "",
                "test_macro_f1": "",
                "eligible_for_main_decision": False,
            }
        ],
        fieldnames=WEIGHTED_ADAPTER_FIELDS,
    )


def _set_cli_value(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command)
    if flag in out:
        idx = out.index(flag)
        if idx + 1 >= len(out):
            out.append(value)
        else:
            out[idx + 1] = value
    else:
        out.extend([flag, value])
    return out


def _ablation_command(base: NativeCommand, ablation_name: str) -> tuple[NativeCommand | None, dict[str, Any], str]:
    command = list(base.command)
    label_feats_enabled = "--label-feats" in command
    num_label_hops = _arg_value(command, "--num-label-hops", "4")
    num_feature_hops = _arg_value(command, "--num-hops", "2")
    graph_edges_enabled = True
    feature_only_mode = False
    notes = ""
    if ablation_name == "default":
        pass
    elif ablation_name == "no_label_feats":
        command = [part for part in command if part != "--label-feats"]
        label_feats_enabled = False
    elif ablation_name == "num_label_hops_0":
        command = _set_cli_value(command, "--num-label-hops", "0")
        num_label_hops = "0"
    elif ablation_name == "num_feature_hops_0":
        command = _set_cli_value(command, "--num-hops", "0")
        num_feature_hops = "0"
    elif ablation_name == "feature_only_mode":
        command = [part for part in command if part != "--label-feats"]
        command = _set_cli_value(command, "--num-label-hops", "0")
        label_feats_enabled = False
        num_label_hops = "0"
        feature_only_mode = True
        notes = "approximated_by_disabling_label_features_in_unmodified_official_sehgnn"
    elif ablation_name == "graph_edges_enabled_false":
        graph_edges_enabled = False
        notes = "not_supported_as_runtime_flag_by_unmodified_official_sehgnn"
        return None, {
            "label_feats_enabled": label_feats_enabled,
            "num_label_hops": num_label_hops,
            "num_feature_hops": num_feature_hops,
            "graph_edges_enabled": graph_edges_enabled,
            "feature_only_mode": feature_only_mode,
            "sehgnn_args_json": json.dumps(command),
        }, notes
    else:
        raise ValueError(f"unsupported ablation_name: {ablation_name}")
    return (
        NativeCommand(command=command, cwd=base.cwd, dataset=base.dataset, seed=base.seed),
        {
            "label_feats_enabled": label_feats_enabled,
            "num_label_hops": num_label_hops,
            "num_feature_hops": num_feature_hops,
            "graph_edges_enabled": graph_edges_enabled,
            "feature_only_mode": feature_only_mode,
            "sehgnn_args_json": json.dumps(command),
        },
        notes,
    )


def _arg_value(command: list[str], flag: str, default: str) -> str:
    if flag not in command:
        return default
    idx = command.index(flag)
    return str(command[idx + 1]) if idx + 1 < len(command) else default


def run_gate21_3(args: argparse.Namespace) -> dict[str, Any]:
    specs = _method_specs(args)
    if _bool_arg(args.dry_run) or _bool_arg(args.plan_only):
        return _write_dry_run(args, specs)
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    dataset = str(args.dataset).upper()
    if dataset != "DBLP":
        raise ValueError("Gate21.3 is intentionally DBLP-only until DBLP relation-channel audit passes")
    manifest_rows = _manifest_rows(args, specs)
    write_csv(out / "gate21_3_run_manifest.csv", manifest_rows, fieldnames=RUN_MANIFEST_FIELDS)
    write_json(out / "gate21_3_plan.json", {"dataset": dataset, "preset": args.preset, "graph_seeds": args.graph_seeds, "training_seeds": args.training_seeds, "methods": [s.method for s in specs], "dry_run": False})
    native_by_seed, export_by_seed, reference_rows = _reference_rows(Path(args.gate21_1_root), dataset)
    raw_rows: list[dict[str, Any]] = list(reference_rows)
    storage_rows: list[dict[str, Any]] = []
    mapping_rows_all: list[dict[str, Any]] = []
    retention_rows_all: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    original = load_hgb_graph(Path(args.data_root), dataset)
    target_type = int(infer_target_node_type(original))
    labels_native, trainval_native, test_native = _native_labels(Path(args.hgb_data_root), dataset, original.num_nodes)
    support_cache: dict[int, tuple[HeteroGraph, np.ndarray]] = {}
    for graph_seed in [int(v) for v in args.graph_seeds]:
        support_cache[graph_seed] = _support_graph(original, "H6", graph_seed=graph_seed, candidate_k=int(args.candidate_k))
    existing = {row.get("run_id"): row for row in _read_csv(out / "gate21_3_raw_rows.csv")} if _bool_arg(args.skip_existing) else {}
    for spec in specs:
        for graph_seed in [int(v) for v in args.graph_seeds]:
            for training_seed in [int(v) for v in args.training_seeds]:
                run_id = _run_id(dataset, spec.method, graph_seed, training_seed)
                if run_id in existing and not _bool_arg(args.force):
                    row = dict(existing[run_id])
                    row["status"] = "skipped_existing"
                    raw_rows.append(row)
                    continue
                try:
                    if spec.method == "target-only-schema-stub":
                        graph, _stub_audit = build_target_only_schema_stub_graph(graph=original, dataset_name=dataset, target_type=TARGET_TYPE_BY_DATASET[dataset])
                        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
                        trainval = trainval_native
                        test = test_native
                        candidate_counts = {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()}
                        retained_counts = dict(candidate_counts)
                        requested_budgets = dict(candidate_counts)
                        min_active = {int(rid): False for rid in graph.relations}
                    else:
                        base_graph, assignment = support_cache[graph_seed]
                        if spec.method == "H6-node30" or spec.method_family in {"label_graph_ablation", "feature_cache_adapter_probe"}:
                            graph = base_graph
                            candidate_counts = {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()}
                            retained_counts = dict(candidate_counts)
                            requested_budgets = dict(candidate_counts)
                            min_active = {int(rid): False for rid in graph.relations}
                        else:
                            graph, candidate_counts, retained_counts, requested_budgets, min_active, score, coverage = _prune_graph(
                                graph=base_graph,
                                original=original,
                                dataset=dataset,
                                spec=spec,
                                graph_seed=graph_seed,
                                target_type=target_type,
                                train_idx=trainval_native,
                                labels=labels_native,
                                min_edges_per_relation=int(args.min_edges_per_relation),
                            )
                            score_rows.extend(score)
                            coverage_rows.extend(coverage)
                        labels, trainval, test = _compressed_labels_and_splits(
                            original=original,
                            graph=graph,
                            assignment=assignment,
                            target_type=target_type,
                            labels_native=labels_native,
                            trainval_native=trainval_native,
                            test_native=test_native,
                        )
                    export_base = Path(args.export_root) if args.export_root else out / "exports" / dataset / f"graph_seed_{graph_seed}"
                    manifest = export_graph_to_sehgnn_hgb(
                        graph=graph,
                        dataset_name=dataset,
                        target_type=TARGET_TYPE_BY_DATASET[dataset],
                        output_dir=export_base,
                        split_mode="official_trainval",
                        train_idx=trainval,
                        val_idx=np.array([], dtype=np.int64),
                        test_idx=test,
                        labels=labels,
                        method_name=spec.method,
                        seed=graph_seed,
                    )
                    export_dir = Path(manifest["export_dir"])
                    export_audit = audit_native_hgb_data_dir(dataset, export_dir.parent, Path(args.sehgnn_root))
                    export_rows.append({**manifest, **export_audit, "method": spec.method, "graph_seed": graph_seed})
                    semantic_ratio = semantic_storage_ratio(graph, original)
                    storage_base = _storage_fields(original, graph, target_type, Path(args.hgb_data_root) / dataset, export_dir)
                    storage = audit_hgb_directory(
                        dataset=dataset,
                        method=spec.method,
                        seed=graph_seed,
                        export_dir=export_dir,
                        native_full_dir=Path(args.hgb_data_root) / dataset,
                        semantic_structural_storage_ratio=float(semantic_ratio),
                        support_node_ratio=_float(storage_base.get("support_node_ratio")),
                        support_edge_ratio=_float(storage_base.get("support_edge_ratio")),
                        total_node_ratio=_float(storage_base.get("total_node_ratio")),
                        total_edge_ratio=_float(storage_base.get("total_edge_ratio")),
                        structural_budget=spec.storage_budget,
                        raw_byte_budget=0.50,
                    ).to_row(method_family=spec.method_family)
                    storage["eligible_for_main_decision"] = spec.eligible_for_main_decision
                    storage_rows.append(storage)
                    mapping_rows, retention_rows = _relation_audit_rows(
                        graph=graph,
                        original=original,
                        dataset=dataset,
                        spec=spec,
                        graph_seed=graph_seed,
                        training_seed=training_seed,
                        candidate_counts=candidate_counts,
                        retained_counts=retained_counts,
                        requested_budgets=requested_budgets,
                        min_active=min_active,
                    )
                    _assert_relation_consistency(export_dir=export_dir, export_audit=export_audit, retention_rows=retention_rows, spec=spec)
                    mapping_rows_all.extend(mapping_rows)
                    retention_rows_all.extend(retention_rows)
                    if _bool_arg(args.skip_official_training) or spec.method_family in {"feature_cache_adapter_probe"}:
                        run_row = {"dataset": dataset, "seed": training_seed, "status": "skipped", "error_message": "skip_official_training_or_adapter_probe"}
                    else:
                        command = build_official_hgb_command(
                            dataset=dataset,
                            seed=training_seed,
                            repo_dir=Path(args.sehgnn_root),
                            data_root=export_dir.parent,
                            device=str(args.device),
                            python_executable=sys.executable,
                        )
                        run_dir = out / "runs" / dataset / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}" / _method_safe(spec.method)
                        run_row = run_native_command(command, stdout_path=run_dir / "stdout.log", stderr_path=run_dir / "stderr.log")
                    raw = _raw_row(
                        dataset=dataset,
                        spec=spec,
                        graph_seed=graph_seed,
                        training_seed=training_seed,
                        run_row=run_row,
                        storage_row=storage,
                        manifest={**manifest, **export_audit},
                        native=native_by_seed.get(training_seed),
                        export_full=export_by_seed.get(training_seed),
                    )
                    raw_rows.append(raw)
                    if spec.method_family == "label_graph_ablation":
                        default_micro = _float(run_row.get("test_micro_f1"))
                        for ablation_name in [
                            "default",
                            "no_label_feats",
                            "num_label_hops_0",
                            "num_feature_hops_0",
                            "feature_only_mode",
                            "graph_edges_enabled_false",
                        ]:
                            if ablation_name == "default":
                                ablation_row = dict(run_row)
                                ab_meta = {
                                    "label_feats_enabled": True,
                                    "num_label_hops": 4,
                                    "num_feature_hops": 2,
                                    "graph_edges_enabled": True,
                                    "feature_only_mode": False,
                                    "sehgnn_args_json": json.dumps(command.command if "command" in locals() else []),
                                }
                                notes = ""
                            else:
                                ab_cmd, ab_meta, notes = _ablation_command(command, ablation_name)
                                if ab_cmd is None:
                                    ablation_row = {"status": "unsupported_by_unmodified_official", "error_message": notes}
                                else:
                                    ab_dir = out / "runs" / dataset / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}" / _method_safe(spec.method) / f"ablation_{ablation_name}"
                                    ablation_row = run_native_command(ab_cmd, stdout_path=ab_dir / "stdout.log", stderr_path=ab_dir / "stderr.log")
                            ab_micro = _float(ablation_row.get("test_micro_f1"))
                            native_micro = _float(native_by_seed.get(training_seed, {}).get("test_micro_f1")) or DEFAULT_NATIVE_MICRO
                            label_rows.append(
                                {
                                    "dataset": dataset,
                                    "method": spec.method,
                                    "graph_seed": int(graph_seed),
                                    "training_seed": int(training_seed),
                                    "ablation_name": ablation_name,
                                    "label_feats_enabled": ab_meta["label_feats_enabled"],
                                    "num_label_hops": ab_meta["num_label_hops"],
                                    "num_feature_hops": ab_meta["num_feature_hops"],
                                    "graph_edges_enabled": ab_meta["graph_edges_enabled"],
                                    "feature_only_mode": ab_meta["feature_only_mode"],
                                    "success": ablation_row.get("status") == "success",
                                    "status": ablation_row.get("status", ""),
                                    "failed_reason": "" if ablation_row.get("status") == "success" else ablation_row.get("error_message", ""),
                                    "test_micro_f1": ablation_row.get("test_micro_f1", ""),
                                    "test_macro_f1": ablation_row.get("test_macro_f1", ""),
                                    "validation_micro_f1": ablation_row.get("validation_micro_f1", ""),
                                    "validation_macro_f1": ablation_row.get("validation_macro_f1", ""),
                                    "recovery_vs_default_method_micro": "" if ab_micro is None or default_micro in {None, 0.0} else ab_micro / default_micro,
                                    "recovery_vs_full_native_micro": "" if ab_micro is None else ab_micro / native_micro,
                                    "sehgnn_args_json": ab_meta["sehgnn_args_json"],
                                    "notes": notes,
                                }
                            )
                    if spec.budget_strategy == "relation_channel_grid":
                        grid_rows.append(_relation_grid_row(dataset=dataset, spec=spec, graph_seed=graph_seed, training_seed=training_seed, raw_row=raw, retention_rows=retention_rows))
                    direction_rows.extend(_directionality_rows(dataset, spec, raw, retention_rows))
                    if spec.method_family == "feature_cache_adapter_probe" or str(args.preset) == "feature_cache_probe":
                        feature_rows.extend(planned_feature_cache_probe_rows(dataset=dataset, base_graph_method=spec.method, graph_seed=graph_seed, training_seed=training_seed, storage_row=storage))
                except Exception as exc:
                    trace_path = out / "tracebacks" / f"{run_id}.txt"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(traceback.format_exc(), encoding="utf-8")
                    raw_rows.append(
                        _raw_row(
                            dataset=dataset,
                            spec=spec,
                            graph_seed=graph_seed,
                            training_seed=training_seed,
                            run_row={"status": "failed_runtime", "error_message": f"{type(exc).__name__}: {exc}"},
                            storage_row={},
                            manifest={},
                            native=native_by_seed.get(training_seed),
                            export_full=export_by_seed.get(training_seed),
                            traceback_path=str(trace_path),
                        )
                    )
    write_csv(out / "gate21_3_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_3_storage_audit.csv", storage_rows, fieldnames=[*STORAGE_AUDIT_FIELDS, "eligible_for_main_decision"])
    write_csv(out / "gate21_3_relation_mapping_audit.csv", mapping_rows_all, fieldnames=RELATION_MAPPING_FIELDS_21_3)
    write_csv(out / "gate21_3_relation_edge_retention.csv", retention_rows_all, fieldnames=RELATION_RETENTION_FIELDS_21_3)
    write_csv(out / "gate21_3_edge_score_diagnostics.csv", score_rows, fieldnames=EDGE_SCORE_V2_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_3_coverage_diagnostics.csv", coverage_rows, fieldnames=COVERAGE_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_3_relation_channel_grid.csv", grid_rows, fieldnames=RELATION_GRID_FIELDS)
    write_csv(out / "gate21_3_directionality_ablation.csv", direction_rows, fieldnames=DIRECTIONALITY_FIELDS)
    write_csv(out / "gate21_3_feature_cache_compression_probe.csv", feature_rows, fieldnames=FEATURE_CACHE_PROBE_FIELDS)
    write_csv(out / "gate21_3_label_graph_ablation.csv", label_rows, fieldnames=LABEL_GRAPH_ABLATION_FIELDS_21_3)
    write_csv(out / "gate21_3_hgb_export_audit.csv", export_rows)
    _write_weighted_probe(out, dataset)
    summary = summarize_gate21_3(out, out)
    return {"methods": len(specs), "raw_rows": len(raw_rows), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--preset", default="core")
    parser.add_argument("--relation-grid-preset", default="core")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.50, 0.40, 0.35, 0.30])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--official-sehgnn-root", dest="sehgnn_root", type=Path)
    parser.add_argument("--native-cache", type=Path, default=None)
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--gate21-1-root", type=Path, default=DEFAULT_GATE21_1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate21_3_relation_channel"))
    parser.add_argument("--best-relation-spec", default="APPA100-PVVP100-PTTP30")
    parser.add_argument("--base-method", default="H6-struct40-best")
    parser.add_argument("--skip-official-training", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--min-edges-per-relation", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_gate21_3(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
