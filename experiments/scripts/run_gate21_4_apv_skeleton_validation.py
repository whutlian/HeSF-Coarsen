from __future__ import annotations

import argparse
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

from experiments.scripts.gate13_task_first_common import load_hgb_graph
from experiments.scripts.run_gate21_1_sehgnn_schema_edge_budget import _compressed_labels_and_splits, _float, _native_labels, _storage_fields, _unweighted_graph
from experiments.scripts.run_gate21_3_relation_channel import (
    COVERAGE_DIAGNOSTIC_FIELDS,
    DIRECTIONALITY_FIELDS,
    LABEL_GRAPH_ABLATION_FIELDS_21_3,
    RAW_FIELDS as RAW_FIELDS_21_3,
    RELATION_GRID_FIELDS,
    RELATION_MAPPING_FIELDS_21_3,
    RELATION_RETENTION_FIELDS_21_3,
    TARGET_TYPE_BY_DATASET,
    WEIGHTED_ADAPTER_FIELDS,
    MethodSpec,
    _assert_relation_consistency,
    _directionality_rows,
    _mock_relation_rows,
    _prune_graph,
    _raw_row,
    _reference_rows,
    _relation_audit_rows,
    _relation_grid_row,
    _run_id,
    _support_graph,
)
from experiments.scripts.summarize_gate21_4_apv_skeleton_validation import summarize_gate21_4
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.cache_hygiene import CacheNamespace, collect_cache_audit_before_after, compute_export_file_list_hash, file_hashes_for_export, prepare_unique_cache_dir
from hesf_coarsen.eval.official.edge_pruning_baselines import semantic_storage_ratio
from hesf_coarsen.eval.official.feature_cache_compression_probe import FEATURE_CACHE_PROBE_FIELDS
from hesf_coarsen.eval.official.relation_channel_skeleton import APV_SKELETON_EDGE_SCORE_STRATEGY, expand_gate21_4_methods
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command, run_native_command
from hesf_coarsen.eval.official.storage_audit import audit_hgb_directory


DEFAULT_GATE21_1_ROOT = Path("outputs/gate21_1_sehgnn_schema_edge_budget")

RAW_FIELDS_21_4 = [
    "dataset",
    "method",
    "canonical_method",
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
    "cache_hygiene_pass",
    "no_test_label_export_leakage",
    "no_test_label_scoring_leakage",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "effective_total_byte_ratio",
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

RUN_MANIFEST_FIELDS_21_4 = [
    "run_id",
    "dataset",
    "method",
    "canonical_method",
    "run_group",
    "graph_seed",
    "training_seed",
    "relation_channel_spec",
    "relation_direction_spec",
    "paper_feature_transform",
    "feature_compression_method",
    "sehgnn_command_json",
    "export_dir",
    "cache_dir",
    "output_dir",
    "status",
]

CACHE_AUDIT_FIELDS_21_4 = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "export_dir",
    "export_file_list_hash",
    "node_dat_hash",
    "link_dat_hash",
    "label_dat_hash",
    "label_test_dat_hash",
    "info_dat_hash",
    "preprocess_cache_dir",
    "cache_dir_exists_before_run",
    "cache_dir_deleted_before_run",
    "force_reprocess_flag",
    "unique_cache_namespace_flag",
    "cache_files_count_before",
    "cache_files_count_after",
    "cache_hash_before",
    "cache_hash_after",
    "cache_reused_flag",
    "cache_hygiene_pass",
    "sehgnn_generated_feature_cache_keys",
    "sehgnn_generated_label_cache_keys",
    "sehgnn_generated_metapath_keys",
    "notes",
]

STORAGE_FIELDS_21_4 = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "semantic_structural_storage_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "total_node_ratio",
    "total_edge_ratio",
    "hgb_raw_file_byte_ratio",
    "effective_total_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "node_dat_bytes",
    "link_dat_bytes",
    "label_dat_bytes",
    "label_test_dat_bytes",
    "info_dat_bytes",
    "metadata_bytes",
    "feature_sidecar_bytes",
    "feature_sidecar_metadata_bytes",
    "preprocessed_cache_bytes",
    "export_total_bytes",
    "native_full_total_bytes",
    "structural_storage50_pass",
    "structural_storage40_pass",
    "structural_storage30_pass",
    "structural_storage20_pass",
    "raw_hgb_byte50_pass",
    "raw_hgb_byte30_pass",
    "effective_byte50_pass",
    "effective_byte30_pass",
    "cache_byte50_pass",
    "cache_byte30_pass",
    "official_sehgnn_unmodified",
    "adapter_family",
    "eligible_for_main_decision",
]


@dataclass(frozen=True)
class Gate214Spec:
    method: str
    canonical_method: str
    method_family: str
    budget_strategy: str
    edge_score_strategy: str
    relation_channel_spec: str
    storage_budget: float | None = None
    official_sehgnn_unmodified: bool = True
    eligible_for_main_decision: bool = True

    def to_gate21_3(self) -> MethodSpec:
        internal_budget = "relation_channel_grid" if self.relation_channel_spec else self.budget_strategy
        internal_score = "random_edge_within_relation" if self.edge_score_strategy == APV_SKELETON_EDGE_SCORE_STRATEGY else self.edge_score_strategy
        return MethodSpec(
            self.method,
            "gate21_4",
            self.method_family,
            internal_budget,
            internal_score,
            self.relation_channel_spec,
            self.storage_budget,
            eligible_for_main_decision=self.eligible_for_main_decision,
        )


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _method_safe(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(method))


def _specs(args: argparse.Namespace) -> list[Gate214Spec]:
    expanded = expand_gate21_4_methods(args.methods)
    specs = [
        Gate214Spec(
            method=row["method"],
            canonical_method=row["canonical_method"],
            method_family=row["method_family"],
            budget_strategy=row["budget_strategy"],
            edge_score_strategy=row["edge_score_strategy"],
            relation_channel_spec=row["relation_channel_spec"],
            storage_budget=row.get("storage_budget"),
            official_sehgnn_unmodified=bool(row["official_sehgnn_unmodified"]),
            eligible_for_main_decision=bool(row["eligible_for_main_decision"]),
        )
        for row in expanded
    ]
    if args.max_runs is not None:
        specs = specs[: int(args.max_runs)]
    return specs


def _run_pairs(spec: Gate214Spec, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[tuple[int | None, int]]:
    if spec.method_family == "reference_full":
        return [(None, int(seed)) for seed in training_seeds]
    return [(int(g), int(t)) for g in graph_seeds for t in training_seeds]


def _manifest_rows(args: argparse.Namespace, specs: Sequence[Gate214Spec]) -> list[dict[str, Any]]:
    dataset = str(args.dataset).upper()
    out = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for graph_seed, training_seed in _run_pairs(spec, args.graph_seeds, args.training_seeds):
            run_id = _run_id(dataset, spec.method, int(graph_seed or 0), int(training_seed))
            export_dir = out / "exports" / dataset / ("graph_seed_none" if graph_seed is None else f"graph_seed_{graph_seed}") / _method_safe(spec.method)
            cache_dir = Path(args.cache_root or out / "cache") / dataset / _method_safe(spec.method) / ("graph_seed_none" if graph_seed is None else f"graph_seed_{graph_seed}") / f"training_seed_{training_seed}" / "export_hash_planned" / "cache"
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": spec.method,
                    "canonical_method": spec.canonical_method,
                    "run_group": "reference" if spec.method_family == "reference_full" else "apv_skeleton_validation",
                    "graph_seed": "" if graph_seed is None else int(graph_seed),
                    "training_seed": int(training_seed),
                    "relation_channel_spec": spec.relation_channel_spec,
                    "relation_direction_spec": spec.relation_channel_spec if spec.method.startswith("H6-dir-") else "",
                    "paper_feature_transform": "",
                    "feature_compression_method": "",
                    "sehgnn_command_json": "",
                    "export_dir": str(export_dir),
                    "cache_dir": str(cache_dir),
                    "output_dir": str(out / "runs" / dataset / ("graph_seed_none" if graph_seed is None else f"graph_seed_{graph_seed}") / f"training_seed_{training_seed}" / _method_safe(spec.method)),
                    "status": "planned" if (_bool_arg(args.dry_run) or _bool_arg(args.plan_only)) else "pending",
                }
            )
    return rows


def _empty_outputs(out: Path) -> None:
    write_csv(out / "gate21_4_raw_rows.csv", [], fieldnames=RAW_FIELDS_21_4)
    write_csv(out / "gate21_4_by_method.csv", [])
    write_csv(out / "gate21_4_relation_channel_grid.csv", [], fieldnames=RELATION_GRID_FIELDS)
    write_csv(out / "gate21_4_relation_mapping_audit.csv", [], fieldnames=[*RELATION_MAPPING_FIELDS_21_3, "canonical_method"])
    write_csv(out / "gate21_4_relation_edge_retention.csv", [], fieldnames=[*RELATION_RETENTION_FIELDS_21_3, "canonical_method"])
    write_csv(out / "gate21_4_hgb_export_audit.csv", [])
    write_csv(out / "gate21_4_storage_audit.csv", [], fieldnames=STORAGE_FIELDS_21_4)
    write_csv(out / "gate21_4_storage_frontier.csv", [])
    write_csv(out / "gate21_4_graph_seed_stability.csv", [])
    write_csv(out / "gate21_4_cache_audit.csv", [], fieldnames=CACHE_AUDIT_FIELDS_21_4)
    write_csv(out / "gate21_4_directionality_ablation.csv", [], fieldnames=DIRECTIONALITY_FIELDS)
    write_csv(out / "gate21_4_feature_channel_ablation.csv", [])
    write_csv(out / "gate21_4_feature_cache_compression_results.csv", [], fieldnames=FEATURE_CACHE_PROBE_FIELDS)
    write_csv(out / "gate21_4_feature_transform_audit.csv", [])
    write_csv(out / "gate21_4_pathaware_v2_validation.csv", [])
    write_csv(out / "gate21_4_edge_score_diagnostics.csv", [], fieldnames=[])
    write_csv(out / "gate21_4_coverage_diagnostics.csv", [], fieldnames=COVERAGE_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_4_label_graph_ablation.csv", [], fieldnames=LABEL_GRAPH_ABLATION_FIELDS_21_3)
    write_csv(out / "gate21_4_weighted_adapter_probe.csv", [], fieldnames=WEIGHTED_ADAPTER_FIELDS)


def _write_dry_run(args: argparse.Namespace, specs: list[Gate214Spec]) -> dict[str, Any]:
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    manifest = _manifest_rows(args, specs)
    write_csv(out / "gate21_4_run_manifest.csv", manifest, fieldnames=RUN_MANIFEST_FIELDS_21_4)
    raw_rows = []
    for row in manifest:
        raw_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "canonical_method": row["canonical_method"],
                "method_family": next(spec.method_family for spec in specs if spec.method == row["method"]),
                "budget_strategy": next(spec.budget_strategy for spec in specs if spec.method == row["method"]),
                "edge_score_strategy": next(spec.edge_score_strategy for spec in specs if spec.method == row["method"]),
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
                "official_sehgnn_unmodified": True,
                "eligible_for_main_decision": next(spec.eligible_for_main_decision for spec in specs if spec.method == row["method"]),
                "cache_hygiene_pass": "",
                "no_test_label_export_leakage": True,
                "no_test_label_scoring_leakage": True,
            }
        )
    write_csv(out / "gate21_4_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS_21_4)
    gate3_specs = [spec.to_gate21_3() for spec in specs if spec.relation_channel_spec]
    mapping_rows, retention_rows = _mock_relation_rows(str(args.dataset).upper(), gate3_specs, args.graph_seeds, args.training_seeds)
    mapping_rows = [{**row, "canonical_method": _canonical_for(row["method"], specs)} for row in mapping_rows]
    retention_rows = [{**row, "canonical_method": _canonical_for(row["method"], specs)} for row in retention_rows]
    write_csv(out / "gate21_4_relation_mapping_audit.csv", mapping_rows, fieldnames=[*RELATION_MAPPING_FIELDS_21_3, "canonical_method"])
    write_csv(out / "gate21_4_relation_edge_retention.csv", retention_rows, fieldnames=[*RELATION_RETENTION_FIELDS_21_3, "canonical_method"])
    write_json(out / "gate21_4_plan.json", {"dataset": str(args.dataset).upper(), "graph_seeds": args.graph_seeds, "training_seeds": args.training_seeds, "methods": [spec.method for spec in specs], "dry_run": True, "plan_only": bool(args.plan_only)})
    summary = summarize_gate21_4(out, out)
    return {"dry_run": True, "planned_runs": len(manifest), "summary": summary}


def _canonical_for(method: str, specs: Sequence[Gate214Spec]) -> str:
    return next((spec.canonical_method for spec in specs if spec.method == method), method)


def _raw21_4(raw21_3: Mapping[str, Any], spec: Gate214Spec, cache_pass: bool | str = "") -> dict[str, Any]:
    row = {field: raw21_3.get(field, "") for field in RAW_FIELDS_21_3}
    row["canonical_method"] = spec.canonical_method
    row["budget_strategy"] = spec.budget_strategy
    row["edge_score_strategy"] = spec.edge_score_strategy
    row["official_sehgnn_unmodified"] = spec.official_sehgnn_unmodified
    row["eligible_for_main_decision"] = spec.eligible_for_main_decision
    row["cache_hygiene_pass"] = cache_pass
    row["effective_total_byte_ratio"] = ""
    return {field: row.get(field, "") for field in RAW_FIELDS_21_4}


def _storage21_4(row: Mapping[str, Any], spec: Gate214Spec, graph_seed: int, training_seed: int) -> dict[str, Any]:
    semantic = _float(row.get("semantic_structural_storage_ratio"))
    raw = _float(row.get("hgb_raw_file_byte_ratio"))
    cache = _float(row.get("preprocessed_cache_byte_ratio"))
    effective = raw
    return {
        "dataset": row.get("dataset", ""),
        "method": spec.method,
        "canonical_method": spec.canonical_method,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
        "support_node_ratio": row.get("support_node_ratio", ""),
        "support_edge_ratio": row.get("support_edge_ratio", ""),
        "total_node_ratio": row.get("total_node_ratio", ""),
        "total_edge_ratio": row.get("total_edge_ratio", ""),
        "hgb_raw_file_byte_ratio": row.get("hgb_raw_file_byte_ratio", ""),
        "effective_total_byte_ratio": effective if effective is not None else "",
        "preprocessed_cache_byte_ratio": row.get("preprocessed_cache_byte_ratio", ""),
        "node_dat_bytes": row.get("node_dat_bytes", ""),
        "link_dat_bytes": row.get("link_dat_bytes", ""),
        "label_dat_bytes": row.get("label_dat_bytes", ""),
        "label_test_dat_bytes": row.get("label_test_dat_bytes", ""),
        "info_dat_bytes": row.get("info_dat_bytes", ""),
        "metadata_bytes": row.get("metadata_sidecar_bytes", ""),
        "feature_sidecar_bytes": 0,
        "feature_sidecar_metadata_bytes": 0,
        "preprocessed_cache_bytes": row.get("export_preprocessed_cache_bytes", ""),
        "export_total_bytes": row.get("export_total_bytes", ""),
        "native_full_total_bytes": row.get("native_full_total_bytes", ""),
        "structural_storage50_pass": _le(semantic, 0.50),
        "structural_storage40_pass": _le(semantic, 0.40),
        "structural_storage30_pass": _le(semantic, 0.30),
        "structural_storage20_pass": _le(semantic, 0.20),
        "raw_hgb_byte50_pass": _le(raw, 0.50),
        "raw_hgb_byte30_pass": _le(raw, 0.30),
        "effective_byte50_pass": _le(effective, 0.50),
        "effective_byte30_pass": _le(effective, 0.30),
        "cache_byte50_pass": _le(cache, 0.50),
        "cache_byte30_pass": _le(cache, 0.30),
        "official_sehgnn_unmodified": spec.official_sehgnn_unmodified,
        "adapter_family": "",
        "eligible_for_main_decision": spec.eligible_for_main_decision,
    }


def _le(value: float | None, threshold: float) -> bool | str:
    return "" if value is None else bool(value <= threshold)


def run_gate21_4(args: argparse.Namespace) -> dict[str, Any]:
    specs = _specs(args)
    if _bool_arg(args.dry_run) or _bool_arg(args.plan_only):
        return _write_dry_run(args, specs)
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    dataset = str(args.dataset).upper()
    if dataset != "DBLP":
        raise ValueError("Gate21.4 remains DBLP-only")
    manifest_rows = _manifest_rows(args, specs)
    write_csv(out / "gate21_4_run_manifest.csv", manifest_rows, fieldnames=RUN_MANIFEST_FIELDS_21_4)
    write_json(out / "gate21_4_plan.json", {"dataset": dataset, "graph_seeds": args.graph_seeds, "training_seeds": args.training_seeds, "methods": [spec.method for spec in specs], "dry_run": False})
    native_by_seed, export_by_seed, reference_rows = _reference_rows(Path(args.gate21_1_root), dataset)
    raw_rows = [_raw21_4(row, next((spec for spec in specs if spec.method == row.get("method")), Gate214Spec(str(row.get("method")), str(row.get("method")), "reference_full", "reference", "reference", "", eligible_for_main_decision=False)), True) for row in reference_rows if row.get("method") in {spec.method for spec in specs}]
    storage_rows: list[dict[str, Any]] = []
    mapping_rows_all: list[dict[str, Any]] = []
    retention_rows_all: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    original = load_hgb_graph(Path(args.data_root), dataset)
    target_type = int(infer_target_node_type(original))
    labels_native, trainval_native, test_native = _native_labels(Path(args.hgb_data_root), dataset, original.num_nodes)
    support_cache = {int(seed): _support_graph(original, "H6", graph_seed=int(seed), candidate_k=int(args.candidate_k)) for seed in args.graph_seeds}
    for spec in specs:
        if spec.method_family == "reference_full":
            continue
        for graph_seed, training_seed in _run_pairs(spec, args.graph_seeds, args.training_seeds):
            assert graph_seed is not None
            gate3_spec = spec.to_gate21_3()
            try:
                base_graph, assignment = support_cache[int(graph_seed)]
                if spec.method == "H6-node30":
                    graph = base_graph
                    candidate_counts = {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()}
                    retained_counts = dict(candidate_counts)
                    requested_budgets = dict(candidate_counts)
                    min_active = {int(rid): False for rid in graph.relations}
                    score_rows = []
                    coverage_rows = []
                else:
                    graph, candidate_counts, retained_counts, requested_budgets, min_active, score_rows, coverage_rows = _prune_graph(
                        graph=base_graph,
                        original=original,
                        dataset=dataset,
                        spec=gate3_spec,
                        graph_seed=int(graph_seed),
                        target_type=target_type,
                        train_idx=trainval_native,
                        labels=labels_native,
                        min_edges_per_relation=int(args.min_edges_per_relation),
                    )
                labels, trainval, test = _compressed_labels_and_splits(
                    original=original,
                    graph=graph,
                    assignment=assignment,
                    target_type=target_type,
                    labels_native=labels_native,
                    trainval_native=trainval_native,
                    test_native=test_native,
                )
                manifest = export_graph_to_sehgnn_hgb(
                    graph=graph,
                    dataset_name=dataset,
                    target_type=TARGET_TYPE_BY_DATASET[dataset],
                    output_dir=out / "exports" / dataset / f"graph_seed_{graph_seed}",
                    split_mode="official_trainval",
                    train_idx=trainval,
                    val_idx=np.array([], dtype=np.int64),
                    test_idx=test,
                    labels=labels,
                    method_name=spec.method,
                    seed=int(graph_seed),
                )
                export_dir = Path(manifest["export_dir"])
                export_audit = audit_native_hgb_data_dir(dataset, export_dir.parent, Path(args.sehgnn_root))
                export_hash = str(manifest.get("file_list_hash") or manifest.get("export_hash") or compute_export_file_list_hash(export_dir))
                namespace = CacheNamespace(dataset, spec.method, int(graph_seed), int(training_seed), export_hash, Path(args.cache_root or out / "cache"))
                before_cache = prepare_unique_cache_dir(namespace, force_reprocess=_bool_arg(args.force_reprocess))
                semantic_ratio = semantic_storage_ratio(graph, original)
                storage_base = _storage_fields(original, graph, target_type, Path(args.hgb_data_root) / dataset, export_dir)
                storage_gate3 = audit_hgb_directory(
                    dataset=dataset,
                    method=spec.method,
                    seed=int(graph_seed),
                    export_dir=export_dir,
                    native_full_dir=Path(args.hgb_data_root) / dataset,
                    semantic_structural_storage_ratio=float(semantic_ratio),
                    support_node_ratio=_float(storage_base.get("support_node_ratio")),
                    support_edge_ratio=_float(storage_base.get("support_edge_ratio")),
                    total_node_ratio=_float(storage_base.get("total_node_ratio")),
                    total_edge_ratio=_float(storage_base.get("total_edge_ratio")),
                    structural_budget=spec.storage_budget,
                    raw_byte_budget=0.50,
                    cache_dir=namespace.cache_dir,
                ).to_row(method_family=spec.method_family)
                mapping_rows, retention_rows = _relation_audit_rows(
                    graph=graph,
                    original=original,
                    dataset=dataset,
                    spec=gate3_spec,
                    graph_seed=int(graph_seed),
                    training_seed=int(training_seed),
                    candidate_counts=candidate_counts,
                    retained_counts=retained_counts,
                    requested_budgets=requested_budgets,
                    min_active=min_active,
                )
                _assert_relation_consistency(export_dir=export_dir, export_audit=export_audit, retention_rows=retention_rows, spec=gate3_spec)
                command = build_official_hgb_command(
                    dataset=dataset,
                    seed=int(training_seed),
                    repo_dir=Path(args.sehgnn_root),
                    data_root=export_dir.parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                if _bool_arg(args.skip_official_training):
                    run_row = {"dataset": dataset, "seed": int(training_seed), "status": "skipped", "error_message": "skip_official_training"}
                else:
                    run_dir = out / "runs" / dataset / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}" / _method_safe(spec.method)
                    run_row = run_native_command(command, stdout_path=run_dir / "stdout.log", stderr_path=run_dir / "stderr.log")
                cache_after = collect_cache_audit_before_after(namespace.cache_dir, before_cache)
                cache_row = {
                    "dataset": dataset,
                    "method": spec.method,
                    "canonical_method": spec.canonical_method,
                    "graph_seed": int(graph_seed),
                    "training_seed": int(training_seed),
                    "export_dir": str(export_dir),
                    **file_hashes_for_export(export_dir),
                    **cache_after,
                }
                cache_rows.append({field: cache_row.get(field, "") for field in CACHE_AUDIT_FIELDS_21_4})
                raw_gate3 = _raw_row(
                    dataset=dataset,
                    spec=gate3_spec,
                    graph_seed=int(graph_seed),
                    training_seed=int(training_seed),
                    run_row=run_row,
                    storage_row=storage_gate3,
                    manifest={**manifest, **export_audit, "file_list_hash": export_hash},
                    native=native_by_seed.get(int(training_seed)),
                    export_full=export_by_seed.get(int(training_seed)),
                )
                raw_rows.append(_raw21_4(raw_gate3, spec, cache_after["cache_hygiene_pass"]))
                storage_rows.append(_storage21_4(storage_gate3, spec, int(graph_seed), int(training_seed)))
                mapping_rows_all.extend({**row, "canonical_method": spec.canonical_method} for row in mapping_rows)
                retention_rows_all.extend({**row, "canonical_method": spec.canonical_method} for row in retention_rows)
                export_rows.append({**manifest, **export_audit, "method": spec.method, "canonical_method": spec.canonical_method, "graph_seed": int(graph_seed)})
                if spec.relation_channel_spec:
                    grid_rows.append(_relation_grid_row(dataset=dataset, spec=gate3_spec, graph_seed=int(graph_seed), training_seed=int(training_seed), raw_row=raw_gate3, retention_rows=retention_rows))
                direction_rows.extend(_directionality_rows(dataset, gate3_spec, raw_gate3, retention_rows))
                _write_score_coverage_sidecars(out, score_rows, coverage_rows)
            except Exception as exc:
                trace_path = out / "tracebacks" / f"{_run_id(dataset, spec.method, int(graph_seed), int(training_seed))}.txt"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(traceback.format_exc(), encoding="utf-8")
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "method": spec.method,
                        "canonical_method": spec.canonical_method,
                        "method_family": spec.method_family,
                        "budget_strategy": spec.budget_strategy,
                        "edge_score_strategy": spec.edge_score_strategy,
                        "relation_channel_spec": spec.relation_channel_spec,
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "run_id": _run_id(dataset, spec.method, int(graph_seed), int(training_seed)),
                        "success": False,
                        "status": "failed_runtime",
                        "failed_reason": f"{type(exc).__name__}: {exc}",
                        "traceback_path": str(trace_path),
                        "official_sehgnn_unmodified": spec.official_sehgnn_unmodified,
                        "eligible_for_main_decision": spec.eligible_for_main_decision,
                    }
                )
                if _bool_arg(args.fail_fast):
                    raise
    write_csv(out / "gate21_4_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS_21_4)
    write_csv(out / "gate21_4_storage_audit.csv", storage_rows, fieldnames=STORAGE_FIELDS_21_4)
    write_csv(out / "gate21_4_relation_mapping_audit.csv", mapping_rows_all, fieldnames=[*RELATION_MAPPING_FIELDS_21_3, "canonical_method"])
    write_csv(out / "gate21_4_relation_edge_retention.csv", retention_rows_all, fieldnames=[*RELATION_RETENTION_FIELDS_21_3, "canonical_method"])
    write_csv(out / "gate21_4_hgb_export_audit.csv", export_rows)
    write_csv(out / "gate21_4_cache_audit.csv", cache_rows, fieldnames=CACHE_AUDIT_FIELDS_21_4)
    write_csv(out / "gate21_4_relation_channel_grid.csv", grid_rows, fieldnames=RELATION_GRID_FIELDS)
    write_csv(out / "gate21_4_directionality_ablation.csv", direction_rows, fieldnames=DIRECTIONALITY_FIELDS)
    _rewrite_manifest_with_results(out, args, manifest_rows, raw_rows, cache_rows)
    summary = summarize_gate21_4(out, out)
    return {"methods": len(specs), "raw_rows": len(raw_rows), "summary": summary}


def _rewrite_manifest_with_results(
    out: Path,
    args: argparse.Namespace,
    manifest_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
) -> None:
    raw_by_key = {_manifest_key(row): row for row in raw_rows}
    cache_by_key = {_manifest_key(row): row for row in cache_rows}
    final_rows = []
    for row in manifest_rows:
        key = _manifest_key(row)
        raw = raw_by_key.get(key, {})
        cache = cache_by_key.get(key, {})
        export_dir = str(cache.get("export_dir") or row.get("export_dir") or "")
        cache_dir = str(cache.get("preprocess_cache_dir") or row.get("cache_dir") or "")
        command_json = ""
        if export_dir:
            try:
                command = build_official_hgb_command(
                    dataset=str(row.get("dataset", args.dataset)),
                    seed=int(row.get("training_seed") or 0),
                    repo_dir=Path(args.sehgnn_root),
                    data_root=Path(export_dir).parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                command_json = json.dumps({"command": list(command.command), "cwd": str(command.cwd), "dataset": command.dataset, "seed": int(command.seed)}, sort_keys=True)
            except Exception:
                command_json = ""
        final_rows.append(
            {
                **row,
                "sehgnn_command_json": command_json or row.get("sehgnn_command_json", ""),
                "export_dir": export_dir,
                "cache_dir": cache_dir,
                "status": raw.get("status") or row.get("status", ""),
            }
        )
    write_csv(out / "gate21_4_run_manifest.csv", final_rows, fieldnames=RUN_MANIFEST_FIELDS_21_4)


def _manifest_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("method", "")),
        str(row.get("graph_seed", "")),
        str(row.get("training_seed", "")),
    )


def _write_score_coverage_sidecars(out: Path, score_rows: Sequence[Mapping[str, Any]], coverage_rows: Sequence[Mapping[str, Any]]) -> None:
    if score_rows:
        write_csv(out / "gate21_4_edge_score_diagnostics.csv", score_rows)
    if coverage_rows:
        write_csv(out / "gate21_4_coverage_diagnostics.csv", coverage_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unique-cache-namespace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-sanity-check", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--min-edges-per-relation", type=int, default=1)
    parser.add_argument("--allow-zero-edge-relation", action="store_true")
    parser.add_argument("--reuse-gate21-0-full-baseline", type=Path, default=None)
    parser.add_argument("--reuse-h6-node30-export", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-official-training", action="store_true")
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--gate21-1-root", type=Path, default=DEFAULT_GATE21_1_ROOT)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_gate21_4(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
