from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate21_1_sehgnn_schema_edge_budget import (
    _compressed_labels_and_splits,
    _edge_count,
    _float,
    _native_labels,
    _storage_fields,
    _unweighted_graph,
)
from experiments.scripts.summarize_gate21_2_h6_path_budget import summarize_gate21_2
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.edge_pruning_baselines import (
    edge_budget_for_storage,
    prune_relationwise,
    semantic_storage_ratio,
)
from hesf_coarsen.eval.official.label_graph_ablation import LABEL_GRAPH_ABLATION_FIELDS, planned_label_graph_ablation_rows
from hesf_coarsen.eval.official.path_aware_edge_scorer import EDGE_SCORE_DIAGNOSTIC_FIELDS
from hesf_coarsen.eval.official.relation_mapping_audit import (
    RELATION_MAPPING_FIELDS,
    RELATION_RETENTION_FIELDS,
    audit_relation_mapping,
)
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.schema_stable_pruning import build_target_only_schema_stub_graph
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command, run_native_command
from hesf_coarsen.eval.official.storage_audit import STORAGE_AUDIT_FIELDS, audit_hgb_directory
from hesf_coarsen.io.schema import HeteroGraph


TARGET_TYPE_BY_DATASET = {"DBLP": "A", "ACM": "P", "IMDB": "M"}
BASELINE_BY_SELECTOR = {"H6": "H6-no-spec-support-only", "flatten": "flatten-sum-support-only"}
DEFAULT_GATE21_1_ROOT = Path("outputs/gate21_1_sehgnn_schema_edge_budget")

RAW_FIELDS = [
    "dataset",
    "seed",
    "model_name",
    "method",
    "method_family",
    "budget_strategy",
    "edge_score_strategy",
    "schema_compatible",
    "uses_weighted_superedges",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "no_test_label_export_leakage",
    "no_test_label_scoring_leakage",
    "success",
    "status",
    "error_type",
    "error_message",
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
    "recovery_vs_export_full_micro",
    "recovery_vs_export_full_macro",
    "mapping_bijective",
    "split_disjoint",
    "relation_order_matches_official",
    "node_type_order_matches_official",
    "schema_complete",
    "stdout_path",
    "stderr_path",
]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    selector: str
    storage_budget: float | None
    budget_strategy: str
    edge_score_strategy: str
    method_family: str = "schema_compatible_subgraph"
    relation_pair_weights: dict[str, float] | None = None

    @property
    def eligible(self) -> bool:
        return self.method_family == "schema_compatible_subgraph"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _budget_token(value: float) -> str:
    return str(int(round(float(value) * 100)))


def _method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    budgets = [float(v) for v in args.storage_budgets]
    budget_strategies = {str(v) for v in args.budget_strategies}
    edge_strategies = {str(v) for v in args.edge_score_strategies}
    specs: list[MethodSpec] = []
    for budget in budgets:
        token = _budget_token(budget)
        if "proportional" in budget_strategies and "current_heuristic" in edge_strategies:
            specs.append(MethodSpec(f"H6-struct{token}-proportional-current", "H6", budget, "proportional", "current_heuristic"))
        if "path_aware" in budget_strategies and "path_aware" in edge_strategies:
            specs.append(MethodSpec(f"H6-struct{token}-pathaware", "H6", budget, "path_aware", "path_aware"))
        if budget in {0.50, 0.40, 0.30} and "pair_grid" in budget_strategies and "path_aware" in edge_strategies:
            specs.append(
                MethodSpec(
                    f"H6-pairgrid-struct{token}-AP50-PT30-PV20",
                    "H6",
                    budget,
                    "pair_grid",
                    "path_aware",
                    relation_pair_weights={"AP_PA": 0.50, "PT_TP": 0.30, "PV_VP": 0.20},
                )
            )
        if budget in {0.50, 0.40, 0.30} and "random_relationwise" in budget_strategies and "random" in edge_strategies:
            specs.append(MethodSpec(f"H6-struct{token}-random-relwise", "H6", budget, "random_relationwise", "random"))
        if budget in {0.50, 0.40, 0.30} and "degree_topk_relationwise" in budget_strategies and "degree" in edge_strategies:
            specs.append(MethodSpec(f"H6-struct{token}-degree-relwise", "H6", budget, "degree_topk_relationwise", "degree"))
    if "proportional" in budget_strategies and "current_heuristic" in edge_strategies:
        for budget in budgets:
            if round(budget, 2) in {0.50, 0.30}:
                specs.append(MethodSpec(f"flatten-struct{_budget_token(budget)}-proportional-current", "flatten", budget, "proportional", "current_heuristic"))
    specs.append(MethodSpec("H6-node30", "H6", None, "node30", "none"))
    specs.append(MethodSpec("target-only-schema-stub", "target-only", None, "diagnostic", "none", method_family="schema_stub_diagnostic"))
    if args.max_runs is not None:
        trainable = [spec for spec in specs if spec.method_family != "schema_stub_diagnostic"][: int(args.max_runs)]
        diagnostics = [spec for spec in specs if spec.method_family == "schema_stub_diagnostic"]
        specs = trainable + diagnostics
    return specs


def _empty_row(dataset: str, seed: int, spec: MethodSpec, *, status: str = "dry_run_planned") -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model_name": "official-SeHGNN",
        "method": spec.method,
        "method_family": spec.method_family,
        "budget_strategy": spec.budget_strategy,
        "edge_score_strategy": spec.edge_score_strategy,
        "schema_compatible": spec.method_family != "weighted_coarse_graph",
        "uses_weighted_superedges": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": spec.eligible,
        "no_test_label_export_leakage": True,
        "no_test_label_scoring_leakage": True,
        "success": False,
        "status": status,
        "error_type": "",
        "error_message": "",
    }


def _reference_rows_from_gate21_1(root: Path, dataset: str, seeds: Sequence[int]) -> tuple[list[dict[str, Any]], dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    raw = _read_csv(Path(root) / "gate21_1_raw_rows.csv")
    native_by_seed: dict[int, Mapping[str, Any]] = {}
    export_by_seed: dict[int, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in raw:
        if str(row.get("dataset", "")) != dataset or int(row.get("seed", -1)) not in set(int(v) for v in seeds):
            continue
        method = str(row.get("method", ""))
        if method == "full-native-SeHGNN":
            native_by_seed[int(row["seed"])] = row
        elif method == "export-full-SeHGNN":
            export_by_seed[int(row["seed"])] = row
        else:
            continue
        rows.append(_reference_row(row, method_family="reference_full"))
    return rows, native_by_seed, export_by_seed


def _reference_row(row: Mapping[str, Any], *, method_family: str) -> dict[str, Any]:
    status = str(row.get("status", ""))
    return {
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "model_name": "official-SeHGNN",
        "method": row.get("method", ""),
        "method_family": method_family,
        "budget_strategy": "reference",
        "edge_score_strategy": "reference",
        "schema_compatible": True,
        "uses_weighted_superedges": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": False,
        "no_test_label_export_leakage": True,
        "no_test_label_scoring_leakage": True,
        "success": status == "success",
        "status": status,
        "error_type": "" if status == "success" else status,
        "error_message": row.get("error", row.get("error_message", "")),
        "validation_micro_f1": row.get("validation_micro_f1", ""),
        "validation_macro_f1": row.get("validation_macro_f1", ""),
        "test_micro_f1": row.get("test_micro_f1", ""),
        "test_macro_f1": row.get("test_macro_f1", ""),
        "test_accuracy_if_single_label": row.get("test_accuracy", row.get("test_accuracy_if_single_label", "")),
        "native_full_test_micro_f1": row.get("native_full_test_micro_f1", row.get("test_micro_f1", "")),
        "native_full_test_macro_f1": row.get("native_full_test_macro_f1", row.get("test_macro_f1", "")),
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


def _support_graph(original: HeteroGraph, selector: str, *, seed: int, candidate_k: int) -> tuple[HeteroGraph, np.ndarray]:
    graph, assignment, _diag = run_support_baseline(
        original,
        baseline=BASELINE_BY_SELECTOR[str(selector)],
        ratio=0.30,
        seed=int(seed),
        candidate_k=int(candidate_k),
    )
    return _unweighted_graph(graph), np.asarray(assignment, dtype=np.int64)


def _storage_reference(native_by_seed: Mapping[int, Mapping[str, Any]], seed: int, metric: str) -> float | None:
    return _float(native_by_seed.get(int(seed), {}).get(metric))


def _run_row_to_raw(
    *,
    dataset: str,
    seed: int,
    spec: MethodSpec,
    run_row: Mapping[str, Any],
    storage_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    native: Mapping[str, Any],
    export_full: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(run_row.get("status", ""))
    test_micro = _float(run_row.get("test_micro_f1"))
    test_macro = _float(run_row.get("test_macro_f1"))
    native_micro = _float(native.get("test_micro_f1"))
    native_macro = _float(native.get("test_macro_f1"))
    export_micro = _float(export_full.get("test_micro_f1"))
    export_macro = _float(export_full.get("test_macro_f1"))
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model_name": "official-SeHGNN",
        "method": spec.method,
        "method_family": spec.method_family,
        "budget_strategy": spec.budget_strategy,
        "edge_score_strategy": spec.edge_score_strategy,
        "schema_compatible": spec.method_family != "weighted_coarse_graph",
        "uses_weighted_superedges": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": spec.eligible,
        "no_test_label_export_leakage": manifest.get("no_test_label_export_leakage", True),
        "no_test_label_scoring_leakage": True,
        "success": status == "success",
        "status": status,
        "error_type": "" if status == "success" else status,
        "error_message": run_row.get("error_message", ""),
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
        "native_full_test_micro_f1": "" if native_micro is None else native_micro,
        "native_full_test_macro_f1": "" if native_macro is None else native_macro,
        "export_full_test_micro_f1": "" if export_micro is None else export_micro,
        "export_full_test_macro_f1": "" if export_macro is None else export_macro,
        "recovery_vs_native_full_micro": "" if native_micro in {None, 0.0} or test_micro is None else test_micro / native_micro,
        "recovery_vs_native_full_macro": "" if native_macro in {None, 0.0} or test_macro is None else test_macro / native_macro,
        "recovery_vs_export_full_micro": "" if export_micro in {None, 0.0} or test_micro is None else test_micro / export_micro,
        "recovery_vs_export_full_macro": "" if export_macro in {None, 0.0} or test_macro is None else test_macro / export_macro,
        "mapping_bijective": manifest.get("mapping_bijective", ""),
        "split_disjoint": manifest.get("split_disjoint", ""),
        "relation_order_matches_official": manifest.get("relation_order_matches_official", ""),
        "node_type_order_matches_official": manifest.get("node_type_order_matches_official", ""),
        "schema_complete": manifest.get("can_load_with_official_data_loader", manifest.get("schema_complete", True)),
        "stdout_path": run_row.get("stdout_path", ""),
        "stderr_path": run_row.get("stderr_path", ""),
    }


def _write_dry_run(args: argparse.Namespace, specs: list[MethodSpec]) -> dict[str, Any]:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in args.seeds]
    dataset = str(args.dataset).upper()
    raw_rows = [
        _empty_row(dataset, seed, MethodSpec("full-native-SeHGNN", "reference", None, "reference", "reference", method_family="reference_full"))
        for seed in seeds
    ]
    raw_rows.extend(
        _empty_row(dataset, seed, MethodSpec("export-full-SeHGNN", "reference", None, "reference", "reference", method_family="reference_full"))
        for seed in seeds
    )
    raw_rows.extend(_empty_row(dataset, seed, spec) for spec in specs for seed in seeds)
    write_csv(out / "gate21_2_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_2_storage_audit.csv", [], fieldnames=[*STORAGE_AUDIT_FIELDS, "eligible_for_main_decision"])
    write_csv(out / "gate21_2_relation_mapping_audit.csv", [], fieldnames=RELATION_MAPPING_FIELDS)
    write_csv(out / "gate21_2_relation_edge_retention.csv", [], fieldnames=RELATION_RETENTION_FIELDS)
    write_csv(out / "gate21_2_edge_score_diagnostics.csv", [], fieldnames=EDGE_SCORE_DIAGNOSTIC_FIELDS)
    ablation_methods = ["full-native-SeHGNN", "H6-struct50-best", "H6-struct30-best", "target-only-schema-stub"]
    write_csv(out / "gate21_2_label_graph_ablation.csv", planned_label_graph_ablation_rows(dataset=dataset, seeds=seeds, methods=ablation_methods), fieldnames=LABEL_GRAPH_ABLATION_FIELDS)
    write_csv(out / "gate21_2_feature_cache_compression_probe.csv", [], fieldnames=["dataset", "seed", "method", "feature_compression_method", "feature_dtype", "feature_dim", "feature_storage_ratio", "raw_hgb_byte_ratio", "preprocessed_cache_byte_ratio", "test_micro_f1", "test_macro_f1", "eligible_for_main_decision"])
    write_csv(out / "gate21_2_weighted_edge_audit.csv", [], fieldnames=["dataset", "method", "seed", "official_preprocess_preserves_edge_values"])
    write_json(
        out / "gate21_2_plan.json",
        {
            "dataset": dataset,
            "seeds": seeds,
            "storage_budgets": [float(v) for v in args.storage_budgets],
            "methods": [spec.method for spec in specs],
            "dry_run": True,
            "allocator_sanity": "planned",
        },
    )
    summary = summarize_gate21_2(out, out)
    return {"dry_run": True, "planned_methods": len(specs), "summary": summary}


def run_gate21_2(args: argparse.Namespace) -> dict[str, Any]:
    specs = _method_specs(args)
    if _bool_arg(args.dry_run):
        return _write_dry_run(args, specs)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dataset = str(args.dataset).upper()
    if dataset != "DBLP":
        raise ValueError("Gate21.2 is intentionally DBLP-only")
    seeds = [int(v) for v in args.seeds]
    raw_rows, native_by_seed, export_by_seed = _reference_rows_from_gate21_1(Path(args.gate21_1_root), dataset, seeds)
    if not native_by_seed or not export_by_seed:
        raise RuntimeError("Gate21.2 requires Gate21.1 native/export-full reference rows; run Gate21.1 first or set --gate21-1-root")
    weighted_src = Path(args.gate21_1_root) / "diagnostics" / "gate21_1_weighted_edge_audit.csv"
    if weighted_src.exists():
        shutil.copyfile(weighted_src, out / "gate21_2_weighted_edge_audit.csv")
    else:
        write_csv(out / "gate21_2_weighted_edge_audit.csv", [], fieldnames=["dataset", "method", "seed", "official_preprocess_preserves_edge_values"])
    graph_seed = int(seeds[0])
    original = load_hgb_graph(Path(args.data_root), dataset)
    target_type = int(infer_target_node_type(original))
    labels_native, trainval_native, test_native = _native_labels(Path(args.hgb_data_root), dataset, original.num_nodes)
    h6_graph, h6_assignment = _support_graph(original, "H6", seed=graph_seed, candidate_k=int(args.candidate_k))
    flatten_graph, flatten_assignment = _support_graph(original, "flatten", seed=graph_seed, candidate_k=int(args.candidate_k))
    support_cache = {"H6": (h6_graph, h6_assignment), "flatten": (flatten_graph, flatten_assignment)}
    storage_rows: list[dict[str, Any]] = []
    relation_mapping_rows: list[dict[str, Any]] = []
    relation_retention_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    stdout_dir = out / "logs"
    export_root = out / "exports"
    for spec in specs:
        if spec.method == "target-only-schema-stub":
            graph, stub_audit = build_target_only_schema_stub_graph(graph=original, dataset_name=dataset, target_type=TARGET_TYPE_BY_DATASET[dataset])
            labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
            trainval = np.asarray(trainval_native, dtype=np.int64)
            test = np.asarray(test_native, dtype=np.int64)
            candidate_counts = {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()}
            retained_counts = dict(candidate_counts)
            requested_budgets = dict(candidate_counts)
            min_active = {int(rid): False for rid in graph.relations}
            semantic_ratio = semantic_storage_ratio(graph, original)
        else:
            base_graph, assignment = support_cache[spec.selector]
            if spec.storage_budget is None:
                graph = base_graph
                candidate_counts = {int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()}
                retained_counts = dict(candidate_counts)
                requested_budgets = dict(candidate_counts)
                min_active = {int(rid): False for rid in graph.relations}
                score_rows.extend([])
            else:
                total_budget = edge_budget_for_storage(original, base_graph, float(spec.storage_budget))
                pruned = prune_relationwise(
                    graph=base_graph,
                    dataset=dataset,
                    method=spec.method,
                    total_edge_budget=total_budget,
                    budget_strategy=spec.budget_strategy,
                    edge_score_strategy=spec.edge_score_strategy,
                    seed=graph_seed,
                    train_idx=trainval_native,
                    val_idx=np.array([], dtype=np.int64),
                    labels=labels_native,
                    features_by_type=base_graph.features,
                    min_edges_per_relation=int(args.min_edges_per_relation),
                    relation_pair_weights=spec.relation_pair_weights,
                    target_type=target_type,
                )
                graph = pruned.graph
                candidate_counts = pruned.candidate_edge_counts
                retained_counts = pruned.retained_edge_counts
                requested_budgets = pruned.requested_relation_budgets
                min_active = pruned.min_edges_constraint_active
                for diag in pruned.edge_score_diagnostics:
                    diag["method"] = spec.method
                    score_rows.append(diag)
            labels, trainval, test = _compressed_labels_and_splits(
                original=original,
                graph=graph,
                assignment=assignment,
                target_type=target_type,
                labels_native=labels_native,
                trainval_native=trainval_native,
                test_native=test_native,
            )
            semantic_ratio = semantic_storage_ratio(graph, original)
        manifest = export_graph_to_sehgnn_hgb(
            graph=graph,
            dataset_name=dataset,
            target_type=TARGET_TYPE_BY_DATASET[dataset],
            output_dir=export_root,
            split_mode="official_trainval",
            train_idx=trainval,
            val_idx=np.array([], dtype=np.int64),
            test_idx=test,
            labels=labels,
            method_name=spec.method,
            seed=graph_seed,
        )
        export_audit = audit_native_hgb_data_dir(dataset, Path(manifest["export_dir"]).parent, Path(args.official_sehgnn_root))
        export_rows.append({**manifest, **export_audit, "method": spec.method})
        storage_base = _storage_fields(original, graph, target_type, Path(args.hgb_data_root) / dataset, Path(manifest["export_dir"]))
        storage = audit_hgb_directory(
            dataset=dataset,
            method=spec.method,
            seed=graph_seed,
            export_dir=Path(manifest["export_dir"]),
            native_full_dir=Path(args.hgb_data_root) / dataset,
            semantic_structural_storage_ratio=float(semantic_ratio),
            support_node_ratio=_float(storage_base.get("support_node_ratio")),
            support_edge_ratio=_float(storage_base.get("support_edge_ratio")),
            total_node_ratio=_float(storage_base.get("total_node_ratio")),
            total_edge_ratio=_float(storage_base.get("total_edge_ratio")),
            structural_budget=spec.storage_budget,
            raw_byte_budget=args.raw_byte_budget,
        ).to_row(method_family=spec.method_family)
        storage["eligible_for_main_decision"] = spec.eligible
        storage_rows.append(storage)
        mapping = audit_relation_mapping(
            graph=graph,
            dataset=dataset,
            method=spec.method,
            seed=graph_seed,
            candidate_edge_counts=candidate_counts,
            retained_edge_counts=retained_counts,
            original_full_edge_counts={int(rid): int(rel.num_edges) for rid, rel in original.relations.items()},
            min_edges_constraint_active=min_active,
        )
        for row in mapping:
            relation_mapping_rows.append(row.to_row())
            relation_retention_rows.append(
                row.to_retention_row(
                    budget_strategy=spec.budget_strategy,
                    edge_score_strategy=spec.edge_score_strategy,
                    requested_relation_budget=requested_budgets.get(int(row.source_relation_id)) if row.source_relation_id is not None else None,
                    actual_relation_budget=retained_counts.get(int(row.source_relation_id)) if row.source_relation_id is not None else None,
                )
            )
        for seed in seeds:
            if _bool_arg(args.skip_official_training):
                run_row = {"dataset": dataset, "seed": seed, "status": "skipped", "error_message": "skip_official_training"}
            else:
                command = build_official_hgb_command(
                    dataset=dataset,
                    seed=seed,
                    repo_dir=Path(args.official_sehgnn_root),
                    data_root=Path(manifest["export_dir"]).parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                run_row = run_native_command(
                    command,
                    stdout_path=stdout_dir / dataset / spec.method / f"{seed}.log",
                    stderr_path=stdout_dir / dataset / spec.method / f"{seed}.stderr",
                )
            raw_rows.append(
                _run_row_to_raw(
                    dataset=dataset,
                    seed=seed,
                    spec=spec,
                    run_row=run_row,
                    storage_row=storage,
                    manifest={**manifest, **export_audit},
                    native=native_by_seed.get(seed, {}),
                    export_full=export_by_seed.get(seed, {}),
                )
            )
    ablation_methods = ["full-native-SeHGNN", "H6-struct50-best", "H6-struct30-best", "target-only-schema-stub"]
    write_csv(out / "gate21_2_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_2_storage_audit.csv", storage_rows, fieldnames=[*STORAGE_AUDIT_FIELDS, "eligible_for_main_decision"])
    write_csv(out / "gate21_2_relation_mapping_audit.csv", relation_mapping_rows, fieldnames=RELATION_MAPPING_FIELDS)
    write_csv(out / "gate21_2_relation_edge_retention.csv", relation_retention_rows, fieldnames=RELATION_RETENTION_FIELDS)
    write_csv(out / "gate21_2_edge_score_diagnostics.csv", score_rows, fieldnames=EDGE_SCORE_DIAGNOSTIC_FIELDS)
    write_csv(out / "gate21_2_label_graph_ablation.csv", planned_label_graph_ablation_rows(dataset=dataset, seeds=seeds, methods=ablation_methods), fieldnames=LABEL_GRAPH_ABLATION_FIELDS)
    write_csv(out / "gate21_2_feature_cache_compression_probe.csv", [], fieldnames=["dataset", "seed", "method", "feature_compression_method", "feature_dtype", "feature_dim", "feature_storage_ratio", "raw_hgb_byte_ratio", "preprocessed_cache_byte_ratio", "test_micro_f1", "test_macro_f1", "eligible_for_main_decision"])
    write_csv(out / "gate21_2_hgb_export_audit.csv", export_rows)
    write_json(
        out / "gate21_2_plan.json",
        {
            "dataset": dataset,
            "seeds": seeds,
            "storage_budgets": [float(v) for v in args.storage_budgets],
            "methods": [spec.method for spec in specs],
            "dry_run": False,
            "skip_official_training": _bool_arg(args.skip_official_training),
        },
    )
    summary = summarize_gate21_2(out, out)
    return {"raw_rows": len(raw_rows), "methods": len(specs), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--storage-budgets", nargs="+", type=float, default=[0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.32, 0.30])
    parser.add_argument("--node-selector", default="H6")
    parser.add_argument("--budget-strategies", nargs="+", default=["proportional", "pair_grid", "path_aware", "random_relationwise", "degree_topk_relationwise"])
    parser.add_argument("--edge-score-strategies", nargs="+", default=["current_heuristic", "path_aware", "random", "degree"])
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gate21_2_h6_path_budget"))
    parser.add_argument("--gate21-1-root", type=Path, default=DEFAULT_GATE21_1_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--skip-official-training", action="store_true")
    parser.add_argument("--reuse-existing-exports", action="store_true")
    parser.add_argument("--reuse-existing-native-full", action="store_true")
    parser.add_argument("--structural-budget-mode", choices=["edge_ratio", "storage_ratio"], default="storage_ratio")
    parser.add_argument("--raw-byte-budget", type=float, default=None)
    parser.add_argument("--min-edges-per-relation", type=int, default=1)
    parser.add_argument("--pair-grid-preset", default="dblp_default")
    parser.add_argument("--validation-greedy-chunk-ratio", type=float, default=0.05)
    parser.add_argument("--no-label-feats-ablation", action="store_true")
    parser.add_argument("--no-feature-cache-audit", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_gate21_2(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
