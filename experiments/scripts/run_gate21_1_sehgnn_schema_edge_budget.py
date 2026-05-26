from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate21_0_sehgnn_native_export import _dir_size, _edge_count, _float, _native_labels, _target_local_ids
from experiments.scripts.summarize_gate21_1_sehgnn_schema_edge_budget import summarize_gate21_1
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.schema_stable_pruning import (
    EdgeBudgetConfig,
    build_schema_stable_edge_budget_graph,
    build_target_only_schema_stub_graph,
)
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
from hesf_coarsen.eval.official.sehgnn_native_runner import NATIVE_METRIC_FIELDS, build_official_hgb_command, run_native_command, run_native_stage
from hesf_coarsen.eval.official.weighted_edge_audit import audit_sehgnn_edge_weight_semantics
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, nodes_of_type


TARGET_TYPE_BY_DATASET = {"DBLP": "A", "ACM": "P", "IMDB": "M"}
BASELINE_BY_PREFIX = {"H6": "H6-no-spec-support-only", "flatten": "flatten-sum-support-only", "TypedHash": "TypedHash-ChebHeat-support-only"}
RAW_FIELDS = [
    "dataset",
    "seed",
    "model_name",
    "method",
    "method_family",
    "schema_compatible",
    "weighted_coarse_graph",
    "uses_weighted_superedges",
    "weighted_edge_preserved",
    "eligible_for_main_decision",
    "status",
    "error",
    "requested_support_node_ratio",
    "actual_support_node_ratio",
    "requested_edge_ratio",
    "actual_support_edge_ratio",
    "requested_storage_ratio",
    "actual_total_storage_ratio_vs_full_graph",
    "support_node_ratio",
    "support_edge_ratio",
    "total_storage_ratio_vs_full_graph",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "test_accuracy",
    "native_full_test_micro_f1",
    "native_full_test_macro_f1",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "mapping_bijective",
    "split_disjoint",
    "no_test_label_export_leakage",
    "relation_order_matches_official",
    "node_type_order_matches_official",
    "schema_complete",
    "stdout_path",
    "stderr_path",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _unweighted_graph(graph: HeteroGraph) -> HeteroGraph:
    relations = {
        int(rid): RelationAdj(
            rel.src.copy(),
            rel.dst.copy(),
            np.ones(rel.num_edges, dtype=np.float32),
            rel.src_type,
            rel.dst_type,
            int(rid),
        )
        for rid, rel in graph.relations.items()
    }
    return HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=relations,
        relation_specs=graph.relation_specs,
        features=None if graph.features is None else {int(k): v.copy() for k, v in graph.features.items()},
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
    )


def _storage_fields(original: HeteroGraph, graph: HeteroGraph, target_type: int, native_full_dir: Path, export_dir: Path) -> dict[str, Any]:
    original_support = int(np.sum(original.node_type != int(target_type)))
    support = int(np.sum(graph.node_type != int(target_type)))
    original_edges = _edge_count(original)
    edges = _edge_count(graph)
    original_storage = max(int(original.num_nodes) + int(original_edges), 1)
    graph_storage = int(graph.num_nodes) + int(edges)
    return {
        "support_node_ratio": float(support / max(original_support, 1)),
        "support_edge_ratio": float(edges / max(original_edges, 1)),
        "total_node_ratio": float(graph.num_nodes / max(original.num_nodes, 1)),
        "total_edge_ratio": float(edges / max(original_edges, 1)),
        "total_storage_ratio_vs_full_graph": float(graph_storage / original_storage),
        "feature_storage_ratio": "",
        "label_storage_ratio": "",
        "edge_storage_ratio": float(edges / max(original_edges, 1)),
        "export_file_bytes": int(_dir_size(export_dir)),
        "native_full_file_bytes": int(_dir_size(native_full_dir)),
    }


def _method_parts(method: str) -> tuple[str, str]:
    if method.startswith("H6-"):
        return "H6", method[len("H6-") :]
    if method.startswith("flatten-"):
        return "flatten", method[len("flatten-") :]
    if method.startswith("TypedHash-"):
        return "TypedHash", method[len("TypedHash-") :]
    if method == "target-only-schema-stub":
        return "target-only-schema-stub", ""
    raise ValueError(f"unsupported Gate21.1 method: {method}")


def _budget_config(method: str, original: HeteroGraph, base: HeteroGraph, seed: int) -> EdgeBudgetConfig | None:
    _family, suffix = _method_parts(method)
    if suffix == "node30":
        return None
    reference_edges = _edge_count(original)
    reference_nodes = int(original.num_nodes)
    if suffix == "node30-edge50":
        edge_ratio = 0.50
        storage_ratio = None
    elif suffix == "node30-edge30":
        edge_ratio = 0.30
        storage_ratio = None
    elif suffix == "storage50":
        edge_ratio = None
        storage_ratio = 0.50
    elif suffix == "storage30":
        edge_ratio = None
        storage_ratio = 0.30
    else:
        raise ValueError(f"unsupported Gate21.1 budget suffix: {suffix}")
    return EdgeBudgetConfig(
        requested_support_node_ratio=0.30,
        requested_edge_ratio=edge_ratio,
        requested_storage_ratio=storage_ratio,
        reference_num_nodes=reference_nodes,
        reference_num_edges=reference_edges,
        min_edges_per_relation_fraction=0.01,
        seed=int(seed),
    )


def _build_schema_method_graph(original: HeteroGraph, method: str, *, seed: int, candidate_k: int, target_type: int) -> tuple[HeteroGraph, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    family, _suffix = _method_parts(method)
    baseline = BASELINE_BY_PREFIX[family]
    base_graph, assignment, diag = run_support_baseline(
        original,
        baseline=baseline,
        ratio=0.30,
        seed=int(seed),
        candidate_k=int(candidate_k),
    )
    base_graph = _unweighted_graph(base_graph)
    config = _budget_config(method, original, base_graph, int(seed))
    if config is None:
        return base_graph, np.asarray(assignment, dtype=np.int64), {"schema_complete": True, "relation_retention": [], **diag}, []
    support_nodes = np.flatnonzero(base_graph.node_type != int(target_type)).astype(np.int64)
    pruned, audit = build_schema_stable_edge_budget_graph(
        graph=base_graph,
        selected_support_nodes=support_nodes,
        dataset_name="DBLP",
        target_type=str(target_type),
        config=config,
    )
    return pruned, np.asarray(assignment, dtype=np.int64), {**diag, **audit}, list(audit.get("relation_retention", []))


def _compressed_labels_and_splits(
    *,
    original: HeteroGraph,
    graph: HeteroGraph,
    assignment: np.ndarray,
    target_type: int,
    labels_native: np.ndarray,
    trainval_native: np.ndarray,
    test_native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_nodes = nodes_of_type(original, int(target_type))
    compressed_target_globals = np.asarray(assignment[target_nodes], dtype=np.int64)
    compressed_target_local = _target_local_ids(graph, int(target_type), compressed_target_globals)
    original_target_local = {int(node): int(pos) for pos, node in enumerate(target_nodes.tolist())}
    if labels_native.ndim == 2:
        labels = np.zeros((graph.num_nodes, labels_native.shape[1]), dtype=np.int64)
    else:
        labels = np.full(graph.num_nodes, -1, dtype=np.int64)
    for original_global, compressed_global in zip(target_nodes.tolist(), compressed_target_globals.tolist()):
        if labels_native.ndim == 2:
            labels[int(compressed_global)] = labels_native[int(original_global)]
        else:
            labels[int(compressed_global)] = int(labels_native[int(original_global)])
    trainval = compressed_target_local[[original_target_local[int(node)] for node in trainval_native.tolist()]]
    test = compressed_target_local[[original_target_local[int(node)] for node in test_native.tolist()]]
    return labels, trainval.astype(np.int64, copy=False), test.astype(np.int64, copy=False)


def _native_metric_lookup(native_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row.get("dataset", "")), str(row.get("seed", ""))): row for row in native_rows}


def _raw_reference_row(dataset: str, seed: int, method: str, row: Mapping[str, Any], native: Mapping[str, Any] | None = None) -> dict[str, Any]:
    native = row if native is None else native
    native_micro = _float(native.get("test_micro_f1"))
    native_macro = _float(native.get("test_macro_f1"))
    test_micro = _float(row.get("test_micro_f1"))
    test_macro = _float(row.get("test_macro_f1"))
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model_name": "official-SeHGNN",
        "method": method,
        "method_family": "reference_full",
        "schema_compatible": True,
        "weighted_coarse_graph": False,
        "uses_weighted_superedges": False,
        "weighted_edge_preserved": False,
        "eligible_for_main_decision": False,
        "status": row.get("status", ""),
        "error": row.get("error_message", ""),
        "test_micro_f1": "" if test_micro is None else test_micro,
        "test_macro_f1": "" if test_macro is None else test_macro,
        "test_accuracy": row.get("test_accuracy_if_single_label", ""),
        "validation_micro_f1": row.get("validation_micro_f1", ""),
        "validation_macro_f1": row.get("validation_macro_f1", ""),
        "native_full_test_micro_f1": "" if native_micro is None else native_micro,
        "native_full_test_macro_f1": "" if native_macro is None else native_macro,
        "recovery_vs_native_full_micro": "" if native_micro in {None, 0.0} or test_micro is None else test_micro / native_micro,
        "recovery_vs_native_full_macro": "" if native_macro in {None, 0.0} or test_macro is None else test_macro / native_macro,
        "mapping_bijective": True,
        "split_disjoint": True,
        "no_test_label_export_leakage": True,
        "relation_order_matches_official": True,
        "node_type_order_matches_official": True,
        "schema_complete": True,
        "stdout_path": row.get("stdout_path", ""),
        "stderr_path": row.get("stderr_path", ""),
    }


def _raw_compressed_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    method_family: str,
    row: Mapping[str, Any],
    native: Mapping[str, Any],
    storage: Mapping[str, Any],
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    eligible: bool,
) -> dict[str, Any]:
    native_micro = _float(native.get("test_micro_f1"))
    native_macro = _float(native.get("test_macro_f1"))
    test_micro = _float(row.get("test_micro_f1"))
    test_macro = _float(row.get("test_macro_f1"))
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model_name": "official-SeHGNN",
        "method": method,
        "method_family": method_family,
        "schema_compatible": bool(audit.get("schema_complete", True)),
        "weighted_coarse_graph": False,
        "uses_weighted_superedges": False,
        "weighted_edge_preserved": False,
        "eligible_for_main_decision": bool(eligible),
        "status": row.get("status", ""),
        "error": row.get("error_message", ""),
        "requested_support_node_ratio": audit.get("requested_support_node_ratio", 0.30),
        "actual_support_node_ratio": storage.get("support_node_ratio", ""),
        "requested_edge_ratio": audit.get("requested_edge_ratio", ""),
        "actual_support_edge_ratio": storage.get("support_edge_ratio", ""),
        "requested_storage_ratio": audit.get("requested_storage_ratio", ""),
        "actual_total_storage_ratio_vs_full_graph": storage.get("total_storage_ratio_vs_full_graph", ""),
        "support_node_ratio": storage.get("support_node_ratio", ""),
        "support_edge_ratio": storage.get("support_edge_ratio", ""),
        "total_storage_ratio_vs_full_graph": storage.get("total_storage_ratio_vs_full_graph", ""),
        "validation_micro_f1": row.get("validation_micro_f1", ""),
        "validation_macro_f1": row.get("validation_macro_f1", ""),
        "test_micro_f1": "" if test_micro is None else test_micro,
        "test_macro_f1": "" if test_macro is None else test_macro,
        "test_accuracy": row.get("test_accuracy_if_single_label", ""),
        "native_full_test_micro_f1": "" if native_micro is None else native_micro,
        "native_full_test_macro_f1": "" if native_macro is None else native_macro,
        "recovery_vs_native_full_micro": "" if native_micro in {None, 0.0} or test_micro is None else test_micro / native_micro,
        "recovery_vs_native_full_macro": "" if native_macro in {None, 0.0} or test_macro is None else test_macro / native_macro,
        "mapping_bijective": manifest.get("mapping_bijective", ""),
        "split_disjoint": manifest.get("split_disjoint", ""),
        "no_test_label_export_leakage": manifest.get("no_test_label_export_leakage", ""),
        "relation_order_matches_official": manifest.get("relation_order_matches_official", ""),
        "node_type_order_matches_official": manifest.get("node_type_order_matches_official", ""),
        "schema_complete": audit.get("schema_complete", True),
        "stdout_path": row.get("stdout_path", ""),
        "stderr_path": row.get("stderr_path", ""),
    }


def _copy_gate21_0_reference(src_root: Path, out_dir: Path, datasets: Sequence[str]) -> bool:
    if not src_root.exists():
        return False
    for rel in (
        Path("native") / "native_metrics.csv",
        Path("native") / "native_summary_by_dataset.csv",
        Path("native") / "native_command_manifest.json",
        Path("native") / "native_environment.json",
        Path("native") / "native_data_audit.csv",
        Path("native") / "native_metric_parser_audit.csv",
    ):
        src = src_root / rel
        if src.exists():
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    for rel in (
        Path("fidelity") / "gate21_0_export_full_metrics.csv",
        Path("fidelity") / "gate21_0_sehgnn_full_fidelity.csv",
        Path("fidelity") / "gate21_0_sehgnn_feature_audit.csv",
        Path("export") / "gate21_0_hgb_export_audit.csv",
    ):
        src = src_root / rel
        if src.exists():
            dst_name = str(rel).replace("gate21_0", "gate21_1")
            dst = out_dir / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    native_rows = _read_csv(out_dir / "native" / "native_metrics.csv")
    return bool(native_rows and all(any(row.get("dataset") == dataset for row in native_rows) for dataset in datasets))


def run_gate21_1(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [str(dataset).upper() for dataset in args.datasets]
    if datasets != ["DBLP"]:
        raise ValueError("Gate21.1 first run is intentionally DBLP-only")
    seeds = [int(seed) for seed in args.seeds]
    if _bool_arg(args.reuse_gate21_0_reference):
        copied = _copy_gate21_0_reference(Path(args.gate21_0_dir), out_dir, datasets)
        if not copied:
            raise RuntimeError("requested Gate21.0 reuse but reference files are missing")
    elif _bool_arg(args.run_native_fidelity):
        run_native_stage(
            repo_dir=Path(args.sehgnn_repo),
            data_root=Path(args.native_data_root),
            datasets=datasets,
            seeds=seeds,
            device=str(args.device),
            out_dir=out_dir,
            python_executable=sys.executable,
        )

    native_rows = [row for row in _read_csv(out_dir / "native" / "native_metrics.csv") if row.get("dataset") in datasets]
    native_by_key = _native_metric_lookup(native_rows)
    raw_rows: list[dict[str, Any]] = []
    for row in native_rows:
        raw_rows.append(_raw_reference_row(str(row["dataset"]), int(row["seed"]), "full-native-SeHGNN", row))

    export_full_rows = [row for row in _read_csv(out_dir / "fidelity" / "gate21_1_export_full_metrics.csv") if row.get("dataset") in datasets]
    export_audits: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    if not export_full_rows and _bool_arg(args.run_native_fidelity):
        export_root = out_dir / "export_full_hgb"
        stdout_dir = out_dir / "fidelity" / "export_full_raw_stdout"
        stderr_dir = out_dir / "fidelity" / "export_full_raw_stderr"
        for dataset in datasets:
            original = load_hgb_graph(Path(args.data_root), dataset)
            labels, trainval, test = _native_labels(Path(args.native_data_root), dataset, original.num_nodes)
            manifest = export_graph_to_sehgnn_hgb(
                graph=original,
                dataset_name=dataset,
                target_type=TARGET_TYPE_BY_DATASET[dataset],
                output_dir=export_root,
                split_mode="official_trainval",
                train_idx=trainval,
                val_idx=np.array([], dtype=np.int64),
                test_idx=test,
                labels=labels,
                method_name="full",
                seed=0,
            )
            audit = audit_native_hgb_data_dir(dataset, Path(manifest["export_dir"]).parent, Path(args.sehgnn_repo))
            export_audits.append({**manifest, **audit})
            for seed in seeds:
                command = build_official_hgb_command(
                    dataset=dataset,
                    seed=seed,
                    repo_dir=Path(args.sehgnn_repo),
                    data_root=Path(manifest["export_dir"]).parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                row = run_native_command(command, stdout_path=stdout_dir / f"{dataset}_{seed}.log", stderr_path=stderr_dir / f"{dataset}_{seed}.stderr")
                export_full_rows.append(row)
                native = native_by_key.get((dataset, str(seed)), {})
                native_micro = _float(native.get("test_micro_f1"))
                native_macro = _float(native.get("test_macro_f1"))
                export_micro = _float(row.get("test_micro_f1"))
                export_macro = _float(row.get("test_macro_f1"))
                fidelity_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "native_official_micro_f1": "" if native_micro is None else native_micro,
                        "native_official_macro_f1": "" if native_macro is None else native_macro,
                        "export_full_micro_f1": "" if export_micro is None else export_micro,
                        "export_full_macro_f1": "" if export_macro is None else export_macro,
                        "micro_gap_native_minus_export": "" if native_micro is None or export_micro is None else native_micro - export_micro,
                        "macro_gap_native_minus_export": "" if native_macro is None or export_macro is None else native_macro - export_macro,
                        "fidelity_pass": bool(row.get("status") == "success" and native_micro is not None and export_micro is not None and abs(native_micro - export_micro) <= 0.02),
                    }
                )
        write_csv(out_dir / "export" / "gate21_1_hgb_export_audit.csv", export_audits)
        write_csv(out_dir / "fidelity" / "gate21_1_export_full_metrics.csv", export_full_rows, fieldnames=NATIVE_METRIC_FIELDS)
        write_csv(out_dir / "fidelity" / "gate21_1_sehgnn_full_fidelity.csv", fidelity_rows)
    else:
        fidelity_rows = _read_csv(out_dir / "fidelity" / "gate21_1_sehgnn_full_fidelity.csv")

    for row in export_full_rows:
        native = native_by_key.get((str(row.get("dataset", "")), str(row.get("seed", ""))), {})
        raw_rows.append(_raw_reference_row(str(row.get("dataset", "")), int(row.get("seed", 0)), "export-full-SeHGNN", row, native=native))

    compressed_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    edge_budget_rows: list[dict[str, Any]] = []
    relation_retention_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    target_stub_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    stdout_dir = out_dir / "compressed" / "compressed_raw_stdout"
    stderr_dir = out_dir / "compressed" / "compressed_raw_stderr"
    export_root = out_dir / "compressed_hgb"
    graph_seed = int(seeds[0])
    for dataset in datasets:
        original = load_hgb_graph(Path(args.data_root), dataset)
        target_type = int(infer_target_node_type(original))
        labels_native, trainval_native, test_native = _native_labels(Path(args.native_data_root), dataset, original.num_nodes)
        if _bool_arg(args.run_weighted_edge_audit):
            weighted_graph, weighted_assignment, _diag = run_support_baseline(
                original,
                baseline="H6-no-spec-support-only",
                ratio=0.30,
                seed=graph_seed,
                candidate_k=int(args.candidate_k),
            )
            weighted_labels, weighted_trainval, weighted_test = _compressed_labels_and_splits(
                original=original,
                graph=weighted_graph,
                assignment=np.asarray(weighted_assignment, dtype=np.int64),
                target_type=target_type,
                labels_native=labels_native,
                trainval_native=trainval_native,
                test_native=test_native,
            )
            weighted_manifest = export_graph_to_sehgnn_hgb(
                graph=weighted_graph,
                dataset_name=dataset,
                target_type=TARGET_TYPE_BY_DATASET[dataset],
                output_dir=out_dir / "diagnostics" / "weighted_probe_hgb",
                split_mode="official_trainval",
                train_idx=weighted_trainval,
                val_idx=np.array([], dtype=np.int64),
                test_idx=weighted_test,
                labels=weighted_labels,
                method_name="H6-node30-weighted-probe",
                seed=graph_seed,
            )
            audit_sehgnn_edge_weight_semantics(
                export_dir=Path(weighted_manifest["export_dir"]),
                dataset_name=dataset,
                sehgnn_repo_dir=Path(args.sehgnn_repo),
                output_dir=out_dir / "diagnostics",
            )
        for method in [str(value) for value in args.methods]:
            if method == "target-only-schema-stub":
                graph, stub_audit = build_target_only_schema_stub_graph(graph=original, dataset_name=dataset, target_type=TARGET_TYPE_BY_DATASET[dataset])
                labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
                trainval = np.asarray(trainval_native, dtype=np.int64)
                test = np.asarray(test_native, dtype=np.int64)
                method_family = "schema_stub_diagnostic"
                eligible = False
                graph_audit: dict[str, Any] = {**stub_audit, "schema_complete": stub_audit["schema_complete"]}
                relation_retention: list[dict[str, Any]] = []
            else:
                graph, assignment, graph_audit, relation_retention = _build_schema_method_graph(
                    original,
                    method,
                    seed=graph_seed,
                    candidate_k=int(args.candidate_k),
                    target_type=target_type,
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
                method_family = "schema_compatible_subgraph"
                eligible = True
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
                method_name=method,
                seed=graph_seed,
            )
            export_audit = audit_native_hgb_data_dir(dataset, Path(manifest["export_dir"]).parent, Path(args.sehgnn_repo))
            export_rows.append({**manifest, **export_audit, "method": method})
            storage = _storage_fields(original, graph, target_type, Path(args.native_data_root) / dataset, Path(manifest["export_dir"]))
            storage_row = {"dataset": dataset, "seed": graph_seed, "method": method, **storage}
            storage_rows.append(storage_row)
            edge_budget_rows.append(
                {
                    "dataset": dataset,
                    "seed": graph_seed,
                    "method": method,
                    "requested_support_node_ratio": graph_audit.get("requested_support_node_ratio", 0.30 if method != "target-only-schema-stub" else 0.0),
                    "actual_support_node_ratio": storage.get("support_node_ratio", ""),
                    "requested_edge_ratio": graph_audit.get("requested_edge_ratio", ""),
                    "actual_support_edge_ratio": storage.get("support_edge_ratio", ""),
                    "requested_storage_ratio": graph_audit.get("requested_storage_ratio", ""),
                    "actual_total_storage_ratio_vs_full_graph": storage.get("total_storage_ratio_vs_full_graph", ""),
                    "node_count_by_type": graph_audit.get("node_count_by_type", ""),
                    "edge_count_by_relation": graph_audit.get("edge_count_by_relation", ""),
                    "schema_complete": graph_audit.get("schema_complete", True),
                    "relation_order_matches_official": manifest.get("relation_order_matches_official", ""),
                    "mapping_bijective": manifest.get("mapping_bijective", ""),
                    "split_disjoint": manifest.get("split_disjoint", ""),
                    "no_test_label_export_leakage": manifest.get("no_test_label_export_leakage", ""),
                }
            )
            for retention in relation_retention:
                relation_retention_rows.append({"dataset": dataset, "seed": graph_seed, "method": method, **retention})
            schema_rows.append(
                {
                    "dataset": dataset,
                    "seed": graph_seed,
                    "method": method,
                    "method_family": method_family,
                    "schema_complete": graph_audit.get("schema_complete", True),
                    "relation_order_matches_official": manifest.get("relation_order_matches_official", ""),
                    "node_type_order_matches_official": manifest.get("node_type_order_matches_official", ""),
                    "can_load_with_official_data_loader": export_audit.get("can_load_with_official_data_loader", ""),
                    "eligible_for_main_decision": eligible,
                }
            )
            if method == "target-only-schema-stub":
                target_stub_rows.append({"dataset": dataset, "seed": graph_seed, **graph_audit})
            for seed in seeds:
                command = build_official_hgb_command(
                    dataset=dataset,
                    seed=seed,
                    repo_dir=Path(args.sehgnn_repo),
                    data_root=Path(manifest["export_dir"]).parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                run_row = run_native_command(command, stdout_path=stdout_dir / f"{dataset}_{method}_{seed}.log", stderr_path=stderr_dir / f"{dataset}_{method}_{seed}.stderr")
                compressed_rows.append({"dataset": dataset, "seed": seed, "method": method, **run_row, **storage})
                native = native_by_key.get((dataset, str(seed)), {})
                raw_rows.append(
                    _raw_compressed_row(
                        dataset=dataset,
                        seed=seed,
                        method=method,
                        method_family=method_family,
                        row=run_row,
                        native=native,
                        storage=storage,
                        manifest=manifest,
                        audit=graph_audit,
                        eligible=eligible,
                    )
                )
    write_csv(out_dir / "gate21_1_raw_rows.csv", raw_rows, fieldnames=RAW_FIELDS)
    write_csv(out_dir / "compressed" / "gate21_1_compressed_metrics.csv", compressed_rows)
    write_csv(out_dir / "compressed" / "gate21_1_compressed_storage_audit.csv", storage_rows)
    write_csv(out_dir / "export" / "gate21_1_hgb_export_audit.csv", export_rows)
    write_csv(out_dir / "diagnostics" / "gate21_1_edge_budget_audit.csv", edge_budget_rows)
    write_csv(out_dir / "diagnostics" / "gate21_1_relation_edge_retention.csv", relation_retention_rows)
    write_csv(out_dir / "diagnostics" / "gate21_1_schema_compatibility.csv", schema_rows)
    write_csv(out_dir / "diagnostics" / "gate21_1_target_only_schema_stub.csv", target_stub_rows)
    summary = summarize_gate21_1(out_dir, out_dir)
    return {"raw_rows": len(raw_rows), "compressed_rows": len(compressed_rows), "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "H6-node30",
            "H6-node30-edge50",
            "H6-node30-edge30",
            "H6-storage50",
            "H6-storage30",
            "flatten-node30",
            "flatten-node30-edge50",
            "flatten-node30-edge30",
            "flatten-storage50",
            "flatten-storage30",
            "TypedHash-node30",
            "target-only-schema-stub",
        ],
    )
    parser.add_argument("--sehgnn-repo", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--native-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate21_1_sehgnn_schema_edge_budget"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--run-native-fidelity", default="true")
    parser.add_argument("--run-weighted-edge-audit", default="true")
    parser.add_argument("--reuse-gate21-0-reference", default="false")
    parser.add_argument("--gate21-0-dir", type=Path, default=Path("outputs/gate21_0_sehgnn_native_export"))
    args = parser.parse_args(argv)
    print(json.dumps(run_gate21_1(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
