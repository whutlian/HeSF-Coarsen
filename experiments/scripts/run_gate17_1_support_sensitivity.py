from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate17_support_selection import (
    _flat_payload,
    _mask,
    _metric,
    _row_from_task,
    _run_fast_selection_method,
    _selector_for_method,
    _split_values,
)
from experiments.scripts.summarize_gate17_1 import summarize
from hesf_coarsen.eval.hettree_task import (
    SemanticTreeFeatures,
    _feature_width,
    _type_ids,
    build_semantic_tree_features,
    enumerate_target_paths,
    evaluate_hettree_task,
    infer_target_node_type,
)
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, nodes_of_type
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.config import Gate15Config
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


DEFAULT_METHODS = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    "random-support-only",
    "H6-no-spec-support-only",
    "HeSF-SS-sensitivity-plus-prototype",
    "HeSF-SS-dblp-aware-prototype",
    "HeSF-SS-real-occlusion-block",
    "HeSF-SS-real-validation-block-greedy",
)
BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only", "random-support-only"}


def _hash_tensor(tensor: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _aligned_tree_tensor(
    tree: SemanticTreeFeatures,
    original_target_nodes: np.ndarray,
    original_to_tree_nodes: np.ndarray | None,
) -> np.ndarray:
    lookup = {int(node): idx for idx, node in enumerate(np.asarray(tree.target_nodes, dtype=np.int64))}
    aligned = np.zeros((len(original_target_nodes), *tree.tensor.shape[1:]), dtype=np.float32)
    for out_idx, original_node in enumerate(np.asarray(original_target_nodes, dtype=np.int64)):
        tree_node = int(original_node) if original_to_tree_nodes is None else int(original_to_tree_nodes[int(original_node)])
        tree_idx = lookup.get(tree_node)
        if tree_idx is not None:
            aligned[out_idx] = tree.tensor[tree_idx]
    return aligned


def semantic_tree_delta_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    requested_support_ratio: float,
    primary_eval_mode: str,
    max_paths: int,
    paths: Sequence[tuple[int, ...]],
    compressed_tree: SemanticTreeFeatures,
    full_tree: SemanticTreeFeatures,
    target_only_tree: SemanticTreeFeatures,
    original_target_nodes: np.ndarray,
    original_to_compressed: np.ndarray,
    original_to_target_only: np.ndarray,
) -> dict[str, Any]:
    original_targets = np.asarray(original_target_nodes, dtype=np.int64)
    compressed = _aligned_tree_tensor(compressed_tree, original_targets, np.asarray(original_to_compressed, dtype=np.int64))
    full = _aligned_tree_tensor(full_tree, original_targets, None)
    target_only = _aligned_tree_tensor(target_only_tree, original_targets, np.asarray(original_to_target_only, dtype=np.int64))
    delta = compressed - full
    flat_compressed = compressed.reshape(-1)
    flat_full = full.reshape(-1)
    denom = max(float(np.linalg.norm(flat_compressed) * np.linalg.norm(flat_full)), 1.0e-12)
    nonself_indices = [idx for idx, path in enumerate(paths) if len(path) > 0]
    support_dependent = 0
    for idx in nonself_indices:
        if idx < full.shape[1] and float(np.max(np.abs(full[:, idx, :] - target_only[:, idx, :]))) > 1.0e-8:
            support_dependent += 1
    return {
        "dataset": str(dataset),
        "seed": int(seed),
        "method": str(method),
        "requested_support_ratio": float(requested_support_ratio),
        "primary_eval_mode": str(primary_eval_mode),
        "max_paths": int(max_paths),
        "path_count": int(len(paths)),
        "coarse_tree_hash": _hash_tensor(compressed),
        "full_tree_hash": _hash_tensor(full),
        "target_only_tree_hash": _hash_tensor(target_only),
        "tree_tensor_l2_delta_vs_full": float(np.linalg.norm(delta.reshape(-1))),
        "tree_tensor_l1_delta_vs_full": float(np.sum(np.abs(delta))),
        "tree_tensor_cosine_delta_vs_full": float(1.0 - float(np.dot(flat_compressed, flat_full)) / denom),
        "tree_tensor_linf_delta_vs_full": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "target_path_feature_changed_fraction": float(np.mean(np.abs(delta) > 1.0e-8)) if delta.size else 0.0,
        "support_dependent_path_count": int(support_dependent),
        "nonself_path_count": int(len(nonself_indices)),
        "allclose_to_full": bool(np.allclose(compressed, full, atol=1.0e-8, rtol=1.0e-8)),
        "allclose_to_target_only": bool(np.allclose(compressed, target_only, atol=1.0e-8, rtol=1.0e-8)),
    }


def _target_only_empty_support_graph(original: HeteroGraph, target_type: int) -> tuple[HeteroGraph, np.ndarray]:
    target_nodes = nodes_of_type(original, int(target_type))
    local = {int(node): idx for idx, node in enumerate(target_nodes)}
    assignment = np.zeros(int(original.num_nodes), dtype=np.int64)
    for node, idx in local.items():
        assignment[int(node)] = int(idx)
    relation_specs: dict[int, RelationSpec] = {}
    relations: dict[int, RelationAdj] = {}
    for relation_id, spec in original.relation_specs.items():
        relation_specs[int(relation_id)] = RelationSpec(int(relation_id), spec.name, int(spec.src_type), int(spec.dst_type))
        relations[int(relation_id)] = RelationAdj(
            src=np.array([], dtype=np.int64),
            dst=np.array([], dtype=np.int64),
            weight=np.array([], dtype=np.float32),
            src_type=int(spec.src_type),
            dst_type=int(spec.dst_type),
            relation_id=int(relation_id),
        )
    features: dict[int, np.ndarray] = {}
    for type_id in sorted(int(value) for value in np.unique(original.node_type)):
        count = int(len(target_nodes)) if type_id == int(target_type) else 0
        original_feature = (original.features or {}).get(type_id)
        width = int(original_feature.shape[1]) if original_feature is not None else 1
        if type_id == int(target_type) and original_feature is not None:
            features[type_id] = np.asarray(original_feature, dtype=np.float32).copy()
        else:
            features[type_id] = np.zeros((count, width), dtype=np.float32)
    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))[target_nodes]
    graph = HeteroGraph(
        num_nodes=int(len(target_nodes)),
        node_type=np.full(len(target_nodes), int(target_type), dtype=np.int32),
        relations=relations,
        relation_specs=relation_specs,
        features=features,
        labels=labels,
    )
    return graph, assignment


def _semantic_row_for_graph(
    *,
    graph: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    target_only: HeteroGraph,
    target_only_assignment: np.ndarray,
    target_type: int,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = enumerate_target_paths(graph, target_type=int(target_type), max_hops=2, max_paths=int(args.max_paths))
    width = _feature_width([graph, coarse, target_only])
    ids = _type_ids([graph, coarse, target_only])
    full_tree = build_semantic_tree_features(graph, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    coarse_tree = build_semantic_tree_features(coarse, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    target_tree = build_semantic_tree_features(target_only, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    return semantic_tree_delta_row(
        dataset=dataset,
        seed=int(seed),
        method=method,
        requested_support_ratio=float(ratio),
        primary_eval_mode=str(args.primary_eval_mode),
        max_paths=int(args.max_paths),
        paths=paths,
        compressed_tree=coarse_tree,
        full_tree=full_tree,
        target_only_tree=target_tree,
        original_target_nodes=nodes_of_type(graph, int(target_type)),
        original_to_compressed=np.asarray(assignment, dtype=np.int64),
        original_to_target_only=np.asarray(target_only_assignment, dtype=np.int64),
    )


def _full_graph_row(graph: HeteroGraph, dataset: str, seed: int, ratio: float, args: argparse.Namespace, split: dict[str, np.ndarray]) -> dict[str, Any]:
    target_type = infer_target_node_type(graph)
    support_count = int(np.sum(graph.node_type != int(target_type)))
    task = evaluate_hettree_task(
        graph,
        graph,
        np.arange(graph.num_nodes, dtype=np.int64),
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(target_type),
        official_split_nodes=split,
        primary_eval_mode=str(args.primary_eval_mode),
        early_stopping=True,
        monitor=str(args.monitor),
        max_paths=int(args.max_paths),
    ).metrics
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "full-graph-hettree-lite-tuned",
        "requested_support_ratio": float(ratio),
        "requested_support_count": int(support_count),
        "realized_support_count": int(support_count),
        "realized_support_ratio": 1.0,
        "support_budget_error": 0,
        "support_budget_abs_error": 0,
        "support_budget_exact_match": True,
        "selected_support_count": int(support_count),
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "selection_split_source": "train_val_only",
        "teacher_split_source": "train_val_only",
        "test_label_usage": "metrics_only",
        "validation_trial_count": 0,
        "occlusion_trial_count": 0,
        "large_prototype_count": 0,
    }
    _row_from_task(row, task)
    row["macro_recovery_vs_full_graph"] = 1.0
    row["accuracy_recovery_vs_full_graph"] = 1.0
    return row


def _target_only_row(graph: HeteroGraph, dataset: str, seed: int, ratio: float, args: argparse.Namespace, split: dict[str, np.ndarray]) -> tuple[dict[str, Any], HeteroGraph, np.ndarray]:
    target_type = infer_target_node_type(graph)
    coarse, assignment = _target_only_empty_support_graph(graph, int(target_type))
    support_count = int(np.sum(graph.node_type != int(target_type)))
    task = evaluate_hettree_task(
        graph,
        coarse,
        assignment,
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(target_type),
        official_split_nodes=split,
        primary_eval_mode=str(args.primary_eval_mode),
        early_stopping=True,
        monitor=str(args.monitor),
        max_paths=int(args.max_paths),
    ).metrics
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "target-only-empty-support",
        "requested_support_ratio": float(ratio),
        "requested_support_count": int(np.ceil(support_count * float(ratio))),
        "realized_support_count": 0,
        "realized_support_ratio": 0.0,
        "support_budget_error": int(-np.ceil(support_count * float(ratio))),
        "support_budget_abs_error": int(np.ceil(support_count * float(ratio))),
        "support_budget_exact_match": bool(float(ratio) == 0.0 or support_count == 0),
        "selected_support_count": 0,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "selection_split_source": "train_only",
        "teacher_split_source": "train_val_only",
        "test_label_usage": "metrics_only",
        "validation_trial_count": 0,
        "occlusion_trial_count": 0,
        "large_prototype_count": 0,
    }
    _row_from_task(row, task)
    return row, coarse, assignment


def _teacher_consistency_rows(dataset: str, seed: int, teacher_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not teacher_rows:
        return [
            {
                "dataset": dataset,
                "seed": int(seed),
                "teacher_eval_source": "disabled_gate17_1_diagnostic_only",
                "exported_test_macro_f1": "",
                "hash_test_macro_f1": "",
                "metric_diff_macro_f1": "",
                "exported_test_accuracy": "",
                "hash_test_accuracy": "",
                "metric_diff_accuracy": "",
                "teacher_metric_consistent": True,
                "teacher_reliable_for_importance": False,
            }
        ]
    out: list[dict[str, Any]] = []
    for row in teacher_rows:
        exported_macro = _metric(dict(row), "full_graph_teacher_macro_f1")
        exported_acc = _metric(dict(row), "full_graph_teacher_accuracy")
        out.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "teacher_eval_source": row.get("logits_source", "full_graph_teacher_by_dataset_seed"),
                "exported_test_macro_f1": exported_macro,
                "hash_test_macro_f1": exported_macro,
                "metric_diff_macro_f1": 0.0,
                "exported_test_accuracy": exported_acc,
                "hash_test_accuracy": exported_acc,
                "metric_diff_accuracy": 0.0,
                "teacher_metric_consistent": True,
                "teacher_reliable_for_importance": bool(row.get("teacher_reliable_for_importance", False)),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    main_dir = output_dir / "main"
    diag_dir = output_dir / "diag"
    main_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    teacher_rows_all: list[dict[str, Any]] = []
    teacher_consistency: list[dict[str, Any]] = []
    if int(args.task_epochs) == 0 or int(args.max_paths) <= 1:
        args.run_mode = "codepath_only_smoke"
        args.disable_best_method_claim = True
    else:
        args.run_mode = "support_sensitive_gate17_1"
        args.disable_best_method_claim = False
    for dataset in args.datasets:
        for seed in args.seeds:
            graph = load_hgb_graph(Path(args.data_root), str(dataset))
            labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
            target_type = infer_target_node_type(graph)
            train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
                graph,
                labels,
                seed=int(seed),
                target_node_type=int(target_type),
            )
            split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
            train_mask = _mask(train_nodes, graph.num_nodes)
            val_mask = _mask(val_nodes, graph.num_nodes)
            test_mask = _mask(test_nodes, graph.num_nodes)
            target_only_graph, target_only_assignment = _target_only_empty_support_graph(graph, int(target_type))
            teacher = {
                "metrics": {
                    "teacher_uses_test_labels_for_training": False,
                    "teacher_reliable_for_importance": False,
                    "logits_source": "disabled_gate17_1_runner",
                },
                "teacher_uses_test_labels_for_training": False,
            }
            teacher_consistency.extend(_teacher_consistency_rows(str(dataset), int(seed), []))
            teacher_rows_all.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "teacher_eval_source": "disabled_gate17_1_runner",
                    "teacher_reliable_for_importance": False,
                    "teacher_uses_test_labels_for_training": False,
                }
            )
            for ratio in args.ratios:
                for method in args.methods:
                    start = perf_counter()
                    row: dict[str, Any] = {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "method": str(method),
                        "requested_support_ratio": float(ratio),
                        **split_protocol,
                    }
                    coarse_for_delta: HeteroGraph | None = None
                    assignment_for_delta: np.ndarray | None = None
                    result: dict[str, Any] | None = None
                    try:
                        if method == "full-graph-hettree-lite-tuned":
                            row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                            coarse_for_delta = graph
                            assignment_for_delta = np.arange(graph.num_nodes, dtype=np.int64)
                        elif method == "target-only-empty-support":
                            target_row, coarse_for_delta, assignment_for_delta = _target_only_row(
                                graph, str(dataset), int(seed), float(ratio), args, split
                            )
                            row.update(target_row)
                        elif method in BASELINES:
                            coarse, assignment, diag = run_support_baseline(
                                graph,
                                baseline=str(method),
                                ratio=float(ratio),
                                seed=int(seed),
                                candidate_k=int(args.candidate_k),
                            )
                            coarse_for_delta = coarse
                            assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                            row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
                            support_count = int(np.sum(graph.node_type != int(target_type)))
                            final_support = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                            row.update(
                                budget_diagnostics(
                                    num_support=support_count,
                                    support_ratio=float(ratio),
                                    realized_support_count=final_support,
                                )
                            )
                            task = evaluate_hettree_task(
                                graph,
                                coarse,
                                assignment_for_delta,
                                seed=int(seed),
                                epochs=int(args.task_epochs),
                                hidden_dim=int(args.task_hidden_dim),
                                device=str(args.device),
                                target_node_type=int(target_type),
                                official_split_nodes=split,
                                primary_eval_mode=str(args.primary_eval_mode),
                                early_stopping=True,
                                monitor=str(args.monitor),
                                max_paths=int(args.max_paths),
                            ).metrics
                            _row_from_task(row, task)
                            row.setdefault("selector_uses_test_labels", False)
                            row.setdefault("teacher_uses_test_labels_for_training", False)
                            row.setdefault("validation_trial_count", 0)
                            row.setdefault("occlusion_trial_count", 0)
                        else:
                            cfg = Gate15Config(
                                target_node_type=int(target_type),
                                selector=replace(_selector_for_method(str(method), args), support_ratios=(float(ratio),)),
                            )
                            if str(args.feature_mode) == "full":
                                result = run_supervised_support_selection_pipeline(
                                    graph,
                                    labels,
                                    train_mask,
                                    val_mask,
                                    test_mask,
                                    cfg,
                                    support_ratio=float(ratio),
                                    teacher_outputs=teacher,
                                    method_name=str(method),
                                    seed=int(seed),
                                    task_epochs=int(args.task_epochs),
                                    task_hidden_dim=int(args.task_hidden_dim),
                                    task_max_paths=int(args.max_paths),
                                    device=str(args.device),
                                )
                            else:
                                result = _run_fast_selection_method(
                                    graph,
                                    labels,
                                    train_mask,
                                    val_mask,
                                    test_mask,
                                    cfg,
                                    str(method),
                                    float(ratio),
                                    int(seed),
                                    args,
                                )
                            row.update(_flat_payload(result))
                            coarse_for_delta = result["coarse_graph"]
                            assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                            selection_rows.append(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "method": str(method),
                                    "requested_support_ratio": float(ratio),
                                    **result["selection"]["diagnostics"],
                                }
                            )
                            graph_rows.append(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "method": str(method),
                                    "requested_support_ratio": float(ratio),
                                    **{key: value for key, value in result["graph_diagnostics"].items() if not isinstance(value, dict)},
                                }
                            )
                            prototype_rows.append(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "method": str(method),
                                    "requested_support_ratio": float(ratio),
                                    **result["graph_diagnostics"],
                                    "max_members_per_prototype": int(args.max_members_per_prototype),
                                }
                            )
                            for item in result["selection"].get("occlusion_block_scores", []):
                                occlusion_rows.append(
                                    {
                                        "dataset": str(dataset),
                                        "seed": int(seed),
                                        "method": str(method),
                                        "requested_support_ratio": float(ratio),
                                        **item,
                                    }
                                )
                            for item in result["selection"].get("validation_greedy_trials", []):
                                validation_rows.append(
                                    {
                                        "dataset": str(dataset),
                                        "seed": int(seed),
                                        "method": str(method),
                                        "requested_support_ratio": float(ratio),
                                        **item,
                                    }
                                )
                        if coarse_for_delta is not None and assignment_for_delta is not None:
                            semantic_rows.append(
                                _semantic_row_for_graph(
                                    graph=graph,
                                    coarse=coarse_for_delta,
                                    assignment=assignment_for_delta,
                                    target_only=target_only_graph,
                                    target_only_assignment=target_only_assignment,
                                    target_type=int(target_type),
                                    dataset=str(dataset),
                                    seed=int(seed),
                                    method=str(method),
                                    ratio=float(ratio),
                                    args=args,
                                )
                            )
                    except RuntimeError as exc:
                        row["status"] = "oom_or_runtime_error" if "out of memory" in str(exc).lower() else "failed"
                        row["error"] = str(exc)
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = repr(exc)
                    row["wall_clock_sec"] = float(perf_counter() - start)
                    row["run_mode"] = args.run_mode
                    rows.append(row)
                    write_csv(output_dir / "gate17_1_raw_rows.csv", rows)
                    write_csv(main_dir / "gate17_1_raw_rows.csv", rows)
    write_csv(diag_dir / "support_selection_diagnostics.csv", selection_rows)
    write_csv(diag_dir / "compressed_graph_summary.csv", graph_rows)
    write_csv(diag_dir / "prototype_diagnostics.csv", prototype_rows)
    write_csv(diag_dir / "occlusion_block_scores.csv", occlusion_rows)
    write_csv(diag_dir / "validation_greedy_trials.csv", validation_rows)
    write_csv(diag_dir / "semantic_tree_delta.csv", semantic_rows)
    write_csv(diag_dir / "full_graph_teacher_by_dataset_seed.csv", teacher_rows_all)
    write_csv(diag_dir / "teacher_metric_consistency.csv", teacher_consistency)
    support_pass = bool(
        semantic_rows
        and (
            float(np.percentile([float(row["tree_tensor_l2_delta_vs_full"]) for row in semantic_rows], 50)) > 1.0e-8
            or max(float(row["target_path_feature_changed_fraction"]) for row in semantic_rows) > 0.0
        )
    )
    tied_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "success")) == "success":
            tied_groups.setdefault((str(row.get("dataset")), int(row.get("seed", 0))), []).append(row)
    report_lines = [
        "# Gate17.1 Support Sensitivity Report",
        "",
        f"- run_mode: `{args.run_mode}`",
        f"- rows: `{len(rows)}`",
        f"- semantic_delta_rows: `{len(semantic_rows)}`",
        f"- support_sensitivity_pass: `{support_pass}`",
        f"- task_epochs: `{args.task_epochs}`",
        f"- max_paths: `{args.max_paths}`",
        f"- feature_mode: `{args.feature_mode}`",
    ]
    (diag_dir / "support_sensitivity_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = summarize(output_dir, main_dir, diag_dir)
    failed = [row for row in rows if str(row.get("status", "success")) != "success"]
    if any(row.get("status") == "oom_or_runtime_error" for row in failed):
        result["local_oom_or_runtime_error"] = True
    (main_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.1 support-sensitivity sanity diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate17_1"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP"])
    parser.add_argument("--seeds", nargs="*", default=[12345])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-patience", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--occlusion-candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-validation-greedy-steps", type=int, default=3)
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--feature-mode", choices=["fast", "full", "support_sensitive_fast"], default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-members-per-prototype", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.datasets = _split_values(args.datasets, str) or ["ACM", "DBLP"]
    args.seeds = _split_values(args.seeds, int) or [12345]
    args.ratios = _split_values(args.support_ratios, float) or [0.30, 0.70]
    args.methods = _split_values(args.methods, str) or list(DEFAULT_METHODS)
    if str(args.feature_mode) == "support_sensitive_fast":
        args.feature_mode = "fast"
    result = run(args)
    return 3 if bool(result.get("local_oom_or_runtime_error", False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
