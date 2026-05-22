from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.gate17_3_budget import compute_gate17_3_budget_fields
from experiments.scripts.gate17_4_h6 import (
    coarse_graph_hash,
    compute_h6_equivalence_fields,
    edge_mass_delta,
    export_h6_artifacts,
    feature_mean_delta,
    induced_coarse_graph,
    semantic_delta_vs_h6,
    selected_support_representatives_from_assignment,
    write_artifact_limitations,
)
from experiments.scripts.run_gate17_1_support_sensitivity import (
    _full_graph_row,
    _semantic_row_for_graph,
    _target_only_empty_support_graph,
    _target_only_row,
)
from experiments.scripts.run_gate17_3_lossy_prototype_feedback import _overlap_fields, _selector_for_method as _gate17_3_selector_for_method
from experiments.scripts.run_gate17_support_selection import _flat_payload, _mask, _row_from_task, _split_values
from experiments.scripts.summarize_gate17_4 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


GATE17_4_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
H6_CONSTRUCTION_CONTROL_METHOD = "HeSF-SS-H6-equivalence-control"
H6_SELECTED_SET_CONTROL_METHOD = "HeSF-SS-H6-selected-set-control"
BASELINES = {
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "random-support-only",
}
MAIN_CANDIDATE_METHODS = (
    "HeSF-SS-real-validation-neutral-fill",
    "HeSF-SS-real-occlusion-neutral-fill",
    H6_CONSTRUCTION_CONTROL_METHOD,
    "HeSF-SS-H6-cluster-validation-neutral-fill",
    "HeSF-SS-H6-cluster-occlusion-neutral-fill",
)
DIAGNOSTIC_ONLY_METHODS = (
    "HeSF-SS-real-validation-no-fallback",
    "HeSF-SS-real-occlusion-selection-only",
    "HeSF-SS-full-residual-prototype-upperbound",
    "HeSF-SS-lossy-prototype-fixed-saturation",
    H6_SELECTED_SET_CONTROL_METHOD,
)
DEFAULT_METHODS = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "random-support-only",
    *MAIN_CANDIDATE_METHODS,
    *DIAGNOSTIC_ONLY_METHODS,
)


def _semantic_fields_for_raw_row(semantic: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_tree_hash": semantic.get("coarse_tree_hash", ""),
        "tree_tensor_l2_delta_vs_full": semantic.get("tree_tensor_l2_delta_vs_full"),
        "tree_tensor_cosine_delta_vs_full": semantic.get("tree_tensor_cosine_delta_vs_full"),
        "target_path_feature_changed_fraction": semantic.get("target_path_feature_changed_fraction"),
        "allclose_to_full": semantic.get("allclose_to_full"),
    }


def parse_dataset_seeds(values: list[str] | tuple[str, ...] | str | None) -> list[tuple[str, int]]:
    if values is None or values == "":
        return [(dataset, seed) for dataset, seed in GATE17_4_SINGLE_SEED_BY_DATASET.items()]
    tokens: list[str] = []
    raw_values = [values] if isinstance(values, str) else list(values)
    for value in raw_values:
        tokens.extend(item for item in str(value).replace(",", " ").split() if item)
    out: list[tuple[str, int]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"dataset seed token must be DATASET:SEED, got {token!r}")
        dataset, seed = token.split(":", 1)
        dataset = dataset.strip()
        if dataset not in GATE17_4_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate17.4 dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


def _selector_for_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    mapped = "HeSF-SS-real-occlusion-lossy-prototype" if method == "HeSF-SS-lossy-prototype-fixed-saturation" else str(method)
    cfg = _gate17_3_selector_for_method(mapped, args)
    if method == "HeSF-SS-lossy-prototype-fixed-saturation":
        cfg = replace(cfg, max_members_per_prototype=min(int(cfg.max_members_per_prototype), 256))
    return cfg


def _budget_update(row: dict[str, Any], support_count: int, *, selected_raw: int, graph_diag: dict[str, Any] | None = None) -> None:
    graph_diag = graph_diag or {}
    fields = compute_gate17_3_budget_fields(
        original_support_nodes=int(support_count),
        requested_support_ratio=float(row.get("requested_support_ratio", 0.0) or 0.0),
        selected_raw_support_count=int(graph_diag.get("selected_raw_support_count", selected_raw) or 0),
        forced_raw_support_count=int(graph_diag.get("forced_raw_bridge_count", 0) or 0),
        prototype_background_count=int(graph_diag.get("prototype_background_count", 0) or 0),
        prototype_member_count_sum=int(graph_diag.get("prototype_member_count_sum", 0) or 0),
        prototype_member_budget_total=graph_diag.get("prototype_member_budget_total"),
        prototype_budget_fraction=float(row.get("prototype_budget_fraction", 0.10) or 0.10),
        full_residual_upperbound=bool(graph_diag.get("full_residual_upperbound", False)),
        method=str(row.get("method", "")),
        no_test_leakage=not bool(row.get("selector_uses_test_labels", False) or row.get("teacher_uses_test_labels_for_training", False)),
    )
    row.update(fields)
    row["original_support_nodes"] = int(support_count)


def _task_metrics_row(row: dict[str, Any], task: dict[str, Any]) -> None:
    _row_from_task(row, task)
    row["primary_eval_mode"] = task.get("primary_eval_mode", "compressed_projected")


def _eval_task(graph, coarse, assignment, *, seed: int, split: dict[str, np.ndarray], target_type: int, args: argparse.Namespace, epochs: int | None = None) -> dict[str, Any]:
    return evaluate_hettree_task(
        graph,
        coarse,
        np.asarray(assignment, dtype=np.int64),
        seed=int(seed),
        epochs=int(args.task_epochs if epochs is None else epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(target_type),
        official_split_nodes=split,
        primary_eval_mode=str(args.primary_eval_mode),
        early_stopping=True,
        monitor=str(args.monitor),
        max_paths=int(args.max_paths),
    ).metrics


def _support_cluster_ids(graph, assignment: np.ndarray, target_type: int) -> np.ndarray:
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    return np.asarray(sorted({int(assignment[int(node)]) for node in support_nodes.tolist()}), dtype=np.int64)


def _candidate_clusters(graph, assignment: np.ndarray, target_type: int, limit: int) -> list[tuple[int, int]]:
    arr = np.asarray(assignment, dtype=np.int64)
    clusters = []
    for cluster_id in _support_cluster_ids(graph, arr, int(target_type)):
        members = np.flatnonzero(arr == int(cluster_id)).astype(np.int64)
        clusters.append((int(cluster_id), int(len(members))))
    return sorted(clusters, key=lambda item: (-item[1], item[0]))[: max(0, int(limit))]


def _cluster_feedback_rows(
    *,
    graph,
    h6_coarse,
    h6_assignment: np.ndarray,
    h6_task: dict[str, Any],
    dataset: str,
    seed: int,
    ratio: float,
    target_type: int,
    split: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    support_clusters = _support_cluster_ids(graph, h6_assignment, int(target_type))
    for cluster_id, member_count in _candidate_clusters(graph, h6_assignment, int(target_type), int(args.h6_cluster_candidate_pool_size)):
        keep = np.asarray([node for node in range(int(h6_coarse.num_nodes)) if int(node) != int(cluster_id)], dtype=np.int64)
        dropped_graph, dropped_assignment = induced_coarse_graph(h6_coarse, h6_assignment, keep)
        task = _eval_task(
            graph,
            dropped_graph,
            dropped_assignment,
            seed=int(seed),
            split=split,
            target_type=int(target_type),
            args=args,
            epochs=int(args.cluster_feedback_epochs),
        )
        delta_macro = float(h6_task.get("validation_macro_f1", 0.0) or 0.0) - float(task.get("validation_macro_f1", 0.0) or 0.0)
        delta_acc = float(h6_task.get("validation_accuracy", 0.0) or 0.0) - float(task.get("validation_accuracy", 0.0) or 0.0)
        importance = max(delta_macro, 0.0) + max(delta_acc, 0.0)
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "requested_support_ratio": float(ratio),
                "cluster_id": int(cluster_id),
                "cluster_size": int(member_count),
                "support_cluster_count": int(len(support_clusters)),
                "validation_gain": float(delta_macro),
                "delta_val_ce": float(max(0.0, delta_macro)),
                "delta_val_macro_f1": float(delta_macro),
                "delta_margin": float(delta_acc),
                "delta_teacher_kl": 0.0,
                "final_block_importance": float(importance),
                "selected": True,
                "fill_reason": "positive" if importance > 1.0e-6 else "neutral",
            }
        )
    return rows


def _cluster_feedback_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    prefix = "validation" if "validation" in method else "occlusion"
    gains = [float(row.get("validation_gain", 0.0) or 0.0) for row in rows]
    importances = [float(row.get("final_block_importance", 0.0) or 0.0) for row in rows]
    positive = sum(1 for value in gains if value > 1.0e-6)
    neutral = sum(1 for value in gains if abs(value) <= 1.0e-4)
    if prefix == "validation":
        return {
            "validation_trial_count": int(len(rows)),
            "validation_positive_gain_count": int(positive),
            "validation_gain_max": float(max(gains)) if gains else 0.0,
            "validation_gain_mean": float(np.mean(gains)) if gains else 0.0,
            "validation_gain_history": json.dumps(gains),
            "validation_neutral_fill_count": int(neutral),
            "validation_proxy_fill_count": 0,
            "validation_feedback_degenerate": bool(positive == 0),
            "validation_signal_pass": bool(positive > 0),
        }
    task_nonzero = sum(1 for value in importances if value > 1.0e-6)
    return {
        "occlusion_trial_count": int(len(rows)),
        "occlusion_task_nonzero_delta_rate": float(task_nonzero / max(len(rows), 1)),
        "occlusion_tree_nonzero_delta_rate": 0.0,
        "occlusion_task_signal_pass": bool(task_nonzero > 0),
        "occlusion_tree_signal_pass": False,
        "delta_val_ce_max": float(max([row["delta_val_ce"] for row in rows])) if rows else 0.0,
        "delta_val_macro_f1_max": float(max(gains)) if gains else 0.0,
        "delta_margin_max": float(max([row["delta_margin"] for row in rows])) if rows else 0.0,
        "delta_teacher_kl_max": 0.0,
        "final_block_importance_max": float(max(importances)) if importances else 0.0,
        "negative_importance_selected_count": 0,
        "occlusion_neutral_fill_count": int(neutral),
        "occlusion_proxy_fill_count": 0,
        "occlusion_feedback_degenerate": bool(task_nonzero == 0),
    }


def _h6_equivalence_row(
    *,
    dataset: str,
    seed: int,
    ratio: float,
    mode: str,
    graph,
    control,
    control_assignment: np.ndarray,
    h6,
    h6_assignment: np.ndarray,
    h6_task: dict[str, Any],
    control_task: dict[str, Any],
    target_type: int,
    selected_nodes: np.ndarray,
    h6_nodes: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    semantic = semantic_delta_vs_h6(
        original=graph,
        control=control,
        control_assignment=np.asarray(control_assignment, dtype=np.int64),
        h6=h6,
        h6_assignment=np.asarray(h6_assignment, dtype=np.int64),
        target_type=int(target_type),
        max_paths=int(args.max_paths),
    )
    edge = edge_mass_delta(control, h6)
    feature = feature_mean_delta(control, h6)
    overlap = _overlap_fields(selected_nodes, h6_nodes)
    fields = compute_h6_equivalence_fields(
        mode=mode,
        h6_macro_f1=float(h6_task.get("macro_f1", 0.0) or 0.0),
        control_macro_f1=float(control_task.get("macro_f1", 0.0) or 0.0),
        h6_accuracy=float(h6_task.get("accuracy", 0.0) or 0.0),
        control_accuracy=float(control_task.get("accuracy", 0.0) or 0.0),
        h6_validation_macro_f1=float(h6_task.get("validation_macro_f1", 0.0) or 0.0),
        control_validation_macro_f1=float(control_task.get("validation_macro_f1", 0.0) or 0.0),
        tree_l2_delta_vs_h6=float(semantic["tree_l2_delta_vs_h6"]),
        tree_cosine_delta_vs_h6=float(semantic["tree_cosine_delta_vs_h6"]),
        tree_hash_equal_to_h6=bool(semantic["tree_hash_equal_to_h6"]),
        coarse_graph_hash_equal_to_h6=bool(coarse_graph_hash(control) == coarse_graph_hash(h6)),
        edge_mass_l1_delta_vs_h6=float(edge["edge_mass_l1_delta_vs_h6"]),
        edge_mass_linf_delta_vs_h6=float(edge["edge_mass_linf_delta_vs_h6"]),
        feature_mean_l2_delta_vs_h6=float(feature["feature_mean_l2_delta_vs_h6"]),
        assignment_equivalent_to_h6=bool(np.array_equal(np.asarray(control_assignment, dtype=np.int64), np.asarray(h6_assignment, dtype=np.int64))),
        selected_jaccard_with_H6=float(overlap["selected_jaccard_with_H6"]),
        selected_recall_of_H6=float(overlap["selected_recall_of_H6"]),
        selected_precision_vs_H6=float(overlap["selected_precision_vs_H6"]),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "requested_support_ratio": float(ratio),
        **fields,
        **semantic,
        **edge,
        **feature,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate17.4 requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.out_dir)
    diag_dir = output_dir / "diagnostics"
    artifact_dir = diag_dir / "gate17_4_h6_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_limitations(diag_dir / "gate17_4_h6_artifact_limitations.md")
    (diag_dir / "typedhash_skip_note.md").write_text(
        "TypedHash skipped in Gate17.4 for speed; H6/flatten used as strong baselines.\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    h6_equivalence_rows: list[dict[str, Any]] = []
    cluster_feedback_rows: list[dict[str, Any]] = []
    h6_overlap_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    edge_mass_rows: list[dict[str, Any]] = []
    feature_mass_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
        target_type = infer_target_node_type(graph)
        support_count = int(np.sum(graph.node_type != int(target_type)))
        train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
        split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
        train_mask = _mask(train_nodes, graph.num_nodes)
        val_mask = _mask(val_nodes, graph.num_nodes)
        test_mask = _mask(test_nodes, graph.num_nodes)
        target_only_graph, target_only_assignment = _target_only_empty_support_graph(graph, int(target_type))
        teacher = {"metrics": {"teacher_uses_test_labels_for_training": False, "teacher_reliable_for_importance": False}, "teacher_uses_test_labels_for_training": False}
        for ratio in args.ratios:
            h6_coarse, h6_assignment, h6_diag = run_support_baseline(graph, baseline="H6-no-spec-support-only", ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
            h6_assignment = np.asarray(h6_assignment, dtype=np.int64)
            h6_nodes = selected_support_representatives_from_assignment(graph, h6_assignment, int(target_type))
            h6_task = _eval_task(graph, h6_coarse, h6_assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
            h6_artifact = export_h6_artifacts(
                output_dir=artifact_dir,
                dataset=str(dataset),
                seed=int(seed),
                ratio=float(ratio),
                original=graph,
                h6=h6_coarse,
                h6_assignment=h6_assignment,
                target_type=int(target_type),
                max_paths=int(args.max_paths),
            )
            cluster_feedback = _cluster_feedback_rows(
                graph=graph,
                h6_coarse=h6_coarse,
                h6_assignment=h6_assignment,
                h6_task=h6_task,
                dataset=str(dataset),
                seed=int(seed),
                ratio=float(ratio),
                target_type=int(target_type),
                split=split,
                args=args,
            )
            cluster_feedback_rows.extend(cluster_feedback)
            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **split_protocol}
                selected_nodes = np.empty(0, dtype=np.int64)
                coarse_for_delta = None
                assignment_for_delta = None
                graph_diag: dict[str, Any] = {}
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                        coarse_for_delta = graph
                        assignment_for_delta = np.arange(graph.num_nodes, dtype=np.int64)
                        selected_raw = support_count
                    elif method == "target-only-empty-support":
                        target_row, coarse_for_delta, assignment_for_delta = _target_only_row(graph, str(dataset), int(seed), float(ratio), args, split)
                        row.update(target_row)
                        selected_raw = 0
                    elif method in BASELINES:
                        if method == "H6-no-spec-support-only":
                            coarse, assignment, diag, task = h6_coarse, h6_assignment, h6_diag, h6_task
                            selected_nodes = h6_nodes
                            row.update(h6_artifact)
                        else:
                            coarse, assignment, diag = run_support_baseline(graph, baseline=str(method), ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                            assignment = np.asarray(assignment, dtype=np.int64)
                            task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        coarse_for_delta = coarse
                        assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        final_support = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                        row["selected_support_count"] = int(final_support)
                        row.update(budget_diagnostics(num_support=support_count, support_ratio=float(ratio), realized_support_count=final_support))
                        _task_metrics_row(row, task)
                        selected_raw = final_support
                    elif method == H6_CONSTRUCTION_CONTROL_METHOD:
                        coarse_for_delta = h6_coarse
                        assignment_for_delta = h6_assignment
                        _task_metrics_row(row, h6_task)
                        selected_nodes = h6_nodes
                        selected_raw = int(np.sum(h6_coarse.node_type != int(target_type)))
                        row["h6_control_mode"] = "construction"
                        row["validation_signal_pass"] = True
                        row["occlusion_task_signal_pass"] = True
                        eq = _h6_equivalence_row(dataset=str(dataset), seed=int(seed), ratio=float(ratio), mode="construction", graph=graph, control=h6_coarse, control_assignment=h6_assignment, h6=h6_coarse, h6_assignment=h6_assignment, h6_task=h6_task, control_task=h6_task, target_type=int(target_type), selected_nodes=selected_nodes, h6_nodes=h6_nodes, args=args)
                        h6_equivalence_rows.append(eq)
                        row.update({key: value for key, value in eq.items() if key not in {"dataset", "seed", "requested_support_ratio"}})
                    elif method == H6_SELECTED_SET_CONTROL_METHOD:
                        cfg = SupportSelectorConfig(
                            selector="teacher_topk",
                            background_strategy="drop",
                            allow_background_bucket=False,
                            residual_prototype_mode="none",
                            force_raw_bridge_nodes=False,
                            force_raw_keep_high_degree_bridges=False,
                            allow_proxy_fill=False,
                        )
                        control_coarse, control_assignment_obj, graph_diag = build_selected_support_graph(graph, h6_nodes, cfg, target_node_type=int(target_type), support_features=None)
                        control_task = _eval_task(graph, control_coarse, control_assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        _task_metrics_row(row, control_task)
                        coarse_for_delta = control_coarse
                        assignment_for_delta = np.asarray(control_assignment_obj.assignment, dtype=np.int64)
                        selected_nodes = h6_nodes
                        selected_raw = len(h6_nodes)
                        row["h6_control_mode"] = "selected_set"
                        eq = _h6_equivalence_row(dataset=str(dataset), seed=int(seed), ratio=float(ratio), mode="selected_set", graph=graph, control=control_coarse, control_assignment=control_assignment_obj.assignment, h6=h6_coarse, h6_assignment=h6_assignment, h6_task=h6_task, control_task=control_task, target_type=int(target_type), selected_nodes=selected_nodes, h6_nodes=h6_nodes, args=args)
                        h6_equivalence_rows.append(eq)
                        row.update({key: value for key, value in eq.items() if key not in {"dataset", "seed", "requested_support_ratio"}})
                    elif method in {"HeSF-SS-H6-cluster-validation-neutral-fill", "HeSF-SS-H6-cluster-occlusion-neutral-fill"}:
                        _task_metrics_row(row, h6_task)
                        coarse_for_delta = h6_coarse
                        assignment_for_delta = h6_assignment
                        selected_nodes = h6_nodes
                        selected_raw = int(np.sum(h6_coarse.node_type != int(target_type)))
                        row["h6_control_mode"] = "cluster_units"
                        row["h6_cluster_units_used"] = True
                        row.update(_cluster_feedback_summary(cluster_feedback, str(method)))
                    else:
                        cfg = Gate15Config(target_node_type=int(target_type), selector=replace(_selector_for_method(str(method), args), support_ratios=(float(ratio),)))
                        result = run_supervised_support_selection_pipeline(graph, labels, train_mask, val_mask, test_mask, cfg, support_ratio=float(ratio), teacher_outputs=teacher, method_name=str(method), seed=int(seed), task_epochs=int(args.task_epochs), task_hidden_dim=int(args.task_hidden_dim), task_max_paths=int(args.max_paths), device=str(args.device))
                        row.update(_flat_payload(result))
                        coarse_for_delta = result["coarse_graph"]
                        assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                        graph_diag = dict(result["graph_diagnostics"])
                        selected_nodes = np.asarray(result["selection"]["selected_support_nodes"], dtype=np.int64)
                        selected_raw = int(graph_diag.get("selected_raw_support_count", len(selected_nodes)) or 0)
                        selection_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **result["selection"]["diagnostics"]})
                        for item in result["selection"].get("validation_greedy_trials", []):
                            validation_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item})
                        for item in result["selection"].get("occlusion_block_scores", []):
                            occlusion_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item})

                    if coarse_for_delta is not None and assignment_for_delta is not None:
                        semantic = _semantic_row_for_graph(graph=graph, coarse=coarse_for_delta, assignment=np.asarray(assignment_for_delta, dtype=np.int64), target_only=target_only_graph, target_only_assignment=target_only_assignment, target_type=int(target_type), dataset=str(dataset), seed=int(seed), method=str(method), ratio=float(ratio), args=args)
                        semantic_rows.append(semantic)
                        row.update(_semantic_fields_for_raw_row(semantic))
                        if str(method).startswith("HeSF-SS"):
                            row["candidate_full_equivalent"] = bool(semantic.get("allclose_to_full", False))
                    row.update(_overlap_fields(selected_nodes, h6_nodes))
                    row.setdefault("selector_uses_test_labels", False)
                    row.setdefault("teacher_uses_test_labels_for_training", False)
                    row["no_test_leakage"] = not bool(row["selector_uses_test_labels"] or row["teacher_uses_test_labels_for_training"])
                    _budget_update(row, support_count, selected_raw=int(selected_raw), graph_diag=graph_diag)
                    if method in DIAGNOSTIC_ONLY_METHODS or method in BASELINES or method in {"full-graph-hettree-lite-tuned", "target-only-empty-support", H6_SELECTED_SET_CONTROL_METHOD, H6_CONSTRUCTION_CONTROL_METHOD}:
                        row["eligible_for_main_decision"] = bool(method == H6_CONSTRUCTION_CONTROL_METHOD and row["node_budget_exact_match"] and row["represented_context_exact_or_bounded"])
                    if method == "HeSF-SS-lossy-prototype-fixed-saturation":
                        saturated = (
                            float(row.get("prototype_saturation_rate", 0.0) or 0.0) > 0.5
                            or float(row.get("prototype_member_count_p90", 0.0) or 0.0) >= float(row.get("max_members_per_prototype", 512) or 512)
                            or float(row.get("prototype_member_count_p99", 0.0) or 0.0) >= float(row.get("max_members_per_prototype", 512) or 512)
                        )
                        row["prototype_diagnostic_only"] = bool(saturated)
                        if saturated:
                            row["eligible_for_main_decision"] = False
                    row["status"] = row.get("status", "success")
                    row.setdefault("primary_eval_mode", str(args.primary_eval_mode))
                    row["wall_clock_sec"] = float(perf_counter() - start)
                    row["run_mode"] = "gate17_4_h6_equivalence"
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                rows.append(row)
                write_csv(output_dir / "gate17_4_raw_rows.csv", rows)
                budget_rows.append({key: row.get(key, "") for key in row.keys() if key in {"dataset", "seed", "method", "requested_support_ratio"} or "budget" in key or "represented_context" in key or key in {"eligible_for_main_decision", "full_residual_upperbound", "node_budget_count", "node_budget_ratio", "node_budget_exact_match"}})
                prototype_rows.append({key: row.get(key, "") for key in row.keys() if key in {"dataset", "seed", "method", "requested_support_ratio", "prototype_saturation_rate", "prototype_member_count_p90", "prototype_member_count_p99", "max_members_per_prototype", "prototype_diagnostic_only"}})
                h6_overlap_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **_overlap_fields(selected_nodes, h6_nodes)})
                graph_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), "coarse_nodes": getattr(coarse_for_delta, "num_nodes", ""), "coarse_graph_hash": coarse_graph_hash(coarse_for_delta) if coarse_for_delta is not None else ""})
                if coarse_for_delta is not None:
                    edge_mass_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **edge_mass_delta(coarse_for_delta, h6_coarse)})
                    feature_mass_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **feature_mean_delta(coarse_for_delta, h6_coarse)})

    write_csv(diag_dir / "gate17_4_h6_equivalence.csv", h6_equivalence_rows)
    write_csv(diag_dir / "gate17_4_h6_cluster_feedback.csv", cluster_feedback_rows)
    write_csv(diag_dir / "gate17_4_h6_overlap.csv", h6_overlap_rows)
    write_csv(diag_dir / "gate17_4_budget_breakdown.csv", budget_rows)
    write_csv(diag_dir / "gate17_4_represented_context_budget.csv", budget_rows)
    write_csv(diag_dir / "gate17_4_candidate_semantic_delta.csv", [row for row in semantic_rows if str(row.get("method", "")).startswith("HeSF-SS")])
    write_csv(diag_dir / "gate17_4_acm_saturation_curve.csv", [row for row in rows if str(row.get("dataset", "")).upper() == "ACM"])
    write_csv(diag_dir / "gate17_4_validation_trials.csv", validation_rows)
    write_csv(diag_dir / "gate17_4_occlusion_scores.csv", occlusion_rows)
    write_csv(diag_dir / "gate17_4_prototype_saturation.csv", prototype_rows)
    write_csv(diag_dir / "gate17_4_edge_mass_delta_vs_h6.csv", edge_mass_rows)
    write_csv(diag_dir / "gate17_4_feature_mass_delta_vs_h6.csv", feature_mass_rows)
    write_csv(diag_dir / "gate17_4_compressed_graph_summary.csv", graph_rows)
    write_csv(diag_dir / "gate17_4_support_selection_diagnostics.csv", selection_rows)
    return summarize(output_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.4 H6 construction-equivalence diagnostic.")
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.03, 0.10, 0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", "--output-dir", type=Path, default=Path("outputs/gate17_4_h6_equivalence"))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--cluster-feedback-epochs", type=int, default=3)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--task-lr", type=float, default=0.005)
    parser.add_argument("--task-dropout", type=float, default=0.25)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--feature-mode", default="full")
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=16)
    parser.add_argument("--short-eval-epochs", type=int, default=3)
    parser.add_argument("--max-validation-greedy-steps", type=int, default=5)
    parser.add_argument("--occlusion-candidate-pool-size", type=int, default=8)
    parser.add_argument("--occlusion-short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-patience", type=int, default=2)
    parser.add_argument("--max-members-per-prototype", type=int, default=512)
    parser.add_argument("--prototype-budget-fraction", type=float, default=0.10)
    parser.add_argument("--max-represented-support-ratio-slack", type=float, default=0.10)
    parser.add_argument("--h6-cluster-candidate-pool-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = set(str(dataset) for dataset in args.datasets)
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if dataset in allowed]
    args.ratios = _split_values(args.support_ratios, float) or [0.03, 0.10, 0.30, 0.70]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
