from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.audit_gate17_code import write_audit
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.summarize_gate17 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.contribution import compute_support_importance
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher


DATASETS = ("ACM", "DBLP", "IMDB")
SEEDS = (12345, 23456, 34567, 45678, 56789)
RATIOS = (0.30, 0.50, 0.70)
BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
PRIMARY_METHODS = (
    "full-graph-hettree-lite-tuned",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
    "HeSF-SS-sensitivity-plus-prototype",
    "HeSF-SS-real-occlusion-block",
    "HeSF-SS-real-validation-block-greedy",
    "HeSF-SS-dblp-aware-prototype",
    "HeSF-SS-occlusion-plus-dblp-prototype",
)
APPENDIX_METHODS = (
    "HeSF-SS-teacher-topk",
    "HeSF-SS-teacher-diverse-topk",
    "HeSF-SS-validation-proxy-diverse",
    "HeSF-SS-hybrid-teacher-response",
    "HeSF-SS-sensitivity-block-selector",
    "HeSF-SS-true-validation-block-greedy",
)


def _split_values(values: list[Any] | tuple[Any, ...] | None, cast=str) -> list[Any]:
    if not values:
        return []
    out: list[Any] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                out.append(cast(item))
    return out


def _mask(nodes: np.ndarray, total: int) -> np.ndarray:
    out = np.zeros(int(total), dtype=bool)
    out[np.asarray(nodes, dtype=np.int64)] = True
    return out


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _argmax_or_unknown(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0 or matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.full(matrix.shape[0] if matrix.ndim else 0, -1, dtype=np.int64)
    ids = np.argmax(matrix, axis=1).astype(np.int64)
    ids[np.sum(matrix, axis=1) <= 1.0e-12] = -1
    return ids


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-12)
    return (matrix / denom).astype(np.float32)


def _fast_support_features(
    graph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_type: int,
) -> dict[str, Any]:
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    relation_ids = sorted(int(item) for item in graph.relations)
    relation_pos = {relation_id: idx for idx, relation_id in enumerate(relation_ids)}
    relation_profile = np.zeros((graph.num_nodes, max(1, len(relation_ids))), dtype=np.float32)
    degree_profile = np.zeros((graph.num_nodes, max(1, len(relation_ids) * 2)), dtype=np.float32)
    class_count = int(np.max(labels[labels >= 0])) + 1 if np.any(labels >= 0) else 1
    class_footprint = np.zeros((graph.num_nodes, class_count), dtype=np.float32)
    anchor_distribution = np.zeros((graph.num_nodes, 3), dtype=np.float32)
    train_targets = np.flatnonzero((graph.node_type == int(target_type)) & np.asarray(train_mask, dtype=bool) & (labels >= 0)).astype(np.int64)
    train_target_set = {int(node) for node in train_targets}
    for relation_id, rel in graph.relations.items():
        pos = relation_pos[int(relation_id)]
        weight = np.asarray(rel.weight, dtype=np.float32)
        np.add.at(relation_profile[:, pos], rel.src, weight)
        np.add.at(relation_profile[:, pos], rel.dst, weight)
        np.add.at(degree_profile[:, pos], rel.src, weight)
        np.add.at(degree_profile[:, pos + len(relation_ids)], rel.dst, weight)
        if int(rel.src_type) == int(target_type):
            target_end = rel.src
            support_end = rel.dst
        elif int(rel.dst_type) == int(target_type):
            target_end = rel.dst
            support_end = rel.src
        else:
            continue
        target_end = np.asarray(target_end, dtype=np.int64)
        support_end = np.asarray(support_end, dtype=np.int64)
        valid = np.asarray(train_mask, dtype=bool)[target_end] & (np.asarray(labels)[target_end] >= 0)
        if not np.any(valid):
            continue
        valid_support = support_end[valid]
        valid_labels = np.asarray(labels, dtype=np.int64)[target_end[valid]]
        valid_weight = weight[valid].astype(np.float32, copy=False)
        np.add.at(class_footprint, (valid_support, valid_labels), valid_weight)
        np.add.at(anchor_distribution[:, 0], valid_support, 1.0)
        np.add.at(anchor_distribution[:, 1], valid_support, valid_weight)
        np.maximum.at(anchor_distribution[:, 2], valid_support, valid_weight)
    component_matrices = {
        "relation_profile": _normalize_rows(relation_profile[support_nodes]),
        "degree_profile": _normalize_rows(degree_profile[support_nodes]),
        "class_footprint": _normalize_rows(class_footprint[support_nodes]),
        "anchor_distribution": _normalize_rows(anchor_distribution[support_nodes]),
        "target_response_signature": _normalize_rows(class_footprint[support_nodes]),
        "relation_response_signature": _normalize_rows(relation_profile[support_nodes]),
    }
    return {
        "support_nodes": support_nodes,
        "support_node_types": graph.node_type[support_nodes].astype(np.int32, copy=False),
        "feature_matrix": np.concatenate(list(component_matrices.values()), axis=1).astype(np.float32, copy=False),
        "component_matrices": component_matrices,
        "all_node_component_matrices": {
            "class_footprint": _normalize_rows(class_footprint),
            "relation_profile": _normalize_rows(relation_profile),
            "anchor_distribution": _normalize_rows(anchor_distribution),
            "degree_profile": _normalize_rows(degree_profile),
        },
        "diagnostics": {
            "zero_footprint_support_share": float(np.mean(np.sum(class_footprint[support_nodes], axis=1) <= 1.0e-12)) if len(support_nodes) else 0.0,
            "support_feature_builder": "gate17_fast_graph_features",
        },
        "target_node_type": int(target_type),
        "selector_uses_test_labels": False,
    }


def _row_from_task(row: dict[str, Any], task: dict[str, Any]) -> None:
    row.update(
        {
            "primary_eval_mode": task.get("primary_eval_mode", "compressed_projected"),
            "primary_task_metric_name": task.get("primary_task_metric_name", "projected_original_macro_f1"),
            "macro_f1": _metric(task, "macro_f1"),
            "micro_f1": _metric(task, "micro_f1"),
            "accuracy": _metric(task, "accuracy"),
            "validation_macro_f1": _metric(task, "validation_macro_f1"),
            "validation_accuracy": _metric(task, "validation_accuracy"),
            "projected_macro_f1": _metric(task, "projected_original_macro_f1"),
            "transfer_macro_f1": _metric(task, "transfer_original_macro_f1"),
            "projected_accuracy": _metric(task, "projected_original_accuracy"),
            "transfer_accuracy": _metric(task, "transfer_original_accuracy"),
            "hybrid_target_macro_f1": _metric(task, "hybrid_target_original_macro_f1"),
            "hybrid_target_accuracy": _metric(task, "hybrid_target_original_accuracy"),
            "projected_vs_transfer_macro_gap": _metric(task, "projected_vs_transfer_macro_gap"),
            "projected_vs_transfer_accuracy_gap": _metric(task, "projected_vs_transfer_accuracy_gap"),
            "best_epoch": int(task.get("best_epoch", -1) or -1),
            "early_stopped": bool(task.get("early_stopped", False)),
            "coarsening_wall_clock": 0.0,
            "task_train_wall_clock": float(task.get("train_time", 0.0) or 0.0),
            "refine_wall_clock": 0.0,
            "total_wall_clock": float(task.get("total_time", 0.0) or 0.0),
            "peak_rss": 0.0,
            "peak_vram_allocated": float(task.get("peak_vram_allocated_mb", 0.0) or 0.0),
            "status": "success" if not task.get("skipped", False) else "skipped",
            "skip_reason": task.get("skip_reason", ""),
            "evaluator_status": "diagnostic_lite_only",
        }
    )


def _selector_for_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    common = {
        "candidate_pool_size": int(args.candidate_pool_size),
        "short_eval_epochs": int(args.short_eval_epochs),
        "max_validation_greedy_steps": int(args.max_validation_greedy_steps),
        "occlusion_candidate_pool_size": int(args.occlusion_candidate_pool_size),
        "occlusion_short_eval_epochs": int(args.occlusion_short_eval_epochs),
        "occlusion_short_patience": int(args.occlusion_short_patience),
        "max_members_per_prototype": int(args.max_members_per_prototype),
    }
    if method == "HeSF-SS-teacher-topk":
        return SupportSelectorConfig(selector="teacher_topk", background_strategy="typed_background", **common)
    if method == "HeSF-SS-teacher-diverse-topk":
        return SupportSelectorConfig(selector="teacher_diverse_topk", background_strategy="typed_background", **common)
    if method == "HeSF-SS-validation-proxy-diverse":
        return SupportSelectorConfig(selector="validation_proxy_diverse", background_strategy="typed_background", **common)
    if method == "HeSF-SS-true-validation-block-greedy":
        return SupportSelectorConfig(selector="true_validation_block_greedy", background_strategy="typed_background", **common)
    if method == "HeSF-SS-sensitivity-block-selector":
        return SupportSelectorConfig(selector="sensitivity_block_selector", background_strategy="typed_background", **common)
    if method == "HeSF-SS-sensitivity-plus-prototype":
        return SupportSelectorConfig(selector="sensitivity_block_selector", background_strategy="class_anchor_relation_prototype", **common)
    if method == "HeSF-SS-real-occlusion-block":
        return SupportSelectorConfig(selector="real_occlusion_block_selector", background_strategy="class_anchor_relation_prototype", **common)
    if method == "HeSF-SS-real-validation-block-greedy":
        return SupportSelectorConfig(selector="real_validation_block_greedy", background_strategy="class_anchor_relation_prototype", **common)
    if method == "HeSF-SS-dblp-aware-prototype":
        return SupportSelectorConfig(selector="sensitivity_block_selector", background_strategy="dblp_aware_prototype", block_key_mode="dblp_aware", **common)
    if method == "HeSF-SS-occlusion-plus-dblp-prototype":
        return SupportSelectorConfig(selector="occlusion_plus_dblp_prototype", background_strategy="dblp_aware_prototype", block_key_mode="dblp_aware", **common)
    if method == "HeSF-SS-hybrid-teacher-response":
        return SupportSelectorConfig(selector="hybrid_teacher_response", background_strategy="class_anchor_relation_prototype", **common)
    raise ValueError(f"unsupported Gate17 method: {method}")


def _importance_stats(importance: np.ndarray, selected_local: np.ndarray) -> dict[str, float]:
    values = np.asarray(importance, dtype=np.float64)
    selected = values[np.asarray(selected_local, dtype=np.int64)] if len(selected_local) else np.empty(0)
    return {
        "support_importance_mean": float(np.mean(values)) if len(values) else 0.0,
        "support_importance_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "selected_importance_mean": float(np.mean(selected)) if len(selected) else 0.0,
        "selected_importance_std": float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0,
    }


def _coverage_count(matrix: np.ndarray) -> int:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return 0
    return int(np.count_nonzero(np.sum(matrix, axis=0) > 1.0e-12))


def _run_fast_selection_method(
    graph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: Gate15Config,
    method: str,
    ratio: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    support_features = _fast_support_features(graph, labels, train_mask, int(cfg.target_node_type))
    importance = compute_support_importance(
        support_features,
        {"metrics": {"teacher_reliable_for_importance": False}},
        mode=str(cfg.selector.selector),
        lambda_response=float(cfg.regularizer.lambda_response),
    )
    trial_split = {"train": _mask(np.flatnonzero(train_mask), graph.num_nodes), "val": _mask(np.flatnonzero(val_mask), graph.num_nodes), "test": _mask(np.flatnonzero(val_mask), graph.num_nodes)}
    support_nodes_all = np.asarray(support_features["support_nodes"], dtype=np.int64)
    proxy_order = np.argsort(-np.asarray(importance["importance"], dtype=np.float64))
    budget_count = int(max(1, np.ceil(len(support_nodes_all) * float(ratio) - 1.0e-12))) if len(support_nodes_all) and float(ratio) > 0.0 else 0
    occlusion_context = support_nodes_all[proxy_order[:budget_count]].astype(np.int64, copy=False)
    trial_cache: dict[tuple[str, tuple[int, ...]], float] = {}

    def validation_score(selected_nodes: np.ndarray) -> float:
        key = tuple(sorted(int(node) for node in np.asarray(selected_nodes, dtype=np.int64).reshape(-1)))
        cache_key = ("validation", key)
        if cache_key in trial_cache:
            return float(trial_cache[cache_key])
        coarse, assignment, _diag = build_selected_support_graph(
            graph,
            np.asarray(key, dtype=np.int64),
            cfg.selector,
            target_node_type=int(cfg.target_node_type),
            support_features=support_features,
        )
        metrics = evaluate_hettree_task(
            graph,
            coarse,
            assignment.assignment,
            seed=int(seed),
            epochs=max(0, int(args.short_eval_epochs)),
            hidden_dim=int(args.task_hidden_dim),
            device=str(args.device),
            target_node_type=int(cfg.target_node_type),
            official_split_nodes=trial_split,
            primary_eval_mode=str(args.primary_eval_mode),
            early_stopping=True,
            monitor=str(args.monitor),
            max_paths=int(args.max_paths),
        ).metrics
        score = _metric(metrics, "validation_macro_f1")
        trial_cache[cache_key] = float(score)
        return float(score)

    def occlusion_score(occluded_nodes: np.ndarray) -> dict[str, float]:
        key = tuple(sorted(int(node) for node in np.asarray(occluded_nodes, dtype=np.int64).reshape(-1)))
        cache_key = ("occlusion", key)
        if cache_key in trial_cache:
            return {"validation_macro_f1": float(trial_cache[cache_key])}
        occluded = set(key)
        retained = np.asarray([int(node) for node in occlusion_context if int(node) not in occluded], dtype=np.int64)
        coarse, assignment, _diag = build_selected_support_graph(
            graph,
            retained,
            cfg.selector,
            target_node_type=int(cfg.target_node_type),
            support_features=support_features,
        )
        metrics = evaluate_hettree_task(
            graph,
            coarse,
            assignment.assignment,
            seed=int(seed),
            epochs=max(0, int(args.occlusion_short_eval_epochs)),
            hidden_dim=int(args.task_hidden_dim),
            device=str(args.device),
            target_node_type=int(cfg.target_node_type),
            official_split_nodes=trial_split,
            primary_eval_mode=str(args.primary_eval_mode),
            early_stopping=True,
            monitor=str(args.monitor),
            max_paths=int(args.max_paths),
        ).metrics
        score = _metric(metrics, "validation_macro_f1")
        trial_cache[cache_key] = float(score)
        return {"validation_macro_f1": float(score)}

    selector_name = str(cfg.selector.selector)
    selected = select_support_nodes(
        support_features,
        importance["importance"],
        float(ratio),
        cfg.selector,
        validation_evaluator=validation_score if selector_name == "real_validation_block_greedy" else None,
        occlusion_evaluator=occlusion_score if selector_name in {"real_occlusion_block_selector", "occlusion_plus_dblp_prototype"} else None,
    )
    coarse, assignment, graph_diag = build_selected_support_graph(
        graph,
        selected["selected_support_nodes"],
        cfg.selector,
        target_node_type=int(cfg.target_node_type),
        support_features=support_features,
    )
    split = {"train": np.flatnonzero(train_mask), "val": np.flatnonzero(val_mask), "test": np.flatnonzero(test_mask)}
    task = evaluate_hettree_task(
        graph,
        coarse,
        assignment.assignment,
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(cfg.target_node_type),
        official_split_nodes=split,
        primary_eval_mode=str(args.primary_eval_mode),
        early_stopping=True,
        monitor=str(args.monitor),
        max_paths=int(args.max_paths),
    ).metrics
    support_count = int(len(support_nodes_all))
    selected_local = np.asarray(selected["selected_local_indices"], dtype=np.int64)
    selection_diag = selected["diagnostics"]
    relation = support_features["component_matrices"].get("relation_profile", np.empty((support_count, 0)))
    anchor = support_features["component_matrices"].get("anchor_distribution", np.empty((support_count, 0)))
    class_fp = support_features["component_matrices"].get("class_footprint", np.empty((support_count, 0)))
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    row = {
        "method": method,
        "primary_method_family": "gate17_fast_support_selection",
        "uses_static_pairwise_coarsening_as_primary": False,
        "requested_support_ratio": float(ratio),
        "requested_support_count": int(selection_diag.get("requested_support_count", len(selected["selected_support_nodes"]))),
        "realized_support_count": int(selection_diag.get("realized_support_count", len(selected["selected_support_nodes"]))),
        "realized_support_ratio": float(selection_diag.get("realized_support_ratio", len(selected["selected_support_nodes"]) / max(support_count, 1))),
        "support_budget_error": int(selection_diag.get("support_budget_error", 0)),
        "support_budget_abs_error": int(selection_diag.get("support_budget_abs_error", abs(int(selection_diag.get("support_budget_error", 0))))),
        "support_budget_exact_match": bool(selection_diag.get("support_budget_exact_match", True)),
        "realized_full_ratio": float(coarse.num_nodes / max(graph.num_nodes, 1)),
        "selected_support_count": int(len(selected["selected_support_nodes"])),
        "background_node_count": int(graph_diag.get("background_node_count", 0)),
        "prototype_background_count": int(graph_diag.get("prototype_background_count", 0)),
        "dropped_support_count": int(graph_diag.get("dropped_support_count", 0)),
        "target_hit": bool(len(np.unique(assignment.assignment[target_nodes])) == len(target_nodes)),
        "teacher_full_graph_val_macro": 0.0,
        "teacher_full_graph_test_macro": 0.0,
        "teacher_reliable_for_importance": False,
        "target_response_error": 0.0,
        "relation_response_error": float(1.0 - _coverage_count(relation[selected_local]) / max(_coverage_count(relation), 1)) if len(selected_local) else 1.0,
        "response_regularizer_value": float(cfg.regularizer.lambda_response),
        "anchor_coverage_before": _coverage_count(anchor),
        "anchor_coverage_after": _coverage_count(anchor[selected_local]) if len(selected_local) else 0,
        "class_coverage_before": _coverage_count(class_fp),
        "class_coverage_after": _coverage_count(class_fp[selected_local]) if len(selected_local) else 0,
        "relation_channel_coverage_before": _coverage_count(relation),
        "relation_channel_coverage_after": _coverage_count(relation[selected_local]) if len(selected_local) else 0,
        "context_collision_rate": selection_diag.get("context_collision_rate", 0.0),
        "selected_support_by_type": selection_diag.get("selected_by_type", {}),
        "selected_support_by_class": selection_diag.get("selected_by_class_footprint", {}),
        "selected_support_by_anchor": selection_diag.get("selected_by_anchor", {}),
        "selected_support_by_relation_bucket": selection_diag.get("selected_by_relation_bucket", {}),
        "prototype_count_by_type": graph_diag.get("prototype_count_by_type", {}),
        "prototype_count_by_class": graph_diag.get("prototype_count_by_class", {}),
        "prototype_count_by_anchor": graph_diag.get("prototype_count_by_anchor", {}),
        "prototype_count_by_relation_bucket": graph_diag.get("prototype_count_by_relation_bucket", {}),
        "prototype_member_count_mean": float(graph_diag.get("prototype_member_count_mean", 0.0) or 0.0),
        "prototype_member_count_p50": float(graph_diag.get("prototype_member_count_p50", 0.0) or 0.0),
        "prototype_member_count_p90": float(graph_diag.get("prototype_member_count_p90", 0.0) or 0.0),
        "prototype_member_count_p99": float(graph_diag.get("prototype_member_count_p99", 0.0) or 0.0),
        "prototype_member_count_max": int(graph_diag.get("prototype_member_count_max", 0) or 0),
        "large_prototype_count": int(graph_diag.get("large_prototype_count", 0) or 0),
        "large_prototype_split_count": int(graph_diag.get("large_prototype_split_count", 0) or 0),
        "forced_raw_bridge_count": int(graph_diag.get("forced_raw_bridge_count", 0) or 0),
        "rare_class_prototype_count": int(graph_diag.get("rare_class_prototype_count", 0) or 0),
        "relation_channel_prototype_count": int(graph_diag.get("relation_channel_prototype_count", 0) or 0),
        "prototype_budget_conflict_count": int(graph_diag.get("prototype_budget_conflict_count", 0) or 0),
        "prototype_key_mode": graph_diag.get("prototype_key_mode", ""),
        "selector_uses_true_validation_feedback": bool(selection_diag.get("selector_uses_true_validation_feedback", False)),
        "validation_trial_count": int(selection_diag.get("validation_trial_count", 0) or 0),
        "validation_candidate_pool_size": int(selection_diag.get("validation_candidate_pool_size", 0) or 0),
        "validation_short_eval_epochs": int(selection_diag.get("validation_short_eval_epochs", 0) or 0),
        "validation_objective": selection_diag.get("validation_objective", ""),
        "validation_greedy_best_gain_mean": float(selection_diag.get("validation_greedy_best_gain_mean", 0.0) or 0.0),
        "validation_greedy_best_gain_max": float(selection_diag.get("validation_greedy_best_gain_max", 0.0) or 0.0),
        "validation_greedy_gain_history": selection_diag.get("validation_greedy_gain_history", []),
        "accepted_block_count": int(selection_diag.get("accepted_block_count", 0) or 0),
        "rejected_block_count": int(selection_diag.get("rejected_block_count", 0) or 0),
        "block_trim_count": int(selection_diag.get("block_trim_count", 0) or 0),
        "oversized_block_count": int(selection_diag.get("oversized_block_count", 0) or 0),
        "selected_block_keys": selection_diag.get("selected_block_keys", []),
        "occlusion_trial_count": int(selection_diag.get("occlusion_trial_count", 0) or 0),
        "occlusion_candidate_pool_size": int(selection_diag.get("occlusion_candidate_pool_size", 0) or 0),
        "occlusion_objective": selection_diag.get("occlusion_objective", ""),
        "occlusion_delta_ce_mean": float(selection_diag.get("occlusion_delta_ce_mean", 0.0) or 0.0),
        "occlusion_delta_ce_max": float(selection_diag.get("occlusion_delta_ce_max", 0.0) or 0.0),
        "occlusion_delta_macro_f1_mean": float(selection_diag.get("occlusion_delta_macro_f1_mean", 0.0) or 0.0),
        "occlusion_delta_margin_mean": float(selection_diag.get("occlusion_delta_margin_mean", 0.0) or 0.0),
        "occlusion_delta_teacher_kl_mean": float(selection_diag.get("occlusion_delta_teacher_kl_mean", 0.0) or 0.0),
        "occlusion_cache_hit_count": int(selection_diag.get("occlusion_cache_hit_count", 0) or 0),
        "occlusion_cache_miss_count": int(selection_diag.get("occlusion_cache_miss_count", 0) or 0),
        "zero_footprint_share": support_features["diagnostics"].get("zero_footprint_support_share", 0.0),
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "selection_split_source": selection_diag.get("selection_split_source", "train_val_only"),
        "teacher_split_source": "train_val_only",
        "test_label_usage": "metrics_only",
        "validation_selection_uses": "validation_macro_f1",
        "selector_feedback_source": selection_diag.get("selector_feedback_source", "proxy_diverse_importance"),
        "evaluator_status": "diagnostic_lite_only",
        **_importance_stats(importance["importance"], selected_local),
    }
    _row_from_task(row, task)
    return row | {
        "selection": selected,
        "graph_diagnostics": graph_diag,
        "importance": importance,
        "support_features": support_features,
    }


def _flat_payload(result: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "coarse_graph",
        "assignment",
        "support_features",
        "importance",
        "selection",
        "graph_diagnostics",
        "task_metrics",
        "teacher_outputs",
    }
    return {key: value for key, value in result.items() if key not in skip}


def _full_graph_row(graph, dataset: str, seed: int, args: argparse.Namespace, split: dict[str, np.ndarray]) -> dict[str, Any]:
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
        "requested_support_ratio": 1.0,
        "requested_support_count": int(support_count),
        "realized_support_count": int(support_count),
        "realized_support_ratio": 1.0,
        "support_budget_error": 0,
        "support_budget_abs_error": 0,
        "support_budget_exact_match": True,
        "realized_full_ratio": 1.0,
        "selected_support_count": int(support_count),
        "background_node_count": 0,
        "prototype_background_count": 0,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "selection_split_source": "train_val_only",
        "teacher_split_source": "train_val_only",
        "test_label_usage": "metrics_only",
        "target_hit": True,
        "validation_trial_count": 0,
        "occlusion_trial_count": 0,
        "large_prototype_count": 0,
    }
    _row_from_task(row, task)
    row["macro_recovery_vs_full_graph"] = 1.0
    row["accuracy_recovery_vs_full_graph"] = 1.0
    return row


def _run_group(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    ratios: list[float] | None = None,
    include_full_graph: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    graph = load_hgb_graph(Path(args.data_root), dataset)
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
    if bool(args.use_teacher):
        teacher = train_full_graph_lite_teacher(
            graph,
            labels,
            train_mask,
            val_mask,
            test_mask,
            Gate15Config(target_node_type=int(target_type)).teacher,
            output_dir=Path(args.output_dir) / "teacher_cache" / f"{dataset}_seed{seed}",
            seed=int(seed),
            epochs=int(args.teacher_epochs),
            hidden_dim=int(args.teacher_hidden_dim),
            device=str(args.device),
            restarts=int(args.teacher_restarts),
        )
        teacher_rows = [{"dataset": dataset, "seed": int(seed), **teacher["metrics"]}]
        teacher_grid_rows = [{"dataset": dataset, "seed": int(seed), **item} for item in teacher.get("grid_results", [])]
    else:
        teacher = {
            "metrics": {
                "teacher_uses_test_labels_for_training": False,
                "teacher_reliable_for_importance": False,
                "logits_source": "disabled_gate17_runner",
            },
            "teacher_uses_test_labels_for_training": False,
        }
        teacher_rows = []
        teacher_grid_rows = []
    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    if include_full_graph and "full-graph-hettree-lite-tuned" in args.methods:
        rows.append(_full_graph_row(graph, dataset, int(seed), args, split))
    ceiling = rows[0] if rows else None
    support_count = int(np.sum(graph.node_type != int(target_type)))
    for ratio in list(args.ratios if ratios is None else ratios):
        for method in args.methods:
            if method == "full-graph-hettree-lite-tuned":
                continue
            start = perf_counter()
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "requested_support_ratio": float(ratio),
                **split_protocol,
            }
            try:
                if method in BASELINES:
                    coarse, assignment, diag = run_support_baseline(
                        graph,
                        baseline=method,
                        ratio=float(ratio),
                        seed=int(seed),
                        candidate_k=int(args.candidate_k),
                    )
                    row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
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
                        np.asarray(assignment, dtype=np.int64),
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
                    row.setdefault("selection_split_source", "train_only")
                    row.setdefault("teacher_split_source", "train_val_only")
                    row.setdefault("test_label_usage", "metrics_only")
                    row.setdefault("validation_trial_count", 0)
                    row.setdefault("occlusion_trial_count", 0)
                    row.setdefault("large_prototype_count", 0)
                else:
                    cfg = Gate15Config(
                        target_node_type=int(target_type),
                        selector=replace(_selector_for_method(method, args), support_ratios=(float(ratio),)),
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
                            method_name=method,
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
                            method,
                            float(ratio),
                            int(seed),
                            args,
                        )
                    row.update(_flat_payload(result))
                    selection_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            **result["selection"]["diagnostics"],
                        }
                    )
                    graph_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            **{key: value for key, value in result["graph_diagnostics"].items() if not isinstance(value, dict)},
                        }
                    )
                    prototype_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            "background_strategy": result["graph_diagnostics"].get("background_strategy", ""),
                            "prototype_key_mode": result["graph_diagnostics"].get("prototype_key_mode", ""),
                            "prototype_background_count": result["graph_diagnostics"].get("prototype_background_count", 0),
                            "prototype_count_by_type": result["graph_diagnostics"].get("prototype_count_by_type", {}),
                            "prototype_count_by_class": result["graph_diagnostics"].get("prototype_count_by_class", {}),
                            "prototype_count_by_anchor": result["graph_diagnostics"].get("prototype_count_by_anchor", {}),
                            "prototype_count_by_relation_bucket": result["graph_diagnostics"].get("prototype_count_by_relation_bucket", {}),
                            "prototype_member_count_mean": result["graph_diagnostics"].get("prototype_member_count_mean", 0.0),
                            "prototype_member_count_p50": result["graph_diagnostics"].get("prototype_member_count_p50", 0.0),
                            "prototype_member_count_p90": result["graph_diagnostics"].get("prototype_member_count_p90", 0.0),
                            "prototype_member_count_p99": result["graph_diagnostics"].get("prototype_member_count_p99", 0.0),
                            "prototype_member_count_max": result["graph_diagnostics"].get("prototype_member_count_max", 0),
                            "large_prototype_count": result["graph_diagnostics"].get("large_prototype_count", 0),
                            "large_prototype_split_count": result["graph_diagnostics"].get("large_prototype_split_count", 0),
                            "forced_raw_bridge_count": result["graph_diagnostics"].get("forced_raw_bridge_count", 0),
                            "rare_class_prototype_count": result["graph_diagnostics"].get("rare_class_prototype_count", 0),
                            "relation_channel_prototype_count": result["graph_diagnostics"].get("relation_channel_prototype_count", 0),
                            "prototype_budget_conflict_count": result["graph_diagnostics"].get("prototype_budget_conflict_count", 0),
                        }
                    )
                    for item in result["selection"].get("occlusion_block_scores", []):
                        occlusion_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "method": method,
                                "requested_support_ratio": float(ratio),
                                **item,
                            }
                        )
                    for item in result["selection"].get("validation_greedy_trials", []):
                        validation_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "method": method,
                                "requested_support_ratio": float(ratio),
                                **item,
                            }
                        )
                    components = result["importance"].get("components", {})
                    importance_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            "importance_mean": row.get("support_importance_mean", 0.0),
                            "selected_importance_mean": row.get("selected_importance_mean", 0.0),
                            **{key: float(np.mean(value)) for key, value in components.items()},
                        }
                    )
                if ceiling is not None:
                    row["macro_recovery_vs_full_graph"] = _metric(row, "macro_f1") / max(_metric(ceiling, "macro_f1"), 1.0e-12)
                    row["accuracy_recovery_vs_full_graph"] = _metric(row, "accuracy") / max(_metric(ceiling, "accuracy"), 1.0e-12)
            except RuntimeError as exc:
                row["status"] = "oom_or_runtime_error" if "out of memory" in str(exc).lower() else "failed"
                row["error"] = str(exc)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
            row["wall_clock_sec"] = float(perf_counter() - start)
            rows.append(row)
    return {
        "rows": rows,
        "teacher_rows": teacher_rows,
        "teacher_grid_rows": teacher_grid_rows,
        "importance_rows": importance_rows,
        "selection_rows": selection_rows,
        "graph_rows": graph_rows,
        "occlusion_rows": occlusion_rows,
        "validation_rows": validation_rows,
        "prototype_rows": prototype_rows,
    }


def _write_evaluator_outputs(path: Path, rows: list[dict[str, Any]]) -> None:
    evaluator_rows = [
        {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "method": row.get("method", ""),
            "requested_support_ratio": row.get("requested_support_ratio", ""),
            "primary_eval_mode": row.get("primary_eval_mode", ""),
            "macro_f1": row.get("macro_f1", ""),
            "projected_macro_f1": row.get("projected_macro_f1", ""),
            "transfer_macro_f1": row.get("transfer_macro_f1", ""),
            "projected_vs_transfer_macro_gap": row.get("projected_vs_transfer_macro_gap", ""),
            "accuracy": row.get("accuracy", ""),
            "projected_accuracy": row.get("projected_accuracy", ""),
            "transfer_accuracy": row.get("transfer_accuracy", ""),
            "projected_vs_transfer_accuracy_gap": row.get("projected_vs_transfer_accuracy_gap", ""),
        }
        for row in rows
    ]
    write_csv(path / "evaluator_comparison.csv", evaluator_rows)
    gaps = [
        float(row.get("projected_vs_transfer_macro_gap", 0.0) or 0.0)
        for row in rows
        if row.get("status", "success") == "success"
    ]
    report = [
        "# Gate17 Evaluator Report",
        "",
        f"- rows: `{len(rows)}`",
        "- primary_eval_mode: `compressed_projected`",
        f"- projected_vs_transfer_macro_gap_mean: `{float(np.mean(gaps)) if gaps else 0.0}`",
        f"- primary_metric_mismatch: `{sum(1 for row in rows if row.get('status', 'success') == 'success' and str(row.get('primary_eval_mode')) == 'compressed_projected' and abs(float(row.get('macro_f1', 0.0) or 0.0) - float(row.get('projected_macro_f1', 0.0) or 0.0)) > 1.0e-12)}`",
    ]
    (path / "evaluator_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_smoke_report(path: Path, rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row.get("status") != "success"]
    lines = [
        "# Gate17 Smoke Report",
        "",
        f"- rows: `{len(rows)}`",
        f"- failed_or_skipped: `{len(failed)}`",
        f"- selector_uses_test_labels: `{any(str(row.get('selector_uses_test_labels', 'False')).lower() not in {'false', '0', ''} for row in rows)}`",
        f"- teacher_uses_test_labels_for_training: `{any(str(row.get('teacher_uses_test_labels_for_training', 'False')).lower() not in {'false', '0', ''} for row in rows)}`",
        f"- budget_fields_present: `{all('support_budget_exact_match' in row for row in rows)}`",
        f"- validation_trial_count_present: `{all('validation_trial_count' in row for row in rows)}`",
        f"- occlusion_trial_count_present: `{all('occlusion_trial_count' in row for row in rows)}`",
        f"- large_prototype_count_present: `{all('large_prototype_count' in row for row in rows)}`",
    ]
    if failed:
        lines += ["", "## Failed/Skipped", ""]
        for row in failed:
            lines.append(f"- {row.get('dataset')} {row.get('seed')} {row.get('method')}: {row.get('status')} {row.get('error', row.get('skip_reason', ''))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_output_dirs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if args.diagnostics_dir is None:
        args.diagnostics_dir = output_dir.parent / "gate17_diagnostics" if output_dir.name == "gate17" else output_dir / "diagnostics"
    if args.tables_dir is None:
        args.tables_dir = output_dir.parent / "gate17_tables" if output_dir.name == "gate17" else output_dir / "tables"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17 support selection experiments.")
    parser.add_argument("--output-dir", "--output-root", type=Path, default=Path("outputs/gate17"))
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="*", default=list(SEEDS))
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=list(RATIOS))
    parser.add_argument("--methods", nargs="*", default=list(PRIMARY_METHODS))
    parser.add_argument("--task-epochs", type=int, default=3)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--teacher-hidden-dim", type=int, default=32)
    parser.add_argument("--teacher-restarts", type=int, default=1)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=2)
    parser.add_argument("--short-eval-epochs", type=int, default=1)
    parser.add_argument("--max-validation-greedy-steps", type=int, default=1)
    parser.add_argument("--occlusion-candidate-pool-size", type=int, default=2)
    parser.add_argument("--occlusion-short-eval-epochs", type=int, default=1)
    parser.add_argument("--occlusion-short-patience", type=int, default=1)
    parser.add_argument("--max-members-per-prototype", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-teacher", action="store_true")
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--feature-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit-groups", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.datasets = _split_values(args.datasets, str) or list(DATASETS)
    args.seeds = _split_values(args.seeds, int) or list(SEEDS)
    args.ratios = _split_values(args.support_ratios, float) or list(RATIOS)
    args.methods = _split_values(args.methods, str) or list(PRIMARY_METHODS)
    if args.smoke:
        args.datasets = ["ACM"]
        args.seeds = [12345]
        args.ratios = [0.30]
        args.methods = [
            "full-graph-hettree-lite-tuned",
            "H6-no-spec-support-only",
            "HeSF-SS-real-occlusion-block",
        ]
    _resolve_output_dirs(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.diagnostics_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)
    write_audit(Path(args.output_dir).parent / "gate17_code_audit")
    groups = [
        (dataset, seed, [float(ratio)], int(ratio_index) == 0)
        for dataset in args.datasets
        for seed in args.seeds
        for ratio_index, ratio in enumerate(args.ratios)
    ]
    if args.limit_groups is not None:
        groups = groups[: max(0, int(args.limit_groups))]
    all_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    teacher_grid_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []

    def consume(payload: dict[str, list[dict[str, Any]]]) -> None:
        all_rows.extend(payload["rows"])
        teacher_rows.extend(payload["teacher_rows"])
        teacher_grid_rows.extend(payload["teacher_grid_rows"])
        importance_rows.extend(payload["importance_rows"])
        selection_rows.extend(payload["selection_rows"])
        graph_rows.extend(payload["graph_rows"])
        occlusion_rows.extend(payload["occlusion_rows"])
        validation_rows.extend(payload["validation_rows"])
        prototype_rows.extend(payload["prototype_rows"])
        write_csv(Path(args.output_dir) / "gate17_raw_rows.csv", all_rows)
        write_csv(Path(args.tables_dir) / "gate17_raw_rows.csv", all_rows)
        write_csv(Path(args.diagnostics_dir) / "support_selection_diagnostics.csv", selection_rows)
        write_csv(Path(args.diagnostics_dir) / "compressed_graph_summary.csv", graph_rows)
        write_csv(Path(args.diagnostics_dir) / "occlusion_block_scores.csv", occlusion_rows)
        write_csv(Path(args.diagnostics_dir) / "validation_greedy_trials.csv", validation_rows)
        write_csv(Path(args.diagnostics_dir) / "prototype_diagnostics.csv", prototype_rows)
        write_csv(Path(args.diagnostics_dir) / "full_graph_teacher_by_dataset_seed.csv", teacher_rows)
        write_csv(Path(args.diagnostics_dir) / "teacher_config_sweep.csv", teacher_grid_rows)

    if int(args.jobs) <= 1:
        for dataset, seed, ratios, include_full_graph in groups:
            consume(_run_group(args, str(dataset), int(seed), list(ratios), bool(include_full_graph)))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {
                pool.submit(_run_group, args, str(dataset), int(seed), list(ratios), bool(include_full_graph)): (dataset, seed, ratios)
                for dataset, seed, ratios, include_full_graph in groups
            }
            for future in as_completed(futures):
                consume(future.result())
    write_csv(Path(args.diagnostics_dir) / "support_importance.csv", importance_rows)
    _write_evaluator_outputs(Path(args.diagnostics_dir), all_rows)
    _write_smoke_report(Path(args.output_dir) / "gate17_smoke_report.md", all_rows)
    result = summarize(Path(args.output_dir), Path(args.tables_dir))
    failed = [row for row in all_rows if row.get("status") != "success"]
    if any(row.get("status") == "oom_or_runtime_error" for row in failed):
        return 3
    return 0 if not failed and result.get("success", 0) else 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
