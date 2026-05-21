from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.eval.hettree_task import evaluate_hettree_task
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config
from hesf_coarsen.task_first.selection.contribution import compute_support_importance
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from hesf_coarsen.task_first.selection.support_features import build_support_selection_features
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher


_METHOD_NAME = {
    "teacher_topk": "HeSF-SS-teacher-topk",
    "teacher_diverse_topk": "HeSF-SS-teacher-diverse-topk",
    "hybrid_teacher_response": "HeSF-SS-hybrid-teacher-response",
    "validation_greedy": "HeSF-SS-validation-proxy-diverse",
    "validation_proxy_diverse": "HeSF-SS-validation-proxy-diverse",
    "true_validation_block_greedy": "HeSF-SS-true-validation-block-greedy",
    "sensitivity_block_selector": "HeSF-SS-sensitivity-block-selector",
    "mlp_importance": "HeSF-SS-mlp-importance",
}


def _mask_nodes(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _importance_stats(importance: np.ndarray, selected_local: np.ndarray) -> dict[str, float]:
    values = np.asarray(importance, dtype=np.float64)
    selected = values[np.asarray(selected_local, dtype=np.int64)] if len(selected_local) else np.empty(0)
    return {
        "support_importance_mean": float(np.mean(values)) if len(values) else 0.0,
        "support_importance_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "selected_importance_mean": float(np.mean(selected)) if len(selected) else 0.0,
        "selected_importance_std": float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0,
    }


def _response_retention(support_features: dict[str, Any], selected_local: np.ndarray) -> float:
    response = support_features["component_matrices"].get("target_response_signature", np.empty((0, 0)))
    if response.size == 0:
        return 0.0
    all_mass = float(np.sum(np.linalg.norm(response, axis=1)))
    selected_mass = float(np.sum(np.linalg.norm(response[selected_local], axis=1))) if len(selected_local) else 0.0
    return float(selected_mass / max(all_mass, 1.0e-12))


def run_supervised_support_selection_pipeline(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: Gate15Config,
    *,
    support_ratio: float | None = None,
    teacher_outputs: dict[str, Any] | None = None,
    method_name: str | None = None,
    output_dir: str | Path | None = None,
    seed: int = 12345,
    task_epochs: int = 10,
    task_hidden_dim: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    if not cfg.keep_all_target_nodes or not cfg.support_only:
        raise ValueError("Gate15 requires target nodes to be preserved and support-only selection")
    labels = np.asarray(labels)
    ratio = float(cfg.selector.support_ratios[0] if support_ratio is None else support_ratio)
    teacher = teacher_outputs
    if teacher is None and cfg.teacher.enabled:
        teacher = train_full_graph_lite_teacher(
            graph,
            labels,
            train_mask,
            val_mask,
            test_mask,
            cfg.teacher,
            output_dir=Path(output_dir) / "teacher" if output_dir is not None else None,
            seed=int(seed),
            epochs=int(task_epochs),
            hidden_dim=int(task_hidden_dim),
            device=str(device),
        )
    support_features = build_support_selection_features(
        graph,
        labels,
        train_mask,
        int(cfg.target_node_type),
        teacher,
        cfg.feature,
    )
    importance = compute_support_importance(
        support_features,
        teacher,
        mode=str(cfg.selector.selector),
        lambda_response=float(cfg.regularizer.lambda_response),
    )
    selected = select_support_nodes(
        support_features,
        importance["importance"],
        ratio,
        cfg.selector,
    )
    coarse, assignment, graph_diag = build_selected_support_graph(
        graph,
        selected["selected_support_nodes"],
        cfg.selector,
        target_node_type=int(cfg.target_node_type),
        support_features=support_features,
    )
    task = evaluate_hettree_task(
        graph,
        coarse,
        assignment.assignment,
        seed=int(seed),
        epochs=int(task_epochs),
        hidden_dim=int(task_hidden_dim),
        device=str(device),
        target_node_type=int(cfg.target_node_type),
        official_split_nodes={
            "train": _mask_nodes(train_mask),
            "val": _mask_nodes(val_mask),
            "test": _mask_nodes(test_mask),
        },
        primary_eval_mode="compressed_projected",
        early_stopping=True,
        monitor="projected_val_macro_f1",
    ).metrics
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    support_count = int(np.sum(graph.node_type != int(cfg.target_node_type)))
    target_hit = bool(len(np.unique(assignment.assignment[target_nodes])) == len(target_nodes))
    teacher_metrics = (teacher or {}).get("metrics", {})
    teacher_macro = float(teacher_metrics.get("full_graph_teacher_macro_f1", 0.0) or 0.0)
    teacher_acc = float(teacher_metrics.get("full_graph_teacher_accuracy", 0.0) or 0.0)
    selected_local = np.asarray(selected["selected_local_indices"], dtype=np.int64)
    selection_diag = selected["diagnostics"]
    method = method_name or _METHOD_NAME.get(str(cfg.selector.selector), f"HeSF-SS-{cfg.selector.selector}")
    row = {
        "method": method,
        "primary_method_family": "supervised_support_selection",
        "uses_static_pairwise_coarsening_as_primary": False,
        "requested_support_ratio": ratio,
        "requested_support_count": int(selection_diag.get("requested_support_count", len(selected["selected_support_nodes"]))),
        "realized_support_count": int(selection_diag.get("realized_support_count", len(selected["selected_support_nodes"]))),
        "realized_support_ratio": float(selection_diag.get("realized_support_ratio", len(selected["selected_support_nodes"]) / max(support_count, 1))),
        "support_budget_error": int(selection_diag.get("support_budget_error", 0)),
        "support_budget_exact_match": bool(selection_diag.get("support_budget_exact_match", True)),
        "realized_full_ratio": float(coarse.num_nodes / max(graph.num_nodes, 1)),
        "selected_support_count": int(len(selected["selected_support_nodes"])),
        "background_node_count": int(graph_diag.get("background_node_count", 0)),
        "prototype_background_count": int(graph_diag.get("prototype_background_count", 0)),
        "dropped_support_count": int(graph_diag.get("dropped_support_count", 0)),
        "target_hit": target_hit,
        "macro_f1": _metric(task, "macro_f1"),
        "micro_f1": _metric(task, "micro_f1"),
        "accuracy": _metric(task, "accuracy"),
        "primary_eval_mode": task.get("primary_eval_mode", "compressed_projected"),
        "primary_task_metric_name": task.get("primary_task_metric_name", "projected_original_macro_f1"),
        "transfer_macro_f1": _metric(task, "transfer_original_macro_f1"),
        "transfer_accuracy": _metric(task, "transfer_original_accuracy"),
        "projected_macro_f1": _metric(task, "projected_original_macro_f1"),
        "projected_accuracy": _metric(task, "projected_original_accuracy"),
        "hybrid_target_macro_f1": _metric(task, "hybrid_target_original_macro_f1"),
        "hybrid_target_accuracy": _metric(task, "hybrid_target_original_accuracy"),
        "projected_vs_transfer_macro_gap": _metric(task, "projected_vs_transfer_macro_gap"),
        "projected_vs_transfer_accuracy_gap": _metric(task, "projected_vs_transfer_accuracy_gap"),
        "validation_macro_f1": _metric(task, "validation_macro_f1"),
        "validation_accuracy": _metric(task, "validation_accuracy"),
        "best_epoch": int(task.get("best_epoch", -1) or -1),
        "early_stopped": bool(task.get("early_stopped", False)),
        "macro_recovery_vs_full_graph": float(_metric(task, "macro_f1") / teacher_macro) if teacher_macro else 0.0,
        "accuracy_recovery_vs_full_graph": float(_metric(task, "accuracy") / teacher_acc) if teacher_acc else 0.0,
        "teacher_full_graph_val_macro": float(teacher_metrics.get("validation_macro_f1", 0.0) or 0.0),
        "teacher_full_graph_test_macro": float(teacher_metrics.get("full_graph_teacher_macro_f1", 0.0) or 0.0),
        "teacher_full_graph_projected_macro": float(teacher_metrics.get("full_graph_teacher_projected_macro_f1", 0.0) or 0.0),
        "teacher_full_graph_transfer_macro": float(teacher_metrics.get("full_graph_teacher_transfer_macro_f1", 0.0) or 0.0),
        "teacher_seed_restart_id": teacher_metrics.get("teacher_best_config_hash", ""),
        "teacher_best_epoch": int(teacher_metrics.get("teacher_best_epoch", -1) or -1),
        "teacher_best_config_hash": teacher_metrics.get("teacher_best_config_hash", ""),
        "teacher_reliable_for_importance": bool(teacher_metrics.get("teacher_reliable_for_importance", False)),
        "target_response_error": float(1.0 - _response_retention(support_features, selected_local)),
        "relation_response_error": float(1.0 - selection_diag.get("relation_channel_coverage_after", 0) / max(selection_diag.get("relation_channel_coverage_before", 1), 1)),
        "response_regularizer_value": float(cfg.regularizer.lambda_response),
        "anchor_coverage_before": selection_diag.get("anchor_coverage_before", 0),
        "anchor_coverage_after": selection_diag.get("anchor_coverage_after", 0),
        "class_coverage_before": selection_diag.get("class_coverage_before", 0),
        "class_coverage_after": selection_diag.get("class_coverage_after", 0),
        "relation_channel_coverage_before": selection_diag.get("relation_channel_coverage_before", 0),
        "relation_channel_coverage_after": selection_diag.get("relation_channel_coverage_after", 0),
        "context_collision_rate": selection_diag.get("context_collision_rate", 0.0),
        "selected_support_by_type": selection_diag.get("selected_by_type", {}),
        "selected_support_by_class": selection_diag.get("selected_by_class_footprint", {}),
        "selected_support_by_anchor": selection_diag.get("selected_by_anchor", {}),
        "selected_support_by_relation_bucket": selection_diag.get("selected_by_relation_bucket", {}),
        "prototype_count_by_type": graph_diag.get("prototype_count_by_type", {}),
        "prototype_count_by_class": graph_diag.get("prototype_count_by_class", {}),
        "prototype_count_by_anchor": graph_diag.get("prototype_count_by_anchor", {}),
        "prototype_count_by_relation_bucket": graph_diag.get("prototype_count_by_relation_bucket", {}),
        "teacher_KL_retained": 0.0,
        "val_loss_delta_retained": 0.0,
        "selector_uses_true_validation_feedback": bool(selection_diag.get("selector_uses_true_validation_feedback", False)),
        "zero_footprint_share": support_features["diagnostics"].get("zero_footprint_support_share", 0.0),
        "known_unknown_merge_count": 0,
        "unknown_unknown_merge_count": 0,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": bool((teacher or {}).get("teacher_uses_test_labels_for_training", False)),
        "test_label_usage": "metrics_only",
        "validation_selection_uses": "validation_macro_f1",
        "selector_feedback_source": "true_validation_feedback"
        if bool(selection_diag.get("selector_uses_true_validation_feedback", False))
        else "proxy_diverse_importance",
        "evaluator_status": "diagnostic_lite_only",
        "status": "success" if not task.get("skipped", False) else "skipped",
        "skip_reason": task.get("skip_reason", ""),
        **_importance_stats(importance["importance"], selected_local),
    }
    return row | {
        "coarse_graph": coarse,
        "assignment": assignment,
        "support_features": support_features,
        "importance": importance,
        "selection": selected,
        "graph_diagnostics": graph_diag,
        "task_metrics": task,
        "teacher_outputs": teacher,
    }
