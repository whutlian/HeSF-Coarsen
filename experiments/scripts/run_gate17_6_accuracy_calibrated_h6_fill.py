from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.gate17_4_h6 import selected_support_representatives_from_assignment
from experiments.scripts.run_gate17_1_support_sensitivity import (
    _full_graph_row,
    _semantic_row_for_graph,
    _target_only_empty_support_graph,
    _target_only_row,
)
from experiments.scripts.run_gate17_3_lossy_prototype_feedback import _overlap_fields
from experiments.scripts.run_gate17_4_h6_equivalence import (
    _budget_update,
    _h6_equivalence_row,
    _semantic_fields_for_raw_row,
    _task_metrics_row,
)
from experiments.scripts.run_gate17_5_h6_cluster_gating import (
    _budget_aliases,
    _cluster_overlap_fields,
    _h6_delta_alias_fields,
    _requested_support_count,
    _run_h6_cluster_method as _gate17_5_run_h6_cluster_method,
    _semantic_and_h6_delta,
    _selector_for_method as _gate17_5_selector_for_method,
)
from experiments.scripts.run_gate17_support_selection import _flat_payload, _mask, _split_values
from experiments.scripts.summarize_gate17_6 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.h6_cluster_gating import extract_h6_cluster_units, h6_fill_support_nodes
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


GATE17_6_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
H6_CONSTRUCTION_CONTROL_METHOD = "HeSF-SS-H6-equivalence-control"
H6_SELECTED_SET_CONTROL_METHOD = "HeSF-SS-H6-selected-set-control"
BASELINES = (
    "random-support-only",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
)
MAIN_CANDIDATE_METHODS = (
    "HeSF-SS-validation-only-neutral-fill",
    "HeSF-SS-validation-H6-fill",
    "HeSF-SS-validation-H6-fill-acc0.25",
    "HeSF-SS-validation-H6-fill-acc0.50",
    "HeSF-SS-validation-H6-fill-acc1.00",
    "HeSF-SS-H6-fill-only",
    "HeSF-SS-random-fill-after-validation",
)
ACCURACY_CALIBRATED_METHODS = {
    "HeSF-SS-validation-H6-fill-acc0.25": 0.25,
    "HeSF-SS-validation-H6-fill-acc0.50": 0.50,
    "HeSF-SS-validation-H6-fill-acc1.00": 1.00,
}
DIAGNOSTIC_ONLY_METHODS = (
    "HeSF-SS-real-occlusion-neutral-fill",
    "HeSF-SS-H6-cluster-validation-coverage-gated",
    H6_CONSTRUCTION_CONTROL_METHOD,
    H6_SELECTED_SET_CONTROL_METHOD,
    "HeSF-SS-full-residual-prototype-upperbound",
)
DEFAULT_METHODS = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    *BASELINES,
    *MAIN_CANDIDATE_METHODS,
    *DIAGNOSTIC_ONLY_METHODS,
)


def parse_dataset_seeds(values: list[str] | tuple[str, ...] | str | None) -> list[tuple[str, int]]:
    if values is None or values == "":
        return [(dataset, seed) for dataset, seed in GATE17_6_SINGLE_SEED_BY_DATASET.items()]
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
        if dataset not in GATE17_6_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate17.6 dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def accuracy_objective_components(
    *,
    validation_macro_f1: float,
    validation_accuracy: float,
    validation_micro_f1: float | None,
    requested_support_count: int,
    selected_support_count: int,
    baseline_num_predicted_classes: int,
    num_predicted_classes: int,
    alpha_accuracy: float,
    delta_micro: float,
    beta_underfill: float,
    gamma_class_collapse: float,
    validation_loss_or_ce: float | None = None,
) -> dict[str, Any]:
    micro_available = validation_micro_f1 is not None
    micro = float(validation_accuracy if validation_micro_f1 is None else validation_micro_f1)
    underfill = max(0, int(requested_support_count) - int(selected_support_count)) / max(1, int(requested_support_count))
    class_collapse = max(0, int(baseline_num_predicted_classes) - int(num_predicted_classes)) / max(1, int(baseline_num_predicted_classes))
    macro_component = float(validation_macro_f1)
    accuracy_component = float(alpha_accuracy) * float(validation_accuracy)
    micro_component = float(delta_micro) * float(micro)
    underfill_component = -float(beta_underfill) * float(underfill)
    collapse_component = -float(gamma_class_collapse) * float(class_collapse)
    return {
        "validation_macro_f1": float(validation_macro_f1),
        "validation_accuracy": float(validation_accuracy),
        "validation_micro_f1": float(micro),
        "validation_micro_f1_available": bool(micro_available),
        "validation_loss_or_ce": float(max(0.0, 1.0 - validation_accuracy) if validation_loss_or_ce is None else validation_loss_or_ce),
        "class_collapse_penalty": float(class_collapse),
        "underfill_penalty": float(underfill),
        "score_macro_component": float(macro_component),
        "score_accuracy_component": float(accuracy_component),
        "score_micro_component": float(micro_component),
        "score_underfill_component": float(underfill_component),
        "score_class_collapse_component": float(collapse_component),
        "score_total": float(macro_component + accuracy_component + micro_component + underfill_component + collapse_component),
    }


def per_class_audit_rows(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    selected_support_labels: Iterable[int] | None = None,
    baseline_per_class: Mapping[int, Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=np.int64).reshape(-1)
    valid = (truth >= 0) & (pred >= 0)
    truth = truth[valid]
    pred = pred[valid]
    baseline = baseline_per_class or {}
    support_labels = np.asarray(list(selected_support_labels or []), dtype=np.int64).reshape(-1)
    support_labels = support_labels[support_labels >= 0]
    labels = sorted({int(value) for value in np.concatenate([truth, pred, np.asarray(list(baseline.keys()), dtype=np.int64) if baseline else np.asarray([], dtype=np.int64)]).tolist()})
    total = max(1, int(len(truth)))
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    true_totals = {label: int(np.sum(truth == label)) for label in labels}
    pred_totals = {label: int(np.sum(pred == label)) for label in labels}
    for label in labels:
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
        recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn))
        support_count = int(np.sum(support_labels == label))
        base = baseline.get(int(label), {})
        per_class.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "support_ratio": float(ratio),
                "requested_support_ratio": float(ratio),
                "class_id": int(label),
                "test_label_count": int(true_totals[label]),
                "predicted_count": int(pred_totals[label]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy_contribution": float(tp / total),
                "delta_precision_vs_best_strong": float(precision - _float_value(base.get("precision", precision))),
                "delta_recall_vs_best_strong": float(recall - _float_value(base.get("recall", recall))),
                "delta_f1_vs_best_strong": float(f1 - _float_value(base.get("f1", f1))),
                "support_selected_count_for_class": int(support_count),
                "support_selected_ratio_for_class": float(support_count / max(1, len(support_labels))),
            }
        )
    for true_label in labels:
        for pred_label in labels:
            count = int(np.sum((truth == true_label) & (pred == pred_label)))
            if count == 0:
                continue
            confusion.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "method": method,
                    "support_ratio": float(ratio),
                    "requested_support_ratio": float(ratio),
                    "true_class": int(true_label),
                    "pred_class": int(pred_label),
                    "count": int(count),
                    "normalized_by_true": float(count / max(1, true_totals[true_label])),
                    "normalized_by_pred": float(count / max(1, pred_totals[pred_label])),
                }
            )
    return per_class, confusion


def _selector_for_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    mapped = {
        "HeSF-SS-validation-only-neutral-fill": "HeSF-SS-real-validation-neutral-fill",
        "HeSF-SS-validation-H6-fill": "HeSF-SS-real-validation-H6-fill",
        "HeSF-SS-random-fill-after-validation": "HeSF-SS-real-validation-neutral-fill",
        "HeSF-SS-H6-fill-only": "HeSF-SS-real-validation-H6-fill",
    }.get(str(method), str(method))
    if method in ACCURACY_CALIBRATED_METHODS:
        mapped = "HeSF-SS-real-validation-neutral-fill"
    cfg = _gate17_5_selector_for_method(mapped, args)
    if method in {"HeSF-SS-validation-H6-fill", "HeSF-SS-random-fill-after-validation", "HeSF-SS-validation-only-neutral-fill"}:
        cfg = replace(cfg, allow_proxy_fill=False, neutral_fill=True, neutral_fill_max_drop=float(args.neutral_fill_max_drop))
    if method in ACCURACY_CALIBRATED_METHODS:
        cfg = replace(
            cfg,
            min_gain=float(args.min_validation_gain),
            allow_proxy_fill=False,
            neutral_fill=True,
            neutral_fill_max_drop=float(args.neutral_fill_max_drop),
            allow_negative_fill=True,
            negative_fill_max_drop=float(args.negative_fill_max_drop),
            validation_score_mode="accuracy_calibrated",
            alpha_accuracy=float(ACCURACY_CALIBRATED_METHODS[method]),
            delta_micro=float(args.delta_micro),
            beta_underfill=float(args.beta_underfill),
            gamma_class_collapse=float(args.gamma_class_collapse),
        )
    return cfg


def _run_selection_pipeline(
    *,
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    target_type: int,
    ratio: float,
    method: str,
    seed: int,
    teacher: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = Gate15Config(target_node_type=int(target_type), selector=replace(_selector_for_method(str(method), args), support_ratios=(float(ratio),)))
    return run_supervised_support_selection_pipeline(
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


def _eval_task(
    graph: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    *,
    seed: int,
    split: dict[str, np.ndarray],
    target_type: int,
    args: argparse.Namespace,
    epochs: int | None = None,
    return_predictions: bool = True,
) -> dict[str, Any]:
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
        return_predictions=bool(return_predictions),
    ).metrics


def _random_fill_support_nodes(
    *,
    graph: HeteroGraph,
    target_type: int,
    selected_support_nodes: np.ndarray,
    requested_support_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = [int(node) for node in np.asarray(selected_support_nodes, dtype=np.int64).reshape(-1)]
    selected_set = set(selected)
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    remaining = np.asarray([int(node) for node in support_nodes.tolist() if int(node) not in selected_set], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if len(remaining):
        rng.shuffle(remaining)
    fill = remaining[: max(0, int(requested_support_count) - len(selected))]
    out = np.asarray(sorted([*selected, *[int(node) for node in fill.tolist()]][: int(requested_support_count)]), dtype=np.int64)
    return out, {
        "random_fill_support_count": int(len(fill)),
        "random_fill_budget_fraction": float(len(fill) / max(1, int(requested_support_count))),
    }


def _entropy_from_counts(counts: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in counts if float(value) > 0.0], dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    probs = arr / max(float(np.sum(arr)), 1.0e-12)
    return float(-np.sum(probs * np.log2(probs)) / max(math.log2(len(probs)), 1.0)) if len(probs) > 1 else 0.0


def _cluster_structure_fields(
    graph: HeteroGraph,
    *,
    member_nodes: np.ndarray,
    target_type: int,
    labels: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, Any]:
    members = {int(node) for node in np.asarray(member_nodes, dtype=np.int64).reshape(-1).tolist()}
    target_anchors: set[int] = set()
    relation_mass: dict[int, float] = {}
    class_mass: dict[int, float] = {}
    degree_mass = 0.0
    for relation_id, rel in graph.relations.items():
        src = np.asarray(rel.src, dtype=np.int64)
        dst = np.asarray(rel.dst, dtype=np.int64)
        weight = np.asarray(rel.weight, dtype=np.float32)
        src_member = np.asarray([int(node) in members for node in src.tolist()], dtype=bool)
        dst_member = np.asarray([int(node) in members for node in dst.tolist()], dtype=bool)
        incident = src_member | dst_member
        mass = float(np.sum(weight[incident])) if np.any(incident) else 0.0
        if mass > 0.0:
            relation_mass[int(relation_id)] = mass
            degree_mass += mass
        if int(rel.src_type) == int(target_type):
            target_end = src[dst_member]
        elif int(rel.dst_type) == int(target_type):
            target_end = dst[src_member]
        else:
            target_end = np.asarray([], dtype=np.int64)
        for node in target_end.tolist():
            target_anchors.add(int(node))
            if bool(train_mask[int(node)]) and int(labels[int(node)]) >= 0:
                class_mass[int(labels[int(node)])] = class_mass.get(int(labels[int(node)]), 0.0) + 1.0
    target_count = int(np.sum(graph.node_type == int(target_type)))
    return {
        "h6_cluster_edge_mass": float(sum(relation_mass.values())),
        "h6_cluster_target_anchor_coverage": float(len(target_anchors) / max(1, target_count)),
        "h6_cluster_relation_channel_count": int(len(relation_mass)),
        "h6_cluster_relation_channel_entropy": float(_entropy_from_counts(list(relation_mass.values()))),
        "h6_cluster_class_footprint_entropy": float(_entropy_from_counts(list(class_mass.values()))),
        "h6_cluster_member_count": int(len(members)),
        "h6_cluster_degree_mass": float(degree_mass),
    }


def _h6_cluster_feedback(
    *,
    graph: HeteroGraph,
    h6_coarse: HeteroGraph,
    h6_assignment: np.ndarray,
    h6_task: dict[str, Any],
    dataset: str,
    seed: int,
    ratio: float,
    target_type: int,
    split: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    units = extract_h6_cluster_units(graph, h6_assignment, int(target_type))
    candidates = sorted(units, key=lambda unit: (-unit.member_count, unit.cluster_id))[: int(args.cluster_gating_candidate_pool_size)]
    all_cluster_ids = {int(unit.cluster_id) for unit in units}
    rows: list[dict[str, Any]] = []
    for unit in candidates:
        keep = sorted(all_cluster_ids - {int(unit.cluster_id)})
        from hesf_coarsen.task_first.selection.h6_cluster_gating import build_gated_h6_graph

        dropped_graph, dropped_assignment, _kept = build_gated_h6_graph(
            original=graph,
            h6_coarse=h6_coarse,
            h6_assignment=h6_assignment,
            selected_cluster_ids=keep,
            target_type=int(target_type),
        )
        task = _eval_task(
            graph,
            dropped_graph,
            dropped_assignment,
            seed=int(seed),
            split=split,
            target_type=int(target_type),
            args=args,
            epochs=int(args.cluster_gating_feedback_epochs),
            return_predictions=False,
        )
        raw_gain = float(h6_task.get("validation_macro_f1", 0.0) or 0.0) - float(task.get("validation_macro_f1", 0.0) or 0.0)
        fields = _cluster_structure_fields(graph, member_nodes=unit.member_nodes, target_type=int(target_type), labels=labels, train_mask=train_mask)
        cluster_score = (
            raw_gain
            + float(args.lambda_edge) * float(fields["h6_cluster_edge_mass"])
            + float(args.lambda_anchor) * float(fields["h6_cluster_target_anchor_coverage"])
            + float(args.lambda_relation) * float(fields["h6_cluster_relation_channel_entropy"])
            + float(args.lambda_class) * float(fields["h6_cluster_class_footprint_entropy"])
        )
        rows.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "requested_support_ratio": float(ratio),
                "cluster_id": int(unit.cluster_id),
                "h6_cluster_type": int(unit.cluster_type),
                "h6_cluster_size": int(unit.member_count),
                "raw_validation_gain": float(raw_gain),
                "validation_gain": float(cluster_score),
                "cluster_score": float(cluster_score),
                "validation_macro_f1_if_dropped": float(task.get("validation_macro_f1", 0.0) or 0.0),
                "validation_ce_delta": float(max(raw_gain, 0.0)),
                "occlusion_delta_ce": float(max(raw_gain, 0.0)),
                "occlusion_delta_macro_f1": float(raw_gain),
                "occlusion_delta_margin": float(float(h6_task.get("validation_accuracy", 0.0) or 0.0) - float(task.get("validation_accuracy", 0.0) or 0.0)),
                "cluster_edge_mass": float(fields["h6_cluster_edge_mass"]),
                "cluster_target_anchor_coverage": float(fields["h6_cluster_target_anchor_coverage"]),
                "cluster_relation_channel_profile": json.dumps({key: fields[key] for key in fields if key.startswith("h6_cluster_relation")}, sort_keys=True),
                "selected": False,
                **fields,
            }
        )
    return rows


def _support_labels(labels: np.ndarray, selected_nodes: np.ndarray) -> list[int]:
    arr = np.asarray(selected_nodes, dtype=np.int64).reshape(-1)
    return [int(labels[int(node)]) for node in arr.tolist() if 0 <= int(node) < len(labels) and int(labels[int(node)]) >= 0]


def _prediction_payload(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    task: Mapping[str, Any],
    selected_labels: Sequence[int],
) -> dict[str, Any] | None:
    y_true = task.get("projected_test_true_labels")
    y_pred = task.get("projected_test_pred_labels")
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        return None
    return {
        "dataset": str(dataset),
        "seed": int(seed),
        "method": str(method),
        "ratio": float(ratio),
        "y_true": [int(value) for value in y_true],
        "y_pred": [int(value) for value in y_pred],
        "selected_support_labels": [int(value) for value in selected_labels],
    }


def _provenance_fields(
    *,
    selected_before_fill: np.ndarray,
    selected_after_fill: np.ndarray,
    selection_diag: Mapping[str, Any],
    requested_count: int,
    h6_fill_count: int = 0,
    random_fill_count: int = 0,
    h6_fill_only_count: int = 0,
) -> dict[str, Any]:
    selected_count = int(len(np.asarray(selected_after_fill, dtype=np.int64).reshape(-1)))
    positive = int(selection_diag.get("validation_positive_fill_count", selection_diag.get("positive_gain_block_count", 0)) or 0)
    neutral = int(selection_diag.get("validation_neutral_fill_count", selection_diag.get("neutral_fill_block_count", 0)) or 0)
    negative = int(selection_diag.get("validation_negative_fill_count", selection_diag.get("negative_fill_block_count", 0)) or 0)
    return {
        "validation_positive": int(positive),
        "validation_neutral": int(neutral),
        "validation_negative_allowed": int(negative),
        "h6_fill": int(h6_fill_count),
        "random_fill": int(random_fill_count),
        "h6_fill_only": int(h6_fill_only_count),
        "validation_core_support_count": int(positive),
        "validation_neutral_support_count": int(neutral),
        "validation_negative_support_count": int(negative),
        "h6_fill_support_count": int(h6_fill_count),
        "random_fill_support_count": int(random_fill_count),
        "h6_fill_only_support_count": int(h6_fill_only_count),
        "selected_support_count": int(selected_count),
        "requested_support_count": int(requested_count),
        "underfill_count": int(max(0, int(requested_count) - selected_count)),
        "underfill_ratio": float(max(0, int(requested_count) - selected_count) / max(1, int(requested_count))),
        "validation_selected_before_fill_count": int(len(np.asarray(selected_before_fill, dtype=np.int64).reshape(-1))),
    }


def _add_accuracy_objective_row_fields(row: dict[str, Any], task: Mapping[str, Any], requested_count: int, selected_count: int, alpha: float, args: argparse.Namespace) -> dict[str, Any]:
    components = accuracy_objective_components(
        validation_macro_f1=_float_value(task.get("validation_macro_f1")),
        validation_accuracy=_float_value(task.get("validation_accuracy")),
        validation_micro_f1=_float_value(task.get("validation_micro_f1")) if task.get("validation_micro_f1") not in {"", None} else None,
        requested_support_count=int(requested_count),
        selected_support_count=int(selected_count),
        baseline_num_predicted_classes=int(task.get("num_classes", task.get("num_predicted_classes", 0)) or 0),
        num_predicted_classes=int(task.get("num_predicted_classes", task.get("num_classes", 0)) or 0),
        alpha_accuracy=float(alpha),
        delta_micro=float(args.delta_micro),
        beta_underfill=float(args.beta_underfill),
        gamma_class_collapse=float(args.gamma_class_collapse),
    )
    row.update(components)
    row["alpha_accuracy"] = float(alpha)
    row["delta_micro"] = float(args.delta_micro)
    row["beta_underfill"] = float(args.beta_underfill)
    row["gamma_class_collapse"] = float(args.gamma_class_collapse)
    row["num_predicted_classes"] = int(task.get("num_predicted_classes", 0) or 0)
    row["predicted_class_histogram"] = json.dumps(task.get("predicted_class_histogram", {}), sort_keys=True)
    return components


def _best_strong_by_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, float], Mapping[str, Any]]:
    buckets: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("method")) not in {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only"}:
            continue
        if str(row.get("status", "success")) != "success":
            continue
        key = (str(row.get("dataset")), str(row.get("seed")), round(_float_value(row.get("requested_support_ratio")), 10))
        buckets.setdefault(key, []).append(row)
    return {key: max(group, key=lambda row: (_float_value(row.get("macro_f1")), _float_value(row.get("accuracy")))) for key, group in buckets.items()}


def _write_fill_ablation(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    best_strong = _best_strong_by_bucket(rows)
    validation_only = {
        (str(row.get("dataset")), str(row.get("seed")), round(_float_value(row.get("requested_support_ratio")), 10)): row
        for row in rows
        if str(row.get("method")) == "HeSF-SS-validation-only-neutral-fill" and str(row.get("status", "success")) == "success"
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        if not method.startswith("HeSF-SS-"):
            continue
        key = (str(row.get("dataset")), str(row.get("seed")), round(_float_value(row.get("requested_support_ratio")), 10))
        baseline = best_strong.get(key)
        base_validation = validation_only.get(key)
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": method,
                "support_ratio": row.get("requested_support_ratio"),
                "requested_support_count": row.get("requested_support_count", ""),
                "selected_support_count": row.get("selected_support_count", ""),
                "validation_core_support_count": row.get("validation_core_support_count", 0),
                "validation_neutral_support_count": row.get("validation_neutral_support_count", 0),
                "validation_negative_support_count": row.get("validation_negative_support_count", 0),
                "h6_fill_support_count": row.get("h6_fill_support_count", 0),
                "random_fill_support_count": row.get("random_fill_support_count", 0),
                "h6_fill_only_support_count": row.get("h6_fill_only_support_count", 0),
                "h6_fill_budget_fraction": row.get("h6_fill_budget_fraction", 0.0),
                "random_fill_budget_fraction": row.get("random_fill_budget_fraction", 0.0),
                "underfill_count": row.get("underfill_count", 0),
                "underfill_ratio": row.get("underfill_ratio", 0.0),
                "overlap_validation_with_h6": row.get("overlap_validation_with_h6", row.get("cluster_overlap_with_h6", "")),
                "overlap_fill_with_h6": row.get("overlap_fill_with_h6", row.get("h6_fill_overlap_with_validation_blocks", "")),
                "overlap_random_fill_with_h6": row.get("overlap_random_fill_with_h6", ""),
                "macro_f1": row.get("macro_f1", ""),
                "accuracy": row.get("accuracy", ""),
                "validation_macro_f1": row.get("validation_macro_f1", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
                "delta_macro_vs_best_strong": "" if baseline is None else float(round(_float_value(row.get("macro_f1")) - _float_value(baseline.get("macro_f1")), 12)),
                "delta_accuracy_vs_best_strong": "" if baseline is None else float(round(_float_value(row.get("accuracy")) - _float_value(baseline.get("accuracy")), 12)),
                "macro_delta_from_fill": "" if base_validation is None else float(round(_float_value(row.get("macro_f1")) - _float_value(base_validation.get("macro_f1")), 12)),
                "accuracy_delta_from_fill": "" if base_validation is None else float(round(_float_value(row.get("accuracy")) - _float_value(base_validation.get("accuracy")), 12)),
                "validation_only_exact_budget": "" if base_validation is None else bool(base_validation.get("node_budget_exact_match", base_validation.get("support_budget_exact_match", False))),
            }
        )
    write_csv(output_dir / "diagnostics" / "gate17_6_fill_ablation.csv", out)


def _write_per_class_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]], payloads: Sequence[Mapping[str, Any]]) -> None:
    row_lookup = {(str(row.get("dataset")), str(row.get("seed")), round(_float_value(row.get("requested_support_ratio")), 10), str(row.get("method"))): row for row in rows}
    best_strong = _best_strong_by_bucket(rows)
    payload_lookup = {(str(item["dataset"]), str(item["seed"]), round(float(item["ratio"]), 10), str(item["method"])): item for item in payloads}
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for payload in payloads:
        key = (str(payload["dataset"]), str(payload["seed"]), round(float(payload["ratio"]), 10))
        baseline_row = best_strong.get(key)
        baseline_per_class: dict[int, Mapping[str, float]] = {}
        if baseline_row is not None:
            baseline_payload = payload_lookup.get((*key, str(baseline_row.get("method"))))
            if baseline_payload is not None:
                base_rows, _base_confusion = per_class_audit_rows(
                    dataset=str(baseline_payload["dataset"]),
                    seed=int(baseline_payload["seed"]),
                    method=str(baseline_payload["method"]),
                    ratio=float(baseline_payload["ratio"]),
                    y_true=baseline_payload["y_true"],
                    y_pred=baseline_payload["y_pred"],
                    selected_support_labels=baseline_payload.get("selected_support_labels", []),
                )
                baseline_per_class = {int(row["class_id"]): row for row in base_rows}
        pc, cm = per_class_audit_rows(
            dataset=str(payload["dataset"]),
            seed=int(payload["seed"]),
            method=str(payload["method"]),
            ratio=float(payload["ratio"]),
            y_true=payload["y_true"],
            y_pred=payload["y_pred"],
            selected_support_labels=payload.get("selected_support_labels", []),
            baseline_per_class=baseline_per_class,
        )
        source_row = row_lookup.get((*key, str(payload["method"])), {})
        for row in pc:
            row["requested_support_count"] = source_row.get("requested_support_count", "")
        per_class_rows.extend(pc)
        confusion_rows.extend(cm)
    write_csv(output_dir / "diagnostics" / "gate17_6_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "diagnostics" / "gate17_6_confusion_matrix_by_method.csv", confusion_rows)


def _write_accuracy_components(output_dir: Path, rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> None:
    fields = {
        "dataset",
        "seed",
        "method",
        "requested_support_ratio",
        "validation_macro_f1",
        "validation_accuracy",
        "validation_micro_f1",
        "validation_loss_or_ce",
        "class_collapse_penalty",
        "underfill_penalty",
        "score_total",
        "score_macro_component",
        "score_accuracy_component",
        "score_micro_component",
        "score_underfill_component",
        "score_class_collapse_component",
        "validation_micro_f1_available",
        "alpha_accuracy",
        "delta_micro",
        "beta_underfill",
        "gamma_class_collapse",
        "num_predicted_classes",
        "predicted_class_histogram",
    }
    out = [{key: row.get(key, "") for key in fields} for row in rows if str(row.get("method", "")).startswith("HeSF-SS-validation-H6-fill")]
    out.extend({key: row.get(key, "") for key in fields | {"step", "candidate_rank", "block_key", "accepted"}} for row in validation_rows)
    write_csv(output_dir / "diagnostics" / "gate17_6_accuracy_objective_components.csv", out)


def _write_method_audit(output_dir: Path) -> None:
    rows = []
    for method in DEFAULT_METHODS:
        if method in {"full-graph-hettree-lite-tuned", "target-only-empty-support"}:
            path = "experiments/scripts/run_gate17_1_support_sensitivity.py"
        elif method in BASELINES:
            path = "experiments/scripts/gate13_task_first_common.py::run_support_baseline"
        elif method == "HeSF-SS-H6-cluster-validation-coverage-gated":
            path = "experiments/scripts/run_gate17_6_accuracy_calibrated_h6_fill.py::_h6_cluster_feedback"
        elif method in {H6_CONSTRUCTION_CONTROL_METHOD, H6_SELECTED_SET_CONTROL_METHOD}:
            path = "experiments/scripts/run_gate17_4_h6_equivalence.py"
        else:
            path = "experiments/scripts/run_gate17_6_accuracy_calibrated_h6_fill.py"
        rows.append({"method": method, "code_path": path, "gate17_6_role": "diagnostic_only" if method in DIAGNOSTIC_ONLY_METHODS else "main_or_baseline"})
    write_csv(output_dir / "gate17_6_method_to_code_path.csv", rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate17.6 requires --primary-eval-mode compressed_projected")
    if str(args.monitor) != "projected_val_macro_f1":
        raise ValueError("Gate17.6 requires --monitor projected_val_macro_f1")
    output_dir = Path(args.output_dir)
    diag_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    h6_equivalence_rows: list[dict[str, Any]] = []
    h6_cluster_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    prediction_payloads: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []

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
            requested_count = _requested_support_count(support_count, float(ratio))
            h6_coarse, h6_assignment, h6_diag = run_support_baseline(graph, baseline="H6-no-spec-support-only", ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
            h6_assignment = np.asarray(h6_assignment, dtype=np.int64)
            h6_nodes = selected_support_representatives_from_assignment(graph, h6_assignment, int(target_type))
            h6_task = _eval_task(graph, h6_coarse, h6_assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
            h6_cluster_feedback = _h6_cluster_feedback(
                graph=graph,
                h6_coarse=h6_coarse,
                h6_assignment=h6_assignment,
                h6_task=h6_task,
                dataset=str(dataset),
                seed=int(seed),
                ratio=float(ratio),
                target_type=int(target_type),
                split=split,
                labels=labels,
                train_mask=train_mask,
                args=args,
            )
            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **split_protocol}
                selected_nodes = np.empty(0, dtype=np.int64)
                selected_before_fill = np.empty(0, dtype=np.int64)
                selected_raw = 0
                graph_diag: dict[str, Any] = {}
                selection_diag: dict[str, Any] = {}
                task_for_prediction: dict[str, Any] | None = None
                coarse_for_delta = None
                assignment_for_delta = None
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                        coarse_for_delta = graph
                        assignment_for_delta = np.arange(graph.num_nodes, dtype=np.int64)
                        selected_raw = support_count
                    elif method == "target-only-empty-support":
                        target_row, coarse_for_delta, assignment_for_delta = _target_only_row(graph, str(dataset), int(seed), float(ratio), args, split)
                        row.update(target_row)
                    elif method in BASELINES:
                        if method == "H6-no-spec-support-only":
                            coarse, assignment, diag, task = h6_coarse, h6_assignment, h6_diag, h6_task
                            selected_nodes = h6_nodes
                        else:
                            coarse, assignment, diag = run_support_baseline(graph, baseline=str(method), ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                            assignment = np.asarray(assignment, dtype=np.int64)
                            selected_nodes = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                            task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        coarse_for_delta = coarse
                        assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        selected_raw = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                        row.update(budget_diagnostics(num_support=support_count, support_ratio=float(ratio), realized_support_count=selected_raw))
                        _task_metrics_row(row, task)
                        task_for_prediction = task
                    elif method == H6_CONSTRUCTION_CONTROL_METHOD:
                        coarse_for_delta = h6_coarse
                        assignment_for_delta = h6_assignment
                        selected_nodes = h6_nodes
                        selected_raw = int(np.sum(h6_coarse.node_type != int(target_type)))
                        _task_metrics_row(row, h6_task)
                        task_for_prediction = h6_task
                        row["h6_control_mode"] = "construction"
                        eq = _h6_equivalence_row(dataset=str(dataset), seed=int(seed), ratio=float(ratio), mode="construction", graph=graph, control=h6_coarse, control_assignment=h6_assignment, h6=h6_coarse, h6_assignment=h6_assignment, h6_task=h6_task, control_task=h6_task, target_type=int(target_type), selected_nodes=selected_nodes, h6_nodes=h6_nodes, args=args)
                        h6_equivalence_rows.append(eq)
                        row.update({key: value for key, value in eq.items() if key not in {"dataset", "seed", "requested_support_ratio"}})
                    elif method == H6_SELECTED_SET_CONTROL_METHOD:
                        cfg = SupportSelectorConfig(selector="teacher_topk", background_strategy="drop", allow_background_bucket=False, residual_prototype_mode="none", force_raw_bridge_nodes=False, force_raw_keep_high_degree_bridges=False, allow_proxy_fill=False)
                        control_coarse, control_assignment_obj, graph_diag = build_selected_support_graph(graph, h6_nodes, cfg, target_node_type=int(target_type), support_features=None)
                        control_task = _eval_task(graph, control_coarse, control_assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        coarse_for_delta = control_coarse
                        assignment_for_delta = np.asarray(control_assignment_obj.assignment, dtype=np.int64)
                        selected_nodes = h6_nodes
                        selected_raw = len(h6_nodes)
                        _task_metrics_row(row, control_task)
                        task_for_prediction = control_task
                        row["h6_control_mode"] = "selected_set"
                        eq = _h6_equivalence_row(dataset=str(dataset), seed=int(seed), ratio=float(ratio), mode="selected_set", graph=graph, control=control_coarse, control_assignment=control_assignment_obj.assignment, h6=h6_coarse, h6_assignment=h6_assignment, h6_task=h6_task, control_task=control_task, target_type=int(target_type), selected_nodes=selected_nodes, h6_nodes=h6_nodes, args=args)
                        h6_equivalence_rows.append(eq)
                        row.update({key: value for key, value in eq.items() if key not in {"dataset", "seed", "requested_support_ratio"}})
                    elif method == "HeSF-SS-H6-cluster-validation-coverage-gated":
                        coarse_for_delta, assignment_for_delta, selected_nodes, payload, annotated = _gate17_5_run_h6_cluster_method(
                            graph=graph,
                            h6_coarse=h6_coarse,
                            h6_assignment=h6_assignment,
                            h6_task=h6_task,
                            feedback_rows=h6_cluster_feedback,
                            dataset=str(dataset),
                            seed=int(seed),
                            ratio=float(ratio),
                            target_type=int(target_type),
                            split=split,
                            support_count=int(support_count),
                            method=str(method),
                            args=args,
                        )
                        task_for_prediction = _eval_task(graph, coarse_for_delta, assignment_for_delta, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        _task_metrics_row(row, task_for_prediction)
                        row.update({key: value for key, value in payload.items() if key not in row})
                        selected_raw = int(payload.get("h6_cluster_member_count_selected", len(selected_nodes)) or 0)
                        h6_cluster_rows.extend(annotated)
                    elif method == "HeSF-SS-H6-fill-only":
                        filled, fill_diag = h6_fill_support_nodes(
                            graph=graph,
                            h6_assignment=h6_assignment,
                            target_type=int(target_type),
                            selected_support_nodes=np.asarray([], dtype=np.int64),
                            requested_support_count=int(requested_count),
                        )
                        selected_nodes = filled
                        selected_before_fill = np.asarray([], dtype=np.int64)
                        cfg = _selector_for_method(str(method), args)
                        coarse_for_delta, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=None)
                        task = _eval_task(graph, coarse_for_delta, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        assignment_for_delta = np.asarray(assignment_obj.assignment, dtype=np.int64)
                        _task_metrics_row(row, task)
                        task_for_prediction = task
                        selected_raw = int(len(selected_nodes))
                        row.update(fill_diag)
                        row.update(_provenance_fields(selected_before_fill=selected_before_fill, selected_after_fill=selected_nodes, selection_diag={}, requested_count=int(requested_count), h6_fill_only_count=int(selected_raw)))
                    else:
                        result = _run_selection_pipeline(graph=graph, labels=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, target_type=int(target_type), ratio=float(ratio), method=str(method), seed=int(seed), teacher=teacher, args=args)
                        row.update(_flat_payload(result))
                        selection_diag = dict(result.get("selection", {}).get("diagnostics", {}))
                        graph_diag = dict(result["graph_diagnostics"])
                        selected_before_fill = np.asarray(result["selection"]["selected_support_nodes"], dtype=np.int64)
                        selected_nodes = selected_before_fill
                        coarse_for_delta = result["coarse_graph"]
                        assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                        task_for_prediction = result.get("task_metrics")
                        if method in {"HeSF-SS-validation-H6-fill", *ACCURACY_CALIBRATED_METHODS}:
                            filled, fill_diag = h6_fill_support_nodes(
                                graph=graph,
                                h6_assignment=h6_assignment,
                                target_type=int(target_type),
                                selected_support_nodes=selected_before_fill,
                                requested_support_count=int(requested_count),
                            )
                            selected_nodes = filled
                            cfg = _selector_for_method(str(method), args)
                            coarse_for_delta, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=None)
                            task = _eval_task(graph, coarse_for_delta, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                            assignment_for_delta = np.asarray(assignment_obj.assignment, dtype=np.int64)
                            _task_metrics_row(row, task)
                            task_for_prediction = task
                            row.update(fill_diag)
                            row.update(_provenance_fields(selected_before_fill=selected_before_fill, selected_after_fill=selected_nodes, selection_diag=selection_diag, requested_count=int(requested_count), h6_fill_count=int(fill_diag.get("h6_fill_support_count", 0))))
                        elif method == "HeSF-SS-random-fill-after-validation":
                            filled, fill_diag = _random_fill_support_nodes(
                                graph=graph,
                                target_type=int(target_type),
                                selected_support_nodes=selected_before_fill,
                                requested_support_count=int(requested_count),
                                seed=int(seed),
                            )
                            selected_nodes = filled
                            cfg = _selector_for_method(str(method), args)
                            coarse_for_delta, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=None)
                            task = _eval_task(graph, coarse_for_delta, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                            assignment_for_delta = np.asarray(assignment_obj.assignment, dtype=np.int64)
                            _task_metrics_row(row, task)
                            task_for_prediction = task
                            row.update(fill_diag)
                            row.update(_provenance_fields(selected_before_fill=selected_before_fill, selected_after_fill=selected_nodes, selection_diag=selection_diag, requested_count=int(requested_count), random_fill_count=int(fill_diag.get("random_fill_support_count", 0))))
                        else:
                            task = _eval_task(graph, coarse_for_delta, assignment_for_delta, seed=int(seed), split=split, target_type=int(target_type), args=args)
                            _task_metrics_row(row, task)
                            task_for_prediction = task
                            row.update(_provenance_fields(selected_before_fill=selected_before_fill, selected_after_fill=selected_nodes, selection_diag=selection_diag, requested_count=int(requested_count)))
                        selected_raw = int(len(selected_nodes))
                        for item in result["selection"].get("validation_greedy_trials", []):
                            trial = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item}
                            validation_rows.append(trial)
                        if method in ACCURACY_CALIBRATED_METHODS and task_for_prediction is not None:
                            objective_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **_add_accuracy_objective_row_fields(row, task_for_prediction, int(requested_count), int(selected_raw), ACCURACY_CALIBRATED_METHODS[str(method)], args)})
                        else:
                            alpha = 0.0
                            if str(method) == "HeSF-SS-validation-H6-fill":
                                objective_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **_add_accuracy_objective_row_fields(row, task_for_prediction or {}, int(requested_count), int(selected_raw), alpha, args)})

                    if coarse_for_delta is not None and assignment_for_delta is not None:
                        semantic = _semantic_and_h6_delta(row=row, graph=graph, coarse=coarse_for_delta, assignment=np.asarray(assignment_for_delta, dtype=np.int64), target_only_graph=target_only_graph, target_only_assignment=target_only_assignment, h6_coarse=h6_coarse, h6_assignment=h6_assignment, target_type=int(target_type), dataset=str(dataset), seed=int(seed), ratio=float(ratio), args=args)
                        semantic_rows.append(semantic)
                    row.update(_overlap_fields(selected_nodes, h6_nodes))
                    all_h6_clusters = {unit.cluster_id for unit in extract_h6_cluster_units(graph, h6_assignment, int(target_type))}
                    selected_clusters = {int(h6_assignment[int(node)]) for node in selected_nodes.tolist() if 0 <= int(node) < len(h6_assignment)}
                    row.update(_cluster_overlap_fields(selected_clusters, all_h6_clusters))
                    row["overlap_validation_with_h6"] = float(row.get("cluster_overlap_with_h6", 0.0) or 0.0)
                    if method == "HeSF-SS-random-fill-after-validation":
                        random_added = set(selected_nodes.tolist()) - set(selected_before_fill.tolist())
                        row["overlap_random_fill_with_h6"] = float(len(random_added & set(h6_nodes.tolist())) / max(1, len(random_added)))
                    row.setdefault("selector_uses_test_labels", False)
                    row.setdefault("teacher_uses_test_labels_for_training", False)
                    row["no_test_leakage"] = not bool(row["selector_uses_test_labels"] or row["teacher_uses_test_labels_for_training"])
                    _budget_update(row, support_count, selected_raw=int(selected_raw), graph_diag=graph_diag)
                    _budget_aliases(row, selected_raw=int(selected_raw), graph_diag=graph_diag)
                    row["diagnostic_only"] = bool(method in DIAGNOSTIC_ONLY_METHODS or method in BASELINES or method in {"full-graph-hettree-lite-tuned", "target-only-empty-support", "HeSF-SS-H6-fill-only"})
                    row["eligible_for_main_decision"] = bool(
                        str(method).startswith("HeSF-SS-")
                        and method not in DIAGNOSTIC_ONLY_METHODS
                        and method != "HeSF-SS-H6-fill-only"
                        and not row["diagnostic_only"]
                    )
                    row["status"] = row.get("status", "success")
                    row.setdefault("primary_eval_mode", str(args.primary_eval_mode))
                    row.setdefault("primary_task_metric_name", "projected_original_macro_f1")
                    row.setdefault("projected_macro_f1", row.get("macro_f1", ""))
                    row.setdefault("transfer_macro_f1", "")
                    row.setdefault("projected_vs_transfer_macro_gap", "")
                    row["run_mode"] = "gate17_6_accuracy_calibrated_h6_fill"
                    row["method_role"] = "H6-assisted validation fill" if str(method).startswith("HeSF-SS-validation-H6-fill") else ""
                    row["wall_clock_sec"] = float(perf_counter() - start)
                    if task_for_prediction is not None:
                        payload = _prediction_payload(
                            dataset=str(dataset),
                            seed=int(seed),
                            method=str(method),
                            ratio=float(ratio),
                            task=task_for_prediction,
                            selected_labels=_support_labels(labels, selected_nodes),
                        )
                        if payload is not None:
                            prediction_payloads.append(payload)
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                rows.append(row)
                write_csv(output_dir / "gate17_6_raw_rows.csv", rows)
                budget_rows.append({key: row.get(key, "") for key in row.keys() if key in {"dataset", "seed", "method", "requested_support_ratio"} or "budget" in key or "represented" in key or key in {"eligible_for_main_decision", "underfill_ratio", "overfill_ratio", "effective_support_node_count", "effective_support_node_ratio", "requested_support_count", "selected_support_count"}})

    _write_per_class_outputs(output_dir, rows, prediction_payloads)
    _write_fill_ablation(output_dir, rows)
    _write_accuracy_components(output_dir, rows, [*validation_rows, *objective_rows])
    write_csv(diag_dir / "gate17_6_h6_cluster_coverage_diagnostics.csv", h6_cluster_rows)
    write_csv(diag_dir / "gate17_6_h6_equivalence.csv", h6_equivalence_rows)
    write_csv(diag_dir / "gate17_6_candidate_semantic_delta.csv", [row for row in semantic_rows if str(row.get("method", "")).startswith("HeSF-SS")])
    write_csv(diag_dir / "gate17_6_validation_trials.csv", validation_rows)
    write_csv(diag_dir / "gate17_6_budget_breakdown.csv", budget_rows)
    _write_method_audit(output_dir)
    return summarize(output_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.6 accuracy-calibrated H6-assisted validation fill diagnostic.")
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=Path("outputs/gate17_6_accuracy_calibrated_h6_fill"))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--cluster-gating-feedback-epochs", type=int, default=2)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
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
    parser.add_argument("--include-typedhash", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--alpha-accuracy-grid", nargs="*", default=[0.0, 0.25, 0.50, 1.00])
    parser.add_argument("--delta-micro", type=float, default=0.25)
    parser.add_argument("--beta-underfill", type=float, default=0.10)
    parser.add_argument("--gamma-class-collapse", type=float, default=0.05)
    parser.add_argument("--min-validation-gain", type=float, default=1.0e-4)
    parser.add_argument("--h6-fill-weight", type=float, default=0.25)
    parser.add_argument("--cluster-gating-candidate-pool-size", type=int, default=16)
    parser.add_argument("--cluster-gating-min-gain", type=float, default=1.0e-4)
    parser.add_argument("--budget-penalty-lambda", type=float, default=0.05)
    parser.add_argument("--underfill-penalty-lambda", type=float, default=0.10)
    parser.add_argument("--neutral-fill-max-drop", type=float, default=1.0e-4)
    parser.add_argument("--negative-fill-max-drop", type=float, default=5.0e-4)
    parser.add_argument("--lambda-edge", type=float, default=0.10)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--lambda-relation", type=float, default=0.05)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = set(str(dataset) for dataset in args.datasets)
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if dataset in allowed]
    args.ratios = _split_values(args.support_ratios, float) or [0.30, 0.70]
    if not _bool_arg(args.include_typedhash):
        args.methods = [method for method in args.methods if method != "TypedHash-ChebHeat-support-only"]
    elif "TypedHash-ChebHeat-support-only" not in args.methods:
        args.methods = [*args.methods, "TypedHash-ChebHeat-support-only"]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
