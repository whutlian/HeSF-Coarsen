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
from experiments.scripts.gate17_4_h6 import (
    coarse_graph_hash,
    edge_mass_delta,
    feature_mean_delta,
    semantic_delta_vs_h6,
    selected_support_representatives_from_assignment,
)
from experiments.scripts.run_gate17_1_support_sensitivity import (
    _full_graph_row,
    _semantic_row_for_graph,
    _target_only_empty_support_graph,
    _target_only_row,
)
from experiments.scripts.run_gate17_3_lossy_prototype_feedback import _overlap_fields
from experiments.scripts.run_gate17_4_h6_equivalence import (
    _budget_update,
    _eval_task,
    _h6_equivalence_row,
    _selector_for_method as _gate17_4_selector_for_method,
    _semantic_fields_for_raw_row,
    _task_metrics_row,
)
from experiments.scripts.run_gate17_support_selection import _flat_payload, _mask, _split_values
from experiments.scripts.summarize_gate17_5 import summarize
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.h6_cluster_gating import (
    build_gated_h6_graph,
    extract_h6_cluster_units,
    h6_fill_support_nodes,
    select_h6_clusters_by_budget,
)
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


GATE17_5_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
H6_CONSTRUCTION_CONTROL_METHOD = "HeSF-SS-H6-equivalence-control"
H6_SELECTED_SET_CONTROL_METHOD = "HeSF-SS-H6-selected-set-control"
BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "random-support-only",
)
MAIN_CANDIDATE_METHODS = (
    "HeSF-SS-real-validation-neutral-fill",
    "HeSF-SS-real-validation-budget-penalty-fill",
    "HeSF-SS-real-validation-H6-fill",
    "HeSF-SS-H6-cluster-validation-gated",
    "HeSF-SS-H6-cluster-validation-budget-penalty",
)
H6_CLUSTER_GATED_METHODS = {
    "HeSF-SS-H6-cluster-validation-gated",
    "HeSF-SS-H6-cluster-validation-budget-penalty",
}
DIAGNOSTIC_ONLY_METHODS = (
    "HeSF-SS-real-occlusion-neutral-fill",
    H6_CONSTRUCTION_CONTROL_METHOD,
    H6_SELECTED_SET_CONTROL_METHOD,
    "HeSF-SS-full-residual-prototype-upperbound",
    "HeSF-SS-lossy-prototype-fixed-saturation",
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
        return [(dataset, seed) for dataset, seed in GATE17_5_SINGLE_SEED_BY_DATASET.items()]
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
        if dataset not in GATE17_5_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate17.5 dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


def _selector_for_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    mapped = {
        "HeSF-SS-real-validation-budget-penalty-fill": "HeSF-SS-real-validation-neutral-fill",
        "HeSF-SS-real-validation-H6-fill": "HeSF-SS-real-validation-neutral-fill",
        "HeSF-SS-lossy-prototype-fixed-saturation": "HeSF-SS-lossy-prototype-fixed-saturation",
    }.get(method, method)
    cfg = _gate17_4_selector_for_method(mapped, args)
    if method == "HeSF-SS-real-validation-budget-penalty-fill":
        cfg = replace(
            cfg,
            min_gain=float(args.cluster_gating_min_gain),
            neutral_fill=True,
            neutral_fill_max_drop=float(args.neutral_fill_max_drop),
            allow_negative_fill=True,
            negative_fill_max_drop=float(args.negative_fill_max_drop),
            allow_proxy_fill=False,
            budget_penalty_lambda=float(args.budget_penalty_lambda),
            underfill_penalty_lambda=float(args.underfill_penalty_lambda),
        )
    if method == "HeSF-SS-real-validation-H6-fill":
        cfg = replace(cfg, allow_proxy_fill=False, neutral_fill=True, neutral_fill_max_drop=float(args.neutral_fill_max_drop))
    return cfg


def _requested_support_count(support_count: int, ratio: float) -> int:
    return int(np.ceil(int(support_count) * float(ratio) - 1.0e-12))


def _budget_aliases(row: dict[str, Any], *, selected_raw: int, graph_diag: dict[str, Any] | None = None) -> None:
    graph_diag = graph_diag or {}
    forced = int(graph_diag.get("forced_raw_bridge_count", 0) or row.get("forced_raw_support_count", 0) or 0)
    proto = int(graph_diag.get("prototype_background_count", 0) or row.get("prototype_background_count", 0) or 0)
    row["selected_budget_support_count"] = int(selected_raw)
    row["forced_raw_support_count"] = int(forced)
    row["effective_support_node_count"] = int(row.get("node_budget_count", selected_raw + forced + proto) or 0)
    row["effective_support_node_ratio"] = float(row.get("node_budget_ratio", 0.0) or 0.0)
    row["represented_support_context_count"] = int(row.get("represented_context_count", row.get("represented_support_context_count", 0)) or 0)
    row["represented_support_context_ratio"] = float(row.get("represented_context_ratio", 0.0) or 0.0)
    requested = float(row.get("requested_support_ratio", 0.0) or 0.0)
    effective = float(row.get("effective_support_node_ratio", 0.0) or 0.0)
    row.setdefault("underfill_ratio", max(0.0, requested - effective))
    row.setdefault("overfill_ratio", max(0.0, effective - requested))


def _cluster_overlap_fields(selected_cluster_ids: set[int], all_h6_cluster_ids: set[int]) -> dict[str, Any]:
    intersection = selected_cluster_ids & all_h6_cluster_ids
    union = selected_cluster_ids | all_h6_cluster_ids
    return {
        "cluster_overlap_with_h6": float(len(intersection) / max(len(union), 1)),
        "cluster_recall_of_h6": float(len(intersection) / max(len(all_h6_cluster_ids), 1)),
        "cluster_precision_vs_h6": float(len(intersection) / max(len(selected_cluster_ids), 1)),
    }


def _h6_delta_alias_fields(edge: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_mass_l1_delta_vs_h6_by_relation": edge.get("edge_mass_l1_delta_vs_h6", ""),
        "feature_mean_l2_delta_vs_h6_by_type": feature.get("feature_mean_l2_delta_vs_h6", ""),
    }


def _semantic_and_h6_delta(
    *,
    row: dict[str, Any],
    graph,
    coarse,
    assignment: np.ndarray,
    target_only_graph,
    target_only_assignment: np.ndarray,
    h6_coarse,
    h6_assignment: np.ndarray,
    target_type: int,
    dataset: str,
    seed: int,
    ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    semantic = _semantic_row_for_graph(
        graph=graph,
        coarse=coarse,
        assignment=np.asarray(assignment, dtype=np.int64),
        target_only=target_only_graph,
        target_only_assignment=target_only_assignment,
        target_type=int(target_type),
        dataset=str(dataset),
        seed=int(seed),
        method=str(row.get("method")),
        ratio=float(ratio),
        args=args,
    )
    row.update(_semantic_fields_for_raw_row(semantic))
    h6_delta = semantic_delta_vs_h6(
        original=graph,
        control=coarse,
        control_assignment=np.asarray(assignment, dtype=np.int64),
        h6=h6_coarse,
        h6_assignment=np.asarray(h6_assignment, dtype=np.int64),
        target_type=int(target_type),
        max_paths=int(args.max_paths),
    )
    row.update(h6_delta)
    row["tree_l2_delta_vs_full"] = row.get("tree_tensor_l2_delta_vs_full", "")
    row["coarse_graph_hash"] = coarse_graph_hash(coarse)
    row["candidate_full_equivalent"] = bool(semantic.get("allclose_to_full", False))
    row["candidate_h6_equivalent"] = bool(h6_delta.get("tree_hash_equal_to_h6", False))
    edge = edge_mass_delta(coarse, h6_coarse)
    feature = feature_mean_delta(coarse, h6_coarse)
    row.update(edge)
    row.update(feature)
    row.update(_h6_delta_alias_fields(edge, feature))
    return semantic


def _run_selection_pipeline(
    *,
    graph,
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


def _h6_cluster_feedback(
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
    units = extract_h6_cluster_units(graph, h6_assignment, int(target_type))
    candidates = sorted(units, key=lambda unit: (-unit.member_count, unit.cluster_id))[: int(args.cluster_gating_candidate_pool_size)]
    rows: list[dict[str, Any]] = []
    all_cluster_ids = {unit.cluster_id for unit in units}
    for unit in candidates:
        keep = sorted(all_cluster_ids - {int(unit.cluster_id)})
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
        )
        gain = float(h6_task.get("validation_macro_f1", 0.0) or 0.0) - float(task.get("validation_macro_f1", 0.0) or 0.0)
        rows.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "requested_support_ratio": float(ratio),
                "cluster_id": int(unit.cluster_id),
                "h6_cluster_type": int(unit.cluster_type),
                "h6_cluster_size": int(unit.member_count),
                "validation_gain": float(gain),
                "validation_macro_f1_if_dropped": float(task.get("validation_macro_f1", 0.0) or 0.0),
                "validation_ce_delta": float(max(gain, 0.0)),
                "occlusion_delta_ce": float(max(gain, 0.0)),
                "occlusion_delta_macro_f1": float(gain),
                "occlusion_delta_margin": float(float(h6_task.get("validation_accuracy", 0.0) or 0.0) - float(task.get("validation_accuracy", 0.0) or 0.0)),
                "cluster_edge_mass": 0.0,
                "cluster_relation_channel_profile": "",
                "cluster_target_anchor_coverage": 0.0,
                "selected": False,
            }
        )
    return rows


def _run_h6_cluster_method(
    *,
    graph,
    h6_coarse,
    h6_assignment: np.ndarray,
    h6_task: dict[str, Any],
    feedback_rows: list[dict[str, Any]],
    dataset: str,
    seed: int,
    ratio: float,
    target_type: int,
    split: dict[str, np.ndarray],
    support_count: int,
    method: str,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    units = extract_h6_cluster_units(graph, h6_assignment, int(target_type))
    scores = {int(row["cluster_id"]): float(row.get("validation_gain", 0.0) or 0.0) for row in feedback_rows}
    selection = select_h6_clusters_by_budget(
        units,
        support_count=int(support_count),
        requested_support_ratio=float(ratio),
        validation_scores=scores,
        min_gain=float(args.cluster_gating_min_gain),
        neutral_fill_max_drop=float(args.neutral_fill_max_drop),
        negative_fill_max_drop=float(args.negative_fill_max_drop),
        budget_penalty_lambda=float(args.budget_penalty_lambda),
        underfill_penalty_lambda=float(args.underfill_penalty_lambda),
    )
    gated_graph, gated_assignment, kept = build_gated_h6_graph(
        original=graph,
        h6_coarse=h6_coarse,
        h6_assignment=h6_assignment,
        selected_cluster_ids=selection.selected_cluster_ids,
        target_type=int(target_type),
    )
    task = _eval_task(graph, gated_graph, gated_assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
    selected_member_nodes = np.asarray(
        sorted({int(node) for unit in units if int(unit.cluster_id) in set(selection.selected_cluster_ids) for node in unit.member_nodes.tolist()}),
        dtype=np.int64,
    )
    selected_set = set(selection.selected_cluster_ids)
    annotated_feedback = []
    for row in feedback_rows:
        item = dict(row)
        item["method"] = str(method)
        item["selected"] = int(row["cluster_id"]) in selected_set
        annotated_feedback.append(item)
    diag = {
        **selection.to_row(),
        "h6_cluster_count_total": int(len(units)),
        "h6_cluster_count_selected": int(len(selection.selected_cluster_ids)),
        "h6_cluster_member_count_selected": int(selection.member_count_selected),
        "h6_cluster_member_ratio_selected": float(selection.member_ratio_selected),
        "h6_cluster_validation_gain_mean": float(np.mean(list(scores.values()))) if scores else 0.0,
        "h6_cluster_validation_gain_max": float(np.max(list(scores.values()))) if scores else 0.0,
        "h6_cluster_occlusion_delta_mean": float(np.mean(list(scores.values()))) if scores else 0.0,
        "h6_cluster_occlusion_delta_max": float(np.max(list(scores.values()))) if scores else 0.0,
        "cluster_budget_unit": "support_member_count",
        "cluster_budget_definition": "member_weighted",
        "h6_cluster_units_used": True,
        "h6_cluster_gating_rebuilt_graph": True,
        "h6_cluster_copied_h6_task_metrics": False,
        "validation_trial_count": int(len(feedback_rows)),
        "validation_positive_gain_count": int(sum(1 for value in scores.values() if value > 1.0e-6)),
        "validation_gain_max": float(np.max(list(scores.values()))) if scores else 0.0,
        "validation_gain_mean": float(np.mean(list(scores.values()))) if scores else 0.0,
        "validation_gain_history": json.dumps([float(value) for value in scores.values()]),
        "validation_neutral_fill_count": int(selection.neutral_fill_block_count),
        "validation_proxy_fill_count": int(selection.proxy_fill_block_count),
        "validation_feedback_degenerate": bool(not scores or max(scores.values()) <= 1.0e-6),
        "validation_signal_pass": bool(scores and max(scores.values()) > 1.0e-6),
    }
    return gated_graph, gated_assignment, selected_member_nodes, {**diag, **task}, annotated_feedback


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate17.5 requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    diag_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "typedhash_skip_note.md").write_text("TypedHash skipped for speed in Gate17.5; must be restored before Gate18.\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    h6_equivalence_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    dblp_budget_fill_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []

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
                args=args,
            )
            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **split_protocol}
                selected_nodes = np.empty(0, dtype=np.int64)
                selected_raw = 0
                graph_diag: dict[str, Any] = {}
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
                            task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                        coarse_for_delta = coarse
                        assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        selected_raw = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                        row.update(budget_diagnostics(num_support=support_count, support_ratio=float(ratio), realized_support_count=selected_raw))
                        _task_metrics_row(row, task)
                    elif method == H6_CONSTRUCTION_CONTROL_METHOD:
                        coarse_for_delta = h6_coarse
                        assignment_for_delta = h6_assignment
                        selected_nodes = h6_nodes
                        selected_raw = int(np.sum(h6_coarse.node_type != int(target_type)))
                        _task_metrics_row(row, h6_task)
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
                        row["h6_control_mode"] = "selected_set"
                        eq = _h6_equivalence_row(dataset=str(dataset), seed=int(seed), ratio=float(ratio), mode="selected_set", graph=graph, control=control_coarse, control_assignment=control_assignment_obj.assignment, h6=h6_coarse, h6_assignment=h6_assignment, h6_task=h6_task, control_task=control_task, target_type=int(target_type), selected_nodes=selected_nodes, h6_nodes=h6_nodes, args=args)
                        h6_equivalence_rows.append(eq)
                        row.update({key: value for key, value in eq.items() if key not in {"dataset", "seed", "requested_support_ratio"}})
                    elif method in H6_CLUSTER_GATED_METHODS:
                        coarse_for_delta, assignment_for_delta, selected_nodes, payload, annotated = _run_h6_cluster_method(
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
                        _task_metrics_row(row, payload)
                        row.update({key: value for key, value in payload.items() if key not in row})
                        selected_raw = int(payload.get("h6_cluster_member_count_selected", len(selected_nodes)) or 0)
                        cluster_rows.extend(annotated)
                    else:
                        result = _run_selection_pipeline(graph=graph, labels=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, target_type=int(target_type), ratio=float(ratio), method=str(method), seed=int(seed), teacher=teacher, args=args)
                        selected_nodes = np.asarray(result["selection"]["selected_support_nodes"], dtype=np.int64)
                        if method == "HeSF-SS-real-validation-H6-fill":
                            filled, fill_diag = h6_fill_support_nodes(
                                graph=graph,
                                h6_assignment=h6_assignment,
                                target_type=int(target_type),
                                selected_support_nodes=selected_nodes,
                                requested_support_count=_requested_support_count(support_count, float(ratio)),
                            )
                            selected_nodes = filled
                            cfg = _selector_for_method(str(method), args)
                            coarse_for_delta, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=None)
                            task = _eval_task(graph, coarse_for_delta, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args)
                            assignment_for_delta = np.asarray(assignment_obj.assignment, dtype=np.int64)
                            _task_metrics_row(row, task)
                            row.update(fill_diag)
                        else:
                            row.update(_flat_payload(result))
                            coarse_for_delta = result["coarse_graph"]
                            assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                            graph_diag = dict(result["graph_diagnostics"])
                        selected_raw = int(len(selected_nodes))
                        for item in result["selection"].get("validation_greedy_trials", []):
                            validation_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item})
                        if str(dataset).upper() == "DBLP" and "real-validation" in str(method):
                            dblp_budget_fill_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), "selected_support_count": int(selected_raw), "requested_support_count": _requested_support_count(support_count, float(ratio)), "underfill_count": max(0, _requested_support_count(support_count, float(ratio)) - int(selected_raw)), **{key: row.get(key, "") for key in row if "fill" in key or "penalty" in key or key in {"underfill_ratio", "overfill_ratio"}}})

                    if coarse_for_delta is not None and assignment_for_delta is not None:
                        semantic = _semantic_and_h6_delta(row=row, graph=graph, coarse=coarse_for_delta, assignment=np.asarray(assignment_for_delta, dtype=np.int64), target_only_graph=target_only_graph, target_only_assignment=target_only_assignment, h6_coarse=h6_coarse, h6_assignment=h6_assignment, target_type=int(target_type), dataset=str(dataset), seed=int(seed), ratio=float(ratio), args=args)
                        semantic_rows.append(semantic)
                    row.update(_overlap_fields(selected_nodes, h6_nodes))
                    all_h6_clusters = {unit.cluster_id for unit in extract_h6_cluster_units(graph, h6_assignment, int(target_type))}
                    selected_clusters = {int(h6_assignment[int(node)]) for node in selected_nodes.tolist() if 0 <= int(node) < len(h6_assignment)}
                    row.update(_cluster_overlap_fields(selected_clusters, all_h6_clusters))
                    row.setdefault("selector_uses_test_labels", False)
                    row.setdefault("teacher_uses_test_labels_for_training", False)
                    row["no_test_leakage"] = not bool(row["selector_uses_test_labels"] or row["teacher_uses_test_labels_for_training"])
                    _budget_update(row, support_count, selected_raw=int(selected_raw), graph_diag=graph_diag)
                    _budget_aliases(row, selected_raw=int(selected_raw), graph_diag=graph_diag)
                    row["diagnostic_only"] = bool(method in DIAGNOSTIC_ONLY_METHODS or method in BASELINES or method in {"full-graph-hettree-lite-tuned", "target-only-empty-support"})
                    if row["diagnostic_only"]:
                        row["eligible_for_main_decision"] = False
                    row["status"] = row.get("status", "success")
                    row.setdefault("primary_eval_mode", str(args.primary_eval_mode))
                    row.setdefault("primary_task_metric_name", "projected_original_macro_f1")
                    row.setdefault("projected_macro_f1", row.get("macro_f1", ""))
                    row.setdefault("transfer_macro_f1", "")
                    row.setdefault("projected_vs_transfer_macro_gap", "")
                    row["run_mode"] = "gate17_5_h6_cluster_gating"
                    row["wall_clock_sec"] = float(perf_counter() - start)
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                rows.append(row)
                write_csv(output_dir / "gate17_5_raw_rows.csv", rows)
                budget_rows.append({key: row.get(key, "") for key in row.keys() if key in {"dataset", "seed", "method", "requested_support_ratio"} or "budget" in key or "represented" in key or key in {"eligible_for_main_decision", "underfill_ratio", "overfill_ratio", "effective_support_node_count", "effective_support_node_ratio"}})

    write_csv(diag_dir / "gate17_5_h6_cluster_gating.csv", cluster_rows)
    write_csv(diag_dir / "gate17_5_dblp_budget_fill.csv", dblp_budget_fill_rows)
    write_csv(diag_dir / "gate17_5_h6_equivalence.csv", h6_equivalence_rows)
    write_csv(diag_dir / "gate17_5_candidate_semantic_delta.csv", [row for row in semantic_rows if str(row.get("method", "")).startswith("HeSF-SS")])
    write_csv(diag_dir / "gate17_5_validation_trials.csv", validation_rows)
    write_csv(diag_dir / "gate17_5_budget_breakdown.csv", budget_rows)
    return summarize(output_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.5 summary-fixed H6 cluster gating diagnostic.")
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=Path("outputs/gate17_5_h6_cluster_gating"))
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
    parser.add_argument("--include-typedhash", action="store_true", default=False)
    parser.add_argument("--cluster-gating-candidate-pool-size", type=int, default=16)
    parser.add_argument("--cluster-gating-min-gain", type=float, default=1.0e-4)
    parser.add_argument("--budget-penalty-lambda", type=float, default=0.05)
    parser.add_argument("--underfill-penalty-lambda", type=float, default=0.10)
    parser.add_argument("--neutral-fill-max-drop", type=float, default=1.0e-4)
    parser.add_argument("--negative-fill-max-drop", type=float, default=5.0e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = set(str(dataset) for dataset in args.datasets)
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if dataset in allowed]
    args.ratios = _split_values(args.support_ratios, float) or [0.30, 0.70]
    if args.include_typedhash and "TypedHash-ChebHeat-support-only" not in args.methods:
        args.methods = [*args.methods, "TypedHash-ChebHeat-support-only"]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
