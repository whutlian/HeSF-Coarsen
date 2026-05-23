from __future__ import annotations

import argparse
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
from experiments.scripts.gate17_4_h6 import induced_coarse_graph, selected_support_representatives_from_assignment
from experiments.scripts.run_gate17_1_support_sensitivity import _full_graph_row, _target_only_empty_support_graph, _target_only_row
from experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill import (
    _random_fill_support_nodes,
    _requested_support_count,
    _run_selection_pipeline as _gate17_6_run_selection_pipeline,
    _selector_for_method as _gate17_6_selector_for_method,
    _support_labels,
    per_class_audit_rows,
)
from experiments.scripts.run_gate17_support_selection import _fast_support_features, _mask, _row_from_task, _split_values
from experiments.scripts.summarize_gate18r import summarize
from hesf_coarsen.eval.calibration import apply_calibrator, fit_macro_constrained_accuracy_calibrator
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import f1_scores, select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.feature_condensation.eval import evaluate_feature_condensation_method
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.selection.h6_cluster_gating import build_gated_h6_graph, h6_fill_support_nodes
from hesf_coarsen.task_first.units.flatten_units import extract_flatten_units
from hesf_coarsen.task_first.units.gated_builder import build_graph_from_units, select_units_under_budget, selected_member_nodes
from hesf_coarsen.task_first.units.h6_units import extract_h6_units
from hesf_coarsen.task_first.units.scoring import score_units
from hesf_coarsen.task_first.units.typedhash_units import extract_typedhash_units
from hesf_coarsen.task_first.units.union_units import make_union_units
from hesf_coarsen.task_first.units.validation_blocks import extract_validation_block_units


GATE18R_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
BASELINES = (
    "random-support-only",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
)
NEW_CANDIDATE_METHODS = (
    "HeSF-SS-validation-underfill-pareto",
    "HeSF-SS-validation-underfill-pareto-logit-calibrated",
    "HeSF-SS-validation-H6-fill-logit-calibrated",
    "HeSF-ClusterGate-H6-units",
    "HeSF-ClusterGate-TypedHash-units",
    "HeSF-ClusterGate-Flatten-units",
    "HeSF-ClusterGate-UnionUnits",
    "HeSF-ClusterGate-UnionUnits-logit-calibrated",
    "HeSF-STC-path-prune",
    "HeSF-STC-path-prototype",
    "HeSF-STC-feature-cache-distill",
    "HeSF-STC-feature-cache-distill-logit-calibrated",
)
OLD_ABLATION_METHODS = (
    "HeSF-SS-validation-H6-fill",
    "HeSF-SS-H6-fill-only",
    "HeSF-SS-random-fill-after-validation",
    "HeSF-SS-validation-H6-fill-acc0.25",
    "HeSF-SS-validation-H6-fill-acc0.50",
    "HeSF-SS-validation-H6-fill-acc1.00",
)
DEFAULT_METHODS = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    *BASELINES,
    *NEW_CANDIDATE_METHODS,
    *OLD_ABLATION_METHODS,
)
CALIBRATED_METHODS = {
    "HeSF-SS-validation-underfill-pareto-logit-calibrated",
    "HeSF-SS-validation-H6-fill-logit-calibrated",
    "HeSF-ClusterGate-UnionUnits-logit-calibrated",
    "HeSF-STC-feature-cache-distill-logit-calibrated",
}
STC_METHODS = {
    "HeSF-STC-path-prune",
    "HeSF-STC-path-prototype",
    "HeSF-STC-feature-cache-distill",
    "HeSF-STC-feature-cache-distill-logit-calibrated",
}


def parse_dataset_seeds(values: list[str] | tuple[str, ...] | str | None) -> list[tuple[str, int]]:
    if values is None or values == "":
        return [(dataset, seed) for dataset, seed in GATE18R_SINGLE_SEED_BY_DATASET.items()]
    tokens: list[str] = []
    raw_values = [values] if isinstance(values, str) else list(values)
    for value in raw_values:
        tokens.extend(item for item in str(value).replace(",", " ").split() if item)
    out: list[tuple[str, int]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"dataset seed token must be DATASET:SEED, got {token!r}")
        dataset, seed = token.split(":", 1)
        dataset = dataset.strip().upper()
        if dataset not in GATE18R_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate18R dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _scores_from_logits(logits: Any, labels: Any) -> dict[str, Any]:
    logits_arr = np.asarray(logits, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits_arr.size == 0 or len(labels_arr) == 0:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": []}
    pred = np.argmax(logits_arr, axis=1).astype(np.int64, copy=False)
    valid = (labels_arr >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": pred.tolist()}
    scores = f1_scores(labels_arr[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {**scores, "accuracy": float(np.mean(labels_arr[valid] == pred[valid])), "pred": pred.tolist()}


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
    return_logits: bool | None = None,
) -> dict[str, Any]:
    want_logits = bool(args.return_logits if return_logits is None else return_logits)
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
        return_predictions=True,
        return_logits=want_logits,
        return_prediction_payload=want_logits,
    ).metrics


def _apply_calibration(task: Mapping[str, Any], *, baseline_macro: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    val_logits = task.get("projected_val_logits", [])
    val_labels = task.get("projected_val_labels", task.get("projected_val_true_labels", []))
    test_logits = task.get("projected_test_logits", [])
    test_labels = task.get("projected_test_labels", task.get("projected_test_true_labels", []))
    if not val_logits or not test_logits:
        calibrated = dict(task)
        return calibrated, {"calibration_status": "missing_logits", "calibrator_uses_test_labels": False}
    fit = fit_macro_constrained_accuracy_calibrator(
        val_logits,
        val_labels,
        baseline_macro=baseline_macro,
        macro_epsilon=0.005,
        temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        class_bias_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
    )
    calibrated_val = apply_calibrator(val_logits, fit)
    calibrated_test = apply_calibrator(test_logits, fit)
    val_scores = _scores_from_logits(calibrated_val, val_labels)
    test_scores = _scores_from_logits(calibrated_test, test_labels)
    out = dict(task)
    out.update(
        {
            "uncalibrated_macro_f1": _float(task.get("macro_f1")),
            "uncalibrated_accuracy": _float(task.get("accuracy")),
            "uncalibrated_validation_macro_f1": _float(task.get("validation_macro_f1")),
            "uncalibrated_validation_accuracy": _float(task.get("validation_accuracy")),
            "macro_f1": float(test_scores["macro_f1"]),
            "micro_f1": float(test_scores["micro_f1"]),
            "accuracy": float(test_scores["accuracy"]),
            "validation_macro_f1": float(val_scores["macro_f1"]),
            "validation_micro_f1": float(val_scores["micro_f1"]),
            "validation_accuracy": float(val_scores["accuracy"]),
            "projected_test_pred": test_scores["pred"],
            "projected_val_pred": val_scores["pred"],
            "projected_test_pred_labels": test_scores["pred"],
            "projected_val_pred_labels": val_scores["pred"],
            "calibrated": True,
            "calibrator_uses_test_labels": False,
        }
    )
    return out, {"calibration_status": "success", **fit}


def _task_to_row(row: dict[str, Any], task: Mapping[str, Any]) -> None:
    _row_from_task(row, dict(task))
    row["calibrator_uses_test_labels"] = bool(task.get("calibrator_uses_test_labels", False))
    if "uncalibrated_macro_f1" in task:
        row["uncalibrated_macro_f1"] = task.get("uncalibrated_macro_f1")
        row["uncalibrated_accuracy"] = task.get("uncalibrated_accuracy")
        row["uncalibrated_validation_macro_f1"] = task.get("uncalibrated_validation_macro_f1")
        row["uncalibrated_validation_accuracy"] = task.get("uncalibrated_validation_accuracy")


def _budget_fields(row: dict[str, Any], *, support_count: int, selected_count: int, graph_diag: Mapping[str, Any] | None = None) -> None:
    graph_diag = graph_diag or {}
    requested = _float(row.get("requested_support_ratio"))
    actual = float(selected_count / max(1, int(support_count)))
    row["selected_support_count"] = int(selected_count)
    row["requested_support_count"] = _requested_support_count(int(support_count), requested)
    row["actual_support_ratio"] = actual
    row["realized_support_ratio"] = actual
    row["effective_support_node_ratio"] = _float(graph_diag.get("effective_support_node_ratio", graph_diag.get("node_budget_ratio", actual)), actual)
    row["represented_support_context_ratio"] = _float(graph_diag.get("represented_support_context_ratio", graph_diag.get("represented_context_ratio", actual)), actual)
    row["requested_support_ratio_is_upper_bound"] = True


def _finalize_row(row: dict[str, Any], *, method: str, args: argparse.Namespace) -> None:
    row.setdefault("status", "success")
    row.setdefault("primary_eval_mode", str(args.primary_eval_mode))
    row.setdefault("selector_uses_test_labels", False)
    row.setdefault("teacher_uses_test_labels_for_training", False)
    row.setdefault("calibrator_uses_test_labels", False)
    row["no_test_leakage"] = not bool(row.get("selector_uses_test_labels") or row.get("teacher_uses_test_labels_for_training") or row.get("calibrator_uses_test_labels"))
    row["typedhash_included"] = bool(args.include_typedhash)
    row["diagnostic_only"] = bool(method in BASELINES or method in OLD_ABLATION_METHODS or method in {"full-graph-hettree-lite-tuned", "target-only-empty-support"})
    row["eligible_for_main_decision"] = bool(method in NEW_CANDIDATE_METHODS and method not in BASELINES)
    row["run_mode"] = "gate18r_accuracy_first_reset"
    row["primary_metric_priority"] = "accuracy_first_macro_guardrail"


def _prediction_audit(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    task: Mapping[str, Any],
    selected_labels: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_true = task.get("projected_test_labels", task.get("projected_test_true_labels"))
    y_pred = task.get("projected_test_pred", task.get("projected_test_pred_labels"))
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        return [], []
    return per_class_audit_rows(
        dataset=str(dataset),
        seed=int(seed),
        method=str(method),
        ratio=float(ratio),
        y_true=[int(value) for value in y_true],
        y_pred=[int(value) for value in y_pred],
        selected_support_labels=selected_labels,
    )


def _unit_inventory_rows(dataset: str, seed: int, ratio: float, units: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "requested_support_ratio": float(ratio),
                "unit_source": unit.source,
                "unit_id": unit.unit_id,
                "member_count": unit.member_count,
                "support_type_distribution": unit.support_type_distribution,
                "relation_profile": unit.relation_profile,
                "edge_mass": float(unit.edge_mass),
                "target_anchor_coverage": float(unit.target_anchor_coverage),
                "class_footprint": unit.class_footprint,
                "structure_available": bool(unit.metadata.get("unit_structure_available", False)),
                **{key: value for key, value in unit.metadata.items() if not isinstance(value, (dict, list, tuple))},
            }
        )
    return rows


def _unit_score_rows(dataset: str, seed: int, ratio: float, units: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "seed": int(seed),
            "requested_support_ratio": float(ratio),
            "unit_source": unit.source,
            "unit_id": unit.unit_id,
            "member_count": unit.member_count,
            "score": float(unit.metadata.get("score", 0.0)),
            "validation_accuracy_gain": float(unit.metadata.get("validation_accuracy_gain", 0.0)),
            "validation_macro_gain": float(unit.metadata.get("validation_macro_gain", 0.0)),
            "normalized_edge_mass": float(unit.metadata.get("normalized_edge_mass", 0.0)),
            "target_anchor_coverage": float(unit.target_anchor_coverage),
            "relation_channel_diversity": float(unit.metadata.get("relation_channel_diversity", 0.0)),
            "class_balance_score": float(unit.metadata.get("class_balance_score", 0.0)),
            "scoring_formula": unit.metadata.get("scoring_formula", ""),
        }
        for unit in units
    ]


def _unit_overlap_rows(dataset: str, seed: int, ratio: float, method_to_nodes: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    items = sorted((method, {int(node) for node in np.asarray(nodes, dtype=np.int64).tolist()}) for method, nodes in method_to_nodes.items())
    for i, (left_name, left) in enumerate(items):
        for right_name, right in items[i + 1 :]:
            union = left | right
            inter = left & right
            out.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "requested_support_ratio": float(ratio),
                    "left_method": left_name,
                    "right_method": right_name,
                    "left_count": int(len(left)),
                    "right_count": int(len(right)),
                    "intersection_count": int(len(inter)),
                    "jaccard": float(len(inter) / max(1, len(union))),
                }
            )
    return out


def _selected_set_cfg() -> SupportSelectorConfig:
    return SupportSelectorConfig(
        selector="teacher_topk",
        background_strategy="drop",
        allow_background_bucket=False,
        residual_prototype_mode="none",
        force_raw_bridge_nodes=False,
        force_raw_keep_high_degree_bridges=False,
        allow_proxy_fill=False,
    )


def _run_structured_unit_method(
    *,
    method: str,
    graph: HeteroGraph,
    target_type: int,
    selected_units: Sequence[Any],
    baseline_coarse: HeteroGraph,
    baseline_assignment: np.ndarray,
) -> tuple[HeteroGraph, np.ndarray, np.ndarray]:
    selected_ids = [int(unit.metadata.get("assignment_cluster_id")) for unit in selected_units if "assignment_cluster_id" in unit.metadata]
    if method == "HeSF-ClusterGate-H6-units":
        coarse, assignment, _keep = build_gated_h6_graph(
            original=graph,
            h6_coarse=baseline_coarse,
            h6_assignment=np.asarray(baseline_assignment, dtype=np.int64),
            selected_cluster_ids=selected_ids,
            target_type=int(target_type),
        )
        return coarse, assignment, selected_member_nodes(list(selected_units))
    assignment_arr = np.asarray(baseline_assignment, dtype=np.int64)
    target_nodes = np.flatnonzero(graph.node_type == int(target_type)).astype(np.int64)
    target_clusters = {int(assignment_arr[int(node)]) for node in target_nodes.tolist()}
    keep = np.asarray(sorted(target_clusters | set(selected_ids)), dtype=np.int64)
    coarse, mapped = induced_coarse_graph(baseline_coarse, assignment_arr, keep)
    return coarse, mapped, selected_member_nodes(list(selected_units))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate18R requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    unit_inventory_rows: list[dict[str, Any]] = []
    unit_score_rows: list[dict[str, Any]] = []
    selected_unit_rows: list[dict[str, Any]] = []
    unit_overlap_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1), dtype=np.int64)
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
            baseline_cache: dict[str, tuple[HeteroGraph, np.ndarray, dict[str, Any], dict[str, Any], np.ndarray]] = {}
            support_features: dict[str, Any] | None = None
            selection_result: dict[str, Any] | None = None
            selection_nodes = np.empty(0, dtype=np.int64)
            selection_graph_diag: dict[str, Any] = {}
            selection_task: dict[str, Any] | None = None
            unit_method_cache: dict[str, list[Any]] = {}
            unit_inventory_logged: set[str] = set()

            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **split_protocol}
                selected_nodes = np.empty(0, dtype=np.int64)
                task: dict[str, Any] | None = None
                graph_diag: dict[str, Any] = {}
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                        _budget_fields(row, support_count=support_count, selected_count=support_count)
                    elif method == "target-only-empty-support":
                        target_row, _coarse, _assignment = _target_only_row(graph, str(dataset), int(seed), float(ratio), args, split)
                        row.update(target_row)
                        _budget_fields(row, support_count=support_count, selected_count=0)
                    elif method in BASELINES:
                        if method not in baseline_cache:
                            coarse, assignment, diag = run_support_baseline(graph, baseline=str(method), ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                            assignment = np.asarray(assignment, dtype=np.int64)
                            task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=bool(args.return_logits))
                            selected = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                            baseline_cache[str(method)] = (coarse, assignment, diag, task, selected)
                        coarse, assignment, diag, task, selected_nodes = baseline_cache[str(method)]
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        _task_to_row(row, task)
                        _budget_fields(row, support_count=support_count, selected_count=int(diag.get("final_support_nodes", len(selected_nodes))), graph_diag=diag)
                    elif method in {
                        "HeSF-SS-validation-underfill-pareto",
                        "HeSF-SS-validation-underfill-pareto-logit-calibrated",
                        "HeSF-SS-validation-H6-fill-logit-calibrated",
                        "HeSF-SS-validation-H6-fill",
                        "HeSF-SS-random-fill-after-validation",
                        "HeSF-SS-validation-H6-fill-acc0.25",
                        "HeSF-SS-validation-H6-fill-acc0.50",
                        "HeSF-SS-validation-H6-fill-acc1.00",
                    }:
                        if selection_result is None:
                            selection_result = _gate17_6_run_selection_pipeline(
                                graph=graph,
                                labels=labels,
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                target_type=int(target_type),
                                ratio=float(ratio),
                                method="HeSF-SS-validation-only-neutral-fill",
                                seed=int(seed),
                                teacher=teacher,
                                args=args,
                            )
                            support_features = dict(selection_result.get("support_features", {}))
                            selection_nodes = np.asarray(selection_result["selection"]["selected_support_nodes"], dtype=np.int64)
                            selection_graph_diag = dict(selection_result.get("graph_diagnostics", {}))
                            selection_task = _eval_task(
                                graph,
                                selection_result["coarse_graph"],
                                np.asarray(selection_result["assignment"].assignment, dtype=np.int64),
                                seed=int(seed),
                                split=split,
                                target_type=int(target_type),
                                args=args,
                                return_logits=True,
                            )
                            for item in selection_result["selection"].get("validation_greedy_trials", []):
                                validation_rows.append({"dataset": str(dataset), "seed": int(seed), "method": "HeSF-SS-validation-underfill-pareto", "requested_support_ratio": float(ratio), **item})
                        assert selection_task is not None
                        h6_coarse, h6_assignment, _h6_diag, h6_task, h6_nodes = baseline_cache.get("H6-no-spec-support-only", (None, None, None, None, None))
                        if h6_coarse is None:
                            h6_coarse, h6_assignment, h6_diag = run_support_baseline(graph, baseline="H6-no-spec-support-only", ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                            h6_assignment = np.asarray(h6_assignment, dtype=np.int64)
                            h6_task = _eval_task(graph, h6_coarse, h6_assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                            h6_nodes = selected_support_representatives_from_assignment(graph, h6_assignment, int(target_type))
                            baseline_cache["H6-no-spec-support-only"] = (h6_coarse, h6_assignment, h6_diag, h6_task, h6_nodes)
                        if method in {"HeSF-SS-validation-underfill-pareto", "HeSF-SS-validation-underfill-pareto-logit-calibrated"}:
                            selected_nodes = selection_nodes
                            task = dict(selection_task)
                            graph_diag = dict(selection_graph_diag)
                        elif method == "HeSF-SS-random-fill-after-validation":
                            selected_nodes, fill_diag = _random_fill_support_nodes(graph=graph, target_type=int(target_type), selected_support_nodes=selection_nodes, requested_support_count=int(requested_count), seed=int(seed))
                            cfg = _gate17_6_selector_for_method(str(method), args)
                            coarse, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=support_features)
                            task = _eval_task(graph, coarse, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                            row.update(fill_diag)
                        else:
                            if method == "HeSF-SS-H6-fill-only":
                                base_nodes = np.empty(0, dtype=np.int64)
                            else:
                                base_nodes = selection_nodes
                            selected_nodes, fill_diag = h6_fill_support_nodes(graph=graph, h6_assignment=np.asarray(h6_assignment, dtype=np.int64), target_type=int(target_type), selected_support_nodes=base_nodes, requested_support_count=int(requested_count))
                            cfg = _gate17_6_selector_for_method("HeSF-SS-validation-H6-fill", args)
                            coarse, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, cfg, target_node_type=int(target_type), support_features=support_features)
                            task = _eval_task(graph, coarse, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                            row.update(fill_diag)
                            if method.startswith("HeSF-SS-validation-H6-fill-acc"):
                                row["legacy_accuracy_alpha"] = float(method.rsplit("acc", 1)[1])
                        if method in CALIBRATED_METHODS and task is not None:
                            task, calib = _apply_calibration(task, baseline_macro=_float(selection_task.get("validation_macro_f1")) if selection_task else None)
                            calibration_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **calib})
                        _task_to_row(row, task or {})
                        _budget_fields(row, support_count=support_count, selected_count=int(len(selected_nodes)), graph_diag=graph_diag)
                    elif method == "HeSF-SS-H6-fill-only":
                        h6_coarse, h6_assignment, _h6_diag, _h6_task, _h6_nodes = baseline_cache.get("H6-no-spec-support-only", (None, None, None, None, None))
                        if h6_assignment is None:
                            h6_coarse, h6_assignment, h6_diag = run_support_baseline(graph, baseline="H6-no-spec-support-only", ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                            h6_assignment = np.asarray(h6_assignment, dtype=np.int64)
                            h6_task = _eval_task(graph, h6_coarse, h6_assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                            h6_nodes = selected_support_representatives_from_assignment(graph, h6_assignment, int(target_type))
                            baseline_cache["H6-no-spec-support-only"] = (h6_coarse, h6_assignment, h6_diag, h6_task, h6_nodes)
                        selected_nodes, fill_diag = h6_fill_support_nodes(graph=graph, h6_assignment=np.asarray(h6_assignment, dtype=np.int64), target_type=int(target_type), selected_support_nodes=np.empty(0, dtype=np.int64), requested_support_count=int(requested_count))
                        coarse, assignment_obj, graph_diag = build_selected_support_graph(graph, selected_nodes, _selected_set_cfg(), target_node_type=int(target_type), support_features=support_features)
                        task = _eval_task(graph, coarse, assignment_obj.assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                        row.update(fill_diag)
                        _task_to_row(row, task)
                        _budget_fields(row, support_count=support_count, selected_count=int(len(selected_nodes)), graph_diag=graph_diag)
                    elif method.startswith("HeSF-ClusterGate-"):
                        if support_features is None:
                            support_features = _fast_support_features(graph, labels, train_mask, int(target_type))
                        if not unit_method_cache:
                            for baseline in ("H6-no-spec-support-only", "TypedHash-ChebHeat-support-only", "flatten-sum-support-only"):
                                if baseline not in baseline_cache:
                                    coarse, assignment, diag = run_support_baseline(graph, baseline=baseline, ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                                    assignment = np.asarray(assignment, dtype=np.int64)
                                    btask = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                                    selected = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                                    baseline_cache[baseline] = (coarse, assignment, diag, btask, selected)
                            h6_units = extract_h6_units(graph, baseline_cache["H6-no-spec-support-only"][1], int(target_type), labels=labels, splits=split)
                            typed_units = extract_typedhash_units(graph, baseline_cache["TypedHash-ChebHeat-support-only"][1], int(target_type), labels=labels, splits=split)
                            flat_units = extract_flatten_units(graph, baseline_cache["flatten-sum-support-only"][1], int(target_type), labels=labels, splits=split)
                            block_units = extract_validation_block_units(graph, support_features, "class_anchor_relation", int(target_type), labels=labels, splits=split)
                            unit_method_cache["HeSF-ClusterGate-H6-units"] = score_units(h6_units)
                            unit_method_cache["HeSF-ClusterGate-TypedHash-units"] = score_units(typed_units)
                            unit_method_cache["HeSF-ClusterGate-Flatten-units"] = score_units(flat_units)
                            union_scored = score_units(make_union_units(h6_units, typed_units, flat_units, block_units, deduplicate=True))
                            unit_method_cache["HeSF-ClusterGate-UnionUnits"] = union_scored
                            unit_method_cache["HeSF-ClusterGate-UnionUnits-logit-calibrated"] = union_scored
                        source_units = unit_method_cache[str(method)]
                        if str(method) not in unit_inventory_logged:
                            unit_inventory_rows.extend(_unit_inventory_rows(str(dataset), int(seed), float(ratio), source_units))
                            unit_score_rows.extend(_unit_score_rows(str(dataset), int(seed), float(ratio), source_units))
                            unit_inventory_logged.add(str(method))
                        selected_units = select_units_under_budget(source_units, support_count=int(support_count), requested_support_ratio=float(ratio), allow_underfill=True)
                        selected_nodes = selected_member_nodes(selected_units)
                        for unit in selected_units:
                            selected_unit_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), "unit_source": unit.source, "unit_id": unit.unit_id, "member_count": unit.member_count, "score": float(unit.metadata.get("score", 0.0))})
                        if method == "HeSF-ClusterGate-H6-units":
                            coarse, assignment, selected_nodes = _run_structured_unit_method(method=str(method), graph=graph, target_type=int(target_type), selected_units=selected_units, baseline_coarse=baseline_cache["H6-no-spec-support-only"][0], baseline_assignment=baseline_cache["H6-no-spec-support-only"][1])
                            graph_diag = {"selected_unit_count": len(selected_units)}
                        elif method == "HeSF-ClusterGate-TypedHash-units":
                            coarse, assignment, selected_nodes = _run_structured_unit_method(method=str(method), graph=graph, target_type=int(target_type), selected_units=selected_units, baseline_coarse=baseline_cache["TypedHash-ChebHeat-support-only"][0], baseline_assignment=baseline_cache["TypedHash-ChebHeat-support-only"][1])
                            graph_diag = {"selected_unit_count": len(selected_units)}
                        elif method == "HeSF-ClusterGate-Flatten-units":
                            coarse, assignment, selected_nodes = _run_structured_unit_method(method=str(method), graph=graph, target_type=int(target_type), selected_units=selected_units, baseline_coarse=baseline_cache["flatten-sum-support-only"][0], baseline_assignment=baseline_cache["flatten-sum-support-only"][1])
                            graph_diag = {"selected_unit_count": len(selected_units)}
                        else:
                            coarse, assignment, graph_diag = build_graph_from_units(graph, list(selected_units), target_type=int(target_type), selector_config=_selected_set_cfg(), support_features=support_features)
                        task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                        if method in CALIBRATED_METHODS:
                            task, calib = _apply_calibration(task)
                            calibration_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **calib})
                        _task_to_row(row, task)
                        _budget_fields(row, support_count=support_count, selected_count=int(len(selected_nodes)), graph_diag=graph_diag)
                        unit_overlap_rows.extend(_unit_overlap_rows(str(dataset), int(seed), float(ratio), {str(method): selected_nodes, "H6": baseline_cache["H6-no-spec-support-only"][4], "TypedHash": baseline_cache["TypedHash-ChebHeat-support-only"][4], "flatten": baseline_cache["flatten-sum-support-only"][4]}))
                    elif method in STC_METHODS:
                        task = evaluate_feature_condensation_method(
                            graph,
                            method=str(method),
                            requested_ratio=float(ratio),
                            target_type=int(target_type),
                            labels=labels,
                            split=split,
                            seed=int(seed),
                            epochs=int(args.task_epochs),
                            max_paths=int(args.max_paths),
                            hidden_dim=int(args.task_hidden_dim),
                            device=str(args.device),
                        )
                        if method in CALIBRATED_METHODS:
                            task, calib = _apply_calibration(task)
                            calibration_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **calib})
                        _task_to_row(row, task)
                        row.update({key: value for key, value in task.items() if key in {"feature_cache_size_ratio", "path_channel_count_ratio", "semantic_tree_l2_delta_vs_full", "teacher_kl_vs_full", "full_teacher_logit_agreement", "compression_axis", "compression_ratio", "feature_model", "feature_model_skipped"}})
                        row["actual_support_ratio"] = 0.0
                        row["effective_support_node_ratio"] = 0.0
                        row["represented_support_context_ratio"] = 0.0
                        row["selected_support_count"] = 0
                        row["requested_support_count"] = requested_count
                        row["requested_support_ratio_is_upper_bound"] = True
                        feature_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **{key: row.get(key, "") for key in ["feature_cache_size_ratio", "path_channel_count_ratio", "semantic_tree_l2_delta_vs_full", "teacher_kl_vs_full", "full_teacher_logit_agreement", "macro_f1", "accuracy", "validation_macro_f1", "validation_accuracy", "compression_axis", "compression_ratio"]}})
                    else:
                        row["status"] = "skipped"
                        row["skip_reason"] = "method_not_enabled_for_gate18r"
                    if task is not None:
                        pc, cm = _prediction_audit(dataset=str(dataset), seed=int(seed), method=str(method), ratio=float(ratio), task=task, selected_labels=_support_labels(labels, selected_nodes))
                        per_class_rows.extend(pc)
                        confusion_rows.extend(cm)
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                row["wall_clock_sec"] = float(perf_counter() - start)
                _finalize_row(row, method=str(method), args=args)
                rows.append(row)
                write_csv(output_dir / "gate18r_raw_rows.csv", rows)

    write_csv(output_dir / "gate18r_raw_rows.csv", rows)
    write_csv(output_dir / "gate18r_calibration.csv", calibration_rows)
    write_csv(output_dir / "gate18r_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "gate18r_confusion_matrix_by_method.csv", confusion_rows)
    write_csv(output_dir / "gate18r_unit_inventory.csv", unit_inventory_rows)
    write_csv(output_dir / "gate18r_unit_scores.csv", unit_score_rows)
    write_csv(output_dir / "gate18r_selected_units.csv", selected_unit_rows)
    write_csv(output_dir / "gate18r_unit_overlap.csv", unit_overlap_rows)
    write_csv(output_dir / "gate18r_feature_condensation.csv", feature_rows)
    write_csv(output_dir / "gate18r_validation_trials.csv", validation_rows)
    result = summarize(output_dir, output_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate18R accuracy-first reset: feature/cluster distillation under budget.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate18r"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.30, 0.50, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--task-epochs", type=int, default=50)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--feature-mode", default="full")
    parser.add_argument("--include-typedhash", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--return-logits", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
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
    parser.add_argument("--alpha-accuracy-grid", nargs="*", default=[0.0, 0.25, 0.50, 1.00])
    parser.add_argument("--delta-micro", type=float, default=0.25)
    parser.add_argument("--beta-underfill", type=float, default=0.10)
    parser.add_argument("--gamma-class-collapse", type=float, default=0.05)
    parser.add_argument("--min-validation-gain", type=float, default=1.0e-4)
    parser.add_argument("--h6-fill-weight", type=float, default=0.25)
    parser.add_argument("--cluster-gating-candidate-pool-size", type=int, default=16)
    parser.add_argument("--cluster-gating-feedback-epochs", type=int, default=2)
    parser.add_argument("--cluster-gating-min-gain", type=float, default=1.0e-4)
    parser.add_argument("--budget-penalty-lambda", type=float, default=0.05)
    parser.add_argument("--underfill-penalty-lambda", type=float, default=0.10)
    parser.add_argument("--neutral-fill-max-drop", type=float, default=1.0e-4)
    parser.add_argument("--negative-fill-max-drop", type=float, default=5.0e-4)
    parser.add_argument("--lambda-edge", type=float, default=0.10)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--lambda-relation", type=float, default=0.10)
    parser.add_argument("--lambda-class", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = {str(dataset).upper() for dataset in args.datasets}
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if dataset in allowed]
    args.ratios = _split_values(args.support_ratios, float) or [0.30, 0.50, 0.70]
    if not _bool_arg(args.include_typedhash):
        args.methods = [method for method in args.methods if method not in {"TypedHash-ChebHeat-support-only", "HeSF-ClusterGate-TypedHash-units"}]
    elif "TypedHash-ChebHeat-support-only" not in args.methods:
        args.methods = [*args.methods, "TypedHash-ChebHeat-support-only"]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
