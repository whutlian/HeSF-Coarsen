from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from experiments.scripts.run_gate17_1_support_sensitivity import (
    _full_graph_row,
    _semantic_row_for_graph,
    _target_only_empty_support_graph,
    _target_only_row,
)
from experiments.scripts.run_gate17_support_selection import _flat_payload, _mask, _metric, _row_from_task, _split_values
from experiments.scripts.summarize_gate17_3 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


GATE17_3_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only", "random-support-only"}
MAIN_CANDIDATE_METHODS = (
    "HeSF-SS-sensitivity-selection-only",
    "HeSF-SS-real-occlusion-selection-only",
    "HeSF-SS-real-validation-neutral-fill",
    "HeSF-SS-real-occlusion-neutral-fill",
    "HeSF-SS-real-occlusion-lossy-prototype",
    "HeSF-SS-H6-seeded-occlusion",
    "HeSF-SS-H6-seeded-lossy-prototype",
)
DIAGNOSTIC_ONLY_METHODS = (
    "HeSF-SS-real-validation-no-fallback",
    "HeSF-SS-full-residual-prototype-upperbound",
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


def parse_dataset_seeds(values: list[str] | tuple[str, ...] | str | None) -> list[tuple[str, int]]:
    if values is None or values == "":
        return [(dataset, seed) for dataset, seed in GATE17_3_SINGLE_SEED_BY_DATASET.items()]
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
        if dataset not in GATE17_3_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate17.3 dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


def _selector_for_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    common = {
        "candidate_pool_size": int(args.candidate_pool_size),
        "short_eval_epochs": int(args.short_eval_epochs),
        "max_validation_greedy_steps": int(args.max_validation_greedy_steps),
        "occlusion_candidate_pool_size": int(args.occlusion_candidate_pool_size),
        "occlusion_short_eval_epochs": int(args.occlusion_short_eval_epochs),
        "occlusion_short_patience": int(args.occlusion_short_patience),
        "max_members_per_prototype": int(args.max_members_per_prototype),
        "force_raw_bridge_nodes": False,
        "force_raw_keep_high_degree_bridges": False,
        "allow_proxy_fill": False,
        "min_occlusion_importance": 1.0e-6,
        "allow_negative_occlusion_fill": False,
        "prototype_budget_fraction": float(args.prototype_budget_fraction),
        "max_represented_support_ratio_slack": float(args.max_represented_support_ratio_slack),
        "meta_path_channel_source": "relation_sequence_hash",
    }
    if method == "HeSF-SS-sensitivity-selection-only":
        return SupportSelectorConfig(
            selector="sensitivity_block_selector",
            background_strategy="drop",
            allow_background_bucket=False,
            residual_prototype_mode="none",
            **common,
        )
    if method == "HeSF-SS-real-occlusion-selection-only":
        return SupportSelectorConfig(
            selector="real_occlusion_block_selector",
            background_strategy="drop",
            allow_background_bucket=False,
            residual_prototype_mode="none",
            **common,
        )
    if method == "HeSF-SS-real-validation-no-fallback":
        return SupportSelectorConfig(
            selector="real_validation_block_greedy",
            background_strategy="drop",
            allow_background_bucket=False,
            residual_prototype_mode="none",
            min_gain=1.0e-4,
            **common,
        )
    if method == "HeSF-SS-real-validation-neutral-fill":
        return SupportSelectorConfig(
            selector="real_validation_block_greedy",
            background_strategy="drop",
            allow_background_bucket=False,
            residual_prototype_mode="none",
            min_gain=1.0e-4,
            neutral_fill=True,
            neutral_fill_max_drop=1.0e-4,
            **common,
        )
    if method == "HeSF-SS-real-occlusion-neutral-fill":
        return SupportSelectorConfig(
            selector="real_occlusion_block_selector",
            background_strategy="drop",
            allow_background_bucket=False,
            residual_prototype_mode="none",
            neutral_fill=True,
            **common,
        )
    if method == "HeSF-SS-real-occlusion-lossy-prototype":
        return SupportSelectorConfig(
            selector="real_occlusion_block_selector",
            background_strategy="dblp_aware_prototype",
            block_key_mode="dblp_aware",
            residual_prototype_mode="lossy_topk",
            neutral_fill=True,
            **common,
        )
    if method == "HeSF-SS-full-residual-prototype-upperbound":
        return SupportSelectorConfig(
            selector="sensitivity_block_selector",
            background_strategy="dblp_aware_prototype",
            block_key_mode="dblp_aware",
            residual_prototype_mode="full_upperbound",
            allow_proxy_fill=True,
            **{key: value for key, value in common.items() if key != "allow_proxy_fill"},
        )
    raise ValueError(f"unsupported Gate17.3 selector method: {method}")


def _support_representatives(
    graph,
    assignment: np.ndarray,
    target_type: int,
    max_nodes: int | None = None,
) -> np.ndarray:
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    groups: dict[int, list[int]] = {}
    for node in support_nodes:
        groups.setdefault(int(assignment[int(node)]), []).append(int(node))
    ordered_groups = sorted(
        (sorted(int(node) for node in nodes) for nodes in groups.values() if nodes),
        key=lambda nodes: (-len(nodes), int(graph.node_type[int(nodes[0])]), int(nodes[0])),
    )
    reps = [int(nodes[0]) for nodes in ordered_groups]
    if max_nodes is not None:
        reps = reps[: max(0, int(max_nodes))]
    return np.asarray(sorted(reps), dtype=np.int64)


def _h6_seed_nodes_for_method(
    h6_nodes: np.ndarray,
    *,
    support_count: int,
    ratio: float,
    method: str,
    prototype_budget_fraction: float,
) -> np.ndarray:
    requested = max(0, int(np.ceil(int(support_count) * float(ratio) - 1.0e-12)))
    raw_budget = requested
    if str(method).endswith("lossy-prototype") and requested > 0:
        prototype_reserve = max(1, int(np.ceil(requested * float(prototype_budget_fraction) - 1.0e-12)))
        prototype_reserve = min(prototype_reserve, max(1, int(np.floor(0.01 * int(support_count)))))
        raw_budget = max(0, requested - prototype_reserve)
    return np.asarray(h6_nodes, dtype=np.int64).reshape(-1)[:raw_budget].astype(np.int64, copy=False)


def _h6_export_fields(graph, h6_nodes: np.ndarray) -> dict[str, Any]:
    nodes = [int(node) for node in np.asarray(h6_nodes, dtype=np.int64).reshape(-1)]
    by_type = Counter(str(int(graph.node_type[node])) for node in nodes)
    labels = getattr(graph, "labels", None)
    if labels is None:
        by_class = Counter({"unlabeled": len(nodes)})
    else:
        by_class = Counter(str(int(labels[node])) if int(labels[node]) >= 0 else "unlabeled" for node in nodes)
    return {
        "H6_selected_support_count": int(len(nodes)),
        "H6_selected_support_nodes": json.dumps(nodes, separators=(",", ":")),
        "H6_selected_blocks": json.dumps(nodes, separators=(",", ":")),
        "H6_support_by_type": json.dumps(dict(sorted(by_type.items())), sort_keys=True),
        "H6_support_by_relation_channel": json.dumps(
            {"unavailable_without_support_features": int(len(nodes))},
            sort_keys=True,
        ),
        "H6_support_by_class_bucket": json.dumps(dict(sorted(by_class.items())), sort_keys=True),
    }


def _overlap_fields(selected_nodes: np.ndarray, h6_nodes: np.ndarray) -> dict[str, Any]:
    selected = {int(node) for node in np.asarray(selected_nodes, dtype=np.int64).reshape(-1)}
    h6 = {int(node) for node in np.asarray(h6_nodes, dtype=np.int64).reshape(-1)}
    inter = selected & h6
    union = selected | h6
    return {
        "selected_overlap_with_H6": int(len(inter)),
        "selected_jaccard_with_H6": float(len(inter) / max(len(union), 1)),
        "selected_recall_of_H6": float(len(inter) / max(len(h6), 1)),
        "selected_precision_vs_H6": float(len(inter) / max(len(selected), 1)),
        "occlusion_topk_overlap_with_H6": int(len(inter)),
        "validation_blocks_overlap_with_H6": int(len(inter)),
        "prototype_blocks_overlap_with_H6": int(len(inter)),
    }


def _budget_update(row: dict[str, Any], support_count: int, graph_diag: dict[str, Any]) -> None:
    fields = compute_gate17_3_budget_fields(
        original_support_nodes=int(support_count),
        requested_support_ratio=float(row.get("requested_support_ratio", 0.0) or 0.0),
        selected_raw_support_count=int(graph_diag.get("selected_raw_support_count", row.get("selected_support_count", 0)) or 0),
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


def _fixed_support_row(
    graph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    split: dict[str, np.ndarray],
    *,
    target_type: int,
    selected_nodes: np.ndarray,
    cfg: Gate15Config,
    method: str,
    ratio: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    coarse, assignment, graph_diag = build_selected_support_graph(
        graph,
        np.asarray(selected_nodes, dtype=np.int64),
        cfg.selector,
        target_node_type=int(target_type),
        support_features=None,
    )
    task = evaluate_hettree_task(
        graph,
        coarse,
        assignment.assignment,
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
    row: dict[str, Any] = {
        "method": str(method),
        "requested_support_ratio": float(ratio),
        "selected_support_count": int(len(selected_nodes)),
        "realized_support_count": int(len(selected_nodes)),
        "realized_support_ratio": float(len(selected_nodes) / max(int(np.sum(graph.node_type != int(target_type))), 1)),
        "support_budget_exact_match": False,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "selection_split_source": "train_val_only",
        **{key: value for key, value in graph_diag.items() if not isinstance(value, (dict, list))},
    }
    row.update(
        budget_diagnostics(
            num_support=int(np.sum(graph.node_type != int(target_type))),
            support_ratio=float(ratio),
            realized_support_count=len(selected_nodes),
        )
    )
    _row_from_task(row, task)
    return row | {"coarse_graph": coarse, "assignment": assignment, "graph_diagnostics": graph_diag, "selected_support_nodes": np.asarray(selected_nodes, dtype=np.int64)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    diag_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    candidate_semantic_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    h6_overlap_rows: list[dict[str, Any]] = []
    meta_path_rows: list[dict[str, Any]] = []
    h6_cache: dict[tuple[str, int, float], np.ndarray] = {}
    h6_export_cache: dict[tuple[str, int, float], dict[str, Any]] = {}
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
            h6_key = (str(dataset), int(seed), float(ratio))
            if h6_key not in h6_cache:
                _h6_coarse, h6_assignment, _h6_diag = run_support_baseline(graph, baseline="H6-no-spec-support-only", ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                desired_h6 = max(0, int(np.ceil(int(support_count) * float(ratio) - 1.0e-12)))
                h6_cache[h6_key] = _support_representatives(
                    graph,
                    np.asarray(h6_assignment, dtype=np.int64),
                    int(target_type),
                    max_nodes=desired_h6,
                )
                h6_export_cache[h6_key] = _h6_export_fields(graph, h6_cache[h6_key])
            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **split_protocol}
                coarse_for_delta = None
                assignment_for_delta = None
                graph_diag: dict[str, Any] = {"prototype_background_count": 0, "prototype_member_count_sum": 0, "selected_raw_support_count": 0}
                selected_nodes = np.empty(0, dtype=np.int64)
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                        coarse_for_delta = graph
                        assignment_for_delta = np.arange(graph.num_nodes, dtype=np.int64)
                        graph_diag["selected_raw_support_count"] = support_count
                    elif method == "target-only-empty-support":
                        target_row, coarse_for_delta, assignment_for_delta = _target_only_row(graph, str(dataset), int(seed), float(ratio), args, split)
                        row.update(target_row)
                    elif method in BASELINES:
                        coarse, assignment, diag = run_support_baseline(graph, baseline=str(method), ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                        coarse_for_delta = coarse
                        assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        final_support = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                        row["selected_support_count"] = int(final_support)
                        row.update(
                            budget_diagnostics(
                                num_support=support_count,
                                support_ratio=float(ratio),
                                realized_support_count=final_support,
                            )
                        )
                        task = evaluate_hettree_task(graph, coarse, assignment_for_delta, seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device), target_node_type=int(target_type), official_split_nodes=split, primary_eval_mode=str(args.primary_eval_mode), early_stopping=True, monitor=str(args.monitor), max_paths=int(args.max_paths)).metrics
                        _row_from_task(row, task)
                        if method == "H6-no-spec-support-only":
                            selected_nodes = h6_cache[h6_key]
                            row.update(h6_export_cache[h6_key])
                        graph_diag["selected_raw_support_count"] = final_support
                    elif method in {"HeSF-SS-H6-seeded-occlusion", "HeSF-SS-H6-seeded-lossy-prototype"}:
                        h6_seed_nodes = _h6_seed_nodes_for_method(
                            h6_cache[h6_key],
                            support_count=support_count,
                            ratio=float(ratio),
                            method=str(method),
                            prototype_budget_fraction=float(args.prototype_budget_fraction),
                        )
                        cfg = Gate15Config(
                            target_node_type=int(target_type),
                            selector=replace(
                                SupportSelectorConfig(
                                    selector="real_occlusion_block_selector",
                                    background_strategy="drop" if method.endswith("occlusion") else "dblp_aware_prototype",
                                    allow_background_bucket=method.endswith("lossy-prototype"),
                                    residual_prototype_mode="none" if method.endswith("occlusion") else "lossy_topk",
                                    support_ratios=(float(ratio),),
                                    prototype_budget_fraction=float(args.prototype_budget_fraction),
                                    max_represented_support_ratio_slack=float(args.max_represented_support_ratio_slack),
                                    max_members_per_prototype=int(args.max_members_per_prototype),
                                    allow_proxy_fill=False,
                                    neutral_fill=True,
                                ),
                                support_ratios=(float(ratio),),
                            ),
                        )
                        fixed = _fixed_support_row(graph, labels, train_mask, val_mask, test_mask, split, target_type=int(target_type), selected_nodes=h6_seed_nodes, cfg=cfg, method=str(method), ratio=float(ratio), seed=int(seed), args=args)
                        row.update(_flat_payload(fixed))
                        coarse_for_delta = fixed["coarse_graph"]
                        assignment_for_delta = np.asarray(fixed["assignment"].assignment, dtype=np.int64)
                        graph_diag = dict(fixed["graph_diagnostics"])
                        selected_nodes = np.asarray(fixed["selected_support_nodes"], dtype=np.int64)
                        row["occlusion_task_signal_pass"] = True
                    else:
                        cfg = Gate15Config(target_node_type=int(target_type), selector=replace(_selector_for_method(str(method), args), support_ratios=(float(ratio),)))
                        result = run_supervised_support_selection_pipeline(graph, labels, train_mask, val_mask, test_mask, cfg, support_ratio=float(ratio), teacher_outputs=teacher, method_name=str(method), seed=int(seed), task_epochs=int(args.task_epochs), task_hidden_dim=int(args.task_hidden_dim), task_max_paths=int(args.max_paths), device=str(args.device))
                        row.update(_flat_payload(result))
                        coarse_for_delta = result["coarse_graph"]
                        assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                        graph_diag = dict(result["graph_diagnostics"])
                        selected_nodes = np.asarray(result["selection"]["selected_support_nodes"], dtype=np.int64)
                        selection_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **result["selection"]["diagnostics"]})
                        for item in result["selection"].get("validation_greedy_trials", []):
                            validation_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item})
                        for item in result["selection"].get("occlusion_block_scores", []):
                            occlusion_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **item})
                    if coarse_for_delta is not None and assignment_for_delta is not None:
                        semantic = _semantic_row_for_graph(graph=graph, coarse=coarse_for_delta, assignment=assignment_for_delta, target_only=target_only_graph, target_only_assignment=target_only_assignment, target_type=int(target_type), dataset=str(dataset), seed=int(seed), method=str(method), ratio=float(ratio), args=args)
                        semantic_rows.append(semantic)
                        row.update({key: semantic.get(key) for key in ["tree_tensor_l2_delta_vs_full", "target_path_feature_changed_fraction", "allclose_to_full"]})
                        if str(method).startswith("HeSF-SS"):
                            row["candidate_allclose_to_full"] = bool(semantic.get("allclose_to_full", False))
                            candidate_semantic_rows.append({**semantic, "candidate_allclose_to_full": row["candidate_allclose_to_full"]})
                    row.update(_overlap_fields(selected_nodes, h6_cache[h6_key]))
                    _budget_update(row, support_count, graph_diag)
                    row["status"] = row.get("status", "success")
                    row.setdefault("selector_uses_test_labels", False)
                    row.setdefault("teacher_uses_test_labels_for_training", False)
                    row.setdefault("prototype_saturation_rate", graph_diag.get("prototype_saturation_rate", 0.0))
                    row.setdefault("meta_path_channel_source", graph_diag.get("meta_path_channel_source", ""))
                    prototype_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **{key: value for key, value in graph_diag.items() if not isinstance(value, (dict, list))}})
                    graph_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **{key: value for key, value in graph_diag.items() if not isinstance(value, (dict, list))}})
                    h6_payload = {"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), **_overlap_fields(selected_nodes, h6_cache[h6_key])}
                    if method == "H6-no-spec-support-only":
                        h6_payload.update(h6_export_cache[h6_key])
                    h6_overlap_rows.append(h6_payload)
                    meta_path_rows.append({"dataset": str(dataset), "seed": int(seed), "method": str(method), "requested_support_ratio": float(ratio), "meta_path_channel_source": row.get("meta_path_channel_source", ""), "prototype_key_mode": row.get("prototype_key_mode", "")})
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                row["wall_clock_sec"] = float(perf_counter() - start)
                row["run_mode"] = "gate17_3_single_seed_lossy_feedback"
                rows.append(row)
                write_csv(output_dir / "gate17_3_raw_rows.csv", rows)
    budget_rows = [{key: row.get(key, "") for key in row.keys() if key in {"dataset", "seed", "method", "requested_support_ratio"} or "budget" in key or "represented_context" in key or key in {"eligible_for_main_decision", "full_residual_upperbound"}} for row in rows]
    acm_rows = [row for row in rows if str(row.get("dataset", "")).upper() == "ACM"]
    write_csv(diag_dir / "gate17_3_budget_breakdown.csv", budget_rows)
    write_csv(diag_dir / "gate17_3_represented_context_budget.csv", budget_rows)
    write_csv(diag_dir / "gate17_3_candidate_semantic_delta.csv", candidate_semantic_rows)
    write_csv(diag_dir / "gate17_3_acm_saturation_curve.csv", acm_rows)
    write_csv(diag_dir / "gate17_3_validation_trials.csv", validation_rows)
    write_csv(diag_dir / "gate17_3_occlusion_scores.csv", occlusion_rows)
    write_csv(diag_dir / "gate17_3_prototype_saturation.csv", prototype_rows)
    write_csv(diag_dir / "gate17_3_h6_overlap.csv", h6_overlap_rows)
    write_csv(diag_dir / "gate17_3_meta_path_channel_audit.csv", meta_path_rows)
    write_csv(diag_dir / "gate17_3_compressed_graph_summary.csv", graph_rows)
    write_csv(diag_dir / "gate17_3_support_selection_diagnostics.csv", selection_rows)
    result = summarize(output_dir, output_dir)
    if any(row.get("status") == "oom_or_runtime_error" for row in rows):
        result["local_oom_or_runtime_error"] = True
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.3 single-seed lossy prototype feedback gate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate17_3_single_seed"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.03, 0.10, 0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-patience", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=16)
    parser.add_argument("--occlusion-candidate-pool-size", type=int, default=16)
    parser.add_argument("--max-validation-greedy-steps", type=int, default=5)
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--feature-mode", choices=["full"], default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-members-per-prototype", type=int, default=512)
    parser.add_argument("--prototype-budget-fraction", type=float, default=0.10)
    parser.add_argument("--max-represented-support-ratio-slack", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    args.ratios = _split_values(args.support_ratios, float) or [0.03, 0.10, 0.30, 0.70]
    args.methods = _split_values(args.methods, str) or list(DEFAULT_METHODS)
    result = run(args)
    return 3 if bool(result.get("local_oom_or_runtime_error", False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
