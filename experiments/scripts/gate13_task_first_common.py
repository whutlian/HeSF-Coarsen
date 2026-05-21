from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, markdown_table, write_csv, write_json
from experiments.scripts.run_task_first_gate12_hgb import _candidate_config, _compose, _flatten, _rss_mb, _train_mask_for_current
from hesf_coarsen.baselines.type_isolated_lsh import _node_embedding, _signatures
from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.candidates.bucket import generate_bucket_candidates_chunked
from hesf_coarsen.candidates.capped_twohop import generate_capped_twohop_candidates_chunked
from hesf_coarsen.candidates.onehop import generate_onehop_candidates_chunked
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.edge_list import load_graph
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.matching.greedy import run_greedy_cluster_matching
from hesf_coarsen.partition.type_partition import default_partition
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch
from hesf_coarsen.sketch.simhash import compute_simhash_buckets
from hesf_coarsen.task_first.candidates import (
    build_hybrid_task_aware_candidates,
    build_class_footprint_knn_candidates,
    build_relation_response_knn_candidates,
    build_target_anchor_co_support_candidates,
    build_target_response_knn_candidates,
    build_target_response_signature_knn_candidates,
)
from hesf_coarsen.task_first.config import (
    SupportCoverageConfig,
    SupportPurityConfig,
    TaskFirstConfig,
    TaskFirstScoringConfig,
)
from hesf_coarsen.task_first.pipeline import (
    build_support_only_task_first_coarsening,
    build_target_preserve_assignment_template,
)
from hesf_coarsen.task_first.state import build_task_first_state


DATASETS = {"ACM": "acm_hesf", "DBLP": "dblp_hesf", "IMDB": "imdb_hesf"}
DEFAULT_SEEDS = (12345, 23456, 34567, 45678, 56789)
PRIMARY_RATIOS = (0.048, 0.096)
DIAGNOSTIC_RATIOS = (0.20, 0.50)


def ratio_token(ratio: float) -> str:
    return f"{float(ratio):.4f}".replace(".", "p").rstrip("0").rstrip("p")


def method_token(method: str) -> str:
    return method.lower().replace("-", "_").replace(" ", "_")


def load_hgb_graph(data_root: Path, dataset: str) -> HeteroGraph:
    return load_graph(Path(data_root) / DATASETS[str(dataset).upper()])


def task_first_config(
    method: str,
    *,
    target_type: int,
    ratio: float | None = None,
    ratio_mode: str = "support",
    max_support_merges: int | None = None,
    pair_delta_mode: str = "response_signature",
    coverage_mode: str = "combined",
    purity_policy: str = "unknown_blocks_known",
    js_threshold: float = 0.35,
    lambda_rel_response: float | None = None,
) -> TaskFirstConfig:
    method_name = str(method)
    base_method = method_name.replace("-response", "").replace("-static", "")
    scoring = TaskFirstScoringConfig(pair_delta_mode=pair_delta_mode)
    if base_method.startswith("HeSF-TC-S"):
        scoring = replace(
            scoring,
            lambda_target_spec=2.0,
            lambda_rel_response=1.0,
            lambda_support_coverage=1.0,
            lambda_support_purity=1.0,
            lambda_feat=0.1,
        )
    elif "no-rel" in base_method:
        scoring = replace(scoring, lambda_rel_response=0.0)
    elif "no-target-spec" in base_method:
        scoring = replace(scoring, lambda_target_spec=0.0)
    elif "no-coverage" in base_method:
        scoring = replace(scoring, lambda_support_coverage=0.0)
    elif "no-purity" in base_method:
        scoring = replace(scoring, lambda_support_purity=0.0)
    elif not base_method.startswith("HeSF-TC-P") and base_method not in {
        "HeSF-TC-coverage-v2",
        "HeSF-TC-purity-v2",
        "HeSF-TC-coverage-v2-purity-v2",
        "HeSF-TC-stateful-v1",
        "HeSF-TC-stateful-v1-coverage-v2",
        "HeSF-TC-stateful-v1-purity-v2",
        "HeSF-TC-stateful-v1-coverage-v2-purity-v2",
    }:
        raise ValueError(f"unsupported HeSF-TC method: {method}")
    if lambda_rel_response is not None:
        scoring = replace(scoring, lambda_rel_response=float(lambda_rel_response))
    kwargs: dict[str, Any] = {}
    if max_support_merges is not None:
        kwargs["max_support_merges"] = int(max_support_merges)
    elif ratio is not None and ratio_mode == "full":
        kwargs["target_ratio"] = float(ratio)
    elif ratio is not None:
        kwargs["support_ratio"] = float(ratio)
    return TaskFirstConfig(
        target_node_type=int(target_type),
        support_coverage=SupportCoverageConfig(mode=coverage_mode),
        support_purity=SupportPurityConfig(
            zero_policy="purity_v2" if "purity-v2" in base_method else purity_policy,
            support_footprint_mode="hybrid_propagated" if "purity-v2" in base_method else "onehop_train",
            js_merge_block_threshold=float(js_threshold),
        ),
        scoring=scoring,
        **kwargs,
    )


def build_random_support_candidates(
    graph: HeteroGraph,
    *,
    target_type: int,
    seed: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    store = ArrayCandidateStore(graph.node_type, K=int(candidate_k), same_type_only=True)
    start = perf_counter()
    emitted = 0
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        if type_id == int(target_type):
            continue
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if len(nodes) < 2:
            continue
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        span = min(max(1, int(candidate_k)), max(1, len(shuffled) - 1))
        for offset in range(1, span + 1):
            left = shuffled[:-offset]
            right = shuffled[offset:]
            scores = rng.random(len(left), dtype=np.float32)
            store.add_many(left, right, scores, "random_support")
            emitted += int(len(left))
    return store, {
        "candidate_source": "random_support",
        "random_support_sec": float(perf_counter() - start),
        "candidate_pairs_emitted": int(emitted),
        "candidate_pairs_retained": int(store.pair_count()),
        "source_counts": store.source_counts(),
    }


def build_sketch_candidates(
    graph: HeteroGraph,
    *,
    seed: int,
    candidate_k: int,
    candidate_source: str = "sketch",
    twohop_budget_per_node: int = 2,
    twohop_time_budget_sec: float = 2.0,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    config = _candidate_config(seed, candidate_k, twohop_budget_per_node, twohop_time_budget_sec)
    partition = default_partition(graph)
    start = perf_counter()
    sketch = compute_lowpass_sketch(graph, config)
    diag: dict[str, Any] = {"candidate_source": candidate_source, "sketch_sec": float(perf_counter() - start)}
    store = ArrayCandidateStore(graph.node_type, K=int(candidate_k), same_type_only=True)
    start = perf_counter()
    diag["onehop"] = generate_onehop_candidates_chunked(
        graph,
        sketch,
        partition,
        config,
        store,
        edge_chunk_size=int(config["candidates"]["edge_chunk_size"]),
    )
    diag["onehop_sec"] = float(perf_counter() - start)
    start = perf_counter()
    diag["twohop"] = generate_capped_twohop_candidates_chunked(
        graph,
        sketch,
        partition,
        config,
        store,
        middle_chunk_size=int(config["candidates"]["middle_chunk_size"]),
        edge_chunk_size=int(config["candidates"]["edge_chunk_size"]),
    )
    diag["twohop_sec"] = float(perf_counter() - start)
    start = perf_counter()
    buckets = compute_simhash_buckets(
        sketch,
        graph.node_type,
        partition,
        bits=int(config["candidates"]["simhash_bits"]),
        seed=int(seed),
    )
    diag["bucket"] = generate_bucket_candidates_chunked(
        buckets,
        graph.node_type,
        partition,
        config,
        store,
        node_chunk_size=int(config["candidates"]["node_chunk_size"]),
    )
    diag["bucket_sec"] = float(perf_counter() - start)
    diag["candidate_pairs_retained"] = int(store.pair_count())
    diag["candidate_pairs_emitted"] = int(store.pair_count())
    diag["source_counts"] = store.source_counts()
    return store, diag


def build_gate13_candidates(
    graph: HeteroGraph,
    *,
    state,
    target_type: int,
    seed: int,
    candidate_k: int,
    candidate_source: str,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    source = str(candidate_source)
    if source == "random_support":
        return build_random_support_candidates(graph, target_type=target_type, seed=seed, candidate_k=candidate_k)
    if source == "sketch":
        return build_sketch_candidates(graph, seed=seed, candidate_k=candidate_k)
    if source == "graph_sketch":
        return build_sketch_candidates(graph, seed=seed, candidate_k=candidate_k, candidate_source="graph_sketch")
    if source == "target_anchor_co_support":
        return build_target_anchor_co_support_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    if source == "class_footprint_knn":
        return build_class_footprint_knn_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    if source == "target_response_knn":
        return build_target_response_knn_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    if source == "target_response_signature_knn":
        return build_target_response_signature_knn_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    if source == "relation_response_knn":
        return build_relation_response_knn_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    if source == "hybrid_task_aware":
        return build_hybrid_task_aware_candidates(graph, state, target_type=target_type, candidate_k=candidate_k)
    raise ValueError(f"unsupported candidate_source: {candidate_source}")


def cap_candidate_store(
    store: ArrayCandidateStore,
    graph: HeteroGraph,
    *,
    max_pairs: int | None,
) -> tuple[ArrayCandidateStore, int]:
    if max_pairs is None or int(max_pairs) <= 0 or int(store.pair_count()) <= int(max_pairs):
        return store, int(store.pair_count())
    capped = ArrayCandidateStore(graph.node_type, K=store.K, same_type_only=store.same_type_only)
    kept = 0
    for block in store.iter_pair_blocks():
        for u, v, score in np.asarray(block):
            if kept >= int(max_pairs):
                return capped, kept
            source = store.source_for_pair(int(u), int(v)) or "unknown"
            capped.add(int(u), int(v), float(score), source)
            kept += 1
    return capped, kept


def run_multilevel_task_first(
    original: HeteroGraph,
    *,
    method: str,
    ratio: float,
    ratio_mode: str,
    seed: int,
    max_levels: int,
    per_level_ratio: float,
    candidate_k: int,
    candidate_source: str,
    pair_delta_mode: str,
    coverage_mode: str = "combined",
    purity_policy: str = "unknown_blocks_known",
    js_threshold: float = 0.35,
    lambda_rel_response: float | None = None,
    candidate_pair_cap: int | None = None,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    target_type = infer_target_node_type(original)
    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
        original,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
    )
    original_support = int(np.sum(original.node_type != int(target_type)))
    if ratio_mode == "support":
        desired_final_support = max(0, int(np.ceil(original_support * float(ratio) - 1.0e-12)))
    else:
        requested_total = int(np.ceil(original.num_nodes * float(ratio) - 1.0e-12))
        target_count = int(np.sum(original.node_type == int(target_type)))
        desired_final_support = max(0, requested_total - target_count)
    current = original
    cumulative = np.arange(original.num_nodes, dtype=np.int64)
    levels: list[dict[str, Any]] = []
    peak_rss = _rss_mb()
    start_total = perf_counter()
    final_stop_reason = "reached_requested_support_ratio"
    for level in range(1, int(max_levels) + 1):
        current_support = int(np.sum(current.node_type != int(target_type)))
        if current_support <= desired_final_support:
            break
        next_support = max(desired_final_support, int(np.ceil(current_support * float(per_level_ratio) - 1.0e-12)))
        max_support_merges = max(0, current_support - next_support)
        if max_support_merges <= 0:
            final_stop_reason = "merge_budget_floor"
            break
        train_mask = _train_mask_for_current(current, train_nodes, cumulative)
        cfg = task_first_config(
            method,
            target_type=int(target_type),
            max_support_merges=int(max_support_merges),
            pair_delta_mode=pair_delta_mode,
            coverage_mode=coverage_mode,
            purity_policy=purity_policy,
            js_threshold=js_threshold,
            lambda_rel_response=lambda_rel_response,
        )
        labels_current = np.asarray(current.labels if current.labels is not None else np.full(current.num_nodes, -1))
        state = build_task_first_state(current, labels_current, train_mask, cfg)
        candidate_start = perf_counter()
        store, candidate_diag = build_gate13_candidates(
            current,
            state=state,
            target_type=int(target_type),
            seed=int(seed) + level * 1009,
            candidate_k=int(candidate_k),
            candidate_source=candidate_source,
        )
        uncapped_pairs = int(store.pair_count())
        store, capped_pairs = cap_candidate_store(store, current, max_pairs=candidate_pair_cap)
        if candidate_pair_cap is not None:
            candidate_diag["candidate_pair_cap"] = int(candidate_pair_cap)
            candidate_diag["candidate_pairs_before_cap"] = int(uncapped_pairs)
            candidate_diag["candidate_pairs_retained"] = int(capped_pairs)
        coarsen_start = perf_counter()
        result = build_support_only_task_first_coarsening(current, store, labels_current, train_mask, cfg)
        selected = int(result.diagnostics.get("selected_support_merges", 0))
        level_row = {
            "level": int(level),
            "input_nodes": int(current.num_nodes),
            "input_support_nodes": int(current_support),
            "requested_support_ratio": float(ratio) if ratio_mode == "support" else "",
            "requested_full_ratio": float(ratio) if ratio_mode == "full" else "",
            "original_support_nodes": int(original_support),
            "current_support_nodes": int(current_support),
            "desired_final_support_nodes": int(desired_final_support),
            "desired_next_support_nodes": int(next_support),
            "max_support_merges": int(max_support_merges),
            "candidate_total_sec": float(perf_counter() - candidate_start),
            "coarsen_sec": float(perf_counter() - coarsen_start),
            "candidate_pair_count": int(candidate_diag.get("candidate_pairs_retained", candidate_diag.get("pair_count", 0))),
            **{f"candidate_{key}": value for key, value in candidate_diag.items() if not isinstance(value, dict)},
            **result.diagnostics,
            "output_nodes": int(result.graph.num_nodes),
            "output_support_nodes": int(np.sum(result.graph.node_type != int(target_type))),
        }
        levels.append(level_row)
        final_stop_reason = str(result.diagnostics.get("stop_reason", "not_stopped"))
        current = result.graph
        cumulative = _compose(cumulative, result.assignment)
        peak_rss = max(peak_rss, _rss_mb())
        if selected <= 0:
            break
    target_nodes = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    final_target_supernodes = cumulative[target_nodes]
    target_hit = bool(len(np.unique(final_target_supernodes)) == len(target_nodes))
    final_support_nodes = int(np.sum(current.node_type != int(target_type)))
    if final_support_nodes <= desired_final_support:
        final_stop_reason = "reached_requested_support_ratio"
    elif len(levels) >= int(max_levels):
        final_stop_reason = "max_levels_reached"
    diagnostics: dict[str, Any] = {
        "target_node_type": int(target_type),
        "train_nodes": int(len(train_nodes)),
        "val_nodes": int(len(val_nodes)),
        "test_nodes": int(len(test_nodes)),
        **split_protocol,
        "requested_ratio": float(ratio),
        "ratio_mode": ratio_mode,
        "candidate_source": str(candidate_source),
        "pair_delta_mode": str(pair_delta_mode),
        "coverage_mode": str(coverage_mode),
        "purity_policy": str(purity_policy),
        "js_threshold": float(js_threshold),
        "original_nodes": int(original.num_nodes),
        "original_support_nodes": int(original_support),
        "desired_final_support_nodes": int(desired_final_support),
        "final_nodes": int(current.num_nodes),
        "final_support_nodes": final_support_nodes,
        "realized_full_ratio": float(current.num_nodes / max(original.num_nodes, 1)),
        "realized_support_ratio": float(final_support_nodes / max(original_support, 1)),
        "target_hit": target_hit,
        "levels": levels,
        "num_levels": int(len(levels)),
        "total_coarsen_sec": float(perf_counter() - start_total),
        "peak_rss_mb": peak_rss,
        "stop_reason": final_stop_reason,
    }
    if levels:
        numeric_keys: set[str] = set()
        for level in levels:
            for key, value in level.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    numeric_keys.add(str(key))
        for key in sorted(numeric_keys):
            values = [float(level.get(key, 0.0) or 0.0) for level in levels]
            diagnostics[f"{key}_last"] = values[-1]
            diagnostics[f"{key}_mean"] = float(np.mean(values))
    return current, cumulative, diagnostics


def evaluate_graph(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    *,
    seed: int,
    task_epochs: int,
    task_hidden_dim: int,
    device: str,
    lr: float = 0.005,
    dropout: float = 0.25,
    max_hops: int = 2,
    max_paths: int | None = 32,
) -> dict[str, Any]:
    return evaluate_hettree_task(
        original,
        coarse,
        np.asarray(assignment, dtype=np.int64),
        seed=int(seed),
        epochs=int(task_epochs),
        hidden_dim=int(task_hidden_dim),
        lr=float(lr),
        dropout=float(dropout),
        max_hops=int(max_hops),
        max_paths=None if max_paths is None else int(max_paths),
        device=str(device),
    ).metrics


def run_full_graph_ceiling_row(args: argparse.Namespace, dataset: str, seed: int, model: str = "hettree_lite") -> dict[str, Any]:
    graph = load_hgb_graph(Path(args.data_root), dataset)
    target_type = infer_target_node_type(graph)
    labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
    train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
        graph,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
    )
    assignment = np.arange(graph.num_nodes, dtype=np.int64)
    task = evaluate_graph(
        graph,
        graph,
        assignment,
        seed=int(seed),
        task_epochs=int(args.task_epochs),
        task_hidden_dim=int(args.task_hidden_dim),
        lr=float(getattr(args, "task_lr", 0.005)),
        dropout=float(getattr(args, "task_dropout", 0.25)),
        max_hops=int(getattr(args, "task_max_hops", 2)),
        max_paths=int(getattr(args, "task_max_paths", 32)),
        device=str(args.device),
    )
    row = {
        "dataset": str(dataset).upper(),
        "seed": int(seed),
        "model": str(model),
        "split_policy": split_protocol.get("split_policy", split_protocol.get("split_protocol", "deterministic_random")),
        "macro_f1": task.get("macro_f1"),
        "micro_f1": task.get("micro_f1"),
        "accuracy": task.get("accuracy"),
        "validation_macro_f1": task.get("validation_macro_f1"),
        "validation_micro_f1": task.get("validation_micro_f1"),
        "validation_accuracy": task.get("validation_accuracy"),
        "train_nodes": int(len(train_nodes)),
        "val_nodes": int(len(val_nodes)),
        "test_nodes": int(len(test_nodes)),
        "num_classes_train": int(len(set(labels[train_nodes].astype(int).tolist()))) if len(train_nodes) else 0,
        "num_classes_val": int(len(set(labels[val_nodes].astype(int).tolist()))) if len(val_nodes) else 0,
        "num_classes_test": int(len(set(labels[test_nodes].astype(int).tolist()))) if len(test_nodes) else 0,
        "epochs": int(args.task_epochs),
        "hidden_dim": int(args.task_hidden_dim),
        "device": task.get("device", str(args.device)),
        "status": "success",
    }
    return row


def _support_baseline_assignment(
    graph: HeteroGraph,
    store: ArrayCandidateStore,
    *,
    target_type: int,
    max_support_merges: int,
) -> Assignment:
    template = build_target_preserve_assignment_template(graph, TaskFirstConfig(target_node_type=int(target_type)))
    rows: list[list[float]] = []
    for block in store.iter_pair_blocks():
        for u, v, score in np.asarray(block):
            u = int(u)
            v = int(v)
            if graph.node_type[u] == int(target_type) or graph.node_type[v] == int(target_type):
                continue
            rows.append([float(u), float(v), float(score)])
    scored = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    raw = run_greedy_cluster_matching(
        graph,
        scored,
        {"coarsening": {"same_type_only": True, "same_partition_only": True, "max_cluster_size": 4, "max_matched_pairs": int(max_support_merges)}},
        partition_id=graph.partitions,
        source_lookup=store.source_for_pair,
    )
    target_nodes = np.flatnonzero(graph.node_type == int(target_type)).astype(np.int64)
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    assignment = np.empty(graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(template.supernode_type[int(template.assignment[int(node)])]))
    raw_to_new: dict[int, int] = {}
    for node in support_nodes:
        root = int(raw.assignment[int(node)])
        if root not in raw_to_new:
            raw_to_new[root] = len(super_types)
            super_types.append(int(graph.node_type[int(node)]))
        assignment[int(node)] = raw_to_new[root]
    return Assignment(
        assignment,
        np.asarray(super_types, dtype=np.int32),
        diagnostics=raw.diagnostics,
    )


def _store_from_embedding(
    graph: HeteroGraph,
    embedding: np.ndarray,
    *,
    target_type: int,
    candidate_k: int,
    source: str,
) -> ArrayCandidateStore:
    store = ArrayCandidateStore(graph.node_type, K=int(candidate_k), same_type_only=True)
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        if type_id == int(target_type):
            continue
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if len(nodes) < 2:
            continue
        local = embedding[nodes].astype(np.float64)
        keys = local @ np.linspace(1.0, 2.0, local.shape[1], dtype=np.float64)
        order = nodes[np.argsort(keys, kind="mergesort")]
        span = min(max(1, int(candidate_k)), max(1, len(order) - 1))
        for offset in range(1, span + 1):
            left = order[:-offset]
            right = order[offset:]
            scores = np.asarray([
                float(np.sum((embedding[int(u)] - embedding[int(v)]) ** 2))
                for u, v in zip(left, right)
            ], dtype=np.float32)
            store.add_many(left, right, scores, source)
    return store


def run_support_baseline(
    original: HeteroGraph,
    *,
    baseline: str,
    ratio: float,
    seed: int,
    candidate_k: int,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    target_type = infer_target_node_type(original)
    support_count = int(np.sum(original.node_type != int(target_type)))
    desired_support = max(0, int(np.ceil(support_count * float(ratio) - 1.0e-12)))
    max_merges = max(0, support_count - desired_support)
    start = perf_counter()
    if baseline == "random-support-only":
        store, diag = build_random_support_candidates(original, target_type=target_type, seed=seed, candidate_k=candidate_k)
    elif baseline == "sketch-support-only-basic":
        store, diag = build_sketch_candidates(original, seed=seed, candidate_k=candidate_k, candidate_source="sketch-support-only-basic")
    elif baseline == "flatten-sum-support-only":
        embedding = np.zeros((original.num_nodes, 1), dtype=np.float32)
        if original.features:
            for type_id, feature in original.features.items():
                nodes = np.flatnonzero(original.node_type == int(type_id)).astype(np.int64)
                embedding[nodes, 0] = np.sum(np.asarray(feature, dtype=np.float32), axis=1)
        store = _store_from_embedding(original, embedding, target_type=target_type, candidate_k=candidate_k, source=baseline)
        diag = {"candidate_source": baseline, "candidate_pairs_retained": int(store.pair_count()), "source_counts": store.source_counts()}
    elif baseline == "H6-no-spec-support-only":
        cfg = task_first_config("HeSF-TC-no-target-spec", target_type=target_type, pair_delta_mode="local_surrogate")
        labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
        train_nodes, _val, _test, _split = select_task_protocol_split(original, labels, seed=int(seed), target_node_type=int(target_type))
        train_mask = np.zeros(original.num_nodes, dtype=bool)
        train_mask[train_nodes] = True
        state = build_task_first_state(original, labels, train_mask, cfg)
        embedding = state.support_relation_footprints
        store = _store_from_embedding(original, embedding, target_type=target_type, candidate_k=candidate_k, source=baseline)
        diag = {"candidate_source": baseline, "candidate_pairs_retained": int(store.pair_count()), "source_counts": store.source_counts()}
    elif baseline == "TypedHash-ChebHeat-support-only":
        embedding = _node_embedding(original, int(seed), dim=8, assignment_source="chebheat_sketch")
        signatures = _signatures(embedding, int(seed), bits=20).reshape(-1, 1).astype(np.float32)
        store = _store_from_embedding(original, signatures, target_type=target_type, candidate_k=candidate_k, source=baseline)
        diag = {
            "candidate_source": baseline,
            "assignment_source": "chebheat_sketch",
            "hash_bits": 20,
            "candidate_pairs_retained": int(store.pair_count()),
            "source_counts": store.source_counts(),
        }
    else:
        raise ValueError(f"unsupported support baseline: {baseline}")
    assignment = _support_baseline_assignment(original, store, target_type=target_type, max_support_merges=max_merges)
    coarse = coarsen_graph(original, assignment)
    target_nodes = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    target_hit = bool(len(np.unique(assignment.assignment[target_nodes])) == len(target_nodes))
    diagnostics = {
        "method": baseline,
        "target_node_type": int(target_type),
        "requested_ratio": float(ratio),
        "ratio_mode": "support",
        "original_nodes": int(original.num_nodes),
        "original_support_nodes": int(support_count),
        "desired_final_support_nodes": int(desired_support),
        "final_nodes": int(coarse.num_nodes),
        "final_support_nodes": int(np.sum(coarse.node_type != int(target_type))),
        "realized_full_ratio": float(coarse.num_nodes / max(original.num_nodes, 1)),
        "realized_support_ratio": float(np.sum(coarse.node_type != int(target_type)) / max(support_count, 1)),
        "target_hit": target_hit,
        "selected_support_merges": int(len(assignment.diagnostics.get("_selected_merge_pairs", []))),
        "num_levels": 1,
        "total_coarsen_sec": float(perf_counter() - start),
        "peak_rss_mb": _rss_mb(),
        **{f"candidate_{key}": value for key, value in diag.items() if not isinstance(value, dict)},
    }
    return coarse, assignment.assignment, diagnostics


def add_task_and_optional_spectral(
    row: dict[str, Any],
    *,
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> None:
    task = evaluate_graph(
        original,
        coarse,
        assignment,
        seed=int(seed),
        task_epochs=int(args.task_epochs),
        task_hidden_dim=int(args.task_hidden_dim),
        lr=float(getattr(args, "task_lr", 0.005)),
        dropout=float(getattr(args, "task_dropout", 0.25)),
        max_hops=int(getattr(args, "task_max_hops", 2)),
        max_paths=int(getattr(args, "task_max_paths", 32)),
        device=str(args.device),
    )
    _flatten("task.", task, row)
    if bool(getattr(args, "spectral", False)):
        spectral = compute_spectral_diagnostics(
            original,
            coarse,
            Assignment(np.asarray(assignment, dtype=np.int64), coarse.node_type.astype(np.int32, copy=False)),
            seed=int(seed),
            num_signals=int(getattr(args, "spectral_signals", 4)),
            smoothing_steps=1,
            relation_detail=False,
        )
        _flatten("spectral.", spectral, row)


def aggregate_rows(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str], metrics: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out: dict[str, Any] = {name: value for name, value in zip(group_keys, key)}
        out["runs"] = len(group)
        for metric in metrics:
            values = []
            for row in group:
                try:
                    values.append(float(row.get(metric)))
                except (TypeError, ValueError):
                    pass
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out_rows.append(out)
    return out_rows


DEFAULT_METRICS = (
    "realized_support_ratio",
    "realized_full_ratio",
    "target_hit",
    "selected_support_merges",
    "num_levels",
    "target_spec_error_last",
    "relation_response_error_last",
    "support_coverage_error_last",
    "support_purity_error_last",
    "task.macro_f1",
    "task.micro_f1",
    "task.accuracy",
    "total_coarsen_sec",
    "peak_rss_mb",
)


def run_parallel(
    combos: Sequence[tuple],
    worker: Callable[..., dict[str, Any]],
    args: argparse.Namespace,
    output_csv: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if int(args.jobs) <= 1:
        for combo in combos:
            rows.append(worker(args, *combo))
            write_csv(output_csv, rows)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {pool.submit(worker, args, *combo): combo for combo in combos}
            for future in as_completed(futures):
                rows.append(future.result())
                write_csv(output_csv, rows)
    return rows


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(PRIMARY_RATIOS))
    parser.add_argument("--ratio-mode", choices=["support", "full"], default="support")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--per-level-ratio", type=float, default=0.55)
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--task-lr", type=float, default=0.005)
    parser.add_argument("--task-dropout", type=float, default=0.25)
    parser.add_argument("--task-max-hops", type=int, default=2)
    parser.add_argument("--task-max-paths", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--spectral-signals", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser


def write_summary_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"Git commit: `{git_commit_hash()}`\n\n")
        handle.write(markdown_table(rows, columns))
        handle.write("\n")
