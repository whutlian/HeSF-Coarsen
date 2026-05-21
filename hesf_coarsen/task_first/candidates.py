from __future__ import annotations

from itertools import combinations
from time import perf_counter
from typing import Any

import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.io.schema import HeteroGraph


def _support_nodes_by_type(graph: HeteroGraph, target_type: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        if type_id == int(target_type):
            continue
        out[type_id] = np.flatnonzero(graph.node_type == type_id).astype(np.int64)
    return out


def _row_distance(values: np.ndarray, u: int, v: int) -> float:
    if values.size == 0:
        return 0.0
    left = values[int(u)].astype(np.float64)
    right = values[int(v)].astype(np.float64)
    denom = max(float(np.sum(left * left) + np.sum(right * right)), 1.0e-12)
    return float(np.sum((left - right) ** 2) / denom)


def _window_knn_candidates(
    graph: HeteroGraph,
    values: np.ndarray,
    *,
    target_type: int,
    candidate_k: int,
    source: str,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    start = perf_counter()
    store = ArrayCandidateStore(graph.node_type, K=int(candidate_k), same_type_only=True)
    distances: list[float] = []
    emitted = 0
    known = np.sum(np.abs(values), axis=1) > 1.0e-12 if values.size else np.zeros(graph.num_nodes, dtype=bool)
    for _type_id, nodes in _support_nodes_by_type(graph, int(target_type)).items():
        if len(nodes) < 2:
            continue
        local = np.asarray(values[nodes], dtype=np.float64)
        if local.size == 0:
            keys = nodes.astype(np.float64)
        else:
            weights = np.linspace(1.0, 2.0, local.shape[1], dtype=np.float64)
            keys = local @ weights
        order = nodes[np.argsort(keys, kind="mergesort")]
        span = min(max(1, int(candidate_k)), max(1, len(order) - 1))
        for offset in range(1, span + 1):
            left = order[:-offset]
            right = order[offset:]
            scores = np.asarray([_row_distance(values, int(u), int(v)) for u, v in zip(left, right)], dtype=np.float32)
            store.add_many(left, right, scores, source)
            emitted += int(len(left))
            distances.extend(float(value) for value in scores)
    return store, {
        "candidate_source": source,
        f"{source}_sec": float(perf_counter() - start),
        f"{source}_pairs_emitted": int(emitted),
        "candidate_pairs_emitted": int(emitted),
        "candidate_pairs_retained": int(store.pair_count()),
        "known_footprint_support_count": int(np.count_nonzero(known & (graph.node_type != int(target_type)))),
        "zero_footprint_support_count": int(np.count_nonzero((~known) & (graph.node_type != int(target_type)))),
        f"{source}_distance_mean": float(np.mean(distances)) if distances else 0.0,
        f"{source}_distance_p95": float(np.percentile(distances, 95)) if distances else 0.0,
        "source_counts": store.source_counts(),
    }


def build_target_anchor_co_support_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    start = perf_counter()
    store = ArrayCandidateStore(graph.node_type, K=int(candidate_k), same_type_only=True)
    anchor_to_nodes: dict[tuple[int, str], list[int]] = {}
    node_to_anchors: dict[int, set[int]] = {}
    for node, memberships in state.support_anchor_memberships.items():
        for key in memberships:
            anchor_to_nodes.setdefault(key, []).append(int(node))
            node_to_anchors.setdefault(int(node), set()).add(int(key[0]))
    overlaps: list[float] = []
    emitted = 0
    for key in sorted(anchor_to_nodes):
        nodes = sorted(
            set(anchor_to_nodes[key]),
            key=lambda node: (-len(node_to_anchors.get(int(node), set())), int(node)),
        )
        nodes = nodes[: max(2, int(candidate_k) * 2)]
        for u, v in combinations(nodes, 2):
            if graph.node_type[int(u)] == int(target_type) or graph.node_type[int(v)] == int(target_type):
                continue
            left = node_to_anchors.get(int(u), set())
            right = node_to_anchors.get(int(v), set())
            union = max(len(left | right), 1)
            overlap = float(len(left & right) / union)
            store.add(int(u), int(v), 1.0 - overlap, "target_anchor_co_support")
            emitted += 1
            overlaps.append(overlap)
    return store, {
        "candidate_source": "target_anchor_co_support",
        "target_anchor_co_support_sec": float(perf_counter() - start),
        "candidate_pairs_emitted": int(emitted),
        "candidate_pairs_retained": int(store.pair_count()),
        "anchor_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "anchor_overlap_p50": float(np.percentile(overlaps, 50)) if overlaps else 0.0,
        "anchor_overlap_p95": float(np.percentile(overlaps, 95)) if overlaps else 0.0,
        "source_counts": store.source_counts(),
    }


def build_class_footprint_knn_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    store, diag = _window_knn_candidates(
        graph,
        state.support_class_footprints,
        target_type=int(target_type),
        candidate_k=int(candidate_k),
        source="class_footprint_knn",
    )
    diag["footprint_knn_pairs_emitted"] = int(diag.get("candidate_pairs_emitted", 0))
    diag["footprint_distance_mean"] = float(diag.get("class_footprint_knn_distance_mean", 0.0))
    diag["footprint_distance_p95"] = float(diag.get("class_footprint_knn_distance_p95", 0.0))
    return store, diag


def build_target_response_knn_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    store, diag = _window_knn_candidates(
        graph,
        state.support_response_signatures,
        target_type=int(target_type),
        candidate_k=int(candidate_k),
        source="target_response_knn",
    )
    diag["response_knn_pairs_emitted"] = int(diag.get("candidate_pairs_emitted", 0))
    diag["response_distance_mean"] = float(diag.get("target_response_knn_distance_mean", 0.0))
    diag["response_distance_p95"] = float(diag.get("target_response_knn_distance_p95", 0.0))
    return store, diag


def build_target_response_signature_knn_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    store, diag = _window_knn_candidates(
        graph,
        state.support_response_signatures,
        target_type=int(target_type),
        candidate_k=int(candidate_k),
        source="target_response_signature_knn",
    )
    diag["response_signature_knn_pairs_emitted"] = int(diag.get("candidate_pairs_emitted", 0))
    diag["response_signature_distance_mean"] = float(diag.get("target_response_signature_knn_distance_mean", 0.0))
    diag["response_signature_distance_p95"] = float(diag.get("target_response_signature_knn_distance_p95", 0.0))
    return store, diag


def build_relation_response_knn_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    store, diag = _window_knn_candidates(
        graph,
        state.support_relation_footprints,
        target_type=int(target_type),
        candidate_k=int(candidate_k),
        source="relation_response_knn",
    )
    diag["relation_response_knn_pairs_emitted"] = int(diag.get("candidate_pairs_emitted", 0))
    diag["relation_response_distance_mean"] = float(diag.get("relation_response_knn_distance_mean", 0.0))
    diag["relation_response_distance_p95"] = float(diag.get("relation_response_knn_distance_p95", 0.0))
    return store, diag


def _merge_candidate_store(target: ArrayCandidateStore, source: ArrayCandidateStore) -> None:
    for block in source.iter_pair_blocks():
        for u, v, score in np.asarray(block):
            source_name = source.source_for_pair(int(u), int(v)) or "unknown"
            target.add(int(u), int(v), float(score), source_name)


def build_hybrid_task_aware_candidates(
    graph: HeteroGraph,
    state,
    *,
    target_type: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    start = perf_counter()
    target = ArrayCandidateStore(graph.node_type, K=max(int(candidate_k) * 3, int(candidate_k)), same_type_only=True)
    source_diags: dict[str, Any] = {}
    for builder in (
        build_target_anchor_co_support_candidates,
        build_class_footprint_knn_candidates,
        build_target_response_signature_knn_candidates,
    ):
        store, diag = builder(graph, state, target_type=int(target_type), candidate_k=int(candidate_k))
        _merge_candidate_store(target, store)
        for key, value in diag.items():
            if not isinstance(value, dict):
                source_diags[f"{diag.get('candidate_source', builder.__name__)}_{key}"] = value
    return target, {
        "candidate_source": "hybrid_task_aware",
        "candidate_generation_sec": float(perf_counter() - start),
        "candidate_pairs_retained": int(target.pair_count()),
        "source_counts": target.source_counts(),
        **source_diags,
    }
