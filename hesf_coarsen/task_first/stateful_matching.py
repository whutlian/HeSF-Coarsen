from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig


@dataclass
class TaskFirstClusterSignature:
    cluster_id: int
    node_ids: np.ndarray
    node_type: int
    target_response_signature: np.ndarray
    relation_response_signature: np.ndarray
    class_footprint: np.ndarray
    anchor_distribution: np.ndarray
    feature_centroid: np.ndarray | None
    support_size: int


@dataclass
class StatefulMatchingResult:
    assignment: Assignment
    selected_pairs: np.ndarray
    diagnostics: dict


def _row_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = max(float(np.sum(left * left) + np.sum(right * right)), 1.0e-12)
    return float(np.sum((left - right) ** 2) / denom)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    eps = 1.0e-12
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / max(float(left.sum()), eps)
    right = right / max(float(right.sum()), eps)
    middle = 0.5 * (left + right)
    total = 0.0
    for value, base in ((left, middle), (right, middle)):
        mask = value > 0.0
        if np.any(mask):
            total += 0.5 * float(np.sum(value[mask] * np.log((value[mask] + eps) / (base[mask] + eps))))
    return total


def _feature_vector(graph: HeteroGraph, node: int, state) -> np.ndarray | None:
    type_id = int(graph.node_type[int(node)])
    feature = (graph.features or {}).get(type_id)
    if feature is None:
        return None
    local = state.feature_node_positions.get(type_id, {})
    return np.asarray(feature[local[int(node)]], dtype=np.float32)


def _anchor_vector(node: int, anchor_index: dict[int, int], state) -> np.ndarray:
    out = np.zeros(len(anchor_index), dtype=np.float32)
    memberships = getattr(state, "support_anchor_memberships", {}).get(int(node), {})
    for (anchor, _relation), (weight, _norm) in memberships.items():
        if int(anchor) in anchor_index:
            out[anchor_index[int(anchor)]] += float(weight)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _initial_signatures(graph: HeteroGraph, state) -> dict[int, TaskFirstClusterSignature]:
    anchors = sorted({int(anchor) for memberships in state.support_anchor_memberships.values() for anchor, _relation in memberships})
    anchor_index = {anchor: index for index, anchor in enumerate(anchors)}
    signatures: dict[int, TaskFirstClusterSignature] = {}
    for node in np.asarray(state.support_nodes, dtype=np.int64):
        node = int(node)
        signatures[node] = TaskFirstClusterSignature(
            cluster_id=node,
            node_ids=np.asarray([node], dtype=np.int64),
            node_type=int(graph.node_type[node]),
            target_response_signature=np.asarray(state.support_response_signatures[node], dtype=np.float32),
            relation_response_signature=np.asarray(state.support_relation_footprints[node], dtype=np.float32),
            class_footprint=np.asarray(state.support_class_footprints[node], dtype=np.float32),
            anchor_distribution=_anchor_vector(node, anchor_index, state),
            feature_centroid=_feature_vector(graph, node, state),
            support_size=1,
        )
    return signatures


def _merge_signature(left: TaskFirstClusterSignature, right: TaskFirstClusterSignature, cluster_id: int) -> TaskFirstClusterSignature:
    total = max(int(left.support_size + right.support_size), 1)

    def weighted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(a, dtype=np.float32) * float(left.support_size)
             + np.asarray(b, dtype=np.float32) * float(right.support_size))
            / float(total)
        ).astype(np.float32)

    if left.feature_centroid is None or right.feature_centroid is None:
        feature = None
    else:
        feature = weighted(left.feature_centroid, right.feature_centroid)
    return TaskFirstClusterSignature(
        cluster_id=int(cluster_id),
        node_ids=np.concatenate([left.node_ids, right.node_ids]).astype(np.int64),
        node_type=int(left.node_type),
        target_response_signature=weighted(left.target_response_signature, right.target_response_signature),
        relation_response_signature=weighted(left.relation_response_signature, right.relation_response_signature),
        class_footprint=weighted(left.class_footprint, right.class_footprint),
        anchor_distribution=weighted(left.anchor_distribution, right.anchor_distribution),
        feature_centroid=feature,
        support_size=int(total),
    )


def _score_pair(left: TaskFirstClusterSignature, right: TaskFirstClusterSignature, cfg: TaskFirstConfig) -> dict[str, float]:
    context_collision = _row_distance(left.anchor_distribution, right.anchor_distribution)
    class_collision = _js_divergence(left.class_footprint, right.class_footprint)
    target_response = _row_distance(left.target_response_signature, right.target_response_signature)
    relation_response = _row_distance(left.relation_response_signature, right.relation_response_signature)
    if left.feature_centroid is None or right.feature_centroid is None:
        feature_distance = 0.0
    else:
        feature_distance = _row_distance(left.feature_centroid, right.feature_centroid)
    score = (
        float(cfg.scoring.lambda_support_coverage) * context_collision
        + float(cfg.scoring.lambda_support_purity) * class_collision
        + float(cfg.scoring.lambda_target_spec) * target_response
        + float(cfg.scoring.lambda_rel_response) * relation_response
        + float(cfg.scoring.lambda_feat) * feature_distance
    )
    return {
        "score": float(score),
        "context_collision": float(context_collision),
        "class_collision": float(class_collision),
        "target_response_distance": float(target_response),
        "relation_response_distance": float(relation_response),
        "feature_distance": float(feature_distance),
    }


def run_stateful_signature_matching(
    graph: HeteroGraph,
    candidates: ArrayCandidateStore,
    state,
    cfg: TaskFirstConfig,
    *,
    max_support_merges: int | None = None,
    max_cluster_size: int = 4,
) -> StatefulMatchingResult:
    signatures = _initial_signatures(graph, state)
    node_to_cluster = {int(node): int(node) for node in np.asarray(state.support_nodes, dtype=np.int64)}
    active = set(signatures)
    candidate_pairs = [
        (int(u), int(v))
        for block in candidates.iter_pair_blocks()
        for u, v, _score in np.asarray(block)
        if int(u) in node_to_cluster and int(v) in node_to_cluster
    ]
    ranked_pairs: list[tuple[float, int, int]] = []
    for u, v in candidate_pairs:
        left = signatures[node_to_cluster[int(u)]]
        right = signatures[node_to_cluster[int(v)]]
        if left.node_type != right.node_type:
            continue
        ranked_pairs.append((_score_pair(left, right, cfg)["score"], int(u), int(v)))
    ranked_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    selected: list[tuple[int, int]] = []
    selected_scores: list[dict[str, float]] = []
    stale_pops = 0
    rescore_count = 0
    update_count = 0
    total_drift = 0.0
    next_cluster_id = (max(active) + 1) if active else 0
    budget = len(candidate_pairs) if max_support_merges is None else max(0, int(max_support_merges))
    for _initial_score, raw_u, raw_v in ranked_pairs:
        if len(selected) >= budget:
            break
        cu = node_to_cluster.get(int(raw_u))
        cv = node_to_cluster.get(int(raw_v))
        if cu is None or cv is None or cu == cv or cu not in active or cv not in active:
            stale_pops += 1
            continue
        left = signatures[cu]
        right = signatures[cv]
        if left.node_type != right.node_type or left.support_size + right.support_size > int(max_cluster_size):
            continue
        best_score = _score_pair(left, right, cfg)
        rescore_count += 1
        merged = _merge_signature(left, right, next_cluster_id)
        next_cluster_id += 1
        total_drift += _row_distance(left.target_response_signature, merged.target_response_signature)
        total_drift += _row_distance(right.target_response_signature, merged.target_response_signature)
        total_drift += _row_distance(left.class_footprint, right.class_footprint)
        if left.feature_centroid is not None and right.feature_centroid is not None:
            total_drift += _row_distance(left.feature_centroid, right.feature_centroid)
        active.remove(left.cluster_id)
        active.remove(right.cluster_id)
        active.add(merged.cluster_id)
        signatures[merged.cluster_id] = merged
        for node in merged.node_ids:
            node_to_cluster[int(node)] = merged.cluster_id
        selected.append((int(raw_u), int(raw_v)))
        selected_scores.append(best_score)
        update_count += 1
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    assignment = np.empty(graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(graph.node_type[int(node)]))
    cluster_to_supernode: dict[int, int] = {}
    for node in range(graph.num_nodes):
        if int(graph.node_type[node]) == int(cfg.target_node_type):
            continue
        cluster_id = node_to_cluster.get(int(node), int(node))
        if cluster_id not in cluster_to_supernode:
            cluster_to_supernode[cluster_id] = len(super_types)
            super_types.append(int(graph.node_type[node]))
        assignment[node] = cluster_to_supernode[cluster_id]
    selected_arr = np.asarray(selected, dtype=np.int64).reshape(-1, 2)

    def mean_metric(name: str) -> float:
        return float(np.mean([row[name] for row in selected_scores])) if selected_scores else 0.0

    return StatefulMatchingResult(
        assignment=Assignment(
            assignment,
            np.asarray(super_types, dtype=np.int32),
            diagnostics={"_selected_merge_pairs": selected_arr, "matching_method": "stateful_signature_v1"},
        ),
        selected_pairs=selected_arr,
        diagnostics={
            "matching_method": "stateful_signature_v1",
            "stateful_update_count": int(update_count),
            "stale_candidate_pop_count": int(stale_pops),
            "rescore_count": int(rescore_count),
            "selected_merge_score_mean": mean_metric("score"),
            "selected_merge_context_collision_mean": mean_metric("context_collision"),
            "selected_merge_class_collision_mean": mean_metric("class_collision"),
            "selected_merge_target_response_distance_mean": mean_metric("target_response_distance"),
            "selected_merge_relation_response_distance_mean": mean_metric("relation_response_distance"),
            "stateful_signature_drift": float(total_drift),
        },
    )
