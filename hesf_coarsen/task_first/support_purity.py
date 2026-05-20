from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig

FOOTPRINT_KNOWN = 0
FOOTPRINT_UNKNOWN_TARGET_CONNECTED = 1
FOOTPRINT_UNKNOWN_ISOLATED_OR_WEAK = 2


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)
    return (values / denom).astype(np.float32)


def build_support_class_footprints(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: TaskFirstConfig,
) -> np.ndarray:
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    train_targets = target_nodes[train_mask[target_nodes] & (labels[target_nodes] >= 0)]
    if len(train_targets) == 0:
        raise ValueError("TaskFirst requires train target labels for support purity")
    num_classes = int(labels[train_targets].max(initial=0)) + 1
    footprints = np.zeros((graph.num_nodes, num_classes), dtype=np.float32)
    target_train = set(int(node) for node in train_targets)
    for rel in graph.relations.values():
        if rel.src_type == int(cfg.target_node_type) and rel.dst_type != int(cfg.target_node_type):
            for src, dst, weight in zip(rel.src, rel.dst, rel.weight):
                if int(src) in target_train:
                    footprints[int(dst), int(labels[int(src)])] += float(weight)
        elif rel.dst_type == int(cfg.target_node_type) and rel.src_type != int(cfg.target_node_type):
            for src, dst, weight in zip(rel.src, rel.dst, rel.weight):
                if int(dst) in target_train:
                    footprints[int(src), int(labels[int(dst)])] += float(weight)
    return _normalize_rows(footprints)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1.0e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def classify_support_footprints(
    support_class_footprints: np.ndarray,
    support_relation_footprints: np.ndarray,
    support_anchor_memberships: dict[int, dict] | None,
) -> np.ndarray:
    footprints = np.asarray(support_class_footprints, dtype=np.float64)
    relations = np.asarray(support_relation_footprints, dtype=np.float64)
    states = np.full(footprints.shape[0], FOOTPRINT_UNKNOWN_ISOLATED_OR_WEAK, dtype=np.int8)
    known = np.sum(footprints, axis=1) > 1.0e-12
    states[known] = FOOTPRINT_KNOWN
    relation_connected = np.sum(relations, axis=1) > 1.0e-12 if relations.size else np.zeros(len(states), dtype=bool)
    anchor_connected = np.zeros(len(states), dtype=bool)
    if support_anchor_memberships:
        for node, memberships in support_anchor_memberships.items():
            if int(node) < len(anchor_connected) and memberships:
                anchor_connected[int(node)] = True
    unknown_target_connected = (~known) & (relation_connected | anchor_connected)
    states[unknown_target_connected] = FOOTPRINT_UNKNOWN_TARGET_CONNECTED
    return states


def _row_distance(values: np.ndarray, u: int, v: int) -> float:
    if values.size == 0:
        return 0.0
    left = values[int(u)].astype(np.float64)
    right = values[int(v)].astype(np.float64)
    denom = max(float(np.sum(left * left) + np.sum(right * right)), 1.0e-12)
    return float(np.sum((left - right) ** 2) / denom)


def merge_is_purity_allowed(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> bool:
    if not cfg.support_purity.enabled:
        return True
    u = int(u)
    v = int(v)
    threshold = float(cfg.support_purity.js_merge_block_threshold)
    policy = str(cfg.support_purity.zero_policy)
    states = getattr(state, "support_footprint_states", None)
    if states is None or policy == "zero_as_no_conflict":
        divergence = _js_divergence(state.support_class_footprints[u], state.support_class_footprints[v])
        return bool(divergence <= threshold)

    u_known = int(states[u]) == FOOTPRINT_KNOWN
    v_known = int(states[v]) == FOOTPRINT_KNOWN
    if u_known and v_known:
        divergence = _js_divergence(state.support_class_footprints[u], state.support_class_footprints[v])
        return bool(divergence <= threshold)
    if policy == "unknown_blocks_known":
        return bool(not (u_known ^ v_known))
    if policy == "unknown_only_merge":
        return bool((not u_known) and (not v_known))
    if policy == "unknown_propagated":
        if u_known ^ v_known:
            return bool(_row_distance(state.support_relation_footprints, u, v) <= threshold)
        return True
    raise ValueError(f"unsupported support purity zero_policy: {cfg.support_purity.zero_policy}")


def delta_support_purity_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> float:
    if str(cfg.support_purity.zero_policy) != "zero_as_no_conflict":
        states = getattr(state, "support_footprint_states", None)
        if states is not None:
            u_known = int(states[int(u)]) == FOOTPRINT_KNOWN
            v_known = int(states[int(v)]) == FOOTPRINT_KNOWN
            if u_known ^ v_known:
                return 1.0
            if (not u_known) and (not v_known):
                return 0.25 * _row_distance(state.support_relation_footprints, int(u), int(v))
    left = state.support_class_footprints[int(u)].astype(np.float64)
    right = state.support_class_footprints[int(v)].astype(np.float64)
    mean = 0.5 * (left + right)
    return float(0.5 * np.sum((left - mean) ** 2) + 0.5 * np.sum((right - mean) ** 2))


def support_purity_pair_kind(u: int, v: int, state) -> str:
    states = getattr(state, "support_footprint_states", None)
    if states is None:
        return "unknown"
    left = int(states[int(u)])
    right = int(states[int(v)])
    if left == FOOTPRINT_KNOWN and right == FOOTPRINT_KNOWN:
        return "known_known"
    if left == FOOTPRINT_KNOWN or right == FOOTPRINT_KNOWN:
        return "known_unknown"
    return "unknown_unknown"
