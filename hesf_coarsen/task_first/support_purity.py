from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig


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


def merge_is_purity_allowed(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> bool:
    if not cfg.support_purity.enabled:
        return True
    divergence = _js_divergence(
        state.support_class_footprints[int(u)],
        state.support_class_footprints[int(v)],
    )
    return bool(divergence <= float(cfg.support_purity.js_merge_block_threshold))


def delta_support_purity_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> float:
    left = state.support_class_footprints[int(u)].astype(np.float64)
    right = state.support_class_footprints[int(v)].astype(np.float64)
    mean = 0.5 * (left + right)
    return float(0.5 * np.sum((left - mean) ** 2) + 0.5 * np.sum((right - mean) ** 2))
