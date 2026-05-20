from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig


def build_anchor_neighborhoods(
    graph: HeteroGraph,
    train_target_nodes: np.ndarray,
    cfg: TaskFirstConfig,
) -> dict[tuple[int, str], np.ndarray]:
    target_type = int(cfg.target_node_type)
    support_mask = graph.node_type != target_type
    out: dict[tuple[int, str], np.ndarray] = {}
    topk = max(1, int(cfg.support_coverage.topk))
    train_targets = set(int(node) for node in np.asarray(train_target_nodes, dtype=np.int64))
    for relation_id, rel in sorted(graph.relations.items()):
        if rel.src_type == target_type and rel.dst_type != target_type:
            for anchor in sorted(train_targets):
                mask = rel.src == anchor
                nodes = rel.dst[mask]
                weights = rel.weight[mask]
                keep = support_mask[nodes]
                rows = np.column_stack([nodes[keep], weights[keep]]) if np.any(keep) else np.empty((0, 2))
                if rows.size:
                    order = np.argsort(-rows[:, 1], kind="mergesort")[:topk]
                    out[(anchor, str(relation_id))] = rows[order].astype(np.float32)
        elif rel.dst_type == target_type and rel.src_type != target_type:
            for anchor in sorted(train_targets):
                mask = rel.dst == anchor
                nodes = rel.src[mask]
                weights = rel.weight[mask]
                keep = support_mask[nodes]
                rows = np.column_stack([nodes[keep], weights[keep]]) if np.any(keep) else np.empty((0, 2))
                if rows.size:
                    order = np.argsort(-rows[:, 1], kind="mergesort")[:topk]
                    out[(anchor, str(relation_id))] = rows[order].astype(np.float32)
    return out


def delta_support_coverage_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> float:
    u = int(u)
    v = int(v)
    penalties = []
    for rows in state.anchor_neighborhoods.values():
        if rows.size == 0:
            continue
        nodes = rows[:, 0].astype(np.int64)
        weights = rows[:, 1].astype(np.float64)
        u_mask = nodes == u
        v_mask = nodes == v
        if not np.any(u_mask) or not np.any(v_mask):
            continue
        norm = max(float(np.sum(weights * weights)), cfg.support_coverage.epsilon)
        wu = float(weights[u_mask][0])
        wv = float(weights[v_mask][0])
        penalties.append(float((wu * wu + wv * wv) / norm))
    return float(np.mean(penalties) if penalties else 0.0)
