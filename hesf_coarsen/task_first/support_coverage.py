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


def build_support_anchor_memberships(
    anchor_neighborhoods: dict[tuple[int, str], np.ndarray],
    cfg: TaskFirstConfig,
) -> dict[int, dict[tuple[int, str], tuple[float, float]]]:
    memberships: dict[int, dict[tuple[int, str], tuple[float, float]]] = {}
    for key, rows in anchor_neighborhoods.items():
        if rows.size == 0:
            continue
        nodes = rows[:, 0].astype(np.int64)
        weights = rows[:, 1].astype(np.float64)
        norm = max(float(np.sum(weights * weights)), cfg.support_coverage.epsilon)
        for node, weight in zip(nodes, weights):
            memberships.setdefault(int(node), {})[key] = (float(weight), norm)
    return memberships


def _anchor_ids(memberships: dict[tuple[int, str], tuple[float, float]]) -> set[int]:
    return {int(anchor) for anchor, _relation in memberships}


def _jaccard_distance(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return float(1.0 - len(left & right) / union)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1.0e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)
    total = 0.0
    for value, base in ((p, m), (q, m)):
        mask = value > 0.0
        if np.any(mask):
            total += 0.5 * float(np.sum(value[mask] * np.log((value[mask] + eps) / (base[mask] + eps))))
    return float(total)


def coverage_components_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> dict[str, float]:
    u = int(u)
    v = int(v)
    memberships = getattr(state, "support_anchor_memberships", None) or {}
    left = memberships.get(u, {})
    right = memberships.get(v, {})
    common = set(left) & set(right)
    same_anchor = 0.0
    if common:
        penalties = []
        for key in common:
            wu, norm = left[key]
            wv, _right_norm = right[key]
            penalties.append(float((wu * wu + wv * wv) / max(float(norm), cfg.support_coverage.epsilon)))
        same_anchor = float(np.mean(penalties))
    left_anchors = _anchor_ids(left)
    right_anchors = _anchor_ids(right)
    class_context = _js_divergence(
        state.support_class_footprints[u],
        state.support_class_footprints[v],
    )
    cross_anchor = float(_jaccard_distance(left_anchors, right_anchors) * class_context)
    return {
        "same_anchor_loss": same_anchor,
        "cross_anchor_collision_loss": cross_anchor,
        "class_context_collision_loss": class_context,
    }


def coverage_v2_components_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> dict[str, float]:
    u = int(u)
    v = int(v)
    memberships = getattr(state, "support_anchor_memberships", None) or {}
    left = memberships.get(u, {})
    right = memberships.get(v, {})
    left_anchors = _anchor_ids(left)
    right_anchors = _anchor_ids(right)
    anchor_collision = _jaccard_distance(left_anchors, right_anchors)
    class_collision = _js_divergence(
        state.support_class_footprints[u],
        state.support_class_footprints[v],
    )
    receptive_field_diversity_loss = float(anchor_collision * class_collision)
    weighted = (
        float(cfg.support_coverage.w_anchor) * float(anchor_collision)
        + float(cfg.support_coverage.w_class) * float(class_collision)
        + float(cfg.support_coverage.w_div) * float(receptive_field_diversity_loss)
    )
    return {
        "anchor_distribution_collision": float(anchor_collision),
        "class_context_collision": float(class_collision),
        "receptive_field_diversity_loss": float(receptive_field_diversity_loss),
        "coverage_v2_error": float(weighted),
    }


def delta_support_coverage_for_merge(
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> float:
    mode = str(getattr(cfg.support_coverage, "mode", "old_common_anchor_only"))
    if mode == "coverage_v2":
        return float(coverage_v2_components_for_merge(int(u), int(v), state, cfg)["coverage_v2_error"])
    if mode not in {"old_common_anchor_only", "coverage_v1_legacy"}:
        components = coverage_components_for_merge(int(u), int(v), state, cfg)
        if mode == "cross_anchor_collision":
            return float(components["cross_anchor_collision_loss"])
        if mode == "class_context_collision":
            return float(components["class_context_collision_loss"])
        if mode == "combined":
            return float(
                components["same_anchor_loss"]
                + components["cross_anchor_collision_loss"]
                + components["class_context_collision_loss"]
            )
        raise ValueError(f"unsupported support coverage mode: {cfg.support_coverage.mode}")
    u = int(u)
    v = int(v)
    memberships = getattr(state, "support_anchor_memberships", None)
    if memberships is not None:
        left = memberships.get(u, {})
        right = memberships.get(v, {})
        common = set(left) & set(right)
        if not common:
            return 0.0
        penalties = []
        for key in common:
            wu, norm = left[key]
            wv, _right_norm = right[key]
            penalties.append(float((wu * wu + wv * wv) / norm))
        return float(np.mean(penalties))
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
