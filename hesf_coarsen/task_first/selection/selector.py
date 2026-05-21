from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _budget_count(total: int, support_ratio: float) -> int:
    ratio = min(max(float(support_ratio), 0.0), 1.0)
    if total <= 0 or ratio <= 0.0:
        return 0
    return max(1, min(int(total), int(np.ceil(total * ratio - 1.0e-12))))


def _class_ids(class_footprint: np.ndarray) -> np.ndarray:
    if class_footprint.size == 0 or class_footprint.shape[1] == 0:
        return np.full(class_footprint.shape[0], -1, dtype=np.int64)
    mass = class_footprint.sum(axis=1)
    ids = np.argmax(class_footprint, axis=1).astype(np.int64)
    ids[mass <= 1.0e-12] = -1
    return ids


def _coverage_count(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(np.count_nonzero(np.sum(values, axis=0) > 1.0e-12))


def _context_collision(class_ids: np.ndarray, selected_local: list[int]) -> float:
    if len(selected_local) < 2:
        return 0.0
    pairs = 0
    collisions = 0
    for i, left in enumerate(selected_local):
        for right in selected_local[i + 1 :]:
            pairs += 1
            if class_ids[left] >= 0 and class_ids[left] == class_ids[right]:
                collisions += 1
    return float(collisions / max(pairs, 1))


def _topk_order(nodes: np.ndarray, importance: np.ndarray) -> list[int]:
    return sorted(range(len(nodes)), key=lambda idx: (-float(importance[idx]), int(nodes[idx])))


def _diverse_order(
    support_features: dict[str, Any],
    importance: np.ndarray,
    budget: int,
    cfg: SupportSelectorConfig,
) -> list[int]:
    nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    class_fp = support_features["component_matrices"].get("class_footprint", np.empty((len(nodes), 0)))
    anchor = support_features["component_matrices"].get("anchor_distribution", np.empty((len(nodes), 0)))
    class_ids = _class_ids(class_fp)
    anchor_ids = np.argmax(anchor, axis=1).astype(np.int64) if anchor.size else np.full(len(nodes), -1, dtype=np.int64)
    remaining = set(range(len(nodes)))
    selected: list[int] = []
    used_classes: set[int] = set()
    used_anchors: set[int] = set()
    while remaining and len(selected) < budget:
        best_idx = None
        best_score = -1.0e30
        for idx in remaining:
            score = float(importance[idx])
            class_id = int(class_ids[idx])
            anchor_id = int(anchor_ids[idx])
            if cfg.class_balance and class_id >= 0 and class_id not in used_classes:
                score += 0.15
            if cfg.anchor_diversity and anchor_id >= 0 and anchor_id not in used_anchors:
                score += 0.10
            if class_id >= 0 and class_id in used_classes:
                score -= 0.03
            if anchor_id >= 0 and anchor_id in used_anchors:
                score -= 0.02
            if score > best_score or (score == best_score and int(nodes[idx]) < int(nodes[best_idx])):  # type: ignore[index]
                best_idx = idx
                best_score = score
        assert best_idx is not None
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
        if class_ids[best_idx] >= 0:
            used_classes.add(int(class_ids[best_idx]))
        if anchor_ids[best_idx] >= 0:
            used_anchors.add(int(anchor_ids[best_idx]))
    return selected


def select_support_nodes(
    support_features: dict[str, Any],
    support_importance: np.ndarray,
    support_ratio: float,
    cfg: SupportSelectorConfig,
) -> dict[str, Any]:
    support_nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    importance = np.asarray(support_importance, dtype=np.float32).reshape(-1)
    if importance.shape != (len(support_nodes),):
        raise ValueError("support_importance must have one score per support node")
    budget = _budget_count(len(support_nodes), float(support_ratio))
    selector = str(cfg.selector)
    if budget == 0:
        selected_local: list[int] = []
    elif selector in {"teacher_diverse_topk", "validation_greedy", "hybrid_teacher_response"}:
        selected_local = _diverse_order(support_features, importance, budget, cfg)
    elif selector in {"teacher_topk", "mlp_importance"}:
        selected_local = _topk_order(support_nodes, importance)[:budget]
    else:
        raise ValueError(f"unsupported selector: {cfg.selector}")
    selected_nodes = support_nodes[np.asarray(selected_local, dtype=np.int64)] if selected_local else np.empty(0, dtype=np.int64)
    class_fp = support_features["component_matrices"].get("class_footprint", np.empty((len(support_nodes), 0)))
    relation = support_features["component_matrices"].get("relation_profile", np.empty((len(support_nodes), 0)))
    anchor = support_features["component_matrices"].get("anchor_distribution", np.empty((len(support_nodes), 0)))
    class_ids = _class_ids(class_fp)
    type_values, type_counts = np.unique(
        support_features["support_node_types"][selected_local] if selected_local else np.empty(0, dtype=np.int32),
        return_counts=True,
    )
    diagnostics = {
        "selected_support_count": int(len(selected_nodes)),
        "requested_support_ratio": float(support_ratio),
        "realized_support_ratio": float(len(selected_nodes) / max(len(support_nodes), 1)),
        "selected_by_type": {str(int(t)): int(c) for t, c in zip(type_values, type_counts)},
        "selected_by_class_footprint": {
            str(int(label)): int(np.count_nonzero(class_ids[selected_local] == int(label)))
            for label in sorted(set(int(value) for value in class_ids[selected_local])) if selected_local
        },
        "anchor_coverage_before": _coverage_count(anchor),
        "anchor_coverage_after": _coverage_count(anchor[selected_local]) if selected_local else 0,
        "class_coverage_before": _coverage_count(class_fp),
        "class_coverage_after": _coverage_count(class_fp[selected_local]) if selected_local else 0,
        "relation_channel_coverage_before": _coverage_count(relation),
        "relation_channel_coverage_after": _coverage_count(relation[selected_local]) if selected_local else 0,
        "context_collision_rate": _context_collision(class_ids, selected_local),
        "selector_uses_test_labels": False,
    }
    score_rows = [
        {
            "support_node": int(node),
            "score": float(importance[idx]),
            "selected": bool(idx in set(selected_local)),
            "rank": int(rank + 1),
        }
        for rank, idx in enumerate(_topk_order(support_nodes, importance))
        for node in [support_nodes[idx]]
    ]
    return {
        "selected_support_nodes": selected_nodes.astype(np.int64, copy=False),
        "support_selection_scores": score_rows,
        "diagnostics": diagnostics,
        "selected_local_indices": np.asarray(selected_local, dtype=np.int64),
    }
