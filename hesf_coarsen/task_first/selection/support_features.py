from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.task_first.config import SupportPurityConfig, TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_purity import (
    FOOTPRINT_KNOWN,
    FOOTPRINT_UNKNOWN_ISOLATED_OR_WEAK,
    FOOTPRINT_UNKNOWN_TARGET_CONNECTED,
)
from hesf_coarsen.task_first.selection.config import SupportFeatureConfig


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
    return (values / denom).astype(np.float32)


def _feature_width(graph: HeteroGraph) -> int:
    width = 0
    for feature in (graph.features or {}).values():
        width = max(width, int(feature.shape[1]))
    return max(width, 1)


def _raw_feature_block(graph: HeteroGraph) -> np.ndarray:
    width = _feature_width(graph)
    type_ids = sorted(int(value) for value in np.unique(graph.node_type))
    block = np.zeros((graph.num_nodes, width + len(type_ids)), dtype=np.float32)
    for type_id in type_ids:
        typed_nodes = nodes_of_type(graph, type_id)
        feature = (graph.features or {}).get(type_id)
        if feature is not None and len(typed_nodes):
            local_width = min(width, int(feature.shape[1]))
            block[typed_nodes, :local_width] = np.asarray(feature[:, :local_width], dtype=np.float32)
        block[typed_nodes, width + type_ids.index(type_id)] = 1.0
    return block


def _degree_profile(graph: HeteroGraph) -> np.ndarray:
    relation_ids = sorted(int(relation_id) for relation_id in graph.relations)
    profile = np.zeros((graph.num_nodes, max(1, len(relation_ids) * 2)), dtype=np.float32)
    if not relation_ids:
        return profile
    offset = len(relation_ids)
    for pos, relation_id in enumerate(relation_ids):
        rel = graph.relations[relation_id]
        np.add.at(profile[:, pos], rel.src, rel.weight.astype(np.float32, copy=False))
        np.add.at(profile[:, offset + pos], rel.dst, rel.weight.astype(np.float32, copy=False))
    return _normalize_rows(profile)


def _teacher_soft_footprint(
    graph: HeteroGraph,
    target_node_type: int,
    teacher_outputs: dict[str, Any] | None,
    width: int,
) -> np.ndarray:
    out = np.zeros((graph.num_nodes, int(width)), dtype=np.float32)
    if not teacher_outputs or "logits" not in teacher_outputs:
        return out
    logits = np.asarray(teacher_outputs["logits"], dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] == 0:
        return out
    logits = logits[:, : min(int(width), logits.shape[1])]
    probs = _softmax(logits)
    target_type = int(target_node_type)
    for rel in graph.relations.values():
        if rel.src_type == target_type and rel.dst_type != target_type:
            np.add.at(out, rel.dst, probs[rel.src] * rel.weight[:, None])
        elif rel.dst_type == target_type and rel.src_type != target_type:
            np.add.at(out, rel.src, probs[rel.dst] * rel.weight[:, None])
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def _anchor_distribution(state, graph: HeteroGraph) -> np.ndarray:
    target_count = max(1, int(len(state.train_target_nodes)))
    block = np.zeros((graph.num_nodes, 3), dtype=np.float32)
    for node, memberships in state.support_anchor_memberships.items():
        if not memberships:
            continue
        weights = np.asarray([float(value[0]) for value in memberships.values()], dtype=np.float32)
        anchors = {int(anchor) for anchor, _rel in memberships}
        block[int(node), 0] = float(len(anchors) / target_count)
        block[int(node), 1] = float(weights.sum())
        block[int(node), 2] = float(weights.max(initial=0.0))
    return _normalize_rows(block)


def _feature_diagnostics(
    feature_matrix: np.ndarray,
    support_nodes: np.ndarray,
    class_footprint: np.ndarray,
    relation_footprint: np.ndarray,
    footprint_states: np.ndarray,
) -> dict[str, float | int]:
    support_class = class_footprint[support_nodes] if len(support_nodes) else np.empty((0, 0))
    masses = np.sum(support_class, axis=1) if support_class.size else np.zeros(len(support_nodes))
    support_states = footprint_states[support_nodes] if len(support_nodes) and len(footprint_states) else np.empty(0)
    total = max(int(len(support_nodes)), 1)
    return {
        "zero_footprint_support_share": float(np.mean(masses <= 1.0e-12)) if len(support_nodes) else 0.0,
        "known_footprint_support_share": float(np.count_nonzero(support_states == FOOTPRINT_KNOWN) / total)
        if len(support_states)
        else 0.0,
        "unknown_but_structured_share": float(
            np.count_nonzero(support_states == FOOTPRINT_UNKNOWN_TARGET_CONNECTED) / total
        )
        if len(support_states)
        else 0.0,
        "unknown_isolated_or_weak_share": float(
            np.count_nonzero(support_states == FOOTPRINT_UNKNOWN_ISOLATED_OR_WEAK) / total
        )
        if len(support_states)
        else 0.0,
        "support_feature_nan_count": int(np.count_nonzero(~np.isfinite(feature_matrix))),
        "support_feature_zero_row_count": int(
            np.count_nonzero(np.linalg.norm(feature_matrix, axis=1) <= 1.0e-12)
        )
        if feature_matrix.size
        else 0,
        "relation_footprint_nonzero_share": float(
            np.mean(np.sum(relation_footprint[support_nodes], axis=1) > 1.0e-12)
        )
        if len(support_nodes) and relation_footprint.size
        else 0.0,
    }


def build_support_selection_features(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_node_type: int,
    teacher_outputs: dict[str, Any] | None,
    cfg: SupportFeatureConfig,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    target_type = int(target_node_type)
    task_cfg = TaskFirstConfig(
        target_node_type=target_type,
        support_purity=SupportPurityConfig(
            zero_policy="purity_v2",
            support_footprint_mode=(
                "twohop_propagated"
                if cfg.footprint_mode == "twohop_propagated"
                else "hybrid_propagated"
                if cfg.footprint_mode in {"hybrid", "teacher_soft"}
                else "onehop_train"
            ),
        ),
    )
    state = build_task_first_state(graph, labels, train_mask, task_cfg)
    support_nodes = np.flatnonzero(graph.node_type != target_type).astype(np.int64)
    raw = _raw_feature_block(graph)
    degree = _degree_profile(graph)
    class_footprint = np.asarray(state.support_class_footprints, dtype=np.float32)
    teacher_width = max(class_footprint.shape[1], 1)
    teacher_soft = _teacher_soft_footprint(graph, target_type, teacher_outputs, teacher_width)
    if cfg.footprint_mode == "teacher_soft" and teacher_soft.shape[1] == class_footprint.shape[1]:
        class_footprint = _normalize_rows(teacher_soft)
    elif cfg.footprint_mode == "hybrid" and teacher_soft.shape[1] == class_footprint.shape[1]:
        class_footprint = _normalize_rows(class_footprint + 0.5 * teacher_soft)
    relation_profile = np.asarray(state.support_relation_footprints, dtype=np.float32)
    anchor_distribution = _anchor_distribution(state, graph)
    response_signature = np.asarray(state.support_response_signatures, dtype=np.float32)
    blocks: list[np.ndarray] = []
    component_matrices: dict[str, np.ndarray] = {}
    components = [
        ("raw_feature", raw, cfg.include_raw_feature),
        ("degree_profile", degree, cfg.include_degree_profile),
        ("relation_profile", relation_profile, cfg.include_relation_profile),
        ("class_footprint", class_footprint, cfg.include_class_footprint),
        ("anchor_distribution", anchor_distribution, cfg.include_anchor_distribution),
        ("target_response_signature", response_signature, cfg.include_target_response_signature),
        ("relation_response_signature", relation_profile, cfg.include_relation_response_signature),
    ]
    for name, matrix, enabled in components:
        local = np.asarray(matrix[support_nodes], dtype=np.float32)
        component_matrices[name] = local
        if enabled:
            blocks.append(local)
    feature_matrix = np.concatenate(blocks, axis=1).astype(np.float32, copy=False) if blocks else np.zeros((len(support_nodes), 0), dtype=np.float32)
    diagnostics = _feature_diagnostics(
        feature_matrix,
        support_nodes,
        class_footprint,
        relation_profile,
        state.support_footprint_states,
    )
    return {
        "support_nodes": support_nodes,
        "support_node_types": graph.node_type[support_nodes].astype(np.int32, copy=False),
        "feature_matrix": feature_matrix,
        "component_matrices": component_matrices,
        "all_node_component_matrices": {
            "class_footprint": class_footprint,
            "relation_profile": relation_profile,
            "anchor_distribution": anchor_distribution,
            "target_response_signature": response_signature,
        },
        "diagnostics": diagnostics,
        "target_node_type": target_type,
        "selector_uses_test_labels": False,
    }
