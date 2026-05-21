from __future__ import annotations

from typing import Any
from collections import Counter, defaultdict

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _argmax_or_unknown(matrix: np.ndarray, node: int) -> int:
    if matrix.size == 0 or int(node) >= matrix.shape[0] or matrix.shape[1] == 0:
        return -1
    row = np.asarray(matrix[int(node)], dtype=np.float32)
    if float(np.sum(row)) <= 1.0e-12:
        return -1
    return int(np.argmax(row))


def _prototype_key(
    node: int,
    type_id: int,
    support_features: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    if not support_features:
        return (int(type_id), -1, -1, -1)
    all_components = support_features.get("all_node_component_matrices", {})
    class_id = _argmax_or_unknown(np.asarray(all_components.get("class_footprint", np.empty((0, 0)))), int(node))
    anchor_id = _argmax_or_unknown(np.asarray(all_components.get("anchor_distribution", np.empty((0, 0)))), int(node))
    relation_bucket = _argmax_or_unknown(np.asarray(all_components.get("relation_profile", np.empty((0, 0)))), int(node))
    return (int(type_id), int(class_id), int(anchor_id), int(relation_bucket))


def build_selected_support_graph(
    original_graph: HeteroGraph,
    selected_support_nodes: np.ndarray,
    cfg: SupportSelectorConfig,
    *,
    target_node_type: int,
    support_features: dict[str, Any] | None = None,
) -> tuple[HeteroGraph, Assignment, dict[str, Any]]:
    target_type = int(target_node_type)
    selected = {int(node) for node in np.asarray(selected_support_nodes, dtype=np.int64).reshape(-1)}
    target_nodes = np.flatnonzero(original_graph.node_type == target_type).astype(np.int64)
    support_nodes = np.flatnonzero(original_graph.node_type != target_type).astype(np.int64)
    assignment = np.empty(original_graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    background_by_type: dict[int, int] = {}
    prototype_by_key: dict[tuple[int, int, int, int], int] = {}
    prototype_members: dict[int, list[int]] = defaultdict(list)
    selected_by_type: dict[int, int] = {}

    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(original_graph.node_type[int(node)]))
    for node in support_nodes:
        node = int(node)
        type_id = int(original_graph.node_type[node])
        if node in selected:
            assignment[node] = len(super_types)
            super_types.append(type_id)
            selected_by_type[type_id] = selected_by_type.get(type_id, 0) + 1
            continue
        if not cfg.allow_background_bucket:
            assignment[node] = len(super_types)
            super_types.append(type_id)
            continue
        if str(cfg.background_strategy) == "class_anchor_relation_prototype":
            key = _prototype_key(node, type_id, support_features)
            type_count = sum(1 for existing in prototype_by_key if existing[0] == type_id)
            if key not in prototype_by_key and type_count >= int(cfg.max_prototypes_per_type):
                key = (type_id, -1, -1, -1)
            if key not in prototype_by_key:
                prototype_by_key[key] = len(super_types)
                super_types.append(type_id)
            assignment[node] = prototype_by_key[key]
            prototype_members[prototype_by_key[key]].append(node)
            continue
        if type_id not in background_by_type:
            background_by_type[type_id] = len(super_types)
            super_types.append(type_id)
        assignment[node] = background_by_type[type_id]

    assignment_obj = Assignment(assignment, np.asarray(super_types, dtype=np.int32))
    coarse = coarsen_graph(original_graph, assignment_obj)
    background_nodes = set(background_by_type.values()) | set(prototype_by_key.values())
    background_edges_by_relation = {}
    edge_retention_by_relation = {}
    for relation_id, rel in coarse.relations.items():
        incident = np.isin(rel.src, list(background_nodes)) | np.isin(rel.dst, list(background_nodes))
        background_edges_by_relation[str(int(relation_id))] = int(np.count_nonzero(incident))
        original_edges = int(original_graph.relations[int(relation_id)].num_edges)
        edge_retention_by_relation[str(int(relation_id))] = float(rel.num_edges / max(original_edges, 1))
    prototype_count_by_type = Counter(str(key[0]) for key in prototype_by_key)
    prototype_count_by_class = Counter(str(key[1]) for key in prototype_by_key)
    prototype_count_by_anchor = Counter(str(key[2]) for key in prototype_by_key)
    prototype_count_by_relation = Counter(str(key[3]) for key in prototype_by_key)
    member_counts = [len(value) for value in prototype_members.values()]
    diagnostics = {
        "background_node_count": int(len(background_by_type)),
        "prototype_background_count": int(len(prototype_by_key)),
        "prototype_count_by_type": dict(prototype_count_by_type),
        "prototype_count_by_class": dict(prototype_count_by_class),
        "prototype_count_by_anchor": dict(prototype_count_by_anchor),
        "prototype_count_by_relation_bucket": dict(prototype_count_by_relation),
        "prototype_member_count_mean": float(np.mean(member_counts)) if member_counts else 0.0,
        "prototype_member_count_max": int(max(member_counts)) if member_counts else 0,
        "background_edges_by_relation": background_edges_by_relation,
        "dropped_support_count": int(len(support_nodes) - len(selected)),
        "selected_support_count": int(len(selected)),
        "selected_raw_support_count": int(len(selected)),
        "unselected_support_count": int(len(support_nodes) - len(selected)),
        "selected_by_type": {str(key): int(value) for key, value in selected_by_type.items()},
        "edge_retention_by_relation": edge_retention_by_relation,
        "support_context_collision_after_condensation": 0.0,
        "coarse_nodes": int(coarse.num_nodes),
        "coarse_edges": int(sum(rel.num_edges for rel in coarse.relations.values())),
        "background_strategy": str(cfg.background_strategy),
    }
    return coarse, assignment_obj, diagnostics
