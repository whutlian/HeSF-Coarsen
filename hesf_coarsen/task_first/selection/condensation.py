from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


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
        if type_id not in background_by_type:
            background_by_type[type_id] = len(super_types)
            super_types.append(type_id)
        assignment[node] = background_by_type[type_id]

    assignment_obj = Assignment(assignment, np.asarray(super_types, dtype=np.int32))
    coarse = coarsen_graph(original_graph, assignment_obj)
    background_nodes = set(background_by_type.values())
    background_edges_by_relation = {}
    edge_retention_by_relation = {}
    for relation_id, rel in coarse.relations.items():
        incident = np.isin(rel.src, list(background_nodes)) | np.isin(rel.dst, list(background_nodes))
        background_edges_by_relation[str(int(relation_id))] = int(np.count_nonzero(incident))
        original_edges = int(original_graph.relations[int(relation_id)].num_edges)
        edge_retention_by_relation[str(int(relation_id))] = float(rel.num_edges / max(original_edges, 1))
    diagnostics = {
        "background_node_count": int(len(background_by_type)),
        "background_edges_by_relation": background_edges_by_relation,
        "dropped_support_count": int(len(support_nodes) - len(selected)),
        "selected_support_count": int(len(selected)),
        "selected_by_type": {str(key): int(value) for key, value in selected_by_type.items()},
        "edge_retention_by_relation": edge_retention_by_relation,
        "support_context_collision_after_condensation": 0.0,
        "coarse_nodes": int(coarse.num_nodes),
        "coarse_edges": int(sum(rel.num_edges for rel in coarse.relations.values())),
        "background_strategy": str(cfg.background_strategy),
    }
    return coarse, assignment_obj, diagnostics
