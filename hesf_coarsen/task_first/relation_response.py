from __future__ import annotations

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.relation_ops import apply_relation, apply_relation_transpose
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.probes import lift_target_seed


def _assignment_with_pair_merged(graph: HeteroGraph, u: int, v: int, cfg: TaskFirstConfig) -> Assignment:
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    target_set = set(int(node) for node in target_nodes)
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(graph.node_type[int(node)]))
    u = int(u)
    v = int(v)
    if u in target_set or v in target_set:
        raise ValueError("TaskFirst relation-response merge requires support nodes")
    pair_supernode = len(super_types)
    assignment[u] = pair_supernode
    assignment[v] = pair_supernode
    super_types.append(int(graph.node_type[u]))
    for node in range(graph.num_nodes):
        if assignment[node] >= 0:
            continue
        assignment[node] = len(super_types)
        super_types.append(int(graph.node_type[node]))
    return Assignment(assignment, np.asarray(super_types, dtype=np.int32))


def _target_relevant_relation_ids(graph: HeteroGraph, cfg: TaskFirstConfig) -> list[int]:
    target_type = int(cfg.target_node_type)
    return [
        int(relation_id)
        for relation_id, rel in sorted(graph.relations.items())
        if rel.src_type == target_type or rel.dst_type == target_type
    ]


def _relation_response(
    graph: HeteroGraph,
    relation_id: int,
    full_seed: np.ndarray,
    target_nodes: np.ndarray,
    target_type: int,
) -> np.ndarray:
    rel = graph.relations[int(relation_id)]
    if rel.src_type == target_type and rel.dst_type == target_type:
        response = apply_relation(graph, relation_id, full_seed, normalize=True)
    elif rel.src_type == target_type:
        support = apply_relation(graph, relation_id, full_seed, normalize=True)
        response = apply_relation_transpose(graph, relation_id, support, normalize=True)
    elif rel.dst_type == target_type:
        support = apply_relation_transpose(graph, relation_id, full_seed, normalize=True)
        response = apply_relation(graph, relation_id, support, normalize=True)
    else:
        response = np.zeros_like(full_seed, dtype=np.float32)
    return response[np.asarray(target_nodes, dtype=np.int64)].astype(np.float32, copy=False)


def compute_relation_target_responses(
    graph: HeteroGraph,
    target_nodes: np.ndarray,
    target_seed_matrix: np.ndarray,
    cfg: TaskFirstConfig,
) -> dict[str, np.ndarray]:
    full_seed = lift_target_seed(graph, target_nodes, target_seed_matrix)
    return {
        str(relation_id): _relation_response(
            graph,
            relation_id,
            full_seed,
            target_nodes,
            int(cfg.target_node_type),
        )
        for relation_id in _target_relevant_relation_ids(graph, cfg)
    }


def relation_response_error(
    original: HeteroGraph,
    assignment: Assignment,
    state,
    cfg: TaskFirstConfig,
) -> float:
    if not cfg.relation_response.enabled:
        return 0.0
    coarse = coarsen_graph(original, assignment)
    full_seed = lift_target_seed(original, state.target_nodes, state.target_seed_matrix)
    coarse_seed = np.zeros((assignment.num_supernodes, full_seed.shape[1]), dtype=np.float32)
    np.add.at(coarse_seed, assignment.assignment, full_seed)
    counts = assignment.cluster_sizes().astype(np.float32)
    coarse_seed /= np.maximum(counts[:, None], 1.0)
    target_supernodes = assignment.assignment[state.target_nodes]
    errors = []
    for key, original_response in state.relation_target_responses.items():
        relation_id = int(key)
        if relation_id not in coarse.relations:
            continue
        coarse_response = _relation_response(
            coarse,
            relation_id,
            coarse_seed,
            target_supernodes,
            int(cfg.target_node_type),
        )
        denom = max(
            float(np.sum(original_response.astype(np.float64) ** 2)),
            cfg.relation_response.epsilon,
        )
        diff = original_response.astype(np.float64) - coarse_response.astype(np.float64)
        errors.append(float(np.sum(diff * diff) / denom))
    return float(np.mean(errors) if errors else 0.0)


def delta_relation_response_for_merge(
    original: HeteroGraph,
    u_or_assignment,
    v_or_state,
    state_or_cfg=None,
    cfg: TaskFirstConfig | None = None,
) -> float:
    if isinstance(u_or_assignment, Assignment):
        assignment = u_or_assignment
        state = v_or_state
        actual_cfg = state_or_cfg
    else:
        if cfg is None:
            raise TypeError("cfg is required when passing merge endpoints")
        assignment = _assignment_with_pair_merged(original, int(u_or_assignment), int(v_or_state), cfg)
        state = state_or_cfg
        actual_cfg = cfg
    if actual_cfg is None:
        raise TypeError("TaskFirstConfig is required")
    return relation_response_error(original, assignment, state, actual_cfg)
