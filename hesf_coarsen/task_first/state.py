from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.probes import (
    build_target_seed_matrix,
    compute_target_conditioned_filter_bank,
    lift_target_seed,
)
from hesf_coarsen.task_first.relation_response import compute_relation_target_responses
from hesf_coarsen.task_first.relation_response import build_support_relation_footprints
from hesf_coarsen.task_first.support_coverage import (
    build_anchor_neighborhoods,
    build_support_anchor_memberships,
)
from hesf_coarsen.task_first.support_purity import build_support_class_footprints
from hesf_coarsen.task_first.support_purity import classify_support_footprints


@dataclass
class TaskFirstState:
    target_nodes: np.ndarray
    support_nodes: np.ndarray
    train_target_nodes: np.ndarray
    target_seed_matrix: np.ndarray
    target_filter_responses: dict[float, np.ndarray]
    relation_target_responses: dict[str, np.ndarray]
    support_relation_footprints: np.ndarray
    support_class_footprints: np.ndarray
    support_response_signatures: np.ndarray
    support_footprint_states: np.ndarray
    anchor_neighborhoods: dict[tuple[int, str], np.ndarray]
    support_anchor_memberships: dict[int, dict[tuple[int, str], tuple[float, float]]]
    feature_node_positions: dict[int, dict[int, int]]


def _feature_node_positions(graph: HeteroGraph) -> dict[int, dict[int, int]]:
    return {
        int(type_id): {
            int(node): int(pos)
            for pos, node in enumerate(np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64))
        }
        for type_id in sorted(int(value) for value in np.unique(graph.node_type))
    }


def _support_response_signatures(
    graph: HeteroGraph,
    target_nodes: np.ndarray,
    target_seed_matrix: np.ndarray,
    support_class_footprints: np.ndarray,
    support_relation_footprints: np.ndarray,
    cfg: TaskFirstConfig,
) -> np.ndarray:
    blocks = [
        np.asarray(support_class_footprints, dtype=np.float32),
        np.asarray(support_relation_footprints, dtype=np.float32),
    ]
    full_response = lift_target_seed(graph, target_nodes, target_seed_matrix)
    for temperature in cfg.target_spec.temperatures:
        steps = max(2, int(np.ceil(float(temperature) * 2.0)))
        response = np.asarray(full_response, dtype=np.float32)
        for _ in range(steps):
            response = apply_fused_smoothing(graph, response)
        blocks.append(response.astype(np.float32, copy=False))
    signature = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    norms = np.maximum(np.linalg.norm(signature, axis=1, keepdims=True), 1.0e-12)
    return (signature / norms).astype(np.float32)


def build_task_first_state(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: TaskFirstConfig,
) -> TaskFirstState:
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    if labels.shape[0] != graph.num_nodes or train_mask.shape != (graph.num_nodes,):
        raise ValueError("labels and train_mask must have one row per graph node")
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    support_nodes = np.flatnonzero(graph.node_type != int(cfg.target_node_type)).astype(np.int64)
    train_target_nodes = target_nodes[train_mask[target_nodes] & (labels[target_nodes] >= 0)]
    target_seed_matrix = build_target_seed_matrix(target_nodes, labels, train_mask)
    target_filter_responses = compute_target_conditioned_filter_bank(
        graph,
        target_nodes,
        target_seed_matrix,
        cfg,
    )
    relation_target_responses = compute_relation_target_responses(
        graph,
        target_nodes,
        target_seed_matrix,
        cfg,
    )
    support_relation_footprints = build_support_relation_footprints(graph, cfg)
    support_class_footprints = build_support_class_footprints(graph, labels, train_mask, cfg)
    anchor_neighborhoods = build_anchor_neighborhoods(graph, train_target_nodes, cfg)
    support_anchor_memberships = build_support_anchor_memberships(anchor_neighborhoods, cfg)
    support_response_signatures = _support_response_signatures(
        graph,
        target_nodes,
        target_seed_matrix,
        support_class_footprints,
        support_relation_footprints,
        cfg,
    )
    support_footprint_states = classify_support_footprints(
        support_class_footprints,
        support_relation_footprints,
        support_anchor_memberships,
    )
    return TaskFirstState(
        target_nodes=target_nodes,
        support_nodes=support_nodes,
        train_target_nodes=train_target_nodes,
        target_seed_matrix=target_seed_matrix,
        target_filter_responses=target_filter_responses,
        relation_target_responses=relation_target_responses,
        support_relation_footprints=support_relation_footprints,
        support_class_footprints=support_class_footprints,
        support_response_signatures=support_response_signatures,
        support_footprint_states=support_footprint_states,
        anchor_neighborhoods=anchor_neighborhoods,
        support_anchor_memberships=support_anchor_memberships,
        feature_node_positions=_feature_node_positions(graph),
    )
