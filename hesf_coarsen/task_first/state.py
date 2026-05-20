from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.probes import (
    build_target_seed_matrix,
    compute_target_conditioned_filter_bank,
)
from hesf_coarsen.task_first.relation_response import compute_relation_target_responses
from hesf_coarsen.task_first.support_coverage import build_anchor_neighborhoods
from hesf_coarsen.task_first.support_purity import build_support_class_footprints


@dataclass
class TaskFirstState:
    target_nodes: np.ndarray
    support_nodes: np.ndarray
    train_target_nodes: np.ndarray
    target_seed_matrix: np.ndarray
    target_filter_responses: dict[float, np.ndarray]
    relation_target_responses: dict[str, np.ndarray]
    support_class_footprints: np.ndarray
    anchor_neighborhoods: dict[tuple[int, str], np.ndarray]


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
    support_class_footprints = build_support_class_footprints(graph, labels, train_mask, cfg)
    anchor_neighborhoods = build_anchor_neighborhoods(graph, train_target_nodes, cfg)
    return TaskFirstState(
        target_nodes=target_nodes,
        support_nodes=support_nodes,
        train_target_nodes=train_target_nodes,
        target_seed_matrix=target_seed_matrix,
        target_filter_responses=target_filter_responses,
        relation_target_responses=relation_target_responses,
        support_class_footprints=support_class_footprints,
        anchor_neighborhoods=anchor_neighborhoods,
    )
