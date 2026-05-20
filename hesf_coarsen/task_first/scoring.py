from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.probes import target_conditioned_response_error
from hesf_coarsen.task_first.relation_response import relation_response_error
from hesf_coarsen.task_first.support_coverage import delta_support_coverage_for_merge
from hesf_coarsen.task_first.support_purity import delta_support_purity_for_merge


@dataclass
class TaskFirstDelta:
    delta_target_spec: float
    delta_rel_response: float
    delta_support_coverage: float
    delta_support_purity: float
    delta_feat: float
    score_task_first: float = 0.0


def assignment_with_pair_merged(graph: HeteroGraph, u: int, v: int, cfg: TaskFirstConfig) -> Assignment:
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
        raise ValueError("TaskFirst pair merge requires support nodes")
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


def _feature_delta(graph: HeteroGraph, u: int, v: int) -> float:
    if graph.features is None:
        return 0.0
    type_id = int(graph.node_type[int(u)])
    feature = graph.features.get(type_id)
    if feature is None:
        return 0.0
    typed_nodes = np.flatnonzero(graph.node_type == type_id).astype(np.int64)
    local = {int(node): idx for idx, node in enumerate(typed_nodes)}
    left = feature[local[int(u)]].astype(np.float64)
    right = feature[local[int(v)]].astype(np.float64)
    denom = max(float(np.sum(left * left) + np.sum(right * right)), 1.0e-8)
    return float(np.sum((left - right) ** 2) / denom)


def compute_task_first_delta(
    graph: HeteroGraph,
    u: int,
    v: int,
    state,
    cfg: TaskFirstConfig,
) -> TaskFirstDelta:
    pair_assignment = assignment_with_pair_merged(graph, int(u), int(v), cfg)
    delta = TaskFirstDelta(
        delta_target_spec=target_conditioned_response_error(graph, pair_assignment, state, cfg),
        delta_rel_response=relation_response_error(graph, pair_assignment, state, cfg),
        delta_support_coverage=delta_support_coverage_for_merge(int(u), int(v), state, cfg),
        delta_support_purity=delta_support_purity_for_merge(int(u), int(v), state, cfg),
        delta_feat=_feature_delta(graph, int(u), int(v)),
    )
    return score_task_first_delta(delta, cfg)


def score_task_first_delta(delta: TaskFirstDelta, cfg: TaskFirstConfig) -> TaskFirstDelta:
    score = (
        float(cfg.scoring.lambda_target_spec) * float(delta.delta_target_spec)
        + float(cfg.scoring.lambda_rel_response) * float(delta.delta_rel_response)
        + float(cfg.scoring.lambda_support_coverage) * float(delta.delta_support_coverage)
        + float(cfg.scoring.lambda_support_purity) * float(delta.delta_support_purity)
        + float(cfg.scoring.lambda_feat) * float(delta.delta_feat)
    )
    return TaskFirstDelta(
        delta_target_spec=float(delta.delta_target_spec),
        delta_rel_response=float(delta.delta_rel_response),
        delta_support_coverage=float(delta.delta_support_coverage),
        delta_support_purity=float(delta.delta_support_purity),
        delta_feat=float(delta.delta_feat),
        score_task_first=float(score),
    )


def normalize_task_first_deltas(
    deltas: list[TaskFirstDelta],
    cfg: TaskFirstConfig,
) -> list[TaskFirstDelta]:
    if str(cfg.scoring.normalization).lower() != "p95" or not deltas:
        return [score_task_first_delta(delta, cfg) for delta in deltas]
    fields = (
        "delta_target_spec",
        "delta_rel_response",
        "delta_support_coverage",
        "delta_support_purity",
        "delta_feat",
    )
    scales = {}
    for field in fields:
        values = np.asarray([float(getattr(delta, field)) for delta in deltas], dtype=np.float64)
        scales[field] = max(float(np.percentile(np.abs(values), 95)), 1.0e-12)
    normalized = []
    for delta in deltas:
        normalized.append(
            score_task_first_delta(
                TaskFirstDelta(
                    delta_target_spec=float(delta.delta_target_spec) / scales["delta_target_spec"],
                    delta_rel_response=float(delta.delta_rel_response) / scales["delta_rel_response"],
                    delta_support_coverage=float(delta.delta_support_coverage)
                    / scales["delta_support_coverage"],
                    delta_support_purity=float(delta.delta_support_purity)
                    / scales["delta_support_purity"],
                    delta_feat=float(delta.delta_feat) / scales["delta_feat"],
                ),
                cfg,
            )
        )
    return normalized
