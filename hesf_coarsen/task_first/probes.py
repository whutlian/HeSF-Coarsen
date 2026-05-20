from __future__ import annotations

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.task_first.config import TaskFirstConfig


def build_target_seed_matrix(
    target_nodes: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    train_targets = target_nodes[train_mask[target_nodes] & (labels[target_nodes] >= 0)]
    if len(train_targets) == 0:
        raise ValueError("TaskFirst requires at least one labeled train target node")
    num_classes = int(labels[train_targets].max(initial=0)) + 1
    seed = np.zeros((len(target_nodes), num_classes), dtype=np.float32)
    target_pos = {int(node): idx for idx, node in enumerate(target_nodes)}
    for node in train_targets:
        seed[target_pos[int(node)], int(labels[int(node)])] = 1.0
    return seed


def lift_target_seed(
    graph: HeteroGraph,
    target_nodes: np.ndarray,
    target_seed_matrix: np.ndarray,
) -> np.ndarray:
    seed = np.zeros((graph.num_nodes, target_seed_matrix.shape[1]), dtype=np.float32)
    seed[np.asarray(target_nodes, dtype=np.int64)] = target_seed_matrix.astype(
        np.float32,
        copy=False,
    )
    return seed


def _steps_for_temperature(temperature: float) -> int:
    return max(2, int(np.ceil(float(temperature) * 2.0)))


def _lowpass_response(
    graph: HeteroGraph,
    full_seed: np.ndarray,
    temperature: float,
) -> np.ndarray:
    response = np.asarray(full_seed, dtype=np.float32)
    for _ in range(_steps_for_temperature(float(temperature))):
        response = apply_fused_smoothing(graph, response)
    return response.astype(np.float32, copy=False)


def compute_target_conditioned_filter_bank(
    graph: HeteroGraph,
    target_nodes: np.ndarray,
    target_seed_matrix: np.ndarray,
    cfg: TaskFirstConfig,
) -> dict[float, np.ndarray]:
    full_seed = lift_target_seed(graph, target_nodes, target_seed_matrix)
    return {
        float(temperature): _lowpass_response(graph, full_seed, float(temperature))[
            np.asarray(target_nodes, dtype=np.int64)
        ]
        for temperature in cfg.target_spec.temperatures
    }


def target_conditioned_response_error(
    original: HeteroGraph,
    assignment: Assignment,
    state,
    cfg: TaskFirstConfig,
) -> float:
    coarse = coarsen_graph(original, assignment)
    full_seed = lift_target_seed(original, state.target_nodes, state.target_seed_matrix)
    coarse_seed = np.zeros((assignment.num_supernodes, full_seed.shape[1]), dtype=np.float32)
    np.add.at(coarse_seed, assignment.assignment, full_seed)
    counts = assignment.cluster_sizes().astype(np.float32)
    coarse_seed /= np.maximum(counts[:, None], 1.0)

    errors = []
    target_supernodes = assignment.assignment[state.target_nodes]
    for temperature, original_response in state.target_filter_responses.items():
        coarse_response = _lowpass_response(coarse, coarse_seed, float(temperature))[
            target_supernodes
        ]
        denom = max(float(np.sum(original_response.astype(np.float64) ** 2)), cfg.target_spec.epsilon)
        diff = original_response.astype(np.float64) - coarse_response.astype(np.float64)
        errors.append(float(np.sum(diff * diff) / denom))
    return float(np.mean(errors) if errors else 0.0)
