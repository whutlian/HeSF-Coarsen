from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.constraints import allow_task_first_merge
from hesf_coarsen.task_first.probes import target_conditioned_response_error
from hesf_coarsen.task_first.relation_response import relation_response_error
from hesf_coarsen.task_first.scoring import compute_task_first_delta, normalize_task_first_deltas
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_coverage import delta_support_coverage_for_merge
from hesf_coarsen.task_first.support_purity import (
    delta_support_purity_for_merge,
    merge_is_purity_allowed,
)


@dataclass
class SupportCompressedGraph:
    graph: HeteroGraph
    assignment: Assignment
    diagnostics: dict


def _support_only_assignment(
    graph: HeteroGraph,
    scored_pairs: np.ndarray,
    cfg: TaskFirstConfig,
) -> Assignment:
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(graph.node_type[int(node)]))

    used = np.zeros(graph.num_nodes, dtype=bool)
    used[target_nodes] = True
    selected_pairs: list[tuple[int, int]] = []
    if scored_pairs.size:
        pairs = np.asarray(scored_pairs, dtype=np.float64)
        order = np.lexsort((pairs[:, 1], pairs[:, 0], pairs[:, 2]))
        for row_idx in order:
            u = int(pairs[row_idx, 0])
            v = int(pairs[row_idx, 1])
            if used[u] or used[v]:
                continue
            supernode = len(super_types)
            assignment[u] = supernode
            assignment[v] = supernode
            used[u] = True
            used[v] = True
            super_types.append(int(graph.node_type[u]))
            selected_pairs.append((u, v))

    for node in range(graph.num_nodes):
        if assignment[node] >= 0:
            continue
        assignment[node] = len(super_types)
        super_types.append(int(graph.node_type[node]))

    return Assignment(
        assignment,
        np.asarray(super_types, dtype=np.int32),
        diagnostics={"_selected_merge_pairs": np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)},
    )


def build_support_only_task_first_coarsening(
    original: HeteroGraph,
    base_candidates,
    labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: TaskFirstConfig,
) -> SupportCompressedGraph:
    if not cfg.keep_all_target_nodes or not cfg.support_only_coarsening:
        raise ValueError("HeSF-TC v1 requires target singleton and support-only coarsening")
    state = build_task_first_state(original, labels, train_mask, cfg)
    scored_rows: list[list[float]] = []
    scored_pairs: list[tuple[int, int]] = []
    deltas = []
    rejected_constraints = 0
    rejected_purity = 0
    evaluated = 0
    for block in base_candidates.iter_pair_blocks():
        for raw_u, raw_v, _base_score in np.asarray(block):
            u = int(raw_u)
            v = int(raw_v)
            if not merge_is_purity_allowed(u, v, state, cfg):
                rejected_purity += 1
                rejected_constraints += 1
                continue
            if not allow_task_first_merge(original, u, v, None, state, cfg):
                rejected_constraints += 1
                continue
            deltas.append(compute_task_first_delta(original, u, v, state, cfg))
            scored_pairs.append((u, v))
            evaluated += 1
    normalized = normalize_task_first_deltas(deltas, cfg)
    for (u, v), delta in zip(scored_pairs, normalized):
        scored_rows.append([float(u), float(v), float(delta.score_task_first)])
    scored = np.asarray(scored_rows, dtype=np.float64).reshape(-1, 3)
    assignment = _support_only_assignment(original, scored, cfg)
    coarse = coarsen_graph(original, assignment)
    diagnostics = {
        "target_node_type": int(cfg.target_node_type),
        "target_nodes_preserved": True,
        "num_target_nodes": int(len(state.target_nodes)),
        "num_support_nodes": int(len(state.support_nodes)),
        "num_support_candidates_scored": int(evaluated),
        "num_support_candidates_rejected_by_purity": int(rejected_purity),
        "num_support_candidates_rejected_by_constraints": int(rejected_constraints),
        "selected_support_merges": int(len(assignment.diagnostics.get("_selected_merge_pairs", []))),
        "target_spec_error": target_conditioned_response_error(original, assignment, state, cfg),
        "relation_response_error": relation_response_error(original, assignment, state, cfg),
        "support_coverage_error": float(
            np.mean(
                [
                    delta_support_coverage_for_merge(int(u), int(v), state, cfg)
                    for u, v in assignment.diagnostics.get("_selected_merge_pairs", [])
                ]
            )
            if len(assignment.diagnostics.get("_selected_merge_pairs", []))
            else 0.0
        ),
        "support_purity_error": float(
            np.mean(
                [
                    delta_support_purity_for_merge(int(u), int(v), state, cfg)
                    for u, v in assignment.diagnostics.get("_selected_merge_pairs", [])
                ]
            )
            if len(assignment.diagnostics.get("_selected_merge_pairs", []))
            else 0.0
        ),
    }
    return SupportCompressedGraph(graph=coarse, assignment=assignment, diagnostics=diagnostics)
