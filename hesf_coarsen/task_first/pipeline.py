from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.matching.greedy import run_greedy_cluster_matching
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


PIPELINE_STEPS = (
    "build_target_preserve_assignment_template",
    "build_task_first_state",
    "filter_candidates_by_hard_constraints",
    "compute_task_first_local_deltas",
    "run_greedy_cluster_on_support_nodes_only",
    "aggregate_with_target_singleton_block",
    "emit_diagnostics",
)


def build_target_preserve_assignment_template(
    graph: HeteroGraph,
    cfg: TaskFirstConfig,
) -> Assignment:
    if not cfg.keep_all_target_nodes or not cfg.support_only_coarsening:
        raise ValueError("TaskFirst template requires target singleton and support-only coarsening")
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    support_nodes = np.flatnonzero(graph.node_type != int(cfg.target_node_type)).astype(np.int64)
    assignment = np.empty(graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(graph.node_type[int(node)]))
    for node in support_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(graph.node_type[int(node)]))
    return Assignment(
        assignment,
        np.asarray(super_types, dtype=np.int32),
        diagnostics={
            "target_preserve_template": True,
            "target_nodes": target_nodes.tolist(),
            "support_nodes": support_nodes.tolist(),
        },
    )


def _greedy_cluster_config(cfg: TaskFirstConfig) -> dict:
    return {
        "coarsening": {
            "matching_method": "greedy_cluster",
            "same_type_only": bool(cfg.same_type_only),
            "same_partition_only": bool(cfg.same_partition_only),
            "max_cluster_size": 4,
        }
    }


def _target_first_assignment_from_support_clusters(
    graph: HeteroGraph,
    raw_assignment: Assignment,
    template: Assignment,
    cfg: TaskFirstConfig,
) -> Assignment:
    target_nodes = np.flatnonzero(graph.node_type == int(cfg.target_node_type)).astype(np.int64)
    support_nodes = np.flatnonzero(graph.node_type != int(cfg.target_node_type)).astype(np.int64)
    target_raw_clusters = set(int(raw_assignment.assignment[int(node)]) for node in target_nodes)
    support_raw_clusters = {
        int(raw_assignment.assignment[int(node)])
        for node in support_nodes
    }
    if target_raw_clusters & support_raw_clusters:
        raise ValueError("greedy_cluster produced a target-support cluster under TaskFirst")

    assignment = np.empty(graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(template.supernode_type[int(template.assignment[int(node)])]))

    raw_to_target_first: dict[int, int] = {}
    for node in support_nodes:
        raw_cluster = int(raw_assignment.assignment[int(node)])
        supernode = raw_to_target_first.get(raw_cluster)
        if supernode is None:
            supernode = len(super_types)
            raw_to_target_first[raw_cluster] = supernode
            super_types.append(int(graph.node_type[int(node)]))
        assignment[int(node)] = supernode

    selected_pairs = np.asarray(
        raw_assignment.diagnostics.get("_selected_merge_pairs", np.empty((0, 2), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1, 2)
    return Assignment(
        assignment,
        np.asarray(super_types, dtype=np.int32),
        diagnostics={
            "_selected_merge_pairs": selected_pairs,
            "matching_method": "greedy_cluster",
            "selected_merges_by_source": dict(raw_assignment.diagnostics.get("selected_merges_by_source", {})),
        },
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
    template = build_target_preserve_assignment_template(original, cfg)
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
    source_lookup = getattr(base_candidates, "source_for_pair", None)
    raw_assignment = run_greedy_cluster_matching(
        original,
        scored,
        _greedy_cluster_config(cfg),
        partition_id=original.partitions,
        source_lookup=source_lookup if callable(source_lookup) else None,
    )
    assignment = _target_first_assignment_from_support_clusters(original, raw_assignment, template, cfg)
    coarse = coarsen_graph(original, assignment)
    selected_pairs = assignment.diagnostics.get("_selected_merge_pairs", np.empty((0, 2), dtype=np.int64))
    diagnostics = {
        "matching_method": "greedy_cluster",
        "pipeline_steps": list(PIPELINE_STEPS),
        "target_node_type": int(cfg.target_node_type),
        "target_nodes_preserved": True,
        "target_preserve_template_supernodes": int(template.num_supernodes),
        "num_target_nodes": int(len(state.target_nodes)),
        "num_support_nodes": int(len(state.support_nodes)),
        "num_support_candidates_scored": int(evaluated),
        "num_support_candidates_rejected_by_purity": int(rejected_purity),
        "num_support_candidates_rejected_by_constraints": int(rejected_constraints),
        "selected_support_merges": int(len(selected_pairs)),
        "selected_merges_by_source": dict(assignment.diagnostics.get("selected_merges_by_source", {})),
        "target_spec_error": target_conditioned_response_error(original, assignment, state, cfg),
        "relation_response_error": relation_response_error(original, assignment, state, cfg),
        "support_coverage_error": float(
            np.mean(
                [
                    delta_support_coverage_for_merge(int(u), int(v), state, cfg)
                    for u, v in selected_pairs
                ]
            )
            if len(selected_pairs)
            else 0.0
        ),
        "support_purity_error": float(
            np.mean(
                [
                    delta_support_purity_for_merge(int(u), int(v), state, cfg)
                    for u, v in selected_pairs
                ]
            )
            if len(selected_pairs)
            else 0.0
        ),
    }
    return SupportCompressedGraph(graph=coarse, assignment=assignment, diagnostics=diagnostics)
