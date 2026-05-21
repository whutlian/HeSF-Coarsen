from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

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
from hesf_coarsen.task_first.scoring import task_first_delta_distribution
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.stateful_matching import run_stateful_signature_matching
from hesf_coarsen.task_first.support_coverage import (
    coverage_components_for_merge,
    coverage_v2_components_for_merge,
    delta_support_coverage_for_merge,
)
from hesf_coarsen.task_first.support_purity import (
    delta_support_purity_for_merge,
    merge_is_purity_allowed,
    purity_v2_diagnostics,
    support_purity_pair_kind,
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


def task_first_support_merge_budget(graph: HeteroGraph, cfg: TaskFirstConfig) -> dict:
    target_count = int(np.sum(graph.node_type == int(cfg.target_node_type)))
    support_count = int(graph.num_nodes - target_count)
    requested_ratio = cfg.target_ratio
    requested_support_ratio = cfg.support_ratio
    explicit_merges = cfg.max_support_merges
    desired_total_nodes = int(graph.num_nodes)
    desired_support_nodes = support_count
    infeasible = False
    if explicit_merges is not None:
        max_merges = max(0, min(support_count, int(explicit_merges)))
        desired_support_nodes = int(support_count - max_merges)
        desired_total_nodes = int(target_count + desired_support_nodes)
    elif requested_support_ratio is not None:
        ratio = min(max(float(requested_support_ratio), 0.0), 1.0)
        desired_support_nodes = max(0, min(support_count, int(ceil(support_count * ratio - 1.0e-12))))
        max_merges = int(support_count - desired_support_nodes)
        desired_total_nodes = int(target_count + desired_support_nodes)
    elif requested_ratio is not None:
        ratio = min(max(float(requested_ratio), 0.0), 1.0)
        requested_total = max(0, min(int(graph.num_nodes), int(ceil(graph.num_nodes * ratio - 1.0e-12))))
        infeasible = requested_total < target_count
        desired_total_nodes = max(target_count, requested_total)
        desired_support_nodes = max(0, min(support_count, desired_total_nodes - target_count))
        max_merges = int(support_count - desired_support_nodes)
    else:
        max_merges = None
    realized_floor_ratio = float(target_count / max(int(graph.num_nodes), 1))
    return {
        "target_nodes": target_count,
        "support_nodes": support_count,
        "requested_target_ratio": None if requested_ratio is None else float(requested_ratio),
        "requested_support_ratio": None if requested_support_ratio is None else float(requested_support_ratio),
        "requested_ratio_infeasible": bool(infeasible),
        "target_preserve_floor_ratio": realized_floor_ratio,
        "desired_total_nodes": int(desired_total_nodes),
        "desired_support_nodes": int(desired_support_nodes),
        "max_support_merges": None if max_merges is None else int(max_merges),
    }


def task_first_budget_stop_reason(
    *,
    current_support_nodes: int,
    desired_support_nodes: int,
    max_support_merges: int | None,
    candidate_pair_count: int,
    eligible_candidate_pair_count: int,
    selected_support_merges: int,
    max_levels_reached: bool = False,
) -> tuple[str, str]:
    if current_support_nodes <= desired_support_nodes:
        return "reached_requested_support_ratio", ""
    if max_levels_reached:
        return "max_levels_reached", "multilevel loop hit configured max_levels"
    if max_support_merges is not None and int(max_support_merges) <= 0:
        return "merge_budget_floor", "computed max_support_merges <= 0"
    if candidate_pair_count <= 0:
        return "candidate_exhaustion", "candidate source emitted no pairs"
    if eligible_candidate_pair_count <= 0:
        return "constraint_blocked_all_candidates", "purity/support constraints rejected all candidates"
    if selected_support_merges <= 0:
        return "no_selected_merges", "greedy cluster selected zero merges"
    return "not_stopped", ""


def _greedy_cluster_config(graph: HeteroGraph, cfg: TaskFirstConfig) -> dict:
    budget = task_first_support_merge_budget(graph, cfg)
    coarsening = {
        "coarsening": {
            "matching_method": "greedy_cluster",
            "same_type_only": bool(cfg.same_type_only),
            "same_partition_only": bool(cfg.same_partition_only),
            "max_cluster_size": 4,
        }
    }
    if budget["max_support_merges"] is not None:
        coarsening["coarsening"]["max_matched_pairs"] = int(budget["max_support_merges"])
    return coarsening


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
    budget = task_first_support_merge_budget(original, cfg)
    if str(cfg.scoring.pair_delta_mode).lower() == "stateful_signature":
        result = run_stateful_signature_matching(
            original,
            base_candidates,
            state,
            cfg,
            max_support_merges=None if budget.get("max_support_merges") is None else int(budget["max_support_merges"]),
        )
        assignment = result.assignment
        coarse = coarsen_graph(original, assignment)
        selected_pairs = np.asarray(result.selected_pairs, dtype=np.int64).reshape(-1, 2)
        candidate_pair_count = int(getattr(base_candidates, "pair_count", lambda: 0)())
        stop_reason, floor_reason = task_first_budget_stop_reason(
            current_support_nodes=int(np.sum(original.node_type != int(cfg.target_node_type))),
            desired_support_nodes=int(budget.get("desired_support_nodes", 0)),
            max_support_merges=None if budget.get("max_support_merges") is None else int(budget["max_support_merges"]),
            candidate_pair_count=candidate_pair_count,
            eligible_candidate_pair_count=candidate_pair_count,
            selected_support_merges=int(len(selected_pairs)),
        )
        cfg_v1 = replace(cfg, support_coverage=replace(cfg.support_coverage, mode="coverage_v1_legacy"))
        cfg_v2 = replace(cfg, support_coverage=replace(cfg.support_coverage, mode="coverage_v2"))
        cfg_purity_v1 = replace(cfg, support_purity=replace(cfg.support_purity, zero_policy="zero_as_no_conflict"))
        cfg_purity_v2 = replace(cfg, support_purity=replace(cfg.support_purity, zero_policy="purity_v2"))
        coverage_v2_components = [
            coverage_v2_components_for_merge(int(u), int(v), state, cfg_v2)
            for u, v in selected_pairs
        ]
        diagnostics = {
            "matching_method": "stateful_signature_v1",
            "pipeline_steps": list(PIPELINE_STEPS) + ["stateful_signature_matching"],
            **budget,
            "target_node_type": int(cfg.target_node_type),
            "target_nodes_preserved": True,
            "target_preserve_template_supernodes": int(template.num_supernodes),
            "num_target_nodes": int(len(state.target_nodes)),
            "num_support_nodes": int(len(state.support_nodes)),
            "num_support_candidates_scored": int(candidate_pair_count),
            "num_support_candidates_rejected_by_purity": 0,
            "num_support_candidates_rejected_by_constraints": 0,
            "selected_support_merges": int(len(selected_pairs)),
            "selected_pair_keys": [f"{int(u)}-{int(v)}" for u, v in selected_pairs],
            "selected_merges_by_source": {},
            "candidate_pair_count": candidate_pair_count,
            "eligible_candidate_pair_count": candidate_pair_count,
            "stop_reason": stop_reason,
            "floor_reason": floor_reason,
            "stateful_approx_status": "implemented_as_stateful_signature_v1",
            **result.diagnostics,
            **purity_v2_diagnostics(state),
            "target_spec_error": target_conditioned_response_error(original, assignment, state, cfg),
            "relation_response_error": relation_response_error(original, assignment, state, cfg),
            "coverage_v1_error": float(np.mean([delta_support_coverage_for_merge(int(u), int(v), state, cfg_v1) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
            "coverage_v2_error": float(np.mean([delta_support_coverage_for_merge(int(u), int(v), state, cfg_v2) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
            "anchor_collision_rate": float(np.mean([item["anchor_distribution_collision"] > 1.0e-12 for item in coverage_v2_components])) if coverage_v2_components else 0.0,
            "class_context_collision_rate": float(np.mean([item["class_context_collision"] > 1.0e-12 for item in coverage_v2_components])) if coverage_v2_components else 0.0,
            "receptive_field_diversity_loss": float(np.mean([item["receptive_field_diversity_loss"] for item in coverage_v2_components])) if coverage_v2_components else 0.0,
            "purity_v1_error": float(np.mean([delta_support_purity_for_merge(int(u), int(v), state, cfg_purity_v1) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
            "purity_v2_error": float(np.mean([delta_support_purity_for_merge(int(u), int(v), state, cfg_purity_v2) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
        }
        diagnostics["support_coverage_error"] = diagnostics["coverage_v2_error"]
        diagnostics["support_purity_error"] = diagnostics["purity_v2_error"]
        return SupportCompressedGraph(graph=coarse, assignment=assignment, diagnostics=diagnostics)
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
        _greedy_cluster_config(original, cfg),
        partition_id=original.partitions,
        source_lookup=source_lookup if callable(source_lookup) else None,
    )
    assignment = _target_first_assignment_from_support_clusters(original, raw_assignment, template, cfg)
    coarse = coarsen_graph(original, assignment)
    selected_pairs = assignment.diagnostics.get("_selected_merge_pairs", np.empty((0, 2), dtype=np.int64))
    selected_set = {tuple(sorted((int(u), int(v)))) for u, v in np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)}
    selected_deltas = [
        delta
        for pair, delta in zip(scored_pairs, normalized)
        if tuple(sorted((int(pair[0]), int(pair[1])))) in selected_set
    ]
    selected_raw = [
        delta
        for pair, delta in zip(scored_pairs, deltas)
        if tuple(sorted((int(pair[0]), int(pair[1])))) in selected_set
    ]
    coverage_components = [
        coverage_components_for_merge(int(u), int(v), state, cfg)
        for u, v in np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)
    ]
    cfg_v1 = replace(cfg, support_coverage=replace(cfg.support_coverage, mode="coverage_v1_legacy"))
    cfg_v2 = replace(cfg, support_coverage=replace(cfg.support_coverage, mode="coverage_v2"))
    cfg_purity_v1 = replace(cfg, support_purity=replace(cfg.support_purity, zero_policy="zero_as_no_conflict"))
    cfg_purity_v2 = replace(cfg, support_purity=replace(cfg.support_purity, zero_policy="purity_v2"))
    coverage_v2_components = [
        coverage_v2_components_for_merge(int(u), int(v), state, cfg_v2)
        for u, v in np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)
    ]
    purity_kind_counts: dict[str, int] = {}
    for u, v in np.asarray(scored_pairs, dtype=np.int64).reshape(-1, 2) if scored_pairs else np.empty((0, 2), dtype=np.int64):
        kind = support_purity_pair_kind(int(u), int(v), state)
        purity_kind_counts[f"candidate_{kind}_count"] = purity_kind_counts.get(f"candidate_{kind}_count", 0) + 1
    selected_purity_kind_counts: dict[str, int] = {}
    for u, v in np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2):
        kind = support_purity_pair_kind(int(u), int(v), state)
        selected_purity_kind_counts[f"selected_{kind}_count"] = selected_purity_kind_counts.get(f"selected_{kind}_count", 0) + 1
    candidate_pair_count = int(getattr(base_candidates, "pair_count", lambda: 0)())
    stop_reason, floor_reason = task_first_budget_stop_reason(
        current_support_nodes=int(np.sum(original.node_type != int(cfg.target_node_type))),
        desired_support_nodes=int(budget.get("desired_support_nodes", 0)),
        max_support_merges=None if budget.get("max_support_merges") is None else int(budget["max_support_merges"]),
        candidate_pair_count=candidate_pair_count,
        eligible_candidate_pair_count=int(evaluated),
        selected_support_merges=int(len(selected_pairs)),
    )
    diagnostics = {
        "matching_method": "greedy_cluster",
        "pipeline_steps": list(PIPELINE_STEPS),
        **budget,
        "target_node_type": int(cfg.target_node_type),
        "target_nodes_preserved": True,
        "target_preserve_template_supernodes": int(template.num_supernodes),
        "num_target_nodes": int(len(state.target_nodes)),
        "num_support_nodes": int(len(state.support_nodes)),
        "num_support_candidates_scored": int(evaluated),
        "num_support_candidates_rejected_by_purity": int(rejected_purity),
        "num_support_candidates_rejected_by_constraints": int(rejected_constraints),
        "selected_support_merges": int(len(selected_pairs)),
        "selected_pair_keys": [f"{int(u)}-{int(v)}" for u, v in np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)],
        "selected_merges_by_source": dict(assignment.diagnostics.get("selected_merges_by_source", {})),
        "candidate_pair_count": candidate_pair_count,
        "eligible_candidate_pair_count": int(evaluated),
        "stop_reason": stop_reason,
        "floor_reason": floor_reason,
        "stateful_approx_status": "not_implemented",
        **purity_v2_diagnostics(state),
        **task_first_delta_distribution(deltas, normalized, cfg),
        **{f"selected_{key}": value for key, value in task_first_delta_distribution(selected_raw, selected_deltas, cfg).items()},
        **purity_kind_counts,
        **selected_purity_kind_counts,
        "coverage_same_anchor_loss_mean": float(np.mean([item["same_anchor_loss"] for item in coverage_components])) if coverage_components else 0.0,
        "coverage_cross_anchor_collision_loss_mean": float(np.mean([item["cross_anchor_collision_loss"] for item in coverage_components])) if coverage_components else 0.0,
        "coverage_class_context_collision_loss_mean": float(np.mean([item["class_context_collision_loss"] for item in coverage_components])) if coverage_components else 0.0,
        "coverage_zero_delta_pair_share": float(np.mean([delta.delta_support_coverage <= 1.0e-12 for delta in normalized])) if normalized else 0.0,
        "coverage_positive_delta_pair_share": float(np.mean([delta.delta_support_coverage > 1.0e-12 for delta in normalized])) if normalized else 0.0,
        "selected_cross_anchor_collision_share": float(np.mean([item["cross_anchor_collision_loss"] > 1.0e-12 for item in coverage_components])) if coverage_components else 0.0,
        "selected_high_context_collision_share": float(np.mean([item["class_context_collision_loss"] > 0.25 for item in coverage_components])) if coverage_components else 0.0,
        "coverage_v1_error": float(np.mean([delta_support_coverage_for_merge(int(u), int(v), state, cfg_v1) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
        "coverage_v2_error": float(np.mean([delta_support_coverage_for_merge(int(u), int(v), state, cfg_v2) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
        "anchor_collision_rate": float(np.mean([item["anchor_distribution_collision"] > 1.0e-12 for item in coverage_v2_components])) if coverage_v2_components else 0.0,
        "class_context_collision_rate": float(np.mean([item["class_context_collision"] > 1.0e-12 for item in coverage_v2_components])) if coverage_v2_components else 0.0,
        "receptive_field_diversity_loss": float(np.mean([item["receptive_field_diversity_loss"] for item in coverage_v2_components])) if coverage_v2_components else 0.0,
        "purity_v1_error": float(np.mean([delta_support_purity_for_merge(int(u), int(v), state, cfg_purity_v1) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
        "purity_v2_error": float(np.mean([delta_support_purity_for_merge(int(u), int(v), state, cfg_purity_v2) for u, v in selected_pairs])) if len(selected_pairs) else 0.0,
        "known_footprint_count": int(np.count_nonzero(getattr(state, "support_footprint_states", np.empty(0)) == 0)),
        "unknown_target_connected_count": int(np.count_nonzero(getattr(state, "support_footprint_states", np.empty(0)) == 1)),
        "unknown_isolated_count": int(np.count_nonzero(getattr(state, "support_footprint_states", np.empty(0)) == 2)),
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
