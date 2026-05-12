from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph


_REQUIRED_DIAGNOSTIC_FIELDS = (
    "candidate_count_total",
    "candidate_count_max",
    "candidate_count_mean",
    "candidate_count_quantiles",
    "matched_pairs",
    "singleton_ratio",
)


def _relation_weight_sums(graph: HeteroGraph) -> dict[int, float]:
    return {
        int(relation_id): float(np.asarray(rel.weight, dtype=np.float64).sum())
        for relation_id, rel in graph.relations.items()
    }


def _count_schema_type_violations(graph: HeteroGraph) -> int:
    violations = 0
    if graph.node_type.shape != (graph.num_nodes,):
        violations += 1
    if graph.num_nodes < 0:
        violations += 1
    if np.any(graph.node_type < 0):
        violations += int(np.sum(graph.node_type < 0))

    for rel in graph.relations.values():
        if rel.src.shape != rel.dst.shape or rel.src.shape != rel.weight.shape:
            violations += 1
            continue
        if rel.num_edges == 0:
            continue
        src_in_bounds = (rel.src >= 0) & (rel.src < graph.num_nodes)
        dst_in_bounds = (rel.dst >= 0) & (rel.dst < graph.num_nodes)
        violations += int(np.sum(~src_in_bounds) + np.sum(~dst_in_bounds))
        valid = src_in_bounds & dst_in_bounds
        if np.any(valid):
            violations += int(np.sum(graph.node_type[rel.src[valid]] != rel.src_type))
            violations += int(np.sum(graph.node_type[rel.dst[valid]] != rel.dst_type))
    return int(violations)


def _count_invalid_assignments(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
) -> int:
    if assignment.assignment.shape != (original.num_nodes,):
        return int(abs(len(assignment.assignment) - original.num_nodes) + original.num_nodes)

    invalid = 0
    mapped = assignment.assignment
    in_assignment_range = (mapped >= 0) & (mapped < assignment.num_supernodes)
    invalid += int(np.sum(~in_assignment_range))

    if assignment.supernode_type.shape != (assignment.num_supernodes,):
        invalid += 1
    if assignment.num_supernodes != coarse.num_nodes:
        invalid += abs(int(assignment.num_supernodes) - int(coarse.num_nodes))

    in_coarse_range = (mapped >= 0) & (mapped < coarse.num_nodes)
    valid = in_assignment_range & in_coarse_range
    if np.any(valid):
        assigned_types = assignment.supernode_type[mapped[valid]]
        coarse_types = coarse.node_type[mapped[valid]]
        original_types = original.node_type[valid]
        invalid += int(np.sum(assigned_types != original_types))
        invalid += int(np.sum(coarse_types != original_types))
    return int(invalid)


def _count_relation_schema_violations(coarse: HeteroGraph) -> int:
    violations = 0
    if set(coarse.relations) != set(coarse.relation_specs):
        violations += len(set(coarse.relations) ^ set(coarse.relation_specs))

    for relation_id, rel in coarse.relations.items():
        spec = coarse.relation_specs.get(relation_id)
        if spec is None:
            violations += 1
        else:
            violations += int(rel.relation_id != relation_id)
            violations += int(spec.relation_id != relation_id)
            violations += int(rel.src_type != spec.src_type)
            violations += int(rel.dst_type != spec.dst_type)
        if rel.src.shape != rel.dst.shape or rel.src.shape != rel.weight.shape:
            violations += 1
            continue
        if rel.num_edges == 0:
            continue
        in_bounds = (
            (rel.src >= 0)
            & (rel.src < coarse.num_nodes)
            & (rel.dst >= 0)
            & (rel.dst < coarse.num_nodes)
        )
        violations += int(np.sum(~in_bounds))
        if np.any(in_bounds):
            violations += int(np.sum(coarse.node_type[rel.src[in_bounds]] != rel.src_type))
            violations += int(np.sum(coarse.node_type[rel.dst[in_bounds]] != rel.dst_type))
    return int(violations)


def _load_diagnostics(
    diagnostics_path: str | Path | None,
    diagnostics: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, int]:
    if diagnostics is not None:
        return diagnostics, 0
    if diagnostics_path is None:
        return None, len(_REQUIRED_DIAGNOSTIC_FIELDS)

    path = Path(diagnostics_path)
    if not path.exists():
        return None, len(_REQUIRED_DIAGNOSTIC_FIELDS)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, len(_REQUIRED_DIAGNOSTIC_FIELDS)
    if not isinstance(data, dict):
        return None, len(_REQUIRED_DIAGNOSTIC_FIELDS)
    return data, 0


def _count_missing_diagnostics(data: dict[str, Any] | None, initial_missing: int) -> int:
    if data is None:
        return int(initial_missing)
    missing = 0
    for field in _REQUIRED_DIAGNOSTIC_FIELDS:
        if field not in data:
            missing += 1
    quantiles = data.get("candidate_count_quantiles")
    if not isinstance(quantiles, dict):
        missing += 1
    else:
        for field in ("p50", "p95", "p99"):
            if field not in quantiles:
                missing += 1
    return int(missing)


def validate_level_invariants(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    diagnostics_path: str | Path | None = None,
    diagnostics: dict[str, Any] | None = None,
    weight_tol: float = 1e-5,
) -> dict[str, Any]:
    """Validate one coarsening level and report counts instead of raising."""

    original_weights = _relation_weight_sums(original)
    coarse_weights = _relation_weight_sums(coarse)
    expected_relations = {
        relation_id
        for relation_id, rel in original.relations.items()
        if rel.num_edges > 0
    }
    missing_relations = expected_relations - set(coarse.relations)
    weight_errors = {
        relation_id: abs(original_weights.get(relation_id, 0.0) - coarse_weights.get(relation_id, 0.0))
        for relation_id in sorted(set(original_weights) | set(coarse_weights))
    }
    max_weight_error = float(max(weight_errors.values(), default=0.0))

    diagnostics_data, initial_missing = _load_diagnostics(diagnostics_path, diagnostics)
    diagnostics_missing_count = _count_missing_diagnostics(diagnostics_data, initial_missing)

    return {
        "schema_type_violations": _count_schema_type_violations(original)
        + _count_schema_type_violations(coarse),
        "invalid_assignment_count": _count_invalid_assignments(original, coarse, assignment),
        "relation_schema_violations": _count_relation_schema_violations(coarse),
        "diagnostics_missing_count": diagnostics_missing_count,
        "relation_weight_preservation_max_error": max_weight_error,
        "relation_weight_preservation_violation": int(max_weight_error > float(weight_tol)),
        "coarse_node_count_violation": int(coarse.num_nodes > original.num_nodes),
        "missing_relation_count": int(len(missing_relations)),
    }
