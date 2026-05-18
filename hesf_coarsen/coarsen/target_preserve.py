from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


def _resolve_type_id(graph: HeteroGraph, target_node_type: int | str) -> int:
    if isinstance(target_node_type, (int, np.integer)):
        type_id = int(target_node_type)
        if np.any(graph.node_type == type_id):
            return type_id
        raise ValueError(f"target node type {type_id} is not present")
    text = str(target_node_type)
    try:
        return _resolve_type_id(graph, int(text))
    except ValueError:
        pass
    name_to_type: dict[str, int] = {}
    for spec in graph.relation_specs.values():
        parts = str(spec.name).split("__")
        if len(parts) >= 3:
            name_to_type.setdefault(parts[0], int(spec.src_type))
            name_to_type.setdefault(parts[-1], int(spec.dst_type))
    if text in name_to_type:
        return _resolve_type_id(graph, name_to_type[text])
    raise ValueError(f"target node type {target_node_type!r} could not be resolved")


def build_target_preserving_assignment(
    graph: HeteroGraph,
    base_assignment: Assignment,
    *,
    target_node_type: int | str,
    preserve_nodes: Iterable[int] | None = None,
) -> Assignment:
    """Rewrite an all-type assignment so requested target nodes become singleton clusters.

    Support nodes reuse the base coarse grouping when it is type-compatible. The output
    assignment is dense and can be passed directly to ``coarsen_graph``.
    """

    target_type = _resolve_type_id(graph, target_node_type)
    if len(base_assignment.assignment) != graph.num_nodes:
        raise ValueError("base assignment must map every original node")

    target_nodes = nodes_of_type(graph, target_type)
    if preserve_nodes is None:
        preserve_set = set(int(node) for node in target_nodes.tolist())
    else:
        valid = set(int(node) for node in target_nodes.tolist())
        preserve_set = {int(node) for node in preserve_nodes if int(node) in valid}

    new_assignment = np.empty(graph.num_nodes, dtype=np.int64)
    new_types: list[int] = []

    for node in range(graph.num_nodes):
        if node in preserve_set:
            new_assignment[node] = len(new_types)
            new_types.append(int(graph.node_type[node]))

    support_key_to_supernode: dict[tuple[int, int], int] = {}
    for node in range(graph.num_nodes):
        if node in preserve_set:
            continue
        node_type = int(graph.node_type[node])
        key = (node_type, int(base_assignment.assignment[node]))
        supernode = support_key_to_supernode.get(key)
        if supernode is None:
            supernode = len(new_types)
            support_key_to_supernode[key] = supernode
            new_types.append(node_type)
        new_assignment[node] = supernode

    assignment = Assignment(
        assignment=new_assignment,
        supernode_type=np.asarray(new_types, dtype=np.int32),
        diagnostics={
            "target_preserve": {
                "enabled": True,
                "target_node_type": int(target_type),
                "preserved_target_nodes": int(len(preserve_set)),
                "support_supernodes": int(len(support_key_to_supernode)),
                "base_supernodes": int(base_assignment.num_supernodes),
            }
        },
    )
    _validate_same_type(graph, assignment)
    return assignment


def _validate_same_type(graph: HeteroGraph, assignment: Assignment) -> None:
    mapped_type = assignment.supernode_type[assignment.assignment]
    if not np.all(mapped_type == graph.node_type):
        bad = np.flatnonzero(mapped_type != graph.node_type)[:5].tolist()
        raise ValueError(f"target-preserving assignment has type mismatch at nodes {bad}")


def target_preservation_report(
    graph: HeteroGraph,
    assignment: Assignment,
    *,
    target_node_type: int | str,
) -> dict[str, object]:
    target_type = _resolve_type_id(graph, target_node_type)
    target_nodes = nodes_of_type(graph, target_type)
    support_nodes = np.flatnonzero(graph.node_type != target_type).astype(np.int64)
    sizes = assignment.cluster_sizes()
    target_supernodes = assignment.assignment[target_nodes]
    support_supernodes = assignment.assignment[support_nodes] if len(support_nodes) else np.array([], dtype=np.int64)
    target_identity = bool(
        len(np.unique(target_supernodes)) == len(target_nodes)
        and np.all(sizes[target_supernodes] == 1)
        and np.all(assignment.supernode_type[target_supernodes] == target_type)
    )
    per_type: dict[str, dict[str, float | int]] = {}
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        original = int(np.sum(graph.node_type == type_id))
        coarse = int(np.sum(assignment.supernode_type == type_id))
        per_type[str(type_id)] = {
            "original_nodes": original,
            "coarse_nodes": coarse,
            "ratio": float(coarse / max(original, 1)),
        }
    support_clusters: dict[int, list[int]] = defaultdict(list)
    for node in support_nodes.tolist():
        support_clusters[int(assignment.assignment[node])].append(int(node))
    support_coarsened = any(len(nodes) > 1 for nodes in support_clusters.values())
    return {
        "target_node_type": int(target_type),
        "target_identity": target_identity,
        "target_original_nodes": int(len(target_nodes)),
        "target_coarse_nodes": int(len(np.unique(target_supernodes))),
        "target_cluster_size_max": int(sizes[target_supernodes].max(initial=0)),
        "support_original_nodes": int(len(support_nodes)),
        "support_coarse_nodes": int(len(np.unique(support_supernodes))) if len(support_supernodes) else 0,
        "support_coarsened": bool(support_coarsened),
        "global_ratio": float(assignment.num_supernodes / max(graph.num_nodes, 1)),
        "per_type": per_type,
    }
