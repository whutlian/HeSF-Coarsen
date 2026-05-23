from __future__ import annotations

from typing import Mapping

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.units.base import SupportUnit, build_unit_structure_index, unit_structure_from_index


def extract_typedhash_units(
    graph: HeteroGraph,
    typedhash_assignment: np.ndarray,
    target_type: int,
    labels: np.ndarray | None = None,
    splits: Mapping[str, np.ndarray] | None = None,
) -> list[SupportUnit]:
    assignment = np.asarray(typedhash_assignment, dtype=np.int64).reshape(-1)
    support_nodes = np.flatnonzero(np.asarray(graph.node_type) != int(target_type)).astype(np.int64)
    groups: dict[int, list[int]] = {}
    for node in support_nodes.tolist():
        if int(node) < len(assignment):
            groups.setdefault(int(assignment[int(node)]), []).append(int(node))
    index = build_unit_structure_index(graph, target_type=int(target_type), labels=labels, splits=splits)
    return [
        unit_structure_from_index(
            graph,
            members,
            source="TypedHash",
            unit_id=cluster_id,
            index=index,
            metadata={"assignment_cluster_id": int(cluster_id), "unit_family": "TypedHash"},
        )
        for cluster_id, members in sorted(groups.items())
        if members
    ]
