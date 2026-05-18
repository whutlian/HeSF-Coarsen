from __future__ import annotations

from dataclasses import dataclass

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.target_preserve import (
    build_target_preserving_assignment,
    target_preservation_report,
)
from hesf_coarsen.io.schema import HeteroGraph


@dataclass(frozen=True)
class HybridGraph:
    graph: HeteroGraph
    assignment: Assignment
    diagnostics: dict


def build_support_coarsened_hybrid(
    original: HeteroGraph,
    base_assignment: Assignment,
    *,
    target_node_type: int,
) -> HybridGraph:
    assignment = build_target_preserving_assignment(
        original,
        base_assignment,
        target_node_type=int(target_node_type),
    )
    hybrid = coarsen_graph(original, assignment)
    return HybridGraph(
        graph=hybrid,
        assignment=assignment,
        diagnostics=target_preservation_report(
            original,
            assignment,
            target_node_type=int(target_node_type),
        ),
    )
