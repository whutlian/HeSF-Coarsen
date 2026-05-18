import numpy as np

from hesf_coarsen.accuracy.target_support_hybrid import build_support_coarsened_hybrid
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _graph() -> HeteroGraph:
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([2, 3], dtype=np.int64),
            weight=None,
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([2, 3, 4], dtype=np.int64),
            dst=np.array([5, 5, 5], dtype=np.int64),
            weight=None,
            src_type=1,
            dst_type=2,
            relation_id=1,
        ),
    }
    return HeteroGraph(
        num_nodes=6,
        node_type=np.array([0, 0, 1, 1, 1, 2], dtype=np.int32),
        relations=relations,
        relation_specs={
            0: RelationSpec(0, "target__uses__support", 0, 1),
            1: RelationSpec(1, "support__links__context", 1, 2),
        },
    )


def test_hybrid_keeps_targets_and_coarsens_support_schema() -> None:
    graph = _graph()
    base = Assignment(
        np.array([0, 1, 2, 2, 3, 4], dtype=np.int64),
        np.array([0, 0, 1, 1, 2], dtype=np.int32),
    )

    hybrid = build_support_coarsened_hybrid(graph, base, target_node_type=0)

    assert hybrid.diagnostics["target_identity"] is True
    assert hybrid.diagnostics["support_coarsened"] is True
    assert set(hybrid.graph.relation_specs) == set(graph.relation_specs)
    assert hybrid.graph.node_type[hybrid.assignment.assignment[0]] == 0
    assert hybrid.graph.node_type[hybrid.assignment.assignment[1]] == 0
