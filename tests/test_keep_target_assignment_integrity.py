import numpy as np

from hesf_coarsen.accuracy.target_support_hybrid import build_support_coarsened_hybrid
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def test_keep_target_hybrid_preserves_every_target_and_compresses_support() -> None:
    graph = HeteroGraph(
        num_nodes=6,
        node_type=np.array([0, 0, 1, 1, 1, 2], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1], dtype=np.int64),
                dst=np.array([2, 3], dtype=np.int64),
                weight=None,
                src_type=0,
                dst_type=1,
                relation_id=0,
            ),
        },
        relation_specs={0: RelationSpec(0, "target__to__support", 0, 1)},
        labels=np.array([0, 1, -1, -1, -1, -1], dtype=np.int64),
    )
    base = Assignment(
        np.array([0, 1, 2, 2, 3, 4], dtype=np.int64),
        np.array([0, 0, 1, 1, 2], dtype=np.int32),
    )

    hybrid = build_support_coarsened_hybrid(graph, base, target_node_type=0)

    assert hybrid.diagnostics["target_identity"] is True
    assert hybrid.diagnostics["target_cluster_size_max"] == 1
    assert hybrid.diagnostics["support_coarsened"] is True
    assert hybrid.graph.num_nodes < graph.num_nodes
