import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema


def test_edge_aggregation_preserves_relation_weight_and_schema():
    node_type = np.array([0, 0, 1], dtype=np.int32)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([2, 2], dtype=np.int64),
            weight=np.array([1.5, 2.5], dtype=np.float32),
            src_type=0,
            dst_type=1,
            relation_id=0,
        )
    }
    specs = {0: RelationSpec(0, "user_to_item", 0, 1)}
    features = {
        0: np.array([[1.0, 0.0], [3.0, 2.0]], dtype=np.float32),
        1: np.array([[10.0]], dtype=np.float32),
    }
    graph = HeteroGraph(
        num_nodes=3,
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
        features=features,
    )
    assignment = Assignment(
        assignment=np.array([0, 0, 1], dtype=np.int64),
        supernode_type=np.array([0, 1], dtype=np.int32),
    )

    coarse = coarsen_graph(graph, assignment)

    validate_schema(coarse)
    rel = coarse.relations[0]
    assert coarse.num_nodes == 2
    assert rel.src.tolist() == [0]
    assert rel.dst.tolist() == [1]
    assert np.isclose(rel.weight.sum(), 4.0)
    assert coarse.relation_specs[0].src_type == 0
    assert coarse.relation_specs[0].dst_type == 1
    assert np.allclose(coarse.features[0][0], [2.0, 1.0])
