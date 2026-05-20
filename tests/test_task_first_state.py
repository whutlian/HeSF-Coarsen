import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.probes import target_conditioned_response_error
from hesf_coarsen.task_first.state import build_task_first_state


def make_target_support_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    relations = {
        0: RelationAdj(
            src=np.array([0, 0, 1], dtype=np.int64),
            dst=np.array([2, 3, 4], dtype=np.int64),
            weight=np.ones(3, dtype=np.float32),
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([2, 3, 4], dtype=np.int64),
            dst=np.array([0, 0, 1], dtype=np.int64),
            weight=np.ones(3, dtype=np.float32),
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
    }
    specs = {
        0: RelationSpec(0, "target_to_support", 0, 1),
        1: RelationSpec(1, "support_to_target", 1, 0),
    }
    features = {
        0: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        1: np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]], dtype=np.float32),
    }
    labels = np.array([0, 1, -1, -1, -1], dtype=np.int64)
    partitions = np.zeros(5, dtype=np.int32)
    graph = HeteroGraph(
        num_nodes=5,
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
        features=features,
        labels=labels,
        partitions=partitions,
    )
    validate_schema(graph)
    return graph


def test_build_task_first_state_uses_target_train_labels_only():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)

    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert state.target_nodes.tolist() == [0, 1]
    assert state.support_nodes.tolist() == [2, 3, 4]
    assert state.train_target_nodes.tolist() == [0, 1]
    np.testing.assert_allclose(
        state.target_seed_matrix,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert set(state.target_filter_responses) == {0.25, 1.0, 4.0}
    assert state.support_class_footprints.shape == (graph.num_nodes, 2)
    assert state.support_class_footprints[2, 0] > 0.0
    assert state.support_class_footprints[4, 1] > 0.0


def test_target_conditioned_response_error_identity_zero_and_merge_increases():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)
    identity = Assignment(np.arange(graph.num_nodes), graph.node_type.copy())
    merged = Assignment(
        assignment=np.array([0, 1, 2, 3, 2], dtype=np.int64),
        supernode_type=np.array([0, 0, 1, 1], dtype=np.int32),
    )

    assert target_conditioned_response_error(graph, identity, state, cfg) < 1.0e-8
    assert target_conditioned_response_error(graph, merged, state, cfg) > 0.0
