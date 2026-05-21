import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def make_gate15_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    labels = np.array([0, 1, 0, 1, -1, -1, -1, -1], dtype=np.int64)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1, 2, 3], dtype=np.int64),
            dst=np.array([4, 4, 5, 5], dtype=np.int64),
            weight=None,
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([4, 4, 5, 5], dtype=np.int64),
            dst=np.array([0, 1, 2, 3], dtype=np.int64),
            weight=None,
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
        2: RelationAdj(
            src=np.array([4, 5], dtype=np.int64),
            dst=np.array([6, 7], dtype=np.int64),
            weight=None,
            src_type=1,
            dst_type=2,
            relation_id=2,
        ),
        3: RelationAdj(
            src=np.array([6, 7], dtype=np.int64),
            dst=np.array([4, 5], dtype=np.int64),
            weight=None,
            src_type=2,
            dst_type=1,
            relation_id=3,
        ),
    }
    relation_specs = {
        0: RelationSpec(0, "paper__to__author", 0, 1),
        1: RelationSpec(1, "author__to__paper", 1, 0),
        2: RelationSpec(2, "author__to__term", 1, 2),
        3: RelationSpec(3, "term__to__author", 2, 1),
    }
    features = {
        0: np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]],
            dtype=np.float32,
        ),
        1: np.array([[1.0, 0.2, 0.0], [0.1, 0.9, 0.3]], dtype=np.float32),
        2: np.array([[0.7], [0.4]], dtype=np.float32),
    }
    return HeteroGraph(
        num_nodes=8,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features=features,
        labels=labels,
    )


def split_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.array([True, True, False, False, False, False, False, False])
    val_mask = np.array([False, False, True, False, False, False, False, False])
    test_mask = np.array([False, False, False, True, False, False, False, False])
    return train_mask, val_mask, test_mask
