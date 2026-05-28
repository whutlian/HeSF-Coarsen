from __future__ import annotations

import numpy as np
import pytest

from hesf_coarsen.eval.official.safe_feature_transforms import transform_gate21_6_graph_features
from hesf_coarsen.io.schema import HeteroGraph


def _tiny_dblp_graph() -> HeteroGraph:
    node_type = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int32)
    return HeteroGraph(
        num_nodes=6,
        node_type=node_type,
        relations={},
        features={
            0: np.ones((2, 3), dtype=np.float32),
            1: np.ones((2, 4), dtype=np.float32),
            2: np.ones((1, 2), dtype=np.float32),
            3: np.zeros((1, 0), dtype=np.float32),
        },
    )


def test_zero_support_feature_transform_preserves_all_type_dimensions() -> None:
    graph = _tiny_dblp_graph()
    transformed = transform_gate21_6_graph_features(graph, "zero-all-support-preserve-dim")

    before = transformed._gate21_6_transform_audit["original_feature_shape_by_type"]
    after = transformed._gate21_6_transform_audit["transformed_feature_shape_by_type"]

    assert before == after
    assert transformed._gate21_6_transform_audit["shape_preserved_by_type"] == {
        "author": True,
        "paper": True,
        "term": True,
        "venue": True,
    }


def test_dimension_injection_requires_explicit_inject_prefix() -> None:
    with pytest.raises(ValueError, match="inject-"):
        transform_gate21_6_graph_features(_tiny_dblp_graph(), "zero-paper-dim64")
