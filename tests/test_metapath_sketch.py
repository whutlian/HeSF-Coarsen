import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema
from hesf_coarsen.sketch.metapath import compute_metapath_sketch


def _author_paper_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1, 1, 2], dtype=np.int64),
            dst=np.array([3, 3, 4, 4], dtype=np.int64),
            weight=None,
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([3, 3, 4, 4], dtype=np.int64),
            dst=np.array([0, 1, 1, 2], dtype=np.int64),
            weight=None,
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
    }
    specs = {
        0: RelationSpec(0, "author__writes__paper", 0, 1),
        1: RelationSpec(1, "paper__written_by__author", 1, 0),
    }
    graph = HeteroGraph(5, node_type, relations, specs)
    validate_schema(graph)
    return graph


def test_metapath_sketch_is_type_restricted_and_deterministic():
    graph = _author_paper_graph()
    config = {
        "metapath_sketch": {
            "enabled": True,
            "dim": 4,
            "seed": 123,
            "row_normalize": True,
            "paths": [
                {
                    "name": "APA",
                    "start_type": 0,
                    "end_type": 0,
                    "steps": [
                        {"relation_id": 0, "direction": "forward"},
                        {"relation_id": 0, "direction": "backward"},
                    ],
                }
            ],
        }
    }

    first = compute_metapath_sketch(graph, config)
    second = compute_metapath_sketch(graph, config)

    assert first.sketch.shape == (graph.num_nodes, 4)
    assert np.allclose(first.sketch, second.sketch)
    assert np.any(np.linalg.norm(first.sketch[graph.node_type == 0], axis=1) > 0)
    assert np.allclose(first.sketch[graph.node_type == 1], 0.0)
    assert first.diagnostics["enabled"] is True
    assert first.diagnostics["num_paths"] == 1
    assert first.diagnostics["paths"][0]["name"] == "APA"


def test_metapath_sketch_resolves_type_names_from_relation_schema():
    graph = _author_paper_graph()
    config = {
        "metapath_sketch": {
            "enabled": True,
            "dim": 2,
            "seed": 321,
            "row_normalize": True,
            "paths": [
                {
                    "name": "APA",
                    "start_type": "author",
                    "end_type": "author",
                    "steps": [
                        {"relation_id": 0, "direction": "forward"},
                        {"relation_id": 0, "direction": "backward"},
                    ],
                }
            ],
        }
    }

    result = compute_metapath_sketch(graph, config)

    assert result.sketch.shape == (graph.num_nodes, 2)
    assert np.any(np.linalg.norm(result.sketch[graph.node_type == 0], axis=1) > 0)
    assert np.allclose(result.sketch[graph.node_type == 1], 0.0)
    assert result.diagnostics["paths"][0]["start_type_name"] == "author"
    assert result.diagnostics["paths"][0]["end_type_name"] == "author"
