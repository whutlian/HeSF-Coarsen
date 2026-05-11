import json

import numpy as np
import pytest

from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph, save_graph
from hesf_coarsen.io.schema import validate_schema
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch


def test_synthetic_graph_validates_and_round_trips(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=7,
    )

    validate_schema(graph)
    save_graph(graph, tmp_path)
    loaded = load_graph(tmp_path)

    validate_schema(loaded)
    assert loaded.num_nodes == graph.num_nodes
    assert np.array_equal(loaded.node_type, graph.node_type)
    assert set(loaded.relations) == {0, 1, 2, 3, 4}
    assert set(loaded.relation_specs) == set(graph.relation_specs)


def test_schema_validation_rejects_wrong_endpoint_type():
    graph = generate_synthetic_graph(
        num_users=4,
        num_items=3,
        num_tags=2,
        seed=11,
    )
    rel = graph.relations[0]
    rel.dst[0] = 0

    with pytest.raises(ValueError, match="destination endpoint"):
        validate_schema(graph)


def test_lowpass_sketch_is_seeded_and_has_configured_dtype():
    graph = generate_synthetic_graph(
        num_users=6,
        num_items=4,
        num_tags=3,
        seed=3,
    )
    config = dict(DEFAULT_CONFIG)
    config["sketch"] = dict(DEFAULT_CONFIG["sketch"], dim=8, order=2, dtype="float16")

    z1 = compute_lowpass_sketch(graph, config)
    z2 = compute_lowpass_sketch(graph, config)

    assert z1.shape == (graph.num_nodes, 8)
    assert z1.dtype == np.float16
    assert np.array_equal(z1, z2)


def test_cli_diagnostics_file_shape(tmp_path):
    graph = generate_synthetic_graph(
        num_users=5,
        num_items=4,
        num_tags=2,
        seed=19,
    )
    save_graph(graph, tmp_path)

    schema_path = tmp_path / "schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    assert "relations" in schema
    assert len(schema["relations"]) == 5
