import numpy as np

from hesf_coarsen.cli.main import main
from hesf_coarsen.coarsen.aggregate_edges import (
    _merge_sorted_chunks,
    coarsen_graph,
    coarsen_graph_chunked,
)
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph, save_graph
from hesf_coarsen.io.memmap_csr import load_memmap_graph, save_memmap_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema
from hesf_coarsen.ops.torch_dense import (
    get_torch_device,
    torch_pairwise_squared_distance,
    torch_row_normalize,
    torch_weighted_pairwise_dense_cost,
)
from hesf_coarsen.scoring.merge_cost import score_candidate_pairs
from hesf_coarsen.scoring import merge_cost as merge_cost_module
from hesf_coarsen.scoring.relation_profile import compute_relation_profiles
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch


def pair_same_type_assignment(graph):
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types = []
    next_super = 0
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = np.flatnonzero(graph.node_type == type_id)
        for start in range(0, len(nodes), 2):
            group = nodes[start : start + 2]
            for node in group:
                assignment[node] = next_super
            super_types.append(int(type_id))
            next_super += 1
    return Assignment(assignment, np.asarray(super_types, dtype=np.int32))


def relation_table(graph, relation_id):
    rel = graph.relations[relation_id]
    return sorted(
        (int(src), int(dst), float(weight))
        for src, dst, weight in zip(rel.src, rel.dst, rel.weight)
    )


def test_memmap_graph_round_trip_uses_mmap_arrays(tmp_path):
    graph = generate_synthetic_graph(num_users=6, num_items=4, num_tags=3, seed=101)

    save_memmap_graph(graph, tmp_path / "mmap", chunk_size=3)
    loaded = load_memmap_graph(tmp_path / "mmap")

    validate_schema(loaded)
    assert loaded.num_nodes == graph.num_nodes
    assert isinstance(loaded.node_type, np.memmap)
    assert isinstance(loaded.relations[0].src, np.memmap)
    assert np.array_equal(loaded.node_type, graph.node_type)


def test_chunked_aggregation_matches_in_memory_aggregation(tmp_path):
    graph = generate_synthetic_graph(num_users=7, num_items=5, num_tags=3, seed=202)
    assignment = pair_same_type_assignment(graph)

    expected = coarsen_graph(graph, assignment)
    actual = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=2,
        output_dir=tmp_path,
        reducer="sort",
    )

    validate_schema(actual)
    assert actual.num_nodes == expected.num_nodes
    for relation_id in expected.relations:
        assert relation_table(actual, relation_id) == relation_table(expected, relation_id)


def test_sort_reduce_chunked_aggregation_handles_duplicate_keys(tmp_path):
    graph = generate_synthetic_graph(num_users=9, num_items=5, num_tags=3, seed=212)
    assignment = pair_same_type_assignment(graph)

    sort_result = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=1,
        output_dir=tmp_path / "sort",
        reducer="sort",
    )
    hash_result = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=1,
        output_dir=tmp_path / "hash",
        reducer="hash",
    )

    for relation_id in hash_result.relations:
        assert relation_table(sort_result, relation_id) == relation_table(hash_result, relation_id)


def test_sort_reduce_chunked_aggregation_uses_external_shard_merge(tmp_path):
    graph = HeteroGraph(
        num_nodes=6,
        node_type=np.zeros(6, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 2, 3, 0, 1, 4, 5], dtype=np.int64),
                dst=np.array([2, 3, 0, 1, 3, 2, 0, 1], dtype=np.int64),
                weight=np.array([1.0, 2.0, 0.5, 0.75, 3.0, 4.0, 5.5, 0.25], dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "same_type", 0, 0)},
    )
    assignment = Assignment(
        assignment=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        supernode_type=np.zeros(3, dtype=np.int32),
    )

    expected = coarsen_graph(graph, assignment)
    actual = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=2,
        output_dir=tmp_path / "external",
        reducer="sort",
    )

    shard_dir = tmp_path / "external" / "_aggregation_shards" / "relation_0"
    chunk_dir = shard_dir / "chunks"
    assert chunk_dir.exists()
    assert len(list(chunk_dir.glob("*_keys.npy"))) == 4
    assert (shard_dir / "final_src.npy").exists()
    assert isinstance(actual.relations[0].src, np.memmap)
    assert relation_table(actual, 0) == relation_table(expected, 0)
    coarse_keys = actual.relations[0].src * actual.num_nodes + actual.relations[0].dst
    assert np.array_equal(coarse_keys, np.unique(coarse_keys))


def test_k_way_merge_sorted_chunks_handles_interleaved_and_empty_shards(tmp_path):
    chunks = [
        (
            np.array([1, 4, 9], dtype=np.int64),
            np.array([0.25, 1.0, 2.0], dtype=np.float32),
        ),
        (
            np.array([2, 4, 8], dtype=np.int64),
            np.array([0.5, 1.5, 3.0], dtype=np.float32),
        ),
        (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        ),
        (
            np.array([1, 8], dtype=np.int64),
            np.array([0.75, 0.25], dtype=np.float32),
        ),
    ]

    src, dst, weight = _merge_sorted_chunks(chunks, num_supernodes=4, output_dir=tmp_path)

    assert isinstance(src, np.memmap)
    assert (src * 4 + dst).tolist() == [1, 2, 4, 8, 9]
    assert np.allclose(weight, [1.0, 0.5, 2.5, 3.25, 2.0])


def test_chunked_aggregation_preserves_features_and_label_majority(tmp_path):
    graph = HeteroGraph(
        num_nodes=5,
        node_type=np.array([0, 0, 0, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 2], dtype=np.int64),
                dst=np.array([3, 3, 4], dtype=np.int64),
                weight=np.ones(3, dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "u_to_v", 0, 1)},
        features={
            0: np.array([[1.0, 2.0], [3.0, 6.0], [9.0, 10.0]], dtype=np.float32),
            1: np.array([[4.0], [8.0]], dtype=np.float32),
        },
        labels=np.array([2, 1, 1, 4, 4], dtype=np.int64),
    )
    assignment = Assignment(
        assignment=np.array([0, 0, 1, 2, 2], dtype=np.int64),
        supernode_type=np.array([0, 0, 1], dtype=np.int32),
    )

    coarse = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=1,
        output_dir=tmp_path,
        reducer="sort",
    )

    assert np.allclose(coarse.features[0], [[2.0, 4.0], [9.0, 10.0]])
    assert np.allclose(coarse.features[1], [[6.0]])
    assert coarse.labels.tolist() == [1, 1, 4]


def test_memmap_and_chunked_cli_commands(tmp_path):
    graph = generate_synthetic_graph(num_users=5, num_items=4, num_tags=2, seed=303)
    input_dir = tmp_path / "input"
    mmap_dir = tmp_path / "mmap"
    chunked_dir = tmp_path / "chunked"
    save_graph(graph, input_dir)
    assignment = pair_same_type_assignment(graph)
    np.savez(
        tmp_path / "assignment.npz",
        assignment=assignment.assignment,
        supernode_type=assignment.supernode_type,
    )

    main(["export-memmap", "--input", str(input_dir), "--output", str(mmap_dir), "--chunk-size", "2"])
    main(
        [
            "chunked-aggregate",
            "--input",
            str(mmap_dir),
            "--assignment",
            str(tmp_path / "assignment.npz"),
            "--output",
            str(chunked_dir),
            "--chunk-size",
            "2",
            "--reducer",
            "sort",
            "--memmap-input",
        ]
    )

    loaded = load_graph(chunked_dir)
    validate_schema(loaded)
    assert loaded.num_nodes == assignment.num_supernodes


def test_torch_dense_helpers_match_numpy_on_cpu_or_cuda():
    device = get_torch_device("auto", max_fraction=0.01)
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    normalized = torch_row_normalize(X, device=device)
    distances = torch_pairwise_squared_distance(X, np.array([[0, 2], [1, 2]], dtype=np.int64), device=device)

    expected_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    expected_dist = np.sum((X[[0, 1]] - X[[2, 2]]) ** 2, axis=1)
    assert np.allclose(normalized, expected_norm, atol=1e-6)
    assert np.allclose(distances, expected_dist, atol=1e-6)


def test_torch_weighted_pairwise_dense_cost_uses_block_memory_limit():
    device = get_torch_device("auto", max_fraction=0.01)
    X = np.arange(128 * 16, dtype=np.float32).reshape(128, 16) / 100.0
    Y = X[:, :8] * 0.5
    pairs = np.array([[0, 1], [2, 3], [0, 3]], dtype=np.int64)

    costs = torch_weighted_pairwise_dense_cost(
        [(X, 1.0), (Y, 0.25)],
        pairs,
        device=device,
        batch_size=3,
        max_bytes=512,
    )

    expected = np.sum((X[pairs[:, 0]] - X[pairs[:, 1]]) ** 2, axis=1)
    expected += 0.25 * np.sum((Y[pairs[:, 0]] - Y[pairs[:, 1]]) ** 2, axis=1)
    assert X.nbytes > 512
    assert np.allclose(costs, expected, atol=1e-5)


def test_torch_scoring_backend_matches_numpy_backend():
    graph = generate_synthetic_graph(num_users=5, num_items=4, num_tags=2, seed=404)
    pairs = np.array([[0, 1, 0.0], [5, 6, 0.0]], dtype=np.float64)
    z = np.arange(graph.num_nodes * 3, dtype=np.float32).reshape(graph.num_nodes, 3) / 10.0
    profiles = compute_relation_profiles(graph)
    conv = z[:, :2]
    base_config = {
        "scoring": {
            "lambda_spec": 1.0,
            "lambda_rel": 0.2,
            "lambda_feat": 0.0,
            "lambda_conv": 0.3,
            "lambda_boundary": 0.1,
        }
    }
    torch_config = dict(base_config)
    torch_config["acceleration"] = {
        "dense_backend": "torch",
        "device": "cpu",
        "fallback_to_numpy": False,
        "max_dense_bytes": None,
    }

    numpy_scores = score_candidate_pairs(graph, pairs, z, profiles, conv, None, base_config)
    torch_scores = score_candidate_pairs(graph, pairs, z, profiles, conv, None, torch_config)

    assert np.allclose(torch_scores, numpy_scores, atol=1e-6)


def test_torch_scoring_backend_uses_block_local_dense_batches():
    graph = generate_synthetic_graph(num_users=8, num_items=6, num_tags=3, seed=414)
    pairs = np.array([[0, 1, 0.0], [2, 3, 0.0], [8, 9, 0.0]], dtype=np.float64)
    z = np.arange(graph.num_nodes * 32, dtype=np.float32).reshape(graph.num_nodes, 32) / 100.0
    profiles = np.arange(graph.num_nodes * 24, dtype=np.float32).reshape(graph.num_nodes, 24) / 50.0
    conv = z[:, :16]
    base_config = {
        "scoring": {
            "lambda_spec": 1.0,
            "lambda_rel": 0.2,
            "lambda_feat": 0.0,
            "lambda_conv": 0.3,
            "lambda_boundary": 0.0,
        }
    }
    torch_config = dict(base_config)
    torch_config["acceleration"] = {
        "dense_backend": "torch",
        "device": "cpu",
        "fallback_to_numpy": False,
        "max_dense_bytes": 1200,
        "scoring_batch_size": 2,
    }

    assert z.nbytes > torch_config["acceleration"]["max_dense_bytes"]
    numpy_scores = score_candidate_pairs(graph, pairs, z, profiles, conv, None, base_config)
    torch_scores = score_candidate_pairs(graph, pairs, z, profiles, conv, None, torch_config)

    assert np.allclose(torch_scores, numpy_scores, atol=1e-5)


def test_feature_scoring_uses_typewise_blocks_without_global_dense_matrix(monkeypatch):
    graph = HeteroGraph(
        num_nodes=5,
        node_type=np.array([0, 0, 0, 1, 1], dtype=np.int32),
        relations={},
        features={
            0: np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 5.0],
                    [4.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            1: np.array(
                [
                    [10.0, 1.0],
                    [8.0, 4.0],
                ],
                dtype=np.float32,
            ),
        },
    )
    pairs = np.array([[0, 1, 0.0], [3, 4, 0.0]], dtype=np.float64)
    z = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    profiles = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    conv = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    config = {
        "features": {"projected_dim": 4},
        "scoring": {
            "lambda_spec": 0.0,
            "lambda_rel": 0.0,
            "lambda_feat": 1.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 0.0,
        },
        "acceleration": {"dense_backend": "numpy", "scoring_batch_size": 1},
    }

    original_zeros = merge_cost_module.np.zeros

    def reject_global_dense(shape, *args, **kwargs):
        if shape == (graph.num_nodes, 3):
            raise AssertionError("feature scoring must not build a global dense feature matrix")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(merge_cost_module.np, "zeros", reject_global_dense)

    scored = score_candidate_pairs(graph, pairs, z, profiles, conv, graph.features, config)

    assert np.allclose(scored[:, 2], [6.0, 13.0])


def test_feature_projection_is_typewise_low_dim_float16():
    graph = HeteroGraph(
        num_nodes=3,
        node_type=np.array([0, 0, 0], dtype=np.int32),
        relations={},
        features={
            0: np.arange(3 * 8, dtype=np.float32).reshape(3, 8),
        },
    )

    view = merge_cost_module._prepare_typewise_feature_view(
        graph,
        graph.features,
        {"seed": 11, "features": {"projected_dim": 2, "projection_dtype": "float16"}},
    )

    assert view is not None
    assert view.blocks[0].shape == (3, 2)
    assert view.blocks[0].dtype == np.float16


def test_lowpass_sketch_accepts_torch_dense_backend():
    graph = generate_synthetic_graph(num_users=5, num_items=4, num_tags=2, seed=505)
    config = {
        "seed": 505,
        "sketch": {"dim": 4, "order": 2, "num_scales": 1, "dtype": "float32"},
        "acceleration": {
            "dense_backend": "torch",
            "device": "cpu",
            "fallback_to_numpy": False,
            "max_dense_bytes": None,
        },
    }

    z = compute_lowpass_sketch(graph, config)

    assert z.shape == (graph.num_nodes, 4)
    assert z.dtype == np.float32
