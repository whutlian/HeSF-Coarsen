import json

import numpy as np

from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph


def small_config(tmp_path):
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.75,
        max_levels=2,
        per_level_ratio=0.7,
    )
    config["sketch"] = dict(DEFAULT_CONFIG["sketch"], dim=8, order=2, dtype="float32")
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=8,
        twohop_budget_K2=4,
        per_middle_pair_cap=16,
        simhash_bits=4,
        bucket_pair_cap=16,
    )
    config["output"] = {"dir": str(tmp_path)}
    return config


def test_multilevel_pipeline_runs_and_writes_diagnostics(tmp_path):
    graph = generate_synthetic_graph(
        num_users=14,
        num_items=8,
        num_tags=5,
        seed=23,
    )
    results = run_multilevel_coarsening(graph, small_config(tmp_path))

    assert results
    assert results[-1].graph.num_nodes < graph.num_nodes
    diagnostics_path = tmp_path / "level_1" / "diagnostics.json"
    assert diagnostics_path.exists()
    with diagnostics_path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    assert diagnostics["coarse_nodes"] == results[0].graph.num_nodes
    assert diagnostics["matched_pairs"] > 0
    validate_loaded = load_graph(tmp_path / "level_1")
    assert validate_loaded.num_nodes == results[0].graph.num_nodes


def test_multilevel_pipeline_is_deterministic(tmp_path):
    graph = generate_synthetic_graph(
        num_users=12,
        num_items=7,
        num_tags=4,
        seed=31,
    )
    cfg1 = small_config(tmp_path / "a")
    cfg2 = small_config(tmp_path / "b")

    first = run_multilevel_coarsening(graph, cfg1)
    second = run_multilevel_coarsening(graph, cfg2)

    assert np.array_equal(
        first[0].assignment.assignment,
        second[0].assignment.assignment,
    )
    assert first[0].diagnostics["candidate_count_total"] == second[0].diagnostics[
        "candidate_count_total"
    ]


def test_multilevel_pipeline_accepts_array_chunked_candidate_backend(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=41,
    )
    config = small_config(tmp_path)
    config["candidates"] = dict(
        config["candidates"],
        store_backend="array",
        use_chunked_generation=True,
        edge_chunk_size=3,
        middle_chunk_size=4,
        node_chunk_size=5,
        mmap_dir=str(tmp_path / "candidate_mmap"),
        incident_index_mmap_dir=str(tmp_path / "incident_index_mmap"),
    )

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert results[0].diagnostics["candidate_count_max"] <= config["candidates"]["total_budget_K"]
    assert (tmp_path / "candidate_mmap" / "level_1" / "candidate_ids.npy").exists()
    assert (tmp_path / "incident_index_mmap" / "level_1" / "incident_endpoints.npy").exists()


def test_multilevel_pipeline_accepts_partition_ann_candidate_source(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=43,
    )
    config = small_config(tmp_path)
    config["candidates"] = dict(
        config["candidates"],
        enable_onehop=False,
        enable_capped_twohop=False,
        enable_bucket=False,
        enable_partition_ann=True,
        ann_num_projections=3,
        ann_window_size=3,
        ann_budget_K=2,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results
    source_counts = results[0].diagnostics["candidate_source_counts"]
    assert source_counts.get("partition_ann", 0) > 0
