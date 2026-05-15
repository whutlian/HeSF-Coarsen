import json

import numpy as np

from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.eval.diagnostics import compute_large_graph_envelope
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_large_graph_envelope_samples_edges_and_artifacts(tmp_path):
    graph = generate_synthetic_graph(num_users=8, num_items=5, num_tags=3, seed=1001)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "block.bin").write_bytes(b"123456789")
    candidate_counts = np.array([0, 1, 2, 3], dtype=np.int32)
    config = {
        "hardware": {"max_ram_gb": 1},
        "candidates": {"total_budget_K": 4},
        "diagnostics": {"edge_sample_size": 2},
    }

    envelope = compute_large_graph_envelope(
        graph,
        candidate_counts=candidate_counts,
        runtime_by_stage={"sketch": 1.5, "candidates": 2.0},
        config=config,
        artifact_dirs={"candidate": artifact_dir},
    )

    assert envelope["graph_array_bytes"] > 0
    assert envelope["runtime_total_seconds"] == 3.5
    assert envelope["runtime_max_stage"] == "candidates"
    assert envelope["candidate_store_estimated_bytes"] > 0
    assert envelope["artifact_bytes_by_name"]["candidate"] == 9
    assert envelope["edge_sample_size"] == 2
    assert envelope["total_sampled_edges"] <= 2 * len(graph.relations)
    for relation in envelope["relation_edge_samples"].values():
        assert relation["sampled_edges"] <= 2


def test_multilevel_diagnostics_include_large_graph_envelope(tmp_path):
    graph = generate_synthetic_graph(num_users=10, num_items=6, num_tags=4, seed=1002)
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.75,
        max_levels=1,
        per_level_ratio=0.7,
    )
    config["sketch"] = dict(DEFAULT_CONFIG["sketch"], dim=8, order=2, dtype="float32")
    config["candidates"] = dict(DEFAULT_CONFIG["candidates"], total_budget_K=4)
    config["diagnostics"] = {
        "enable_large_graph_envelope": True,
        "edge_sample_size": 3,
    }
    config["output"] = {"dir": str(tmp_path)}

    results = run_multilevel_coarsening(graph, config)

    assert results
    diagnostics_path = tmp_path / "level_1" / "diagnostics.json"
    with diagnostics_path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    envelope = diagnostics["large_graph_envelope"]
    assert envelope["edge_sample_size"] == 3
    assert envelope["graph_array_bytes"] > 0
    assert envelope["runtime_total_seconds"] >= envelope["runtime_by_stage"]["sketch"]


def test_multilevel_diagnostics_include_candidate_generation_breakdown(tmp_path):
    graph = generate_synthetic_graph(num_users=12, num_items=7, num_tags=4, seed=1003)
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.75,
        max_levels=1,
        per_level_ratio=0.75,
    )
    config["sketch"] = dict(DEFAULT_CONFIG["sketch"], dim=8, order=2, dtype="float32")
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=4,
        enable_bucket=True,
        enable_capped_twohop=True,
        enable_fallback=True,
    )
    config["diagnostics"] = dict(DEFAULT_CONFIG["diagnostics"], enable_spectral=False)
    config["output"] = {"dir": str(tmp_path)}

    run_multilevel_coarsening(graph, config)

    diagnostics_path = tmp_path / "level_1" / "diagnostics.json"
    with diagnostics_path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    assert diagnostics["candidate_generation_time"] >= 0.0
    assert diagnostics["candidate_pairs_per_sec"] >= 0.0
    assert diagnostics["candidate_substage_times"]["onehop"] >= 0.0
    assert diagnostics["candidate_substage_times"]["twohop_expansion"] >= 0.0
    assert diagnostics["candidate_substage_times"]["bucket_emit"] >= 0.0
    assert diagnostics["candidate_source_coverage"]["bucket"] >= 0.0
    assert diagnostics["partition_imbalance"]["partition_count"] >= 1
    assert diagnostics["memory_by_candidate_buffers"]["estimated_total_bytes"] > 0
