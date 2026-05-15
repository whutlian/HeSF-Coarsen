import json
import inspect

import numpy as np

from hesf_coarsen.coarsen import aggregate_edges
from hesf_coarsen.coarsen import multilevel as multilevel_module
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj


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


def test_multilevel_pipeline_streams_explicit_mutual_best_without_to_pairs(tmp_path, monkeypatch):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=123,
    )
    config = small_config(tmp_path)
    config["coarsening"] = dict(config["coarsening"], matching_method="mutual_best")
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["candidates"] = dict(config["candidates"], pair_block_size=2)

    def fail_to_pairs(*_args, **_kwargs):
        raise AssertionError("explicit mutual-best pipeline should stream pair blocks")

    monkeypatch.setattr(multilevel_module.BoundedCandidateStore, "to_pairs", fail_to_pairs)
    monkeypatch.setattr(multilevel_module.ArrayCandidateStore, "to_pairs", fail_to_pairs)

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert results[0].graph.num_nodes < graph.num_nodes


def test_multilevel_pipeline_writes_score_term_diagnostics(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=124,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["candidates"] = dict(config["candidates"], pair_block_size=2)

    result = run_multilevel_coarsening(graph, config)[0]

    score_terms = result.diagnostics["score_terms"]
    score_contributions = result.diagnostics["score_contributions"]
    score_share = result.diagnostics["score_contribution_share"]
    assert set(score_terms) == {"spec", "rel", "feat", "conv", "boundary"}
    assert set(score_contributions) == {"spec", "rel", "feat", "conv", "boundary"}
    assert set(score_share) == {"spec", "rel", "feat", "conv", "boundary"}
    assert np.isclose(sum(score_share.values()), 1.0)
    for term_stats in score_terms.values():
        assert term_stats["count"] > 0
        assert term_stats["sample_count"] > 0
        for key in ("mean", "p50", "p95", "p99"):
            assert key in term_stats
            assert np.isfinite(term_stats[key])

    with (tmp_path / "level_1" / "diagnostics.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["score_terms"]["spec"]["count"] == score_terms["spec"]["count"]
    assert "score_contribution_share" in saved


def test_multilevel_pipeline_writes_label_projection_task_metrics(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=126,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)

    result = run_multilevel_coarsening(graph, config)[0]

    task = result.diagnostics["task"]
    assert task["model"] == "majority_label_projection"
    assert task["labeled_nodes"] > 0
    assert 0.0 <= task["micro_f1"] <= 1.0
    assert 0.0 <= task["macro_f1"] <= 1.0


def test_multilevel_pipeline_records_cluster_reduction_diagnostics(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=127,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["coarsening"] = dict(
        config["coarsening"],
        matching_method="greedy_cluster",
        max_cluster_size=4,
    )

    result = run_multilevel_coarsening(graph, config)[0]
    diagnostics = result.diagnostics

    assert diagnostics["cluster_count"] == result.assignment.num_supernodes
    assert diagnostics["node_reduction"] == graph.num_nodes - result.graph.num_nodes
    assert diagnostics["matched_units"] == diagnostics["node_reduction"]
    assert diagnostics["cluster_size_histogram"]
    assert diagnostics["cluster_size_mean"] >= 1.0
    assert "cluster_label_entropy" in diagnostics


def test_multilevel_pipeline_writes_full_cluster_diagnostics_without_nan(tmp_path):
    graph = generate_synthetic_graph(
        num_users=12,
        num_items=8,
        num_tags=4,
        seed=128,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["coarsening"] = dict(
        config["coarsening"],
        target_ratio=0.5,
        matching_method="greedy_cluster",
        max_cluster_size=4,
    )

    diagnostics = run_multilevel_coarsening(graph, config)[0].diagnostics

    required = [
        "cluster_count",
        "cluster_size_mean",
        "cluster_size_p50",
        "cluster_size_p95",
        "cluster_size_p99",
        "cluster_size_histogram",
        "cluster_size_histogram_by_type",
        "node_reduction",
        "node_reduction_by_type",
        "cluster_sketch_spread_mean",
        "cluster_sketch_spread_p95",
        "cluster_relation_profile_variance_mean",
        "cluster_relation_profile_variance_p95",
        "cluster_conv_response_spread_mean",
        "cluster_conv_response_spread_p95",
        "cluster_label_entropy_train_only_mean",
        "cluster_label_entropy_train_only_p95",
        "bad_cluster_count",
        "bad_cluster_fraction",
    ]
    for key in required:
        assert key in diagnostics
    for key in required:
        value = diagnostics[key]
        if isinstance(value, (int, float)):
            assert np.isfinite(value), key
    assert diagnostics["bad_cluster_count"] == 0


def test_terminal_guard_blocks_large_protected_clusters_and_reports_diagnostics():
    graph = HeteroGraph(
        num_nodes=5,
        node_type=np.zeros(5, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 0, 0, 0], dtype=np.int64),
                dst=np.array([1, 2, 3, 4], dtype=np.int64),
                weight=np.ones(4, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
    )
    scored_pairs = np.array(
        [
            [0, 1, 0.0],
            [0, 2, 0.1],
            [0, 3, 0.2],
            [0, 4, 0.3],
        ],
        dtype=np.float64,
    )

    from hesf_coarsen.matching.greedy import run_greedy_cluster_matching

    assignment = run_greedy_cluster_matching(
        graph,
        scored_pairs,
        {
            "coarsening": {
                "matching_method": "greedy_cluster",
                "same_type_only": True,
                "same_partition_only": False,
                "max_cluster_size": 4,
                "terminal_guard": {
                    "enabled": True,
                    "protect_hubs": True,
                    "hub_degree_percentile": 50,
                    "max_terminal_cluster_size": 2,
                },
            }
        },
    )

    assert max(assignment.cluster_sizes()) <= 2
    terminal = assignment.diagnostics["terminal_guard"]
    assert terminal["protected_node_count"] > 0
    assert terminal["protected_by_reason"]["hub"] > 0
    assert terminal["merge_blocked_count"] > 0


def test_repair_bad_clusters_records_objective_and_accept_gate():
    graph = HeteroGraph(
        num_nodes=4,
        node_type=np.zeros(4, dtype=np.int32),
        relations={},
        labels=np.array([0, 0, 1, 1], dtype=np.int32),
    )
    assignment = Assignment(
        assignment=np.zeros(graph.num_nodes, dtype=np.int64),
        supernode_type=np.array([0], dtype=np.int32),
    )
    z = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 0.0],
            [3.1, 0.0],
        ],
        dtype=np.float32,
    )
    config = {
        "coarsening": {
            "cumulative_guard": {
                "enabled": True,
                "repair_bad_clusters": True,
                "repair_strategy": "split_local_swap_accept",
                "accept_only_if_cumulative_improves": True,
            }
        }
    }

    repaired, diagnostics = multilevel_module._repair_bad_clusters(graph, assignment, z, config)

    assert repaired.num_supernodes > assignment.num_supernodes
    assert diagnostics["repair_accepted"] is True
    assert diagnostics["repair_strategy"] == "split_local_swap_accept"
    assert diagnostics["repair_objective"]["cluster_sketch_spread"] > 0.0
    assert "cumulative_energy_delta" in diagnostics["repair_objective"]
    assert "relation_profile_variance" in diagnostics["repair_objective"]
    assert "train_label_entropy" in diagnostics["repair_objective"]
    assert diagnostics["estimated_cumulative_dee_after"] <= diagnostics["estimated_cumulative_dee_before"]


def test_repair_bad_clusters_uses_configured_objective_to_select_clusters():
    graph = HeteroGraph(
        num_nodes=6,
        node_type=np.zeros(6, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([3, 3, 4], dtype=np.int64),
                dst=np.array([4, 5, 5], dtype=np.int64),
                weight=np.ones(3, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        labels=np.array([0, 0, 0, 0, 1, 2], dtype=np.int32),
    )
    assignment = Assignment(
        assignment=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        supernode_type=np.array([0, 0], dtype=np.int32),
    )
    z = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )

    def repair(objective: str):
        config = {
            "coarsening": {
                "cumulative_guard": {
                    "enabled": True,
                    "repair_bad_clusters": True,
                    "repair_strategy": "split_local_swap_accept",
                    "repair_objective": objective,
                }
            }
        }
        return multilevel_module._repair_bad_clusters(graph, assignment, z, config)[1]

    energy_diag = repair("energy")
    relation_diag = repair("relation")
    task_diag = repair("task")

    assert energy_diag["repair_objective_name"] == "energy"
    assert relation_diag["repair_objective_name"] == "relation"
    assert task_diag["repair_objective_name"] == "task"
    assert energy_diag["repair_selected_clusters"] == [0]
    assert relation_diag["repair_selected_clusters"] == [1]
    assert task_diag["repair_selected_clusters"] == [1]
    assert relation_diag["repair_trace_signature"] != energy_diag["repair_trace_signature"]


def test_level_config_caps_matches_by_remaining_target_ratio():
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.5,
        per_level_ratio=0.55,
    )

    far = multilevel_module._config_for_level(config, num_nodes=100, target_nodes=50)
    near = multilevel_module._config_for_level(config, num_nodes=60, target_nodes=50)

    assert far["coarsening"]["level_ratio"] == 0.55
    assert far["coarsening"]["desired_coarse_nodes"] == 55
    assert far["coarsening"]["max_matched_pairs"] == 45
    assert np.isclose(near["coarsening"]["remaining_ratio"], 50 / 60)
    assert near["coarsening"]["level_ratio"] == 50 / 60
    assert near["coarsening"]["desired_coarse_nodes"] == 50
    assert near["coarsening"]["max_matched_pairs"] == 10


def test_multilevel_pipeline_records_target_control_and_resolved_config(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=125,
    )
    config = small_config(tmp_path)
    config["coarsening"] = dict(
        config["coarsening"],
        target_ratio=0.8,
        max_levels=2,
        per_level_ratio=0.55,
    )
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)

    result = run_multilevel_coarsening(graph, config)[0]

    control = result.diagnostics["target_control"]
    assert control["target_ratio"] == 0.8
    assert control["target_nodes"] == int(np.ceil(graph.num_nodes * 0.8))
    assert control["desired_coarse_nodes"] >= control["target_nodes"]
    assert control["max_matched_pairs"] == control["input_nodes"] - control["desired_coarse_nodes"]
    assert result.diagnostics["config"]["coarsening"]["target_ratio"] == 0.8
    assert result.diagnostics["config"]["sketch"]["method"] == config["sketch"]["method"]
    assert "total_budget_K" in result.diagnostics["config"]["candidates"]


def test_multilevel_pipeline_writes_spectral_diagnostics_closed_loop(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=24,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(
        config["diagnostics"],
        enable_spectral=True,
        spectral_num_signals=3,
        spectral_smoothing_steps=1,
        spectral_exact_eigenvalue_max_nodes=64,
        spectral_baseline_max_nodes=64,
        spectral_baselines=["random", "heavy_edge", "graphzoom_style", "convmatch_style"],
    )

    results = run_multilevel_coarsening(graph, config)

    spectral = results[0].diagnostics["spectral"]
    assert "relation_weighted_fused_energy_relative_error" in spectral
    assert "chebheat_sketch_inner_product_relative_error" in spectral
    assert "relation_energy_relative_error_max" in spectral
    assert "sketch_dirichlet_energy_relative_error" in spectral
    assert "exact_eigenvalue_sanity" in spectral
    assert set(spectral["baseline_comparison"]) == {
        "random",
        "heavy_edge",
        "graphzoom_style",
        "convmatch_style",
    }
    with (tmp_path / "level_1" / "diagnostics.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "spectral" in saved
    assert "relation_weighted_fused_energy_relative_error" in saved["spectral"]
    assert "cumulative_spectral" in saved
    assert "relation_weighted_fused_energy_relative_error" in saved["cumulative_spectral"]
    cumulative_payload = np.load(tmp_path / "level_1" / "cumulative_assignment.npz")
    assert cumulative_payload["assignment"].shape == (graph.num_nodes,)


def test_spectral_diagnostics_samples_exact_sanity_when_graph_exceeds_limit(tmp_path):
    from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics

    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=1410,
    )
    result = run_multilevel_coarsening(graph, small_config(tmp_path / "run"))[0]

    diagnostics = compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        exact_eigenvalue_max_nodes=8,
    )

    assert diagnostics["exact_eigenvalue_sanity"]["status"] == "sampled_subgraph"
    assert diagnostics["exact_eigenvalue_sanity"]["mode"] == "sampled_dense_eigvalsh"
    assert diagnostics["exact_eigenvalue_sanity"]["relative_error"] >= 0.0


def test_multilevel_pipeline_runs_lazy_and_chebyshev_sketches(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=29,
    )
    lazy_config = small_config(tmp_path / "lazy")
    lazy_config["sketch"] = dict(lazy_config["sketch"], method="lazy")
    cheb_config = small_config(tmp_path / "cheb")
    cheb_config["sketch"] = dict(
        cheb_config["sketch"],
        method="chebyshev_heat",
        dim=10,
        order=4,
        heat_times=[1.0, 2.0],
    )
    cheb_config["metapath_sketch"] = {
        "enabled": True,
        "dim": 2,
        "max_paths": 1,
        "max_path_length": 2,
        "seed": 123,
        "row_normalize": True,
        "paths": [
            {
                "name": "user_item_user",
                "start_type": 0,
                "end_type": 0,
                "steps": [
                    {"relation_id": 0, "direction": "forward"},
                    {"relation_id": 0, "direction": "backward"},
                ],
            }
        ],
    }

    lazy_results = run_multilevel_coarsening(graph, lazy_config)
    cheb_results = run_multilevel_coarsening(graph, cheb_config)

    assert lazy_results
    assert cheb_results
    assert lazy_results[0].diagnostics["sketch"]["sketch_method"] == "lazy"
    assert cheb_results[0].diagnostics["sketch"]["sketch_method"] == "chebyshev_heat"
    assert cheb_results[0].diagnostics["metapath_sketch"]["enabled"] is True


def test_lazy_sketch_can_use_fused_relation_metapath_operator(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=30,
    )
    config = small_config(tmp_path / "lazy_meta")
    config["sketch"] = dict(config["sketch"], method="lazy", dim=8, order=2, dtype="float32")
    config["metapath_sketch"] = dict(
        config["metapath_sketch"],
        enabled=True,
        max_paths=1,
        operator_weight_total=0.25,
        auto_paths=True,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results[0].diagnostics["sketch"]["sketch_method"] == "lazy"
    assert results[0].diagnostics["metapath_sketch"]["enabled"] is True
    assert results[0].diagnostics["metapath_sketch"]["operator_mode"] == "fused_laplacian"
    assert results[0].diagnostics["fusion"]["relation_operator_weight_total"] < 1.0


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


def test_multilevel_pipeline_uses_chunked_aggregation(tmp_path, monkeypatch):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=42,
    )
    config = small_config(tmp_path)
    config["coarsening"] = dict(
        config["coarsening"],
        aggregation_chunk_size=3,
        aggregation_reducer="sort",
    )
    calls = []

    def fake_chunked_aggregation(graph_arg, assignment, **kwargs):
        calls.append(kwargs)
        return aggregate_edges.coarsen_graph(graph_arg, assignment)

    monkeypatch.setattr(
        multilevel_module,
        "coarsen_graph_chunked",
        fake_chunked_aggregation,
        raising=False,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert calls
    assert calls[0]["chunk_size"] == 3
    assert calls[0]["reducer"] == "sort"
    assert calls[0]["output_dir"] == tmp_path / "level_1"


def test_multilevel_pipeline_records_feature_aggregation_method(tmp_path, monkeypatch):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=45,
    )
    config = small_config(tmp_path)
    config["coarsening"] = dict(
        config["coarsening"],
        feature_aggregation="degree_weighted",
    )
    calls = []

    def fake_chunked_aggregation(graph_arg, assignment, **kwargs):
        calls.append(kwargs)
        return aggregate_edges.coarsen_graph(
            graph_arg,
            assignment,
            feature_aggregation=kwargs.get("feature_aggregation"),
            feature_weights=kwargs.get("feature_weights"),
        )

    monkeypatch.setattr(
        multilevel_module,
        "coarsen_graph_chunked",
        fake_chunked_aggregation,
        raising=False,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert calls[0]["feature_aggregation"] == "degree_weighted"
    assert results[0].diagnostics["feature_aggregation"]["method"] == "degree_weighted"
    assert results[0].diagnostics["feature_aggregation"]["uses_weights"] is True


def test_multilevel_pipeline_writes_level_projected_feature_store(tmp_path):
    graph = generate_synthetic_graph(
        num_users=10,
        num_items=6,
        num_tags=4,
        seed=44,
        feature_dim=12,
    )
    config = small_config(tmp_path / "run")
    config["features"] = dict(
        config["features"],
        projected_dim=3,
        projection_dtype="float16",
        projection_mmap_dir=str(tmp_path / "projected_features"),
        projection_chunk_size=2,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results
    projected_path = tmp_path / "projected_features" / "level_1" / "features_type_0_projected.npy"
    assert projected_path.exists()
    projected = np.load(projected_path, mmap_mode="r")
    assert isinstance(projected, np.memmap)
    assert projected.dtype == np.float16
    assert projected.shape[1] == 3


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


def test_multilevel_pipeline_records_selected_match_sources_and_fallback_fraction(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=46,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)

    results = run_multilevel_coarsening(graph, config)

    selected = results[0].diagnostics["matched_pairs_by_source"]
    assert sum(selected.values()) == results[0].diagnostics["matched_pairs"]
    assert "fallback_selected_fraction" in results[0].diagnostics


def test_multilevel_pipeline_can_disable_fallback_candidates(tmp_path):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=48,
    )
    config = small_config(tmp_path)
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["candidates"] = dict(
        config["candidates"],
        enable_onehop=False,
        enable_capped_twohop=False,
        enable_bucket=False,
        enable_partition_ann=False,
        enable_fallback=False,
    )

    results = run_multilevel_coarsening(graph, config)

    assert results[0].diagnostics["candidate_source_counts"].get("fallback", 0) == 0
    assert results[0].diagnostics["fallback_selected_fraction"] == 0.0


def test_multilevel_pipeline_emits_stage_progress_when_enabled(tmp_path, capsys):
    graph = generate_synthetic_graph(
        num_users=8,
        num_items=5,
        num_tags=3,
        seed=47,
    )
    config = small_config(tmp_path)
    config["progress"] = {
        "enabled": True,
        "backend": "plain",
        "min_interval_seconds": 0,
    }

    run_multilevel_coarsening(graph, config)

    captured = capsys.readouterr()
    assert "level 1" in captured.err
    assert "sketch" in captured.err
    assert "candidates" in captured.err
    assert "chebyshev heat components" in captured.err
    assert "scoring relation profiles" in captured.err
    assert "score dense batches" in captured.err


def test_multilevel_scoring_path_does_not_build_global_feature_dense_matrix():
    source = inspect.getsource(multilevel_module)

    assert "_global_feature_matrix" not in source
    assert "np.concatenate([Z.astype(np.float32), X]" not in source
