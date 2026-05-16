import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph, save_graph


def _tiny_config(tmp_path: Path) -> dict:
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.6,
        max_levels=1,
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


def test_invariant_validator_reports_valid_and_invalid_assignment(tmp_path):
    from hesf_coarsen.eval.invariants import validate_level_invariants

    graph = generate_synthetic_graph(num_users=10, num_items=6, num_tags=4, seed=1201)
    result = run_multilevel_coarsening(graph, _tiny_config(tmp_path / "run"))[0]
    diagnostics_path = tmp_path / "run" / "level_1" / "diagnostics.json"

    valid = validate_level_invariants(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        diagnostics_path=diagnostics_path,
    )
    assert valid["schema_type_violations"] == 0
    assert valid["invalid_assignment_count"] == 0
    assert valid["relation_schema_violations"] == 0
    assert valid["diagnostics_missing_count"] == 0

    broken_assignment = result.assignment
    broken_assignment.supernode_type = broken_assignment.supernode_type.copy()
    broken_assignment.supernode_type[broken_assignment.assignment[0]] = 99
    invalid = validate_level_invariants(
        original=graph,
        coarse=result.graph,
        assignment=broken_assignment,
        diagnostics_path=diagnostics_path,
    )
    assert invalid["invalid_assignment_count"] > 0


def test_experiment_scripts_support_help():
    scripts = [
        "run_sanity.py",
        "run_hgb_stage_b.py",
        "run_hgb_sweep.py",
        "run_hgb_stage_b_ablation.py",
        "run_hgb_task_eval.py",
        "run_hgb_next4_mainline.py",
        "run_hgb_next4_relation_fusion.py",
        "run_hgb_lambda_grid.py",
        "run_hgb_next4_baselines.py",
        "evaluate_refine_curve.py",
        "summarize_stage_b.py",
        "summarize_next4.py",
        "compare_stage_b.py",
        "compare_hgb_ablation.py",
        "run_ogbn_mag_subset.py",
        "run_ogbn_mag_envelope.py",
        "run_ogbn_mag_next4_medium.py",
        "collect_diagnostics.py",
        "summarize_experiments.py",
        "make_synthetic_scale.py",
        "run_synthetic_scale.py",
    ]
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(Path("experiments/scripts") / script), "--help"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_hgb_sweep_config_generation():
    from experiments.scripts.run_hgb_sweep import generate_hgb_sweep_configs

    configs = list(generate_hgb_sweep_configs(datasets=["ACM"]))

    assert len(configs) == 64
    names = {item.run_name for item in configs}
    assert len(names) == 64
    ann = [item for item in configs if item.candidate_sources == "onehop_twohop_bucket_ann"]
    assert ann and all(item.config["candidates"]["enable_partition_ann"] for item in ann)


def test_hgb_sweep_progress_dry_run_writes_progress_config(tmp_path):
    from experiments.scripts.run_hgb_sweep import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--dry-run",
            "--progress",
            "--progress-backend",
            "plain",
            "--progress-interval",
            "0.25",
        ]
    )

    assert exit_code == 0
    config_paths = sorted(tmp_path.glob("hgb_ACM_*/config.yaml"))
    assert config_paths
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    assert config["progress"]["enabled"] is True
    assert config["progress"]["backend"] == "plain"
    assert config["progress"]["min_interval_seconds"] == 0.25


def test_summarizer_writes_csv_and_failures(tmp_path):
    from experiments.scripts.summarize_experiments import summarize_experiments

    good = tmp_path / "runs" / "good" / "level_1"
    good.mkdir(parents=True)
    (good / "diagnostics.json").write_text(
        json.dumps(
            {
                "original_nodes": 10,
                "coarse_nodes": 5,
                "compression_ratio": 0.5,
                "candidate_count_mean": 2.0,
                "candidate_count_max": 4,
                "candidate_count_quantiles": {"p50": 2, "p95": 4, "p99": 4},
                "candidate_source_counts": {"onehop": 3},
                "matched_pairs": 5,
                "singleton_ratio": 0.0,
                "relation_weight_abs_error": {"0": 0.0},
                "runtime_by_stage": {"sketch": 1.0, "candidates": 2.0, "spectral_diagnostics": 0.5},
                "spectral": {
                    "sketch_dirichlet_energy_relative_error": 0.1,
                    "relation_weighted_fused_energy_relative_error": 0.2,
                    "relation_energy_relative_error_max": 0.3,
                    "chebheat_sketch_inner_product_relative_error": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runs" / "good" / "metadata.json").write_text(
        json.dumps({"status": "success", "dataset": "tiny", "run_name": "good"}),
        encoding="utf-8",
    )
    failed = tmp_path / "runs" / "failed"
    failed.mkdir()
    (failed / "metadata.json").write_text(
        json.dumps({"status": "failed", "failure_reason": "boom", "run_name": "failed"}),
        encoding="utf-8",
    )

    summarize_experiments([tmp_path / "runs"], tmp_path / "summary")

    all_rows = list(csv.DictReader((tmp_path / "summary" / "all_runs.csv").open()))
    quality_rows = list(csv.DictReader((tmp_path / "summary" / "quality_summary.csv").open()))
    failed_rows = list(csv.DictReader((tmp_path / "summary" / "failures.csv").open()))
    assert {row["status"] for row in all_rows} == {"success", "failed"}
    assert quality_rows[0]["spectral_sketch_dirichlet_energy_relative_error"] == "0.1"
    assert quality_rows[0]["spectral_relation_weighted_fused_energy_relative_error"] == "0.2"
    assert quality_rows[0]["spectral_relation_energy_relative_error_max"] == "0.3"
    assert quality_rows[0]["spectral_chebheat_sketch_inner_product_relative_error"] == "0.4"
    assert failed_rows[0]["failure_reason"] == "boom"
    assert (tmp_path / "summary" / "report.md").exists()


def test_summarizer_writes_final_cumulative_rows_and_target_errors(tmp_path):
    from experiments.scripts.summarize_experiments import summarize_experiments

    run_dir = tmp_path / "runs" / "hgb_ACM_r0p5_L2_d16_K8_base"
    level_1 = run_dir / "level_1"
    level_2 = run_dir / "level_2"
    level_1.mkdir(parents=True)
    level_2.mkdir()
    common = {
        "config": {
            "coarsening": {"target_ratio": 0.5, "per_level_ratio": 0.55, "max_levels": 2},
            "sketch": {"method": "chebyshev_heat", "dim": 16, "order": 5},
            "fusion": {"relation_weighting": {"method": "inverse_energy"}},
            "metapath_sketch": {"enabled": True, "operator_weight_total": 0.25},
            "scoring": {
                "lambda_spec": 1.0,
                "lambda_rel": 0.5,
                "lambda_feat": 0.2,
                "lambda_conv": 0.5,
                "lambda_boundary": 0.2,
                "normalization": "p95",
            },
            "candidates": {"total_budget_K": 8, "enable_onehop": True},
        },
        "spectral": {
            "sketch_dirichlet_energy_relative_error": 0.1,
            "relation_weighted_fused_energy_relative_error": 0.2,
            "fused_sketch_energy_relative_error": 0.3,
            "relation_energy_relative_error_max": 0.4,
            "chebheat_sketch_inner_product_relative_error": 0.5,
            "exact_eigenvalue_sanity": {
                "status": "computed",
                "mode": "dense_eigvalsh",
                "relative_error": 0.01,
            },
            "baseline_comparison": {
                "random": {"status": "computed"},
                "heavy_edge": {"status": "computed"},
                "graphzoom_style": {"status": "computed"},
                "convmatch_style": {"status": "skipped"},
            },
        },
        "task": {"micro_f1": 0.7, "macro_f1": 0.6},
        "candidate_count_total": 4,
        "candidate_generation_time": 2.0,
        "candidate_retained_pair_count": 3,
        "candidate_pairs_per_sec": 1.5,
        "candidate_substage_times": {
            "onehop": 0.2,
            "incident_index_build": 0.1,
            "twohop_expansion": 0.7,
            "simhash": 0.2,
            "bucket_emit": 0.6,
            "fallback": 0.1,
            "store_finalize": 0.1,
        },
        "candidate_source_coverage": {"bucket": 0.8, "capped_twohop": 0.6},
        "partition_imbalance": {"partition_count": 1, "max_to_mean": 1.0},
        "memory_by_candidate_buffers": {"estimated_total_bytes": 4096},
        "candidate_source_counts": {"bucket": 3, "onehop": 1},
        "generated_candidates_by_source": {"bucket": 3, "onehop": 1},
        "selected_merges_by_source": {"bucket": 2},
        "selected_source_avg_score": {"bucket": 0.42},
        "selected_source_avg_delta_spec": {"bucket": 0.12},
        "selected_source_avg_delta_conv": {"bucket": 0.08},
        "selected_source_cluster_size_hist": {"bucket": {"2": 2}},
        "matched_pairs": 2,
        "matched_units": 3,
        "node_reduction": 3,
        "node_reduction_ratio": 0.03,
        "cluster_count": 97,
        "cluster_size_histogram": {"1": 94, "2": 2, "3": 1},
        "cluster_size_mean": 1.03,
        "cluster_label_entropy": 0.25,
        "matched_pairs_by_source": {"bucket": 2},
        "score_terms": {
            "spec": {"count": 2, "mean": 10.0, "p50": 9.0, "p95": 12.0, "p99": 13.0},
            "rel": {"count": 2, "mean": 1.0, "p50": 1.0, "p95": 2.0, "p99": 2.0},
        },
        "score_contributions": {
            "spec": {"count": 2, "mean": 0.4, "p50": 0.3, "p95": 0.8, "p99": 0.9},
            "rel": {"count": 2, "mean": 0.2, "p50": 0.1, "p95": 0.4, "p99": 0.5},
        },
        "score_contribution_share": {
            "spec": 0.4,
            "rel": 0.2,
            "feat": 0.1,
            "conv": 0.2,
            "boundary": 0.1,
        },
        "runtime_by_stage": {
            "sketch": 1.0,
            "candidates": 2.0,
            "scoring": 0.5,
            "matching": 0.25,
            "aggregation": 0.75,
            "matching_and_aggregation": 1.2,
        },
        "cumulative_spectral": {
            "sketch_dirichlet_energy_relative_error": 0.11,
            "relation_weighted_fused_energy_relative_error": 0.22,
            "fused_sketch_energy_relative_error": 0.33,
            "relation_energy_relative_error_max": 0.44,
            "chebheat_sketch_inner_product_relative_error": 0.55,
            "exact_eigenvalue_sanity": {
                "status": "sampled_subgraph",
                "mode": "sampled_dense_eigvalsh",
                "relative_error": 0.066,
            },
            "baseline_comparison": {
                "random": {
                    "status": "computed",
                    "final_cumulative_ratio": 0.52,
                    "dirichlet_energy_relative_error": 0.21,
                    "fused_sketch_energy_relative_error": 0.31,
                    "relation_energy_relative_error_max": 0.41,
                    "chebheat_sketch_inner_product_relative_error": 0.51,
                    "exact_eigenvalue_sanity": {"relative_error": 0.061},
                    "task_projected_macro_f1": 0.62,
                    "task_refined_macro_f1": 0.68,
                    "task_train_time": 2.0,
                    "task_refine_time": 0.5,
                    "task_total_time": 2.5,
                    "runtime_total": 1.25,
                },
                "heavy_edge": {
                    "status": "computed",
                    "final_cumulative_ratio": 0.52,
                    "task_projected_macro_f1": 0.60,
                    "task_refined_macro_f1": 0.70,
                    "task_train_time": 3.0,
                    "task_refine_time": 0.7,
                    "task_total_time": 3.7,
                },
            },
        },
        "large_graph_envelope": {
            "process_rss_bytes": 2 * 1024**3,
            "cuda_memory": {
                "peak_allocated_bytes": 3 * 1024**3,
                "peak_reserved_bytes": 4 * 1024**3,
            },
        },
    }
    (level_1 / "diagnostics.json").write_text(
        json.dumps({**common, "original_nodes": 100, "coarse_nodes": 70, "compression_ratio": 0.7}),
        encoding="utf-8",
    )
    (level_2 / "diagnostics.json").write_text(
        json.dumps({**common, "original_nodes": 70, "coarse_nodes": 52, "compression_ratio": 52 / 70}),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "success",
                "dataset": "ACM",
                "run_name": "hgb_ACM_r0p5_L2_d16_K8_base",
                "variant": "base",
                "experiment_block": "B1",
                "unique_run_key": "B1:ACM:base:unit",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "task_eval.json").write_text(
        json.dumps(
            {
                "model": "rgcn_lite",
                "device": "cpu",
                "coarse_train_micro_f1": 0.8,
                "coarse_train_macro_f1": 0.75,
                "projected_original_micro_f1": 0.7,
                "projected_original_macro_f1": 0.65,
                "refined_original_micro_f1": 0.77,
                "refined_original_macro_f1": 0.72,
                "refined_original_macro_f1@0": 0.66,
                "refined_original_macro_f1@1": 0.64,
                "refined_original_macro_f1@3": 0.71,
                "refined_original_macro_f1@5": 0.72,
                "best_refined_macro_f1": 0.72,
                "best_refined_epoch": 5,
                "refine_auc_macro_f1": 0.686,
                "refine_time_by_epoch": {"0": 0.0, "1": 0.1, "3": 0.4, "5": 0.8},
                "primary_task_metric_name": "refined_original_macro_f1",
                "primary_task_metric": 0.72,
                "eval_on": "original_test_refined",
                "projection_eval_on": "original_test_projected",
                "refine_eval_on": "original_test_refined",
                "label_coverage_train": 0.91,
                "label_coverage_val": 0.82,
                "label_coverage_test": 0.73,
                "train_only_label_coverage": 0.64,
                "task_split_policy": "synthetic_stratified",
            }
        ),
        encoding="utf-8",
    )

    summarize_experiments([tmp_path / "runs"], tmp_path / "summary")

    final_rows = list(csv.DictReader((tmp_path / "summary" / "final_summary.csv").open()))
    quality_rows = list(csv.DictReader((tmp_path / "summary" / "quality_summary.csv").open()))
    score_rows = list(csv.DictReader((tmp_path / "summary" / "score_term_scale.csv").open()))
    source_rows = list(csv.DictReader((tmp_path / "summary" / "candidate_source_pareto.csv").open()))
    task_rows = list(csv.DictReader((tmp_path / "summary" / "task_summary.csv").open()))
    resource_run_rows = list(csv.DictReader((tmp_path / "summary" / "resource_summary_runlevel.csv").open()))
    target_rows = list(csv.DictReader((tmp_path / "summary" / "target_check.csv").open()))
    paper_rows = list(csv.DictReader((tmp_path / "summary" / "paper_table_mean_std.csv").open()))
    paper_dataset_rows = list(csv.DictReader((tmp_path / "summary" / "paper_table_dataset_variant.csv").open()))
    report = (tmp_path / "summary" / "report.md").read_text(encoding="utf-8")

    assert len(final_rows) == 1
    assert (tmp_path / "summary" / "run_final_summary.csv").exists()
    assert (tmp_path / "summary" / "all_levels.csv").exists()
    assert (tmp_path / "summary" / "baseline_summary.csv").exists()
    assert (tmp_path / "summary" / "compare_by_variant.csv").exists()
    assert (tmp_path / "summary" / "compare_by_dataset_variant.csv").exists()
    assert (tmp_path / "summary" / "compare_by_source.csv").exists()
    assert (tmp_path / "summary" / "compare_by_dim.csv").exists()
    assert (tmp_path / "summary" / "paper_table_mean_std.csv").exists()
    assert (tmp_path / "summary" / "paper_table_dataset_variant.csv").exists()
    assert final_rows[0]["run_count_unique"] == "1"
    assert final_rows[0]["experiment_block"] == "B1"
    assert final_rows[0]["unique_run_key"] == "B1:ACM:base:unit"
    assert final_rows[0]["level_row_count"] == "2"
    assert final_rows[0]["final_level"] == "2"
    assert final_rows[0]["initial_nodes"] == "100"
    assert final_rows[0]["final_nodes"] == "52"
    assert np.isclose(float(final_rows[0]["final_cumulative_ratio"]), 0.52)
    assert np.isclose(float(final_rows[0]["target_abs_error"]), 0.02)
    assert final_rows[0]["target_hit"] == "true"
    assert final_rows[0]["candidate_generation_time"] == "4.0"
    assert final_rows[0]["candidate_retained_pair_count"] == "6"
    assert final_rows[0]["candidate_pairs_per_sec"] == "1.5"
    assert final_rows[0]["candidate_substage_times.twohop_expansion"] == "1.4"
    assert final_rows[0]["runtime_by_stage.matching"] == "0.5"
    assert final_rows[0]["runtime_by_stage.aggregation"] == "1.5"
    assert final_rows[0]["selected_source_avg_score.bucket"] == "0.42"
    assert final_rows[0]["selected_source_avg_delta_spec.bucket"] == "0.12"
    assert final_rows[0]["selected_source_avg_delta_conv.bucket"] == "0.08"
    assert final_rows[0]["selected_source_cluster_size_hist.bucket.2"] == "2"
    assert final_rows[0]["bucket_coverage"] == "0.8"
    assert final_rows[0]["partition_count"] == "1"
    assert final_rows[0]["partition_imbalance_max_to_mean"] == "1.0"
    assert final_rows[0]["candidate_buffer_bytes"] == "4096"
    assert final_rows[0]["best_level"] == "2"
    assert final_rows[0]["config.sketch.method"] == "chebyshev_heat"
    assert final_rows[0]["final_DEE"] == "0.1"
    assert final_rows[0]["final_FWE_weighted"] == "0.2"
    assert final_rows[0]["final_FSE_unweighted"] == "0.3"
    assert final_rows[0]["final_REE_max"] == "0.4"
    assert final_rows[0]["final_SIPE"] == "0.5"
    assert final_rows[0]["cumulative_dee"] == "0.11"
    assert final_rows[0]["cumulative_fwe_weighted"] == "0.22"
    assert final_rows[0]["cumulative_fse_unweighted"] == "0.33"
    assert final_rows[0]["cumulative_ree_max"] == "0.44"
    assert final_rows[0]["cumulative_sipe"] == "0.55"
    assert final_rows[0]["cumulative_sampled_eigen_error"] == "0.066"
    assert final_rows[0]["baseline_random_final_cumulative_ratio"] == "0.52"
    assert final_rows[0]["baseline_random_cumulative_dee"] == "0.21"
    assert final_rows[0]["baseline_random_cumulative_fse_unweighted"] == "0.31"
    assert final_rows[0]["baseline_random_cumulative_ree_max"] == "0.41"
    assert final_rows[0]["baseline_random_cumulative_sipe"] == "0.51"
    assert final_rows[0]["baseline_random_cumulative_sampled_eigen_error"] == "0.061"
    assert final_rows[0]["baseline_random_task_projected_macro_f1"] == "0.62"
    assert final_rows[0]["baseline_random_task_refined_macro_f1"] == "0.68"
    assert final_rows[0]["baseline_random_task_train_time"] == "2.0"
    assert final_rows[0]["baseline_random_task_refine_time"] == "0.5"
    assert final_rows[0]["baseline_random_task_total_time"] == "2.5"
    assert final_rows[0]["baseline_heavy_edge_task_projected_macro_f1"] == "0.6"
    assert final_rows[0]["baseline_heavy_edge_task_refined_macro_f1"] == "0.7"
    assert final_rows[0]["baseline_heavy_edge_task_train_time"] == "3.0"
    assert final_rows[0]["baseline_heavy_edge_task_refine_time"] == "0.7"
    assert final_rows[0]["baseline_heavy_edge_task_total_time"] == "3.7"
    assert final_rows[0]["baseline_projected_macro_f1"] == "0.6"
    assert final_rows[0]["baseline_refined_macro_f1"] == "0.7"
    assert final_rows[0]["baseline_train_time"] == "3.0"
    assert final_rows[0]["baseline_refine_time"] == "0.7"
    assert final_rows[0]["baseline_total_time"] == "3.7"
    assert final_rows[0]["baseline_random_runtime_total"] == "1.25"
    assert final_rows[0]["task_projected_macro_f1"] == "0.65"
    assert final_rows[0]["task_refined_macro_f1"] == "0.72"
    assert final_rows[0]["task_refined_macro_f1@0"] == "0.66"
    assert final_rows[0]["task_refined_macro_f1@1"] == "0.64"
    assert final_rows[0]["task_refined_macro_f1@3"] == "0.71"
    assert final_rows[0]["task_refined_macro_f1@5"] == "0.72"
    assert final_rows[0]["task_best_refined_macro_f1"] == "0.72"
    assert final_rows[0]["task_best_refined_epoch"] == "5"
    assert final_rows[0]["task_refine_auc_macro_f1"] == "0.686"
    assert final_rows[0]["task_coarse_train_macro_f1"] == "0.75"
    assert final_rows[0]["task_primary_metric_name"] == "refined_original_macro_f1"
    assert final_rows[0]["task_primary_macro_f1"] == "0.72"
    assert final_rows[0]["task_macro_f1"] == "0.72"
    assert final_rows[0]["compute_device"] == "cpu"
    assert final_rows[0]["cuda_available"] == "true"
    assert final_rows[0]["cpu_only"] == "false"
    assert final_rows[0]["spectral_baseline_computed_count"] == "3"
    assert final_rows[0]["spectral_exact_eigenvalue_sanity_status"] == "computed"
    assert final_rows[0]["spectral_exact_eigenvalue_sanity_mode"] == "dense_eigvalsh"
    assert final_rows[0]["node_reduction"] == "3"
    assert final_rows[0]["cluster_size_mean"] == "1.03"
    assert final_rows[0]["score_contribution_share_spec"] == "0.4"
    assert np.isclose(float(final_rows[0]["runtime_total_run"]), 9.4)
    assert np.isclose(float(final_rows[0]["peak_rss_gb"]), 2.0)
    assert np.isclose(float(final_rows[0]["peak_cpu_memory_gb"]), 2.0)
    assert np.isclose(float(final_rows[0]["peak_vram_allocated_gb"]), 3.0)
    assert np.isclose(float(final_rows[0]["peak_vram_reserved_gb"]), 4.0)
    assert np.isclose(float(final_rows[0]["peak_gpu_memory_allocated_gb"]), 3.0)
    assert "spectral_fused_sketch_energy_relative_error" in quality_rows[0]
    assert {row["term"] for row in score_rows} == {"spec", "rel", "feat", "conv", "boundary"}
    spec_score = next(row for row in score_rows if row["term"] == "spec")
    assert spec_score["raw_mean"] == "10.0"
    assert spec_score["weighted_normalized_mean"] == "0.4"
    bucket_source = next(row for row in source_rows if row["source"] == "bucket")
    assert bucket_source["candidate_count"] == "3.0"
    assert bucket_source["selected_count"] == "2.0"
    assert bucket_source["avg_score"] == "0.42"
    assert bucket_source["avg_delta_spec"] == "0.12"
    assert bucket_source["avg_delta_conv"] == "0.08"
    assert task_rows[0]["task_projected_macro_f1"] == "0.65"
    assert task_rows[0]["task_refined_macro_f1"] == "0.72"
    assert task_rows[0]["task_best_refined_macro_f1"] == "0.72"
    assert task_rows[0]["task_refine_auc_macro_f1"] == "0.686"
    assert task_rows[0]["task_primary_macro_f1"] == "0.72"
    assert task_rows[0]["label_coverage_train"] == "0.91"
    assert task_rows[0]["label_coverage_val"] == "0.82"
    assert task_rows[0]["label_coverage_test"] == "0.73"
    assert task_rows[0]["train_only_label_coverage"] == "0.64"
    assert task_rows[0]["task_split_policy"] == "synthetic_stratified"
    assert resource_run_rows[0]["peak_rss_gb"] == "2.0"
    assert resource_run_rows[0]["peak_cpu_memory_gb"] == "2.0"
    assert resource_run_rows[0]["peak_vram_allocated_gb"] == "3.0"
    assert resource_run_rows[0]["peak_vram_reserved_gb"] == "4.0"
    assert resource_run_rows[0]["peak_gpu_memory_reserved_gb"] == "4.0"
    assert resource_run_rows[0]["cuda_available"] == "true"
    assert resource_run_rows[0]["cpu_only"] == "false"
    assert target_rows[0]["target_hit_rate"] == "1.0"
    assert paper_rows[0]["variant"] == "base"
    assert paper_rows[0]["compute_device_mark"] == "GPU"
    assert paper_rows[0]["cumulative_dee_mean"] == "0.11"
    assert paper_rows[0]["cumulative_dee_std"] == "0.0"
    assert paper_rows[0]["task_refined_macro_f1_mean_pm_std"] == "0.7200 +/- 0.0000"
    assert paper_dataset_rows[0]["dataset"] == "ACM"
    all_level_rows = list(csv.DictReader((tmp_path / "summary" / "all_levels.csv").open()))
    variant_rows = list(csv.DictReader((tmp_path / "summary" / "compare_by_variant.csv").open()))
    assert len(all_level_rows) == 2
    assert variant_rows[0]["run_count"] == "1"
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "summary" / "figures" / "target_ratio_hit_rate.png").exists()
        assert (tmp_path / "summary" / "figures" / "score_contribution_share.png").exists()
        assert (tmp_path / "summary" / "figures" / "cumulative_vs_final_gap.png").exists()
        assert (tmp_path / "summary" / "figures" / "source_distribution_by_variant.png").exists()
        assert (tmp_path / "summary" / "figures" / "task_vs_cumulative_dee.png").exists()
    assert "Unique runs: 1" in report
    assert "GPU-marked runs: 1" in report
    assert "Level rows: 2" in report
    expected_core_header = (
        "| variant | final ratio | DEE \u2193 | FSE-unweighted \u2193 | "
        "REE-max \u2193 | SIPE \u2193 | macro-F1 \u2191 | runtime \u2193 | peak RAM |"
    )
    assert expected_core_header in report
    assert "task_primary_metric" in report
    assert "peak_vram_reserved_gb" in report
    assert "spectral_baseline_computed_count" in report
    assert "final-level baseline" in report
    assert "cumulative baseline" in report
    assert "HeSF-LVC" in report
    assert "meta-path is optional / disabled" in report


def test_next4_mainline_dry_run_generates_required_variants(tmp_path):
    from experiments.scripts.run_hgb_next4_mainline import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--variants",
            "H0",
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "--target-ratio",
            "0.5",
            "--seeds",
            "12345",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("next4_*/config.yaml")
    }
    assert exit_code == 0
    assert len(configs) == 7
    h0 = next(cfg for name, cfg in configs.items() if "_H0_" in name)
    h2 = next(cfg for name, cfg in configs.items() if "_H2_" in name)
    h3 = next(cfg for name, cfg in configs.items() if "_H3_" in name)
    h4 = next(cfg for name, cfg in configs.items() if "_H4_" in name)
    h6 = next(cfg for name, cfg in configs.items() if "_H6_" in name)
    assert h0["coarsening"]["matching_method"] == "mutual_best"
    assert h0["coarsening"]["max_cluster_size"] == 2
    assert h2["coarsening"]["matching_method"] == "greedy_cluster"
    assert h2["coarsening"]["max_cluster_size"] == 4
    assert h2["sketch"]["dim"] == 16
    assert h2["fusion"]["relation_weighting"]["method"] == "uniform"
    assert h2["metapath_sketch"]["enabled"] is False
    assert h2["scoring"]["lambda_conv"] == 0.5
    assert h3["scoring"]["lambda_conv"] == 0.35
    assert h4["scoring"]["lambda_conv"] == 0.0
    assert h6["scoring"]["lambda_spec"] == 0.0


def test_hgb_lambda_grid_dry_run_generates_lambda_configs(tmp_path):
    from experiments.scripts.run_hgb_lambda_grid import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--variants",
            "H2",
            "--lambda-specs",
            "0",
            "1",
            "--lambda-convs",
            "0",
            "0.5",
            "--lambda-rel",
            "0",
            "--seeds",
            "12345",
            "--dry-run",
        ]
    )

    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.glob("lambda_grid_*/config.yaml"))
    ]
    assert exit_code == 0
    assert len(configs) == 4
    assert {cfg["scoring"]["lambda_spec"] for cfg in configs} == {0.0, 1.0}
    assert {cfg["scoring"]["lambda_conv"] for cfg in configs} == {0.0, 0.5}
    assert {cfg["scoring"]["lambda_rel"] for cfg in configs} == {0.0}


def test_next4_mainline_default_freezes_short_confirmation_matrix(tmp_path):
    from experiments.scripts.run_hgb_next4_mainline import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--target-ratio",
            "0.5",
            "--seeds",
            "12345",
            "--dry-run",
        ]
    )

    run_names = sorted(path.parent.name for path in tmp_path.glob("next4_*/config.yaml"))

    assert exit_code == 0
    assert len(run_names) == 5
    assert {name.split("_")[2] for name in run_names} == {"H0", "H2", "H3", "H4", "H6"}
    assert not any("_H5_" in name for name in run_names)


def test_next4_mainline_limited_twohop_candidate_source_sets_budgeted_mode(tmp_path):
    from experiments.scripts.run_hgb_next4_mainline import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--variants",
            "H2",
            "--target-ratio",
            "0.5",
            "--seeds",
            "12345",
            "--candidate-source",
            "onehop_bucket_limited_twohop",
            "--twohop-budget-per-node",
            "2",
            "--twohop-max-time-budget-sec",
            "15",
            "--dry-run",
        ]
    )

    config_path = next(tmp_path.glob("next4_*/config.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert config["candidates"]["enable_onehop"] is True
    assert config["candidates"]["enable_bucket"] is True
    assert config["candidates"]["enable_capped_twohop"] is True
    assert config["candidates"]["twohop_mode"] == "capped_sampled"
    assert config["candidates"]["twohop_budget_per_node"] == 2
    assert config["candidates"]["twohop_max_time_budget_sec"] == 15.0


def test_ogbn_mag_next4_medium_dry_run_generates_cuda_h2h3h4(tmp_path):
    from experiments.scripts.run_ogbn_mag_next4_medium import main

    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing_subset"),
            "--output",
            str(tmp_path),
            "--variants",
            "H2",
            "H3",
            "H4",
            "--seeds",
            "12345",
            "--device",
            "cuda",
            "--optimized-candidates",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("ogbn_mag_medium_*/config.yaml")
    }

    assert exit_code == 0
    assert len(configs) == 3
    assert {name.split("_")[3] for name in configs} == {"H2", "H3", "H4"}
    h2 = next(cfg for name, cfg in configs.items() if "_H2_" in name)
    h3 = next(cfg for name, cfg in configs.items() if "_H3_" in name)
    h4 = next(cfg for name, cfg in configs.items() if "_H4_" in name)
    assert h2["acceleration"]["device"] == "cuda"
    assert h2["acceleration"]["dense_backend"] == "torch"
    assert h2["candidates"]["store_backend"] == "array"
    assert h2["candidates"]["use_chunked_generation"] is True
    assert h2["candidates"]["enable_capped_twohop"] is False
    assert h2["diagnostics"]["enable_large_graph_envelope"] is True
    assert h3["scoring"]["lambda_conv"] == 0.35
    assert h4["scoring"]["lambda_conv"] == 0.0


def test_ogbn_mag_next4_medium_limited_twohop_mode_sets_budgeted_twohop(tmp_path):
    from experiments.scripts.run_ogbn_mag_next4_medium import main

    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing_subset"),
            "--output",
            str(tmp_path),
            "--variants",
            "H2",
            "--seeds",
            "12345",
            "--device",
            "cuda",
            "--candidate-mode",
            "limited_twohop",
            "--twohop-budget-per-node",
            "2",
            "--twohop-max-time-budget-sec",
            "30",
            "--dry-run",
        ]
    )

    config_path = next(tmp_path.glob("ogbn_mag_medium_*/config.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert config["candidates"]["enable_onehop"] is True
    assert config["candidates"]["enable_bucket"] is True
    assert config["candidates"]["enable_capped_twohop"] is True
    assert config["candidates"]["twohop_mode"] == "capped_sampled"
    assert config["candidates"]["twohop_budget_per_node"] == 2
    assert config["candidates"]["twohop_max_time_budget_sec"] == 30.0


def test_ogbn_mag_next4_medium_accepts_protocol_variant_names(tmp_path):
    from experiments.scripts.run_ogbn_mag_next4_medium import main

    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing_subset"),
            "--output",
            str(tmp_path),
            "--variants",
            "H2-opt",
            "H3-opt",
            "H4-opt",
            "flatten-sum-opt",
            "--seeds",
            "12345",
            "--device",
            "cuda",
            "--candidate-mode",
            "optimized",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("ogbn_mag_medium_*/config.yaml")
    }
    flatten_cfg = next(cfg for name, cfg in configs.items() if "flatten-sum-opt" in name)

    assert exit_code == 0
    assert len(configs) == 4
    assert flatten_cfg["fusion"]["relation_operator_mode"] == "single_relation_sum"
    assert flatten_cfg["scoring"]["relation_profile_mode"] == "single_relation_sum"


def test_next4_relation_fusion_dry_run_generates_required_variants(tmp_path):
    from experiments.scripts.run_hgb_next4_relation_fusion import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--target-ratio",
            "0.5",
            "--seeds",
            "12345",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("next4_rel_*/config.yaml")
    }

    assert exit_code == 0
    assert len(configs) == 4
    full = next(cfg for name, cfg in configs.items() if "_H2-full_" in name)
    flatten = next(cfg for name, cfg in configs.items() if "_H2-single-relation-sum_" in name)
    no_rel = next(cfg for name, cfg in configs.items() if "_H2-no-rel-term_" in name)
    fused_only = next(cfg for name, cfg in configs.items() if "_H2-uniform-fused-only_" in name)
    assert full["fusion"]["relation_operator_mode"] == "relationwise"
    assert full["scoring"]["lambda_rel"] > 0.0
    assert flatten["fusion"]["relation_operator_mode"] == "single_relation_sum"
    assert flatten["scoring"]["relation_profile_mode"] == "single_relation_sum"
    assert no_rel["scoring"]["lambda_rel"] == 0.0
    assert no_rel["scoring"]["relation_guard"]["enabled"] is False
    assert fused_only["scoring"]["lambda_rel"] == 0.0
    assert fused_only["scoring"]["lambda_conv"] == 0.0
    assert fused_only["scoring"]["lambda_feat"] == 0.0
    assert fused_only["scoring"]["lambda_boundary"] == 0.0
    assert fused_only["diagnostics"]["spectral_relation_detail"] is False


def test_next4_mainline_dry_run_generates_terminal_guard_variants(tmp_path):
    from experiments.scripts.run_hgb_next4_mainline import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--variants",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "--target-ratio",
            "0.25",
            "--seeds",
            "12345",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("next4_*/config.yaml")
    }
    assert exit_code == 0
    assert len(configs) == 5
    a0 = next(cfg for name, cfg in configs.items() if "_A0_" in name)
    a1 = next(cfg for name, cfg in configs.items() if "_A1_" in name)
    a2 = next(cfg for name, cfg in configs.items() if "_A2_" in name)
    a3 = next(cfg for name, cfg in configs.items() if "_A3_" in name)
    a4 = next(cfg for name, cfg in configs.items() if "_A4_" in name)
    assert a0["coarsening"]["target_ratio"] == 0.25
    assert a0["coarsening"]["matching_method"] == "greedy_cluster"
    assert a0["coarsening"]["max_cluster_size"] == 4
    assert a0["coarsening"]["terminal_guard"]["enabled"] is False
    assert a1["coarsening"]["terminal_guard"]["protect_hubs"] is True
    assert a2["coarsening"]["terminal_guard"]["protect_rare_relation_carriers"] is True
    assert a3["coarsening"]["terminal_guard"]["protect_train_label_conflict_nodes"] is True
    assert a4["coarsening"]["terminal_guard"]["protect_hubs"] is True
    assert a4["coarsening"]["terminal_guard"]["protect_rare_relation_carriers"] is True
    assert a4["coarsening"]["terminal_guard"]["protect_boundary_nodes"] is True
    assert a4["coarsening"]["terminal_guard"]["protect_train_label_conflict_nodes"] is True


def test_next4_baseline_summary_marks_failed_target_control(tmp_path):
    from experiments.scripts.run_hgb_next4_baselines import main

    summary = tmp_path / "final_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "run_name,dataset,variant,target_ratio,baseline_random_target_hit,baseline_random_target_abs_error,baseline_random_final_cumulative_ratio,baseline_random_cumulative_dee,baseline_random_projected_macro_f1,baseline_random_refined_macro_f1@0,baseline_random_refined_macro_f1@1,baseline_random_refined_macro_f1@3,baseline_random_refined_macro_f1@5,baseline_heavy_edge_target_hit,baseline_heavy_edge_target_abs_error,baseline_heavy_edge_final_cumulative_ratio",
                "r1,ACM,H2,0.5,true,0.01,0.51,0.2,0.6,0.61,0.62,0.63,0.64,false,0.44,0.94",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--summary",
            str(summary),
            "--variants",
            "H2",
            "--baselines",
            "random",
            "heavy_edge",
            "--output",
            str(tmp_path / "out"),
        ]
    )

    rows = list(csv.DictReader((tmp_path / "out" / "baseline_summary.csv").open()))
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert rows[0]["baseline"] == "random"
    assert rows[0]["baseline_target_hit"] == "true"
    assert rows[1]["baseline"] == "heavy_edge"
    assert rows[1]["comparison_status"] == "failed target control"
    assert "failed target control" in report


def test_next4_baseline_script_computes_target_matched_rows(tmp_path):
    from experiments.scripts.run_hgb_next4_baselines import main

    graph = generate_synthetic_graph(num_users=16, num_items=10, num_tags=6, seed=1701)
    save_graph(graph, tmp_path / "data" / "acm_hesf")
    summary = tmp_path / "final_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "run_name,dataset,variant,seed,target_ratio",
                "r1,ACM,H2,12345,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--summary",
            str(summary),
            "--variants",
            "H2",
            "--baselines",
            "random",
            "graphzoom_style",
            "convmatch_style",
            "--graph-root",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--spectral-exact-eigenvalue-max-nodes",
            "64",
        ]
    )

    rows = list(csv.DictReader((tmp_path / "out" / "baseline_summary.csv").open()))
    wide = list(csv.DictReader((tmp_path / "out" / "final_summary_with_baselines.csv").open()))
    assert exit_code == 0
    assert {row["baseline"] for row in rows} == {"random", "graphzoom_style", "convmatch_style"}
    for row in rows:
        assert row["comparison_status"] == "included"
        assert row["baseline_target_hit"] == "True"
        assert float(row["baseline_target_abs_error"]) <= 0.02
        assert row["baseline_cumulative_dee"] != ""
        assert row["baseline_cumulative_sipe"] != ""
    assert wide[0]["baseline_random_target_hit"] == "True"
    assert wide[0]["baseline_graphzoom_style_target_hit"] == "True"


def test_evaluate_refine_curve_uses_cached_task_eval(tmp_path):
    from experiments.scripts.evaluate_refine_curve import main

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "task_eval.json").write_text(
        json.dumps(
            {
                "run_name": "r1",
                "dataset": "ACM",
                "variant": "H2",
                "projected_original_macro_f1": 0.61,
                "refined_original_macro_f1@0": 0.62,
                "refined_original_macro_f1@1": 0.58,
                "refined_original_macro_f1@3": 0.66,
                "refined_original_macro_f1@5": 0.67,
                "best_refined_macro_f1": 0.67,
                "best_refined_epoch": 5,
                "refine_auc_macro_f1": 0.64,
                "full_graph_rgcn_lite_macro_f1": 0.7,
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "run_name,run_dir,dataset,variant\n"
        f"r1,{run_dir},ACM,H2\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--summary",
            str(summary),
            "--variants",
            "H2",
            "--refine-epochs",
            "0",
            "1",
            "3",
            "5",
            "--output",
            str(tmp_path / "out"),
            "--graph-root",
            str(tmp_path / "missing_graph_root"),
        ]
    )

    rows = list(csv.DictReader((tmp_path / "out" / "task_refine_curve.csv").open()))
    assert exit_code == 0
    assert len(rows) == 4
    assert {row["source"] for row in rows} == {"task_eval_cache"}
    by_epoch = {row["refine_epochs"]: row for row in rows}
    assert by_epoch["1"]["refined_original_macro_f1"] == "0.58"
    assert by_epoch["5"]["best_refined_macro_f1"] == "0.67"
    assert by_epoch["5"]["full_graph_macro_f1"] == "0.7"


def test_compare_hgb_ablation_groups_metrics(tmp_path):
    from experiments.scripts.compare_hgb_ablation import compare_hgb_ablation

    input_csv = tmp_path / "all_runs.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "variant", "final_cumulative_ratio", "cumulative_dee"],
        )
        writer.writeheader()
        writer.writerow({"dataset": "ACM", "variant": "base", "final_cumulative_ratio": "0.51", "cumulative_dee": "0.1"})
        writer.writerow({"dataset": "ACM", "variant": "base", "final_cumulative_ratio": "0.49", "cumulative_dee": "0.2"})
    output_csv = tmp_path / "compare.csv"

    compare_hgb_ablation(
        summary=input_csv,
        output=output_csv,
        group_by=["dataset", "variant"],
        metrics=["final_cumulative_ratio", "cumulative_dee"],
    )

    rows = list(csv.DictReader(output_csv.open()))
    assert rows[0]["dataset"] == "ACM"
    assert rows[0]["variant"] == "base"
    assert rows[0]["run_count"] == "2"
    assert np.isclose(float(rows[0]["final_cumulative_ratio_mean"]), 0.5)


def test_compare_hgb_ablation_has_stage_b_default_metrics(tmp_path):
    from experiments.scripts.compare_hgb_ablation import compare_hgb_ablation

    input_csv = tmp_path / "run_final_summary.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "row_type", "final_DEE"])
        writer.writeheader()
        writer.writerow({"dataset": "ACM", "variant": "base", "row_type": "final", "final_DEE": "0.2"})
    output_csv = tmp_path / "compare.csv"

    compare_hgb_ablation(
        summary=input_csv,
        output=output_csv,
        group_by=["dataset", "variant"],
    )

    rows = list(csv.DictReader(output_csv.open()))
    assert rows[0]["run_count"] == "1"
    assert rows[0]["final_DEE_mean"] == "0.2"


def test_stage_b_ablation_dry_run_writes_variants(tmp_path):
    from experiments.scripts.run_hgb_stage_b_ablation import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--sketch-dims",
            "16",
            "32",
            "--sketch-orders",
            "3",
            "--seeds",
            "12345",
            "--variants",
            "base",
            "uniform_weight",
            "no_metapath",
            "lazy_no_metapath",
            "no_conv",
            "--dry-run",
        ]
    )

    configs = {path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in tmp_path.glob("hgb_*/config.yaml")}
    assert exit_code == 0
    if not configs:
        configs = {path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in tmp_path.glob("stageB_*/config.yaml")}
    assert len(configs) == 10
    assert all(name.startswith("stageB_") for name in configs)
    assert any("base" in name for name in configs)
    assert any(cfg["fusion"]["relation_weighting"]["method"] == "uniform" for cfg in configs.values())
    assert any(cfg["metapath_sketch"]["enabled"] is False for cfg in configs.values())
    assert any(cfg["sketch"]["method"] == "lazy" for cfg in configs.values())
    assert any(cfg["scoring"]["lambda_conv"] == 0.0 for cfg in configs.values())
    assert any(cfg["sketch"]["dim"] == 32 for cfg in configs.values())
    assert all(cfg["sketch"]["order"] == 3 for cfg in configs.values())
    assert all(cfg["scoring"]["normalization"] == "p95" for cfg in configs.values())


def test_stage_b_ablation_supports_next_measurement_variants_and_cli(tmp_path):
    from experiments.scripts.run_hgb_stage_b_ablation import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--sketch-order",
            "3",
            "--seeds",
            "12345",
            "--variants",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "V0",
            "V1",
            "V4",
            "C1-stop",
            "C2-repair",
            "C2-size3",
            "--lambda-conv",
            "0.5",
            "--spectral-baseline-max-nodes",
            "50000",
            "--spectral-exact-eigenvalue-max-nodes",
            "64",
            "--cumulative-spectral-exact-eigenvalue-max-nodes",
            "64",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("stageB_*/config.yaml")
    }
    assert exit_code == 0
    assert len(configs) == 12
    a0 = next(cfg for name, cfg in configs.items() if "_A0_" in name)
    a1 = next(cfg for name, cfg in configs.items() if "_A1_" in name)
    a2 = next(cfg for name, cfg in configs.items() if "_A2_" in name)
    a4 = next(cfg for name, cfg in configs.items() if "_A4_" in name)
    v0 = next(cfg for name, cfg in configs.items() if "_V0_" in name)
    v1 = next(cfg for name, cfg in configs.items() if "_V1_" in name)
    v4 = next(cfg for name, cfg in configs.items() if "_V4_" in name)
    c1_stop_name, c1_stop = next((name, cfg) for name, cfg in configs.items() if "_C1-stop_" in name)
    c2_size3_name, c2_size3 = next((name, cfg) for name, cfg in configs.items() if "_C2-size3_" in name)
    c2_repair_name, c2_repair = next(
        (name, cfg) for name, cfg in configs.items() if "_C2-repair_" in name
    )

    assert a0["fusion"]["relation_weighting"]["method"] == "uniform"
    assert a1["fusion"]["relation_weighting"]["method"] == "inverse_sqrt_energy"
    assert a2["metapath_sketch"]["enabled"] is True
    assert a2["metapath_sketch"]["operator_weight_total"] == 0.1
    assert a4["sketch"]["method"] == "lazy"
    assert a4["sketch"]["dim"] == 32
    assert a4["fusion"]["relation_weighting"]["method"] == "uniform"
    assert v0["scoring"]["lambda_conv"] == 0.0
    assert v1["scoring"]["lambda_conv"] == 0.5
    assert v4["fusion"]["relation_weighting"]["method"] == "uniform"
    assert c1_stop["coarsening"]["max_levels"] == 6
    assert "_L6_" in c1_stop_name
    assert c2_size3["coarsening"]["max_cluster_size"] == 3
    assert "_c3_" in c2_size3_name
    assert c2_repair["coarsening"]["matching_method"] == "greedy_cluster"
    assert "_greedy_cluster_c4_" in c2_repair_name
    assert c2_repair["coarsening"]["cumulative_guard"]["repair_bad_clusters"] is True
    assert c2_repair["diagnostics"]["spectral_baseline_max_nodes"] == 50000


def test_stage_b_ablation_supports_m_g_s_next_stage_matrices(tmp_path):
    from experiments.scripts.run_hgb_stage_b_ablation import main

    m_output = tmp_path / "m"
    m_exit = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(m_output),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--sketch-order",
            "3",
            "--seeds",
            "12345",
            "--variants",
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "--dry-run",
        ]
    )
    m_configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in m_output.glob("stageB_*/config.yaml")
    }
    assert m_exit == 0
    assert len(m_configs) == 6
    m0 = next(cfg for name, cfg in m_configs.items() if "_M0_" in name)
    m1 = next(cfg for name, cfg in m_configs.items() if "_M1_" in name)
    m2 = next(cfg for name, cfg in m_configs.items() if "_M2_" in name)
    m3 = next(cfg for name, cfg in m_configs.items() if "_M3_" in name)
    m4 = next(cfg for name, cfg in m_configs.items() if "_M4_" in name)
    m5 = next(cfg for name, cfg in m_configs.items() if "_M5_" in name)
    assert m0["sketch"]["method"] == "chebyshev_heat"
    assert m0["sketch"]["dim"] == 16
    assert m0["fusion"]["relation_weighting"]["method"] == "uniform"
    assert m0["metapath_sketch"]["enabled"] is False
    assert m0["scoring"]["lambda_conv"] == 0.5
    assert m1["scoring"]["lambda_conv"] == 0.25
    assert m2["scoring"]["lambda_conv"] == 0.75
    assert m3["fusion"]["relation_weighting"]["method"] == "capped_inverse_sqrt_energy"
    assert m4["sketch"]["method"] == "lazy"
    assert m4["sketch"]["dim"] == 32
    assert m5["metapath_sketch"]["enabled"] is True
    assert m5["metapath_sketch"]["preset"] == "canonical"
    assert m5["metapath_sketch"]["operator_weight_total"] == 0.1

    g_output = tmp_path / "g"
    g_exit = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(g_output),
            "--target-ratios",
            "0.25",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--seeds",
            "12345",
            "--variants",
            "G0",
            "G1",
            "G2",
            "G3",
            "G4",
            "--dry-run",
        ]
    )
    g_configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in g_output.glob("stageB_*/config.yaml")
    }
    assert g_exit == 0
    assert len(g_configs) == 5
    g0 = next(cfg for name, cfg in g_configs.items() if "_G0_" in name)
    g1 = next(cfg for name, cfg in g_configs.items() if "_G1_" in name)
    g2 = next(cfg for name, cfg in g_configs.items() if "_G2_" in name)
    g3 = next(cfg for name, cfg in g_configs.items() if "_G3_" in name)
    g4 = next(cfg for name, cfg in g_configs.items() if "_G4_" in name)
    assert g0["coarsening"]["matching_method"] == "greedy_cluster"
    assert g0["coarsening"]["max_cluster_size"] == 4
    assert g0["coarsening"]["cumulative_guard"]["repair_bad_clusters"] is False
    assert g1["coarsening"]["cumulative_guard"]["repair_strategy"] == "current"
    assert g2["coarsening"]["cumulative_guard"]["repair_strategy"] == "split_high_spread"
    assert g3["coarsening"]["cumulative_guard"]["repair_strategy"] == "split_local_swap_accept"
    assert g3["coarsening"]["cumulative_guard"]["accept_only_if_cumulative_improves"] is True
    assert g4["coarsening"]["max_cluster_size"] == 3

    s_output = tmp_path / "s"
    s_exit = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(s_output),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--seeds",
            "12345",
            "--variants",
            "S0",
            "S1",
            "S2",
            "S3",
            "--dry-run",
        ]
    )
    s_configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in s_output.glob("stageB_*/config.yaml")
    }
    assert s_exit == 0
    assert len(s_configs) == 4
    s0 = next(cfg for name, cfg in s_configs.items() if "_S0_" in name)
    s1 = next(cfg for name, cfg in s_configs.items() if "_S1_" in name)
    s2 = next(cfg for name, cfg in s_configs.items() if "_S2_" in name)
    s3 = next(cfg for name, cfg in s_configs.items() if "_S3_" in name)
    assert s0["sketch"]["method"] == "chebyshev_heat"
    assert s1["sketch"]["method"] == "lazy"
    assert s1["sketch"]["dim"] == 32
    assert s2["candidates"]["quotas"]["bucket_min_fraction"] == 0.3
    assert s2["candidates"]["quotas"]["twohop_max_fraction"] == 0.7
    assert s2["candidates"]["quotas"]["enforce_on"] == "selected_matches"
    assert s3["candidates"]["quotas"]["bucket_min_fraction"] == 0.3
    assert s3["candidates"]["quotas"]["twohop_max_fraction"] == 0.7
    assert s3["candidates"]["quotas"]["enforce_on"] == "selected_matches"


def test_stage_b_ablation_supports_pdf_next_round_variants(tmp_path):
    from experiments.scripts.run_hgb_stage_b_ablation import main

    exit_code = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(tmp_path),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--seeds",
            "12345",
            "--variants",
            "M0-repeat",
            "M0-conv0.35",
            "M0-conv0.65",
            "M0-relation-guard",
            "G3-fixed",
            "G3-task",
            "G3-relation",
            "--dry-run",
        ]
    )

    configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("stageB_*/config.yaml")
    }
    assert exit_code == 0
    assert len(configs) == 7
    repeat = next(cfg for name, cfg in configs.items() if "_M0-repeat_" in name)
    conv035 = next(cfg for name, cfg in configs.items() if "_M0-conv0.35_" in name)
    conv065 = next(cfg for name, cfg in configs.items() if "_M0-conv0.65_" in name)
    relation_guard = next(cfg for name, cfg in configs.items() if "_M0-relation-guard_" in name)
    g3_fixed = next(cfg for name, cfg in configs.items() if "_G3-fixed_" in name)
    g3_task = next(cfg for name, cfg in configs.items() if "_G3-task_" in name)
    g3_relation = next(cfg for name, cfg in configs.items() if "_G3-relation_" in name)

    assert repeat["sketch"]["method"] == "chebyshev_heat"
    assert repeat["sketch"]["dim"] == 16
    assert repeat["fusion"]["relation_weighting"]["method"] == "uniform"
    assert repeat["metapath_sketch"]["enabled"] is False
    assert repeat["scoring"]["lambda_conv"] == 0.5
    assert conv035["scoring"]["lambda_conv"] == 0.35
    assert conv065["scoring"]["lambda_conv"] == 0.65
    assert relation_guard["scoring"]["relation_guard"]["enabled"] is True
    assert g3_fixed["coarsening"]["matching_method"] == "greedy_cluster"
    assert g3_fixed["coarsening"]["cumulative_guard"]["accept_metric"] == "true_cumulative"
    assert g3_task["coarsening"]["cumulative_guard"]["objective"] == "task"
    assert g3_relation["coarsening"]["cumulative_guard"]["objective"] == "relation"


def test_stage_b_ablation_supports_matching_and_repair_objective_variants(tmp_path):
    from experiments.scripts.run_hgb_stage_b_ablation import main
    from experiments.scripts.run_hgb_task_eval import build_parser as build_task_eval_parser

    p_output = tmp_path / "p"
    p_exit = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(p_output),
            "--target-ratios",
            "0.5",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--seeds",
            "12345",
            "--variants",
            "P0",
            "P1",
            "P2",
            "P3",
            "--dry-run",
        ]
    )
    p_configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in p_output.glob("stageB_*/config.yaml")
    }
    assert p_exit == 0
    assert len(p_configs) == 4
    p0 = next(cfg for name, cfg in p_configs.items() if "_P0_" in name)
    p1 = next(cfg for name, cfg in p_configs.items() if "_P1_" in name)
    p2 = next(cfg for name, cfg in p_configs.items() if "_P2_" in name)
    p3 = next(cfg for name, cfg in p_configs.items() if "_P3_" in name)
    assert p0["sketch"]["method"] == "chebyshev_heat"
    assert p0["sketch"]["dim"] == 16
    assert p0["fusion"]["relation_weighting"]["method"] == "uniform"
    assert p0["metapath_sketch"]["enabled"] is False
    assert p0["scoring"]["lambda_conv"] == 0.5
    assert p0["coarsening"]["matching_method"] == "mutual_best"
    assert p0["coarsening"]["max_cluster_size"] == 2
    assert p1["coarsening"]["matching_method"] == "greedy_cluster"
    assert p1["coarsening"]["max_cluster_size"] == 3
    assert p2["coarsening"]["matching_method"] == "greedy_cluster"
    assert p2["coarsening"]["max_cluster_size"] == 4
    assert p3["coarsening"]["matching_method"] == "greedy_cluster"
    assert p3["coarsening"]["max_cluster_size"] == 4
    assert p3["scoring"]["lambda_conv"] == 0.35

    g_output = tmp_path / "g"
    g_exit = main(
        [
            "--datasets",
            "ACM",
            "--output",
            str(g_output),
            "--target-ratios",
            "0.25",
            "--max-levels",
            "4",
            "--candidate-source",
            "onehop_twohop_bucket",
            "--candidate-K",
            "8",
            "--seeds",
            "12345",
            "--variants",
            "G3-energy",
            "G3-relation",
            "G3-task",
            "--dry-run",
        ]
    )
    g_configs = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in g_output.glob("stageB_*/config.yaml")
    }
    assert g_exit == 0
    assert len(g_configs) == 3
    g3_energy = next(cfg for name, cfg in g_configs.items() if "_G3-energy_" in name)
    g3_relation = next(cfg for name, cfg in g_configs.items() if "_G3-relation_" in name)
    g3_task = next(cfg for name, cfg in g_configs.items() if "_G3-task_" in name)
    assert g3_energy["coarsening"]["cumulative_guard"]["repair_objective"] == "energy"
    assert g3_relation["coarsening"]["cumulative_guard"]["repair_objective"] == "relation"
    assert g3_task["coarsening"]["cumulative_guard"]["repair_objective"] == "task"

    parsed = build_task_eval_parser().parse_args(["--runs-root", "runs", "--refine-epochs-list", "0", "1", "3", "5"])
    assert parsed.refine_epochs_list == [0, 1, 3, 5]


def test_sanity_runner_outputs_summary_and_report(tmp_path):
    from experiments.scripts.run_sanity import run_sanity

    output = tmp_path / "sanity"
    exit_code = run_sanity(output=output, python=sys.executable)

    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert exit_code == 0
    assert rows
    assert all(row["status"] == "success" for row in rows)
    assert all(row["schema_type_violations"] == "0" for row in rows)
    assert all(row["invalid_assignment_count"] == "0" for row in rows)
    assert (output / "report.md").exists()


def test_sanity_runner_writes_per_level_diagnose_outputs(tmp_path):
    from experiments.scripts.run_sanity import run_sanity

    output = tmp_path / "sanity"
    exit_code = run_sanity(output=output, python=sys.executable)

    assert exit_code == 0
    for diagnose_path in sorted(output.glob("sanity_*_level/level_*/diagnose.json")):
        payload = json.loads(diagnose_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload
    assert len(list(output.glob("sanity_*_level/level_*/diagnose.json"))) == 3


def test_sanity_runner_returns_nonzero_when_any_run_fails(tmp_path, monkeypatch):
    import experiments.scripts.run_sanity as run_sanity_module

    def fail_coarsening(*args, **kwargs):
        raise RuntimeError("forced sanity failure")

    monkeypatch.setattr(run_sanity_module, "run_multilevel_coarsening", fail_coarsening)

    output = tmp_path / "sanity"
    exit_code = run_sanity_module.run_sanity(output=output, python=sys.executable)

    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert exit_code == 1
    assert rows
    assert all(row["status"] == "failed" for row in rows)


def test_subset_sampler_is_deterministic_and_writes_diagnostics(tmp_path):
    from experiments.scripts.run_ogbn_mag_subset import sample_relation_aware_subset

    graph = generate_synthetic_graph(num_users=50, num_items=30, num_tags=10, seed=1301)
    save_graph(graph, tmp_path / "input")

    first = sample_relation_aware_subset(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "subset_a",
        target_nodes=40,
        edge_budget=200,
        seed=7,
    )
    second = sample_relation_aware_subset(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "subset_b",
        target_nodes=40,
        edge_budget=200,
        seed=7,
    )

    graph_a = load_graph(first)
    graph_b = load_graph(second)
    assert np.array_equal(graph_a.node_type, graph_b.node_type)
    assert (Path(first) / "subset_diagnostics.json").exists()
    with (Path(first) / "subset_diagnostics.json").open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    assert diagnostics["target_nodes"] == 40
    assert diagnostics["actual_nodes"] <= 40


def test_spectral_diagnostics_api_returns_bounded_metrics(tmp_path):
    from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics

    graph = generate_synthetic_graph(num_users=8, num_items=5, num_tags=3, seed=1401)
    result = run_multilevel_coarsening(graph, _tiny_config(tmp_path / "run"))[0]
    z = np.arange(graph.num_nodes * 3, dtype=np.float32).reshape(graph.num_nodes, 3) / 10.0

    diagnostics = compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        smoothing_steps=1,
        relation_weights={relation_id: 1.0 for relation_id in graph.relations},
        Z=z,
        exact_eigenvalue_max_nodes=64,
        baseline_methods=["random", "heavy_edge", "graphzoom_style", "convmatch_style"],
        baseline_max_nodes=64,
    )

    assert "dirichlet_energy_relative_error" in diagnostics
    assert diagnostics["dirichlet_energy_relative_error"] >= 0.0
    assert diagnostics["relation_energy_relative_error_max"] >= 0.0
    assert "relation_weighted_fused_energy_relative_error" in diagnostics
    assert "chebheat_sketch_inner_product_relative_error" in diagnostics
    assert "exact_eigenvalue_sanity" in diagnostics
    assert set(diagnostics["baseline_comparison"]) == {
        "random",
        "heavy_edge",
        "graphzoom_style",
        "convmatch_style",
    }


def test_spectral_diagnostics_can_skip_per_relation_detail(tmp_path):
    from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics

    graph = generate_synthetic_graph(num_users=8, num_items=5, num_tags=3, seed=1411)
    result = run_multilevel_coarsening(graph, _tiny_config(tmp_path / "run"))[0]
    z = np.arange(graph.num_nodes * 3, dtype=np.float32).reshape(graph.num_nodes, 3) / 10.0

    diagnostics = compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        smoothing_steps=1,
        relation_weights={relation_id: 1.0 for relation_id in graph.relations},
        Z=z,
        relation_detail=False,
    )

    assert "relation_energy_before" not in diagnostics
    assert "relation_energy_after" not in diagnostics
    assert "relation_energy_relative_error" not in diagnostics
    assert "relation_energy_relative_error_max" not in diagnostics
    assert "fused_sketch_energy_relative_error" in diagnostics


def test_spectral_diagnostics_target_matches_cumulative_baselines(tmp_path):
    from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics

    graph = generate_synthetic_graph(num_users=12, num_items=8, num_tags=4, seed=1402)
    result = run_multilevel_coarsening(graph, _tiny_config(tmp_path / "run"))[0]
    z = np.arange(graph.num_nodes * 3, dtype=np.float32).reshape(graph.num_nodes, 3) / 10.0

    diagnostics = compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        smoothing_steps=1,
        relation_weights={relation_id: 1.0 for relation_id in graph.relations},
        Z=z,
        exact_eigenvalue_max_nodes=64,
        baseline_methods=["random", "heavy_edge"],
        baseline_max_nodes=64,
        baseline_target_ratio=0.5,
        baseline_target_tolerance=0.05,
        baseline_max_levels=4,
    )

    for baseline in diagnostics["baseline_comparison"].values():
        assert baseline["status"] == "computed"
        assert baseline["target_ratio"] == 0.5
        assert baseline["target_abs_error"] <= 0.05
        assert baseline["target_hit"] is True
        assert baseline["levels"] >= 1


def test_spectral_diagnostics_can_evaluate_target_matched_baseline_tasks(tmp_path, monkeypatch):
    import hesf_coarsen.eval.spectral_diagnostics as spectral_diagnostics
    from hesf_coarsen.eval.task_gnn import TaskEvalResult

    graph = generate_synthetic_graph(num_users=12, num_items=8, num_tags=4, seed=1403)
    graph.labels = np.arange(graph.num_nodes, dtype=np.int64) % 2
    result = run_multilevel_coarsening(graph, _tiny_config(tmp_path / "run"))[0]
    z = np.arange(graph.num_nodes * 3, dtype=np.float32).reshape(graph.num_nodes, 3) / 10.0
    calls = []

    def fake_evaluate_rgcn_task(original, coarse, original_to_coarse, **params):
        calls.append((original, coarse, original_to_coarse, params))
        return TaskEvalResult(
            {
                "projected_original_macro_f1": 0.61,
                "refined_original_macro_f1": 0.67,
                "refined_original_macro_f1@0": 0.62,
                "refined_original_macro_f1@1": 0.63,
                "refined_original_macro_f1@3": 0.66,
                "refined_original_macro_f1@5": 0.67,
                "best_refined_macro_f1": 0.67,
                "best_refined_epoch": 5,
                "refine_auc_macro_f1": 0.65,
                "refine_time_by_epoch": {"0": 0.0, "1": 0.1, "3": 0.25, "5": 0.3},
                "train_time": 1.2,
                "refine_time": 0.3,
                "total_time": 1.5,
            }
        )

    monkeypatch.setattr(spectral_diagnostics, "evaluate_rgcn_task", fake_evaluate_rgcn_task)

    diagnostics = spectral_diagnostics.compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        smoothing_steps=1,
        relation_weights={relation_id: 1.0 for relation_id in graph.relations},
        Z=z,
        baseline_methods=["random"],
        baseline_max_nodes=64,
        baseline_target_ratio=0.5,
        baseline_target_tolerance=0.05,
        baseline_max_levels=4,
        baseline_task_eval=True,
        baseline_task_eval_params={"epochs": 2, "refine_epochs": 1, "device": "cpu"},
    )

    baseline = diagnostics["baseline_comparison"]["random"]
    assert len(calls) == 1
    assert calls[0][3]["epochs"] == 2
    assert calls[0][3]["refine_epochs"] == 1
    assert calls[0][3]["device"] == "cpu"
    assert baseline["task_projected_macro_f1"] == 0.61
    assert baseline["task_refined_macro_f1"] == 0.67
    assert baseline["task_refined_macro_f1@0"] == 0.62
    assert baseline["task_refined_macro_f1@1"] == 0.63
    assert baseline["task_refined_macro_f1@3"] == 0.66
    assert baseline["task_refined_macro_f1@5"] == 0.67
    assert baseline["task_best_refined_macro_f1"] == 0.67
    assert baseline["task_best_refined_epoch"] == 5
    assert baseline["task_refine_auc_macro_f1"] == 0.65
    assert baseline["task_train_time"] == 1.2
    assert baseline["task_refine_time"] == 0.3
    assert baseline["task_total_time"] == 1.5


def test_synthetic_scale_estimate_has_expected_fields():
    from experiments.scripts.make_synthetic_scale import estimate_scale_bytes

    estimate = estimate_scale_bytes(nodes=1_000_000, edges=10_000_000, feature_dim=32, candidate_k=16)

    assert estimate["relation_arrays_bytes"] > 0
    assert estimate["candidate_store_bytes"] > estimate["sketch_bytes_fp16"]
    assert estimate["expected_disk_footprint_bytes"] >= estimate["relation_arrays_bytes"]
