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
        "summarize_stage_b.py",
        "compare_stage_b.py",
        "compare_hgb_ablation.py",
        "run_ogbn_mag_subset.py",
        "run_ogbn_mag_envelope.py",
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
        },
        "task": {"micro_f1": 0.7, "macro_f1": 0.6},
        "candidate_count_total": 4,
        "candidate_source_counts": {"bucket": 3, "onehop": 1},
        "matched_pairs": 2,
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
        "cumulative_spectral": {
            "sketch_dirichlet_energy_relative_error": 0.11,
            "relation_weighted_fused_energy_relative_error": 0.22,
            "fused_sketch_energy_relative_error": 0.33,
            "relation_energy_relative_error_max": 0.44,
            "chebheat_sketch_inner_product_relative_error": 0.55,
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

    summarize_experiments([tmp_path / "runs"], tmp_path / "summary")

    final_rows = list(csv.DictReader((tmp_path / "summary" / "final_summary.csv").open()))
    quality_rows = list(csv.DictReader((tmp_path / "summary" / "quality_summary.csv").open()))
    score_rows = list(csv.DictReader((tmp_path / "summary" / "score_term_scale.csv").open()))
    source_rows = list(csv.DictReader((tmp_path / "summary" / "candidate_source_pareto.csv").open()))
    task_rows = list(csv.DictReader((tmp_path / "summary" / "task_summary.csv").open()))
    resource_run_rows = list(csv.DictReader((tmp_path / "summary" / "resource_summary_runlevel.csv").open()))
    target_rows = list(csv.DictReader((tmp_path / "summary" / "target_check.csv").open()))
    report = (tmp_path / "summary" / "report.md").read_text(encoding="utf-8")

    assert len(final_rows) == 1
    assert (tmp_path / "summary" / "run_final_summary.csv").exists()
    assert (tmp_path / "summary" / "all_levels.csv").exists()
    assert (tmp_path / "summary" / "compare_by_variant.csv").exists()
    assert (tmp_path / "summary" / "compare_by_source.csv").exists()
    assert (tmp_path / "summary" / "compare_by_dim.csv").exists()
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
    assert final_rows[0]["task_macro_f1"] == "0.6"
    assert final_rows[0]["score_contribution_share_spec"] == "0.4"
    assert np.isclose(float(final_rows[0]["runtime_total_run"]), 0.0)
    assert np.isclose(float(final_rows[0]["peak_rss_gb"]), 2.0)
    assert np.isclose(float(final_rows[0]["peak_vram_allocated_gb"]), 3.0)
    assert np.isclose(float(final_rows[0]["peak_vram_reserved_gb"]), 4.0)
    assert "spectral_fused_sketch_energy_relative_error" in quality_rows[0]
    assert {row["term"] for row in score_rows} == {"spec", "rel", "feat", "conv", "boundary"}
    spec_score = next(row for row in score_rows if row["term"] == "spec")
    assert spec_score["raw_mean"] == "10.0"
    assert spec_score["weighted_normalized_mean"] == "0.4"
    bucket_source = next(row for row in source_rows if row["source"] == "bucket")
    assert bucket_source["candidate_count"] == "3.0"
    assert bucket_source["selected_count"] == "2.0"
    assert task_rows[0]["task_macro_f1"] == "0.6"
    assert resource_run_rows[0]["peak_rss_gb"] == "2.0"
    assert resource_run_rows[0]["peak_vram_allocated_gb"] == "3.0"
    assert resource_run_rows[0]["peak_vram_reserved_gb"] == "4.0"
    assert target_rows[0]["target_hit_rate"] == "1.0"
    all_level_rows = list(csv.DictReader((tmp_path / "summary" / "all_levels.csv").open()))
    variant_rows = list(csv.DictReader((tmp_path / "summary" / "compare_by_variant.csv").open()))
    assert len(all_level_rows) == 2
    assert variant_rows[0]["run_count"] == "1"
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "summary" / "figures" / "target_ratio_hit_rate.png").exists()
        assert (tmp_path / "summary" / "figures" / "score_contribution_share.png").exists()
    assert "Unique runs: 1" in report
    assert "Level rows: 2" in report
    assert "| variant | final ratio | DEE ↓ | FSE-unweighted ↓ | REE-max ↓ | SIPE ↓ | macro-F1 ↑ | runtime ↓ | peak RAM |" in report
    assert "final_DEE" in report


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


def test_synthetic_scale_estimate_has_expected_fields():
    from experiments.scripts.make_synthetic_scale import estimate_scale_bytes

    estimate = estimate_scale_bytes(nodes=1_000_000, edges=10_000_000, feature_dim=32, candidate_k=16)

    assert estimate["relation_arrays_bytes"] > 0
    assert estimate["candidate_store_bytes"] > estimate["sketch_bytes_fp16"]
    assert estimate["expected_disk_footprint_bytes"] >= estimate["relation_arrays_bytes"]
