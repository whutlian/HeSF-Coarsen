import csv
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
        "run_hgb_sweep.py",
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
                "runtime_by_stage": {"sketch": 1.0, "candidates": 2.0},
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
    failed_rows = list(csv.DictReader((tmp_path / "summary" / "failures.csv").open()))
    assert {row["status"] for row in all_rows} == {"success", "failed"}
    assert failed_rows[0]["failure_reason"] == "boom"
    assert (tmp_path / "summary" / "report.md").exists()


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

    diagnostics = compute_spectral_diagnostics(
        original=graph,
        coarse=result.graph,
        assignment=result.assignment,
        seed=11,
        num_signals=3,
        smoothing_steps=1,
    )

    assert "dirichlet_energy_relative_error" in diagnostics
    assert diagnostics["dirichlet_energy_relative_error"] >= 0.0
    assert diagnostics["relation_energy_relative_error_max"] >= 0.0


def test_synthetic_scale_estimate_has_expected_fields():
    from experiments.scripts.make_synthetic_scale import estimate_scale_bytes

    estimate = estimate_scale_bytes(nodes=1_000_000, edges=10_000_000, feature_dim=32, candidate_k=16)

    assert estimate["relation_arrays_bytes"] > 0
    assert estimate["candidate_store_bytes"] > estimate["sketch_bytes_fp16"]
    assert estimate["expected_disk_footprint_bytes"] >= estimate["relation_arrays_bytes"]
