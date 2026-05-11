import json
from pathlib import Path

import numpy as np
import pytest

from hesf_coarsen.cli.main import build_parser
from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph


def resume_config(tmp_path: Path, max_levels: int) -> dict:
    config = dict(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.1,
        max_levels=max_levels,
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


def test_multilevel_writes_checkpoint_and_assignment(tmp_path):
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=101)

    results = run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=1))

    assert results
    level_dir = tmp_path / "level_1"
    checkpoint_path = level_dir / "checkpoint.json"
    assignment_path = level_dir / "assignment.npz"
    assert checkpoint_path.exists()
    assert assignment_path.exists()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    assignment_payload = np.load(assignment_path)
    assert checkpoint["complete"] is True
    assert checkpoint["level"] == 1
    assert checkpoint["coarse_nodes"] == results[0].graph.num_nodes
    assert assignment_payload["assignment"].shape == (graph.num_nodes,)


def test_resume_continues_from_last_completed_level(tmp_path):
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=103)
    run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=1))
    config = resume_config(tmp_path, max_levels=3)
    config["resume"] = {"enabled": True, "allow_legacy_checkpoints": False}

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert results[0].level == 2
    assert (tmp_path / "level_2" / "checkpoint.json").exists()


def test_existing_completed_output_requires_resume(tmp_path):
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=105)
    run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=1))

    with pytest.raises(FileExistsError, match="resume"):
        run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=2))


def test_resume_accepts_legacy_completed_level_when_enabled(tmp_path):
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=107)
    run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=1))
    (tmp_path / "level_1" / "checkpoint.json").unlink()
    config = resume_config(tmp_path, max_levels=3)
    config["resume"] = {"enabled": True, "allow_legacy_checkpoints": True}

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert results[0].level == 2
    assert (tmp_path / "level_2" / "checkpoint.json").exists()


def test_resume_ignores_incomplete_next_level(tmp_path):
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=109)
    run_multilevel_coarsening(graph, resume_config(tmp_path, max_levels=1))
    incomplete = tmp_path / "level_2"
    incomplete.mkdir()
    (incomplete / "diagnostics.json").write_text("{}", encoding="utf-8")
    config = resume_config(tmp_path, max_levels=3)
    config["resume"] = {"enabled": True, "allow_legacy_checkpoints": False}

    results = run_multilevel_coarsening(graph, config)

    assert results
    assert results[0].level == 2
    assert load_graph(incomplete).num_nodes == results[0].graph.num_nodes
    assert (incomplete / "checkpoint.json").exists()


def test_coarsen_cli_accepts_resume_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "coarsen",
            "--config",
            "configs/default.yaml",
            "--input",
            "data/tiny",
            "--output",
            "outputs/tiny",
            "--resume",
            "--allow-legacy-checkpoints",
        ]
    )

    assert args.resume is True
    assert args.allow_legacy_checkpoints is True
