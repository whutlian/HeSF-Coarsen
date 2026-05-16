from copy import deepcopy
from pathlib import Path

import numpy as np

from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import load_config
from hesf_coarsen.eval.invariants import validate_level_invariants
from hesf_coarsen.io.edge_list import generate_synthetic_graph


EXPECTED_PAPER_CONFIGS = {
    "hgb_hesf_lvc_p.yaml": {
        "method": "HeSF-LVC-P",
        "variant": "H2",
        "lambda_spec": 0.25,
        "lambda_conv": 0.0,
        "lambda_rel": 0.0,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_hesf_lvc_s.yaml": {
        "method": "HeSF-LVC-S",
        "variant": "H3",
        "lambda_spec": 0.5,
        "lambda_conv": 0.0,
        "lambda_rel": 0.0,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_hesf_lvc_t.yaml": {
        "method": "HeSF-LVC-T",
        "variant": "H2",
        "lambda_spec": 2.0,
        "lambda_conv": 0.25,
        "lambda_rel": 0.0,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_h0_mutual_best.yaml": {
        "method": "H0-mutual-best",
        "variant": "H0",
        "lambda_spec": 1.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "mutual_best",
        "max_cluster_size": 2,
    },
    "hgb_h6_no_spec.yaml": {
        "method": "H6-no-spec",
        "variant": "H6",
        "lambda_spec": 0.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_flatten_sum.yaml": {
        "method": "flatten-sum",
        "variant": "H2-single-relation-sum",
        "lambda_spec": 1.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_random_target_matched.yaml": {
        "method": "random",
        "variant": "target-matched-baseline",
        "baseline_method": "random",
        "lambda_spec": 1.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_graphzoom_style.yaml": {
        "method": "GraphZoom-style",
        "variant": "target-matched-baseline",
        "baseline_method": "graphzoom_style",
        "lambda_spec": 1.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
    "hgb_convmatch_style.yaml": {
        "method": "ConvMatch-style",
        "variant": "target-matched-baseline",
        "baseline_method": "convmatch_style",
        "lambda_spec": 1.0,
        "lambda_conv": 0.5,
        "lambda_rel": 0.2,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
    },
}

NEXT9_PAPER_CONFIGS = {
    "hgb_hesf_lvc_t_appendix.yaml",
    "hgb_hesf_lvc_p_spectral_guard.yaml",
    "hgb_hesf_lvc_s_spectral_guard.yaml",
    "hgb_hesf_lvc_p_sourceaware_auto.yaml",
    "hgb_hesf_lvc_s_sourceaware_auto.yaml",
    "ogbn_mag_next9_opt_aggregation.yaml",
}


def _tiny_smoke_config(config: dict, tmp_path: Path) -> dict:
    cfg = deepcopy(config)
    cfg["coarsening"] = dict(
        cfg["coarsening"],
        target_ratio=0.7,
        max_levels=1,
        per_level_ratio=0.7,
    )
    cfg["sketch"] = dict(cfg["sketch"], dim=8, order=2, dtype="float32")
    cfg["candidates"] = dict(
        cfg["candidates"],
        total_budget_K=8,
        twohop_budget_K2=4,
        per_middle_pair_cap=16,
        bucket_pair_cap=16,
        simhash_bits=4,
        enable_fallback=True,
    )
    cfg["diagnostics"] = dict(cfg["diagnostics"], enable_spectral=False)
    cfg["output"] = {"dir": str(tmp_path)}
    return cfg


def test_next7_paper_configs_exist_and_load_expected_profiles():
    paper_dir = Path("configs/paper")
    paths = {path.name: path for path in paper_dir.glob("*.yaml")}

    assert set(EXPECTED_PAPER_CONFIGS).issubset(paths)
    assert NEXT9_PAPER_CONFIGS.issubset(paths)
    for name, expected in EXPECTED_PAPER_CONFIGS.items():
        config = load_config(paths[name])

        assert config["paper"]["method"] == expected["method"]
        assert config["paper"]["variant"] == expected["variant"]
        if "baseline_method" in expected:
            assert config["paper"]["baseline_method"] == expected["baseline_method"]
        assert config["coarsening"]["same_type_only"] is True
        assert config["coarsening"]["same_partition_only"] is True
        assert config["coarsening"]["matching_method"] == expected["matching_method"]
        assert config["coarsening"]["max_cluster_size"] == expected["max_cluster_size"]
        assert config["sketch"]["method"] == "chebyshev_heat"
        assert config["sketch"]["dim"] == 16
        assert config["sketch"]["order"] == 5
        assert config["fusion"]["relation_weighting"]["method"] == "uniform"
        assert config["metapath_sketch"]["enabled"] is False
        assert config["scoring"]["lambda_spec"] == expected["lambda_spec"]
        assert config["scoring"]["lambda_conv"] == expected["lambda_conv"]
        assert config["scoring"]["lambda_rel"] == expected["lambda_rel"]
        assert config["candidates"]["total_budget_K"] == 8
        assert config["candidates"]["enable_onehop"] is True
        assert config["candidates"]["enable_capped_twohop"] is True
        assert config["candidates"]["enable_bucket"] is True
        assert "include_metapath_filters" not in config.get("fusion", {})


def test_next7_paper_configs_smoke_on_tiny_graph_and_echo_values(tmp_path):
    graph = generate_synthetic_graph(num_users=7, num_items=5, num_tags=3, seed=1701)

    for path in sorted(Path("configs/paper").glob("*.yaml")):
        config = _tiny_smoke_config(load_config(path), tmp_path / path.stem)

        result = run_multilevel_coarsening(graph, config)[0]
        validation = validate_level_invariants(
            original=graph,
            coarse=result.graph,
            assignment=result.assignment,
            diagnostics_path=tmp_path / path.stem / "level_1" / "diagnostics.json",
        )
        diagnostics_config = result.diagnostics["config"]

        assert validation["schema_type_violations"] == 0
        assert validation["invalid_assignment_count"] == 0
        assert result.diagnostics["candidate_count_max"] <= config["candidates"]["total_budget_K"]
        assert diagnostics_config["paper"]["method"] == config["paper"]["method"]
        assert diagnostics_config["paper"]["variant"] == config["paper"]["variant"]
        assert diagnostics_config["scoring"]["lambda_spec"] == config["scoring"]["lambda_spec"]
        assert diagnostics_config["scoring"]["lambda_conv"] == config["scoring"]["lambda_conv"]
        assert diagnostics_config["scoring"]["lambda_rel"] == config["scoring"]["lambda_rel"]
        assert np.isclose(
            diagnostics_config["coarsening"]["target_ratio"],
            config["coarsening"]["target_ratio"],
        )
