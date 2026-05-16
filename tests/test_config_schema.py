from pathlib import Path

import yaml

from hesf_coarsen.config import DEFAULT_CONFIG, load_config


CONFIG_PATHS = sorted(Path("configs").rglob("*.yaml"))


def test_default_config_uses_canonical_sketch_schema():
    config = load_config()

    assert config["sketch"]["method"] == "chebyshev_heat"
    assert config["sketch"]["chebyshev_scaling"] == "estimate_norm"
    assert config["metapath_sketch"]["enabled"] is False
    assert config["metapath_sketch"]["preset"] == "off"
    assert config["metapath_sketch"]["auto_paths"] is False
    assert config["metapath_sketch"]["operator_weight_total"] == 0.0
    assert config["metapath_sketch"]["weighting"]["method"] == "uniform"
    assert isinstance(config["fusion"]["relation_weighting"], dict)
    assert config["fusion"]["relation_weighting"]["method"] == "uniform"
    assert config["fusion"]["relation_operator_mode"] == "relationwise"
    assert config["fusion"]["symmetric_relation_scale"] == 0.5
    assert config["fusion"]["estimate_operator_norm"] is True
    assert config["fusion"]["chebyshev_rescale_if_needed"] is True
    assert config["coarsening"]["feature_aggregation"] == "mean"
    assert config["scoring"]["spec_volume_weighting"] is True
    assert config["scoring"]["relation_profile_distance"] == "jsd"
    assert config["scoring"]["relation_profile_mode"] == "relationwise"
    assert config["scoring"]["conv_response_operator"] == "fused_operator"
    assert config["scoring"]["boundary_mode"] == "node_risk"
    assert config["scoring"]["boundary_hub_gamma"] > 0.0
    assert config["scoring"]["boundary_terminal_gamma"] > 0.0
    assert config["features"]["projector"] == "gaussian_random"
    assert "include_metapath_filters" not in config["fusion"]
    assert "include_metapath_filters" not in DEFAULT_CONFIG["fusion"]


def test_default_config_promotes_hesf_lvc_mainline():
    config = load_config()

    assert config["coarsening"]["target_ratio"] == 0.5
    assert config["coarsening"]["matching_method"] == "greedy_cluster"
    assert config["coarsening"]["max_cluster_size"] == 4
    assert config["coarsening"]["same_type_only"] is True
    assert config["coarsening"]["same_partition_only"] is True
    assert config["sketch"]["method"] == "chebyshev_heat"
    assert config["sketch"]["dim"] == 16
    assert config["sketch"]["order"] == 5
    assert config["sketch"]["dtype"] == "float16"
    assert config["sketch"]["row_normalize"] is True
    assert config["fusion"]["relation_weighting"]["method"] == "uniform"
    assert config["metapath_sketch"]["enabled"] is False
    assert config["metapath_sketch"]["operator_weight_total"] == 0.0
    assert config["scoring"]["normalization"] == "p95"
    assert config["scoring"]["normalization_scope"] == "level"
    assert config["scoring"]["lambda_spec"] == 1.0
    assert config["scoring"]["lambda_conv"] == 0.5
    assert config["diagnostics"]["spectral_relation_detail"] is True
    assert config["candidates"]["source"] == "onehop_twohop_bucket"
    assert config["candidates"]["total_budget_K"] == 8


def test_explicit_legacy_override_can_still_select_mutual_best(tmp_path):
    override = tmp_path / "legacy.yaml"
    override.write_text(
        """
coarsening:
  matching_method: mutual_best
  max_cluster_size: 2
fusion:
  relation_weighting:
    method: inverse_sqrt_energy
metapath_sketch:
  enabled: true
  operator_weight_total: 0.1
""",
        encoding="utf-8",
    )

    config = load_config(override)

    assert config["coarsening"]["matching_method"] == "mutual_best"
    assert config["coarsening"]["max_cluster_size"] == 2
    assert config["fusion"]["relation_weighting"]["method"] == "inverse_sqrt_energy"
    assert config["metapath_sketch"]["enabled"] is True
    assert config["metapath_sketch"]["operator_weight_total"] == 0.1


def test_shipped_configs_do_not_use_legacy_schema_names():
    assert CONFIG_PATHS
    for path in CONFIG_PATHS:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        loaded = load_config(path)

        assert loaded["sketch"]["method"] in {"lazy", "chebyshev_heat"}
        assert loaded["sketch"]["method"] != "repeated_smoothing"
        if loaded["sketch"]["method"] == "chebyshev_heat":
            assert loaded["sketch"]["chebyshev_scaling"] in {
                "estimate_norm",
                "normalized_laplacian_2",
            }
        assert isinstance(loaded["fusion"]["relation_weighting"], dict)
        assert raw.get("fusion", {}).get("relation_weighting") != "uniform"
        assert "include_metapath_filters" not in raw.get("fusion", {})


def test_main_chebheat_config_enables_metapath_sketch():
    config = load_config("configs/sketch_chebheat_metapath.yaml")

    assert config["sketch"]["method"] == "chebyshev_heat"
    assert config["metapath_sketch"]["enabled"] is True
    assert config["metapath_sketch"]["auto_paths"] is True
    assert config["metapath_sketch"]["dim"] > 0


def test_large_graph_configs_enable_projected_feature_store():
    for path in [
        "configs/ogbn_mag_A_cpu_chunked.yaml",
        "configs/ogbn_mag_B_cpu_ann.yaml",
        "configs/ogbn_mag_C_torch_ann.yaml",
    ]:
        config = load_config(path)

        assert config["features"]["projection_mmap_dir"]
        assert config["features"]["projection_dtype"] == "float16"
        assert int(config["features"]["projection_chunk_size"]) > 0
