from pathlib import Path

import yaml

from hesf_coarsen.config import DEFAULT_CONFIG, load_config


CONFIG_PATHS = sorted(Path("configs").glob("*.yaml"))


def test_default_config_uses_canonical_sketch_schema():
    config = load_config()

    assert config["sketch"]["method"] == "chebyshev_heat"
    assert config["metapath_sketch"]["enabled"] is True
    assert config["metapath_sketch"]["auto_paths"] is True
    assert config["metapath_sketch"]["weighting"]["method"] == "inverse_energy"
    assert isinstance(config["fusion"]["relation_weighting"], dict)
    assert config["fusion"]["relation_weighting"]["method"] == "inverse_energy"
    assert config["fusion"]["symmetric_relation_scale"] == 0.5
    assert config["fusion"]["estimate_operator_norm"] is True
    assert config["fusion"]["chebyshev_rescale_if_needed"] is True
    assert config["coarsening"]["feature_aggregation"] == "mean"
    assert "include_metapath_filters" not in config["fusion"]
    assert "include_metapath_filters" not in DEFAULT_CONFIG["fusion"]


def test_shipped_configs_do_not_use_legacy_schema_names():
    assert CONFIG_PATHS
    for path in CONFIG_PATHS:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        loaded = load_config(path)

        assert loaded["sketch"]["method"] in {"lazy", "chebyshev_heat"}
        assert loaded["sketch"]["method"] != "repeated_smoothing"
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
