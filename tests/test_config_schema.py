from pathlib import Path

import yaml

from hesf_coarsen.config import DEFAULT_CONFIG, load_config


CONFIG_PATHS = sorted(Path("configs").glob("*.yaml"))


def test_default_config_uses_canonical_sketch_schema():
    config = load_config()

    assert config["sketch"]["method"] == "chebyshev_heat"
    assert isinstance(config["fusion"]["relation_weighting"], dict)
    assert config["fusion"]["relation_weighting"]["method"] == "inverse_energy"
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
