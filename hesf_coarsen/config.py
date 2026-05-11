from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 12345,
    "hardware": {
        "gpu": "optional",
        "max_vram_gb": 24,
        "max_ram_gb": 256,
    },
    "acceleration": {
        "dense_backend": "numpy",
        "device": "auto",
        "fallback_to_numpy": True,
        "max_dense_bytes": None,
        "scoring_batch_size": 65_536,
    },
    "progress": {
        "enabled": False,
        "backend": "auto",
        "min_interval_seconds": 1.0,
    },
    "coarsening": {
        "target_ratio": 0.1,
        "max_levels": 6,
        "per_level_ratio": 0.55,
        "same_type_only": True,
        "same_partition_only": True,
    },
    "sketch": {
        "dim": 32,
        "order": 5,
        "num_scales": 2,
        "dtype": "float16",
        "probe": "rademacher",
        "method": "repeated_smoothing",
    },
    "fusion": {
        "relation_weighting": "uniform",
        "include_metapath_filters": False,
    },
    "candidates": {
        "store_backend": "heap",
        "use_chunked_generation": False,
        "mmap_dir": None,
        "incident_index_mmap_dir": None,
        "edge_chunk_size": 1_000_000,
        "middle_chunk_size": 100_000,
        "node_chunk_size": 1_000_000,
        "total_budget_K": 16,
        "twohop_budget_K2": 8,
        "middle_degree_cap_policy": "p99",
        "per_middle_pair_cap": 64,
        "bucket_pair_cap": 64,
        "enable_onehop": True,
        "enable_capped_twohop": True,
        "enable_bucket": True,
        "enable_partition_ann": False,
        "ann_num_projections": 4,
        "ann_window_size": 8,
        "ann_budget_K": 8,
        "simhash_bits": 16,
    },
    "scoring": {
        "lambda_spec": 1.0,
        "lambda_rel": 0.2,
        "lambda_feat": 0.1,
        "lambda_conv": 0.3,
        "lambda_boundary": 0.1,
    },
    "features": {
        "projected_dim": 32,
    },
    "diagnostics": {
        "enable_large_graph_envelope": False,
        "edge_sample_size": 1024,
    },
    "output": {
        "dir": "outputs/default_run",
    },
}


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config at {path} must contain a YAML mapping")
    return deep_update(config, loaded)
