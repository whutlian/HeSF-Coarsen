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
    "resume": {
        "enabled": False,
        "allow_legacy_checkpoints": False,
    },
    "coarsening": {
        "target_ratio": 0.5,
        "max_levels": 6,
        "per_level_ratio": 0.55,
        "same_type_only": True,
        "same_partition_only": True,
        "matching_method": "greedy_cluster",
        "max_cluster_size": 4,
        "feature_aggregation": "mean",
        "feature_aggregation_pagerank_iterations": 20,
        "feature_aggregation_pagerank_damping": 0.85,
        "aggregation_chunk_size": 1_000_000,
        "aggregation_reducer": "sort",
        "terminal_guard": {
            "enabled": False,
            "protect_hubs": False,
            "protect_rare_relation_carriers": False,
            "protect_boundary_nodes": False,
            "protect_train_label_conflict_nodes": False,
            "hub_degree_percentile": 95,
            "rare_relation_min_count": 1,
            "label_entropy_threshold": 0.0,
            "max_terminal_cluster_size": 2,
        },
        "cumulative_guard": {
            "enabled": False,
            "probe_count": 32,
            "max_cumulative_dee": 0.35,
            "max_cumulative_sipe": 0.70,
            "repair_bad_clusters": False,
        },
    },
    "sketch": {
        "dim": 16,
        "order": 5,
        "num_scales": 2,
        "dtype": "float16",
        "probe": "rademacher",
        "method": "chebyshev_heat",
        "heat_times": [1.0, 3.0],
        "chebyshev_scaling": "estimate_norm",
        "chebyshev_quadrature_points": 128,
        "internal_dtype": "float32",
        "row_normalize": True,
    },
    "fusion": {
        "symmetric_relation_operator": True,
        "symmetric_relation_scale": 0.5,
        "relation_operator_mode": "relationwise",
        "estimate_operator_norm": True,
        "operator_norm_iterations": 8,
        "operator_norm_probe_dim": 4,
        "operator_norm_tolerance": 1e-3,
        "chebyshev_rescale_if_needed": True,
        "reverse_relation_policy": "include_all",
        "relation_weighting": {
            "method": "uniform",
            "eta": 0.5,
            "gamma": 1.0,
            "epsilon": 1e-8,
            "sample_edges_per_relation": 200_000,
            "energy_basis": "random",
            "seed": 12345,
        },
    },
    "metapath_sketch": {
        "enabled": False,
        "dim": 8,
        "preset": "off",
        "operator_weight_total": 0.0,
        "weighting": {
            "method": "uniform",
            "eta": 0.5,
            "gamma": 1.0,
            "epsilon": 1e-8,
            "energy_basis": "random",
            "seed": 12345,
        },
        "max_paths": 3,
        "max_path_length": 3,
        "auto_paths": False,
        "seed": 123,
        "row_normalize": True,
        "paths": [],
    },
    "candidates": {
        "source": "onehop_twohop_bucket",
        "store_backend": "heap",
        "use_chunked_generation": False,
        "mmap_dir": None,
        "incident_index_mmap_dir": None,
        "edge_chunk_size": 1_000_000,
        "middle_chunk_size": 100_000,
        "node_chunk_size": 1_000_000,
        "pair_block_size": 65_536,
        "total_budget_K": 8,
        "twohop_budget_K2": 8,
        "twohop_mode": "full",
        "twohop_budget_per_node": 0,
        "twohop_max_time_budget_sec": None,
        "middle_degree_cap_policy": "p99",
        "per_middle_pair_cap": 64,
        "bucket_pair_cap": 64,
        "enable_onehop": True,
        "enable_capped_twohop": True,
        "enable_bucket": True,
        "enable_partition_ann": False,
        "enable_fallback": True,
        "fallback_penalty": 1.0e6,
        "fallback_max_fraction": 1.0,
        "ann_num_projections": 4,
        "ann_window_size": 8,
        "ann_budget_K": 8,
        "simhash_bits": 16,
        "hash_tables": None,
        "multi_probe": False,
        "hamming_radius": 0,
        "adaptive_hamming_radius": False,
        "quotas": {
            "enforce_on": "candidate_retention",
            "bucket_min_fraction": 0.0,
            "twohop_max_fraction": 1.0,
            "fallback_max_fraction": 1.0,
        },
    },
    "scoring": {
        "lambda_spec": 1.0,
        "lambda_rel": 0.2,
        "lambda_feat": 0.1,
        "lambda_conv": 0.5,
        "lambda_boundary": 0.1,
        "normalization": "p95",
        "normalization_scope": "level",
        "spec_volume_weighting": True,
        "spec_volume_epsilon": 1e-12,
        "relation_profile_distance": "jsd",
        "relation_profile_mode": "relationwise",
        "relation_profile_epsilon": 1e-12,
        "relation_guard": {
            "enabled": False,
            "max_ree_increase": 0.02,
            "penalty_weight": 10.0,
        },
        "conv_response_operator": "fused_operator",
        "boundary_mode": "node_risk",
        "boundary_hub_gamma": 0.05,
        "boundary_terminal_gamma": 1.0,
        "boundary_terminal_degree": 1.0,
    },
    "features": {
        "projected_dim": 32,
        "projection_dtype": "float16",
        "projector": "gaussian_random",
        "projection_mmap_dir": None,
        "projection_chunk_size": 100_000,
    },
    "diagnostics": {
        "enable_large_graph_envelope": False,
        "spectral_relation_detail": True,
        "edge_sample_size": 1024,
        "enable_spectral": True,
        "spectral_num_signals": 4,
        "spectral_smoothing_steps": 1,
        "spectral_exact_eigenvalue_max_nodes": 256,
        "spectral_baseline_max_nodes": 5000,
        "score_term_sample_size": 200_000,
        "spectral_baselines": [
            "random",
            "heavy_edge",
            "graphzoom_style",
            "convmatch_style",
        ],
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
