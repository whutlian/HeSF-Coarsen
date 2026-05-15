from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.ops.normalization import relation_degrees
from hesf_coarsen.progress import progress_iter


@dataclass(frozen=True)
class RelationWeightResult:
    weights: dict[int, float]
    energy_estimates: dict[int, float]
    volume_estimates: dict[int, float]
    diagnostics: dict[str, Any]


def _weighting_config(config: dict[str, Any]) -> dict[str, Any]:
    fusion = config.get("fusion", {})
    raw = fusion.get("relation_weighting", "uniform")
    if isinstance(raw, dict):
        result = dict(raw)
    else:
        method = str(raw).lower()
        if method in {"reliability", "reliability_weighted"}:
            method = "inverse_energy"
        result = {"method": method}
    if "eta" not in result and "volume_eta" in fusion:
        result["eta"] = fusion["volume_eta"]
    if "epsilon" not in result and "energy_epsilon" in fusion:
        result["epsilon"] = fusion["energy_epsilon"]
    return result


def _normalize(raw: dict[int, float]) -> dict[int, float]:
    cleaned = {relation_id: max(float(value), 0.0) for relation_id, value in raw.items()}
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        uniform = 1.0 / max(len(cleaned), 1)
        return {relation_id: uniform for relation_id in cleaned}
    return {relation_id: value / total for relation_id, value in cleaned.items()}


def _entropy(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr > 0.0]
    if len(arr) == 0:
        return 0.0
    arr = arr / max(float(arr.sum()), 1e-12)
    return float(-np.sum(arr * np.log(arr)))


def _clip_and_normalize(
    weights: dict[int, float],
    *,
    min_weight: float | None,
    max_weight: float | None,
) -> dict[int, float]:
    if min_weight is None and max_weight is None:
        return _normalize(weights)
    clipped = {
        relation_id: float(
            np.clip(
                value,
                -np.inf if min_weight is None else float(min_weight),
                np.inf if max_weight is None else float(max_weight),
            )
        )
        for relation_id, value in weights.items()
    }
    return _normalize(clipped)


def _relation_volume(graph: HeteroGraph, relation_id: int) -> float:
    rel = graph.relations[int(relation_id)]
    if rel.num_edges == 0:
        return 0.0
    return float(np.sum(rel.weight.astype(np.float64)))


def _random_basis(graph: HeteroGraph, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(graph.num_nodes, int(dim)))


def _feature_projection_dtype(config: dict[str, Any]) -> np.dtype:
    name = str(config.get("features", {}).get("projection_dtype", "float16"))
    dtype = np.dtype(name)
    if dtype not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("features.projection_dtype must be float16 or float32")
    return dtype


def _project_feature_block(
    feature: np.ndarray,
    *,
    type_id: int,
    target_dim: int,
    seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape[1] <= target_dim:
        out = np.zeros((feature.shape[0], target_dim), dtype=dtype)
        out[:, : feature.shape[1]] = feature.astype(dtype, copy=False)
        return out
    rng = np.random.default_rng(int(seed) + 1009 * int(type_id))
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(target_dim)),
        size=(feature.shape[1], target_dim),
    ).astype(np.float32)
    return (feature @ projection).astype(dtype)


def _feature_basis(graph: HeteroGraph, config: dict[str, Any]) -> np.ndarray | None:
    if not graph.features:
        return None
    max_width = max(feature.shape[1] for feature in graph.features.values())
    projected_dim = int(config.get("features", {}).get("projected_dim", 32))
    width = max_width if projected_dim <= 0 else min(max_width, projected_dim)
    dtype = _feature_projection_dtype(config)
    seed = int(config.get("seed", 12345))
    basis = np.zeros((graph.num_nodes, width), dtype=dtype)
    for type_id, feature in graph.features.items():
        basis[nodes_of_type(graph, int(type_id))] = _project_feature_block(
            feature,
            type_id=int(type_id),
            target_dim=width,
            seed=seed,
            dtype=dtype,
        )
    return basis


def _basis_for_energy(
    graph: HeteroGraph,
    config: dict[str, Any],
    weight_cfg: dict[str, Any],
    basis: np.ndarray | None,
    method: str,
) -> tuple[np.ndarray, str]:
    if basis is not None:
        B = np.asarray(basis, dtype=np.float32)
        if B.ndim == 1:
            B = B[:, None]
        if B.shape[0] != graph.num_nodes:
            raise ValueError("basis must have one row per graph node")
        return B, "provided"
    if method == "feature_smoothness":
        feature_basis = _feature_basis(graph, config)
        if feature_basis is not None:
            return feature_basis, "features"
    dim = int(weight_cfg.get("energy_basis_dim", 8))
    seed = int(weight_cfg.get("seed", config.get("seed", 12345)))
    return _random_basis(graph, dim, seed), "random"


def _sample_edge_indices(length: int, sample_size: int | None, seed: int, relation_id: int) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or length <= sample_size:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + int(relation_id) * 7919)
    return np.sort(rng.choice(length, size=int(sample_size), replace=False).astype(np.int64))


def _relation_energy(
    graph: HeteroGraph,
    relation_id: int,
    basis: np.ndarray,
    *,
    epsilon: float,
    sample_edges_per_relation: int | None,
    seed: int,
    chunk_size: int = 200_000,
    progress_config: dict[str, Any] | None = None,
) -> float:
    rel = graph.relations[int(relation_id)]
    denom = float(np.sum(basis.astype(np.float64) * basis.astype(np.float64))) + float(epsilon)
    if rel.num_edges == 0 or denom <= 0.0:
        return 0.0

    src_degree, dst_degree = relation_degrees(graph, rel)
    indices = _sample_edge_indices(rel.num_edges, sample_edges_per_relation, seed, relation_id)
    total = 0.0
    effective_chunk_size = max(int(chunk_size), 1)
    chunk_starts = range(0, len(indices), effective_chunk_size)
    for start in progress_iter(
        chunk_starts,
        total=ceil(len(indices) / effective_chunk_size) if len(indices) else 0,
        desc=f"relation energy r={relation_id}",
        config=progress_config,
        unit="chunk",
    ):
        idx = indices[start : start + effective_chunk_size]
        src = rel.src[idx]
        dst = rel.dst[idx]
        src_scale = np.sqrt(src_degree[src].astype(np.float64) + float(epsilon))[:, None]
        dst_scale = np.sqrt(dst_degree[dst].astype(np.float64) + float(epsilon))[:, None]
        diff = basis[src].astype(np.float64) / src_scale - basis[dst].astype(np.float64) / dst_scale
        total += float(np.sum(rel.weight[idx].astype(np.float64) * np.sum(diff * diff, axis=1)))
    if len(indices) and len(indices) < rel.num_edges:
        total *= float(rel.num_edges / len(indices))
    return float(max(total / denom, 0.0))


def compute_relation_weights(
    graph: HeteroGraph,
    config: dict[str, Any] | None = None,
    *,
    basis: np.ndarray | None = None,
) -> RelationWeightResult:
    """Compute normalized non-negative alpha_r weights for fused sketching."""

    config = config or {}
    weight_cfg = _weighting_config(config)
    method = str(weight_cfg.get("method", "uniform")).lower()
    if method in {"reliability", "reliability_weighted"}:
        method = "inverse_energy"
    if method in {"capped_inverse_sqrt", "clipped_inverse_sqrt", "capped_inverse_sqrt_energy"}:
        method = "capped_inverse_sqrt_energy"
    supported = {
        "uniform",
        "volume",
        "inverse_energy",
        "clipped_inverse_energy",
        "inverse_sqrt_energy",
        "capped_inverse_sqrt_energy",
        "smoothed_inverse_energy",
        "feature_smoothness",
    }
    if method not in supported:
        raise ValueError(f"unsupported fusion.relation_weighting.method: {method}")

    relation_ids = sorted(graph.relations)
    epsilon = float(weight_cfg.get("epsilon", 1e-8))
    eta = float(weight_cfg.get("eta", 0.5))
    default_gamma = 0.5 if method in {"inverse_sqrt_energy", "capped_inverse_sqrt_energy"} else 1.0
    gamma = float(weight_cfg.get("gamma", default_gamma))
    sample_edges = weight_cfg.get("sample_edges_per_relation", None)
    sample_edges = None if sample_edges in (None, "") else int(sample_edges)
    seed = int(weight_cfg.get("seed", config.get("seed", 12345)))

    volumes = {relation_id: _relation_volume(graph, relation_id) for relation_id in relation_ids}
    energies = {relation_id: 0.0 for relation_id in relation_ids}
    basis_source = None
    if method in {
        "inverse_energy",
        "clipped_inverse_energy",
        "inverse_sqrt_energy",
        "capped_inverse_sqrt_energy",
        "smoothed_inverse_energy",
        "feature_smoothness",
    }:
        B, basis_source = _basis_for_energy(graph, config, weight_cfg, basis, method)
        for relation_id in progress_iter(
            relation_ids,
            total=len(relation_ids),
            desc="relation weights",
            config=config,
            unit="relation",
        ):
            energies[relation_id] = _relation_energy(
                graph,
                relation_id,
                B,
                epsilon=epsilon,
                sample_edges_per_relation=sample_edges,
                seed=seed,
                progress_config=config,
            )

    if method == "uniform":
        raw = {relation_id: 1.0 for relation_id in relation_ids}
    elif method == "volume":
        raw = {relation_id: (volumes[relation_id] + epsilon) ** eta for relation_id in relation_ids}
    elif method == "smoothed_inverse_energy":
        temperature = max(float(weight_cfg.get("temperature", 1.0)), 1.0e-6)
        entropy_regularization = float(weight_cfg.get("entropy_regularization", 0.0))
        energy_smoothing = float(
            weight_cfg.get("energy_smoothing", weight_cfg.get("smoothing", epsilon))
        )
        log_scores = np.asarray(
            [
                eta * np.log(volumes[relation_id] + epsilon)
                - gamma * np.log(energies[relation_id] + energy_smoothing + epsilon)
                for relation_id in relation_ids
            ],
            dtype=np.float64,
        )
        log_scores = log_scores / temperature
        log_scores -= float(np.max(log_scores)) if len(log_scores) else 0.0
        exp_scores = np.exp(log_scores)
        total = max(float(exp_scores.sum()), 1.0e-12)
        soft = exp_scores / total
        if entropy_regularization > 0.0 and len(soft):
            mix = min(max(entropy_regularization, 0.0), 1.0)
            soft = (1.0 - mix) * soft + mix * (1.0 / len(soft))
        weights = {relation_id: float(value) for relation_id, value in zip(relation_ids, soft)}
        weights = _clip_and_normalize(
            weights,
            min_weight=(
                None
                if weight_cfg.get("min_weight", None) in (None, "")
                else float(weight_cfg.get("min_weight"))
            ),
            max_weight=(
                None
                if weight_cfg.get("max_weight", None) in (None, "")
                else float(weight_cfg.get("max_weight"))
            ),
        )
        raw = dict(weights)
    else:
        raw = {
            relation_id: (volumes[relation_id] + epsilon) ** eta
            / ((energies[relation_id] + epsilon) ** gamma)
            for relation_id in relation_ids
        }
    weights = weights if method == "smoothed_inverse_energy" else _normalize(raw)
    clip_min = None
    clip_max = None
    if method in {"clipped_inverse_energy", "capped_inverse_sqrt_energy"}:
        clip_min = float(weight_cfg.get("weight_clip_min", weight_cfg.get("clip_min", 0.0)))
        default_clip_max = 0.75 if method == "capped_inverse_sqrt_energy" else 0.5
        clip_max = float(weight_cfg.get("weight_clip_max", weight_cfg.get("clip_max", default_clip_max)))
        clipped = {
            relation_id: float(np.clip(value, clip_min, clip_max))
            for relation_id, value in weights.items()
        }
        weights = _normalize(clipped)
    stats_values = list(weights.values())
    min_weight_value = float(min(stats_values, default=0.0))
    max_weight_value = float(max(stats_values, default=0.0))
    diagnostics: dict[str, Any] = {
        "relation_weighting_method": method,
        "relation_weights": {str(k): float(v) for k, v in weights.items()},
        "relation_weight_stats": {
            "sum": float(sum(stats_values)),
            "min": min_weight_value,
            "max": max_weight_value,
            "num_relations": int(len(stats_values)),
        },
        "relation_weight_entropy": _entropy(stats_values),
        "relation_weight_max_min_ratio": (
            float(max_weight_value / max(min_weight_value, 1.0e-12))
            if stats_values
            else 0.0
        ),
        "relation_energy_estimates": {str(k): float(v) for k, v in energies.items()},
        "relation_volume_estimates": {str(k): float(v) for k, v in volumes.items()},
    }
    if method == "smoothed_inverse_energy":
        diagnostics.update(
            {
                "relation_weighting_base_method": "inverse_energy",
                "temperature": float(weight_cfg.get("temperature", 1.0)),
                "min_weight": weight_cfg.get("min_weight", None),
                "max_weight": weight_cfg.get("max_weight", None),
                "entropy_regularization": float(weight_cfg.get("entropy_regularization", 0.0)),
                "energy_smoothing": float(
                    weight_cfg.get("energy_smoothing", weight_cfg.get("smoothing", epsilon))
                ),
            }
        )
    if method == "capped_inverse_sqrt_energy":
        diagnostics["relation_weighting_base_method"] = "inverse_sqrt_energy"
    if basis_source is not None:
        diagnostics["energy_basis_source"] = basis_source
        diagnostics["energy_basis_object"] = "Z_X"
        diagnostics["energy_estimator"] = "sampled_normalized_edge_energy"
    if clip_min is not None and clip_max is not None:
        diagnostics["weight_clip_min"] = clip_min
        diagnostics["weight_clip_max"] = clip_max
    return RelationWeightResult(weights, energies, volumes, diagnostics)
