from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.progress import progress_iter


SCORE_TERM_NAMES = ("spec", "rel", "feat", "conv", "boundary")


@dataclass(frozen=True)
class TypewiseFeatureView:
    blocks: dict[int, np.ndarray]
    local_index: np.ndarray


@dataclass(frozen=True)
class PairScoringContext:
    graph: HeteroGraph
    Z: np.ndarray
    relation_profiles: np.ndarray
    conv_sketch: np.ndarray
    feature_view: TypewiseFeatureView | None
    config: dict
    partition_id: np.ndarray | None
    lambda_spec: float
    lambda_rel: float
    lambda_feat: float
    lambda_conv: float
    lambda_boundary: float
    relation_profile_distance: str
    relation_profile_epsilon: float
    spec_volume_weighting: bool
    spec_volume_epsilon: float
    boundary_mode: str
    boundary_risk: np.ndarray | None
    node_volume: np.ndarray
    batch_size: int
    use_torch: bool
    acceleration: dict


class ScoreTermAccumulator:
    """Streaming summary for unweighted scoring term distributions."""

    def __init__(self, sample_size: int = 200_000, seed: int = 12345):
        self.sample_size = max(int(sample_size), 0)
        self._rng = np.random.default_rng(int(seed))
        self._count = {name: 0 for name in SCORE_TERM_NAMES}
        self._sum = {name: 0.0 for name in SCORE_TERM_NAMES}
        self._samples = {
            name: np.empty(0, dtype=np.float64) for name in SCORE_TERM_NAMES
        }

    @classmethod
    def from_config(cls, config: dict) -> "ScoreTermAccumulator":
        diagnostics_cfg = config.get("diagnostics", {})
        return cls(
            sample_size=int(diagnostics_cfg.get("score_term_sample_size", 200_000)),
            seed=int(diagnostics_cfg.get("score_term_seed", config.get("seed", 12345))),
        )

    def update(self, terms: dict[str, np.ndarray]) -> None:
        for name in SCORE_TERM_NAMES:
            values = np.asarray(terms.get(name, np.empty(0)), dtype=np.float64).ravel()
            if values.size == 0:
                continue
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            old_count = int(self._count[name])
            self._count[name] = old_count + int(values.size)
            self._sum[name] += float(values.sum())
            if self.sample_size <= 0:
                continue

            current = self._samples[name]
            fill = min(self.sample_size - len(current), len(values))
            if fill > 0:
                current = np.concatenate([current, values[:fill].astype(np.float64, copy=False)])
                values = values[fill:]
            if values.size:
                seen_before = old_count + fill
                positions = seen_before + np.arange(1, values.size + 1, dtype=np.float64)
                keep = self._rng.random(values.size) < (float(self.sample_size) / positions)
                if np.any(keep):
                    slots = self._rng.integers(0, self.sample_size, size=int(np.sum(keep)))
                    current[slots] = values[keep]
            self._samples[name] = current

    def summary(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for name in SCORE_TERM_NAMES:
            count = int(self._count[name])
            samples = self._samples[name]
            if count == 0 or samples.size == 0:
                summary[name] = {
                    "count": count,
                    "sample_count": int(samples.size),
                    "sample_fraction": 0.0,
                    "mean": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }
                continue
            p50, p95, p99 = np.percentile(samples, [50, 95, 99])
            summary[name] = {
                "count": count,
                "sample_count": int(samples.size),
                "sample_fraction": float(samples.size / max(count, 1)),
                "mean": float(self._sum[name] / max(count, 1)),
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
            }
        return summary


def _projection_dtype(config: dict) -> np.dtype:
    name = str(config.get("features", {}).get("projection_dtype", "float16"))
    dtype = np.dtype(name)
    if dtype not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("features.projection_dtype must be float16 or float32")
    return dtype


def _projection_method(config: dict) -> str:
    feature_cfg = config.get("features", {})
    method = feature_cfg.get("projector", feature_cfg.get("projection_method", "gaussian_random"))
    method = str(method).lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian_random",
        "random": "gaussian_random",
        "random_projection": "gaussian_random",
        "incremental": "incremental_pca",
        "ipca": "incremental_pca",
    }
    method = aliases.get(method, method)
    if method not in {"gaussian_random", "sparse_random", "pca", "incremental_pca"}:
        raise ValueError(
            "features.projector/features.projection_method must be one of "
            "gaussian_random, sparse_random, pca, or incremental_pca"
        )
    return method


def _make_projected_output(
    *,
    type_id: int,
    rows: int,
    output_dim: int,
    dtype: np.dtype,
    mmap_dir: Path | None,
) -> np.ndarray:
    if mmap_dir is None:
        return np.empty((rows, output_dim), dtype=dtype)

    mmap_dir.mkdir(parents=True, exist_ok=True)
    projected_path = mmap_dir / f"features_type_{int(type_id)}_projected.npy"
    if projected_path.exists():
        projected = np.lib.format.open_memmap(projected_path, mode="r+")
        if projected.shape == (rows, output_dim) and projected.dtype == dtype:
            return projected
        mmap_handle = getattr(projected, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
    return np.lib.format.open_memmap(
        projected_path,
        mode="w+",
        dtype=dtype,
        shape=(rows, output_dim),
    )


def _dense_random_projection(
    *,
    input_dim: int,
    output_dim: int,
    type_id: int,
    seed: int,
    method: str,
    sparse_density: float | None,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1009 * int(type_id))
    if method == "gaussian_random":
        return rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(float(output_dim)),
            size=(input_dim, output_dim),
        ).astype(np.float32)

    density = 1.0 / np.sqrt(float(input_dim)) if sparse_density is None else float(sparse_density)
    if not 0.0 < density <= 1.0:
        raise ValueError("features.projection_density must be in (0, 1] for sparse_random")
    mask = rng.random((input_dim, output_dim)) < density
    if input_dim > 0:
        empty_cols = np.flatnonzero(~np.any(mask, axis=0))
        for col in empty_cols:
            mask[int(rng.integers(0, input_dim)), col] = True
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(input_dim, output_dim))
    scale = 1.0 / np.sqrt(density * float(output_dim))
    return (mask.astype(np.float32) * signs * np.float32(scale)).astype(np.float32)


def _numpy_pca_components(
    feature: np.ndarray,
    output_dim: int,
    *,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    rows, input_dim = feature.shape
    n_components = min(output_dim, rows, input_dim)
    mean = np.zeros(input_dim, dtype=np.float64)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        mean += feature[start:stop].astype(np.float64, copy=False).sum(axis=0)
    mean = (mean / max(rows, 1)).astype(np.float32)
    components = np.zeros((output_dim, input_dim), dtype=np.float32)
    if n_components == 0:
        return mean, components
    covariance = np.zeros((input_dim, input_dim), dtype=np.float64)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        centered = feature[start:stop].astype(np.float64, copy=False) - mean.astype(np.float64)
        covariance += centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1][:n_components]
    components[:n_components] = eigvecs[:, order].T.astype(np.float32, copy=False)
    return mean, components


def _project_type_feature(
    feature: np.ndarray,
    *,
    type_id: int,
    projected_dim: int,
    seed: int,
    dtype: np.dtype,
    mmap_dir: Path | None = None,
    chunk_size: int = 100_000,
    projection_method: str = "gaussian_random",
    sparse_density: float | None = None,
) -> np.ndarray:
    feature = np.asarray(feature)
    if feature.ndim != 2:
        raise ValueError(f"features for type {type_id} must be 2D")
    output_dim = feature.shape[1] if projected_dim <= 0 else min(feature.shape[1], projected_dim)
    if mmap_dir is None and (projected_dim <= 0 or feature.shape[1] <= projected_dim):
        return feature.astype(dtype, copy=False)

    projected = _make_projected_output(
        type_id=type_id,
        rows=feature.shape[0],
        output_dim=output_dim,
        dtype=dtype,
        mmap_dir=mmap_dir,
    )

    chunk_size = max(int(chunk_size), 1)
    if projected_dim <= 0 or feature.shape[1] <= projected_dim:
        for start in range(0, feature.shape[0], chunk_size):
            stop = min(start + chunk_size, feature.shape[0])
            projected[start:stop] = feature[start:stop].astype(dtype, copy=False)
    elif projection_method in {"pca", "incremental_pca"}:
        # Use NumPy SVD directly for portability. It gives deterministic PCA
        # projections without requiring sklearn/SciPy runtime stability.
        mean, components = _numpy_pca_components(feature, output_dim, chunk_size=chunk_size)
        for start in range(0, feature.shape[0], chunk_size):
            stop = min(start + chunk_size, feature.shape[0])
            block = feature[start:stop].astype(np.float32, copy=False)
            projected[start:stop] = ((block - mean) @ components.T).astype(dtype, copy=False)
    else:
        projection = _dense_random_projection(
            input_dim=feature.shape[1],
            output_dim=output_dim,
            type_id=type_id,
            seed=seed,
            method=projection_method,
            sparse_density=sparse_density,
        )
        for start in range(0, feature.shape[0], chunk_size):
            stop = min(start + chunk_size, feature.shape[0])
            block = feature[start:stop].astype(np.float32, copy=False)
            projected[start:stop] = (block @ projection).astype(dtype, copy=False)
    if isinstance(projected, np.memmap):
        projected.flush()
    return projected


def _prepare_typewise_feature_view(
    graph: HeteroGraph,
    features: dict[int, np.ndarray] | None,
    config: dict,
) -> TypewiseFeatureView | None:
    if not features:
        return None
    feature_cfg = config.get("features", {})
    projected_dim = int(feature_cfg.get("projected_dim", 32))
    dtype = _projection_dtype(config)
    projection_method = _projection_method(config)
    seed = int(config.get("seed", 12345))
    mmap_dir_value = feature_cfg.get("projection_mmap_dir")
    mmap_dir = None if mmap_dir_value in {None, ""} else Path(mmap_dir_value)
    chunk_size = int(feature_cfg.get("projection_chunk_size", 100_000))
    sparse_density_value = feature_cfg.get("projection_density", feature_cfg.get("sparse_density"))
    sparse_density = None if sparse_density_value is None else float(sparse_density_value)
    blocks: dict[int, np.ndarray] = {}
    local_index = np.full(graph.num_nodes, -1, dtype=np.int64)
    for type_id, feature in sorted(features.items()):
        type_id = int(type_id)
        nodes = nodes_of_type(graph, type_id)
        local_index[nodes] = np.arange(len(nodes), dtype=np.int64)
        blocks[type_id] = _project_type_feature(
            feature,
            type_id=type_id,
            projected_dim=projected_dim,
            seed=seed,
            dtype=dtype,
            mmap_dir=mmap_dir,
            chunk_size=chunk_size,
            projection_method=projection_method,
            sparse_density=sparse_density,
        )
    return TypewiseFeatureView(blocks=blocks, local_index=local_index)


def _typewise_feature_pair_cost(
    graph: HeteroGraph,
    batch: np.ndarray,
    feature_view: TypewiseFeatureView | None,
) -> np.ndarray:
    if feature_view is None or len(batch) == 0:
        return np.zeros(len(batch), dtype=np.float32)
    left = batch[:, 0]
    right = batch[:, 1]
    out = np.zeros(len(batch), dtype=np.float32)
    batch_types = graph.node_type[left]
    for type_id in np.unique(batch_types):
        type_id = int(type_id)
        block = feature_view.blocks.get(type_id)
        if block is None:
            continue
        mask = batch_types == type_id
        left_local = feature_view.local_index[left[mask]]
        right_local = feature_view.local_index[right[mask]]
        valid = (left_local >= 0) & (right_local >= 0)
        if not np.any(valid):
            continue
        masked_positions = np.flatnonzero(mask)
        diff = (
            block[left_local[valid]].astype(np.float32)
            - block[right_local[valid]].astype(np.float32)
        )
        out[masked_positions[valid]] = np.sum(diff * diff, axis=1)
    return out


def _incident_weight_mass(graph: HeteroGraph) -> np.ndarray:
    volumes = np.zeros(graph.num_nodes, dtype=np.float32)
    for rel in graph.relations.values():
        weight = rel.weight.astype(np.float32, copy=False)
        np.add.at(volumes, rel.src, weight)
        np.add.at(volumes, rel.dst, weight)
    return volumes


def _squared_l2_pair_cost(values: np.ndarray, batch: np.ndarray) -> np.ndarray:
    if len(batch) == 0:
        return np.empty(0, dtype=np.float32)
    left = batch[:, 0]
    right = batch[:, 1]
    diff = values[left].astype(np.float32) - values[right].astype(np.float32)
    return np.sum(diff * diff, axis=1).astype(np.float32)


def _spectral_pair_cost(
    Z: np.ndarray,
    batch: np.ndarray,
    node_volume: np.ndarray,
    *,
    volume_weighting: bool,
    epsilon: float,
) -> np.ndarray:
    values = _squared_l2_pair_cost(Z, batch)
    if not volume_weighting:
        return values
    left = batch[:, 0]
    right = batch[:, 1]
    left_volume = node_volume[left].astype(np.float32, copy=False)
    right_volume = node_volume[right].astype(np.float32, copy=False)
    factor = (left_volume * right_volume) / np.maximum(left_volume + right_volume, float(epsilon))
    return (values * factor.astype(np.float32, copy=False)).astype(np.float32)


def _jsd_pair_cost(profiles: np.ndarray, batch: np.ndarray, *, epsilon: float) -> np.ndarray:
    if len(batch) == 0:
        return np.empty(0, dtype=np.float32)
    left = batch[:, 0]
    right = batch[:, 1]
    p = profiles[left].astype(np.float64, copy=False)
    q = profiles[right].astype(np.float64, copy=False)
    p_sum = p.sum(axis=1, keepdims=True)
    q_sum = q.sum(axis=1, keepdims=True)
    p = np.divide(p, np.maximum(p_sum, float(epsilon)), out=np.zeros_like(p), where=p_sum > 0.0)
    q = np.divide(q, np.maximum(q_sum, float(epsilon)), out=np.zeros_like(q), where=q_sum > 0.0)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = a > 0.0
        terms = np.zeros_like(a, dtype=np.float64)
        terms[mask] = a[mask] * np.log(a[mask] / np.maximum(b[mask], float(epsilon)))
        return terms.sum(axis=1)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).astype(np.float32)


def _relation_profile_pair_cost(
    relation_profiles: np.ndarray,
    batch: np.ndarray,
    *,
    method: str,
    epsilon: float,
) -> np.ndarray:
    method = str(method).lower()
    if method in {"l2", "squared_l2", "euclidean"}:
        return _squared_l2_pair_cost(relation_profiles, batch)
    if method == "jsd":
        return _jsd_pair_cost(relation_profiles, batch, epsilon=epsilon)
    raise ValueError("scoring.relation_profile_distance must be one of: l2, squared_l2, jsd")


def _partition_boundary_flags(
    graph: HeteroGraph,
    partition_id: np.ndarray | None,
) -> np.ndarray:
    flags = np.zeros(graph.num_nodes, dtype=np.float32)
    if partition_id is None:
        return flags
    partition_id = np.asarray(partition_id)
    for rel in graph.relations.values():
        src_part = partition_id[rel.src]
        dst_part = partition_id[rel.dst]
        cut = src_part != dst_part
        if np.any(cut):
            flags[rel.src[cut]] = 1.0
            flags[rel.dst[cut]] = 1.0
    return flags


def _node_boundary_risk(
    graph: HeteroGraph,
    partition_id: np.ndarray | None,
    scoring: dict,
) -> np.ndarray:
    degree = _incident_weight_mass(graph)
    risk = _partition_boundary_flags(graph, partition_id)
    hub_gamma = float(scoring.get("boundary_hub_gamma", 0.0))
    if hub_gamma != 0.0:
        risk = risk + np.float32(hub_gamma) * np.log1p(degree.astype(np.float32, copy=False))
    terminal_gamma = float(scoring.get("boundary_terminal_gamma", 0.0))
    if terminal_gamma != 0.0:
        threshold = float(scoring.get("boundary_terminal_degree", 1.0))
        risk = risk + np.float32(terminal_gamma) * (degree <= threshold).astype(np.float32)
    return risk.astype(np.float32, copy=False)


def _boundary_pair_penalty(
    graph: HeteroGraph,
    batch: np.ndarray,
    boundary_risk: np.ndarray | None,
    partition_id: np.ndarray | None,
    *,
    mode: str,
) -> np.ndarray:
    if len(batch) == 0:
        return np.empty(0, dtype=np.float32)
    mode = str(mode).lower()
    if mode in {"partition", "partition_pair", "legacy"}:
        if partition_id is None:
            return np.zeros(len(batch), dtype=np.float32)
        return (partition_id[batch[:, 0]] != partition_id[batch[:, 1]]).astype(np.float32)
    if mode != "node_risk":
        raise ValueError("scoring.boundary_mode must be one of: node_risk, partition_pair")
    if boundary_risk is None:
        boundary_risk = _node_boundary_risk(graph, partition_id, {})
    values = np.maximum(boundary_risk[batch[:, 0]], boundary_risk[batch[:, 1]])
    if partition_id is not None:
        values = np.maximum(
            values,
            (partition_id[batch[:, 0]] != partition_id[batch[:, 1]]).astype(np.float32),
        )
    return values.astype(np.float32, copy=False)


def _scoring_pair_cost_batches(
    graph: HeteroGraph,
    candidate_pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    feature_view: TypewiseFeatureView | None,
    *,
    lambda_spec: float,
    lambda_rel: float,
    lambda_feat: float,
    lambda_conv: float,
    lambda_boundary: float,
    relation_profile_distance: str,
    relation_profile_epsilon: float,
    spec_volume_weighting: bool,
    spec_volume_epsilon: float,
    boundary_mode: str,
    boundary_risk: np.ndarray | None,
    partition_id: np.ndarray | None,
    node_volume: np.ndarray,
    batch_size: int,
    config: dict,
    include_conv: bool,
) -> np.ndarray:
    batch_size = max(int(batch_size), 1)
    cost = np.empty(candidate_pairs.shape[0], dtype=np.float32)
    starts = range(0, candidate_pairs.shape[0], batch_size)
    for start in progress_iter(
        starts,
        total=ceil(candidate_pairs.shape[0] / batch_size) if len(candidate_pairs) else 0,
        desc="score dense batches",
        config=config,
        unit="batch",
    ):
        stop = min(start + batch_size, candidate_pairs.shape[0])
        batch = candidate_pairs[start:stop]
        spec_values = _spectral_pair_cost(
            Z,
            batch,
            node_volume,
            volume_weighting=spec_volume_weighting,
            epsilon=spec_volume_epsilon,
        )
        rel_values = _relation_profile_pair_cost(
            relation_profiles,
            batch,
            method=relation_profile_distance,
            epsilon=relation_profile_epsilon,
        )
        conv_values = _squared_l2_pair_cost(conv_sketch, batch) if include_conv else 0.0
        feat_values = _typewise_feature_pair_cost(graph, batch, feature_view)
        boundary_values = _boundary_pair_penalty(
            graph,
            batch,
            boundary_risk,
            partition_id,
            mode=boundary_mode,
        )
        cost[start:stop] = (
            lambda_spec * spec_values
            + lambda_rel * rel_values
            + lambda_feat * feat_values
            + lambda_conv * conv_values
            + lambda_boundary * boundary_values
        )
    return cost


def _scoring_pair_term_batches(
    graph: HeteroGraph,
    candidate_pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    feature_view: TypewiseFeatureView | None,
    *,
    relation_profile_distance: str,
    relation_profile_epsilon: float,
    spec_volume_weighting: bool,
    spec_volume_epsilon: float,
    boundary_mode: str,
    boundary_risk: np.ndarray | None,
    partition_id: np.ndarray | None,
    node_volume: np.ndarray,
    batch_size: int,
    config: dict,
    include_conv: bool,
) -> dict[str, np.ndarray]:
    batch_size = max(int(batch_size), 1)
    terms = {
        name: np.zeros(candidate_pairs.shape[0], dtype=np.float32)
        for name in SCORE_TERM_NAMES
    }
    starts = range(0, candidate_pairs.shape[0], batch_size)
    for start in progress_iter(
        starts,
        total=ceil(candidate_pairs.shape[0] / batch_size) if len(candidate_pairs) else 0,
        desc="score dense batches",
        config=config,
        unit="batch",
    ):
        stop = min(start + batch_size, candidate_pairs.shape[0])
        batch = candidate_pairs[start:stop]
        terms["spec"][start:stop] = _spectral_pair_cost(
            Z,
            batch,
            node_volume,
            volume_weighting=spec_volume_weighting,
            epsilon=spec_volume_epsilon,
        )
        terms["rel"][start:stop] = _relation_profile_pair_cost(
            relation_profiles,
            batch,
            method=relation_profile_distance,
            epsilon=relation_profile_epsilon,
        )
        if include_conv:
            terms["conv"][start:stop] = _squared_l2_pair_cost(conv_sketch, batch)
        terms["feat"][start:stop] = _typewise_feature_pair_cost(graph, batch, feature_view)
        terms["boundary"][start:stop] = _boundary_pair_penalty(
            graph,
            batch,
            boundary_risk,
            partition_id,
            mode=boundary_mode,
        )
    return terms


def _numpy_weighted_pairwise_dense_cost(
    graph: HeteroGraph,
    candidate_pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    feature_view: TypewiseFeatureView | None,
    *,
    lambda_spec: float,
    lambda_rel: float,
    lambda_feat: float,
    lambda_conv: float,
    lambda_boundary: float,
    relation_profile_distance: str,
    relation_profile_epsilon: float,
    spec_volume_weighting: bool,
    spec_volume_epsilon: float,
    boundary_mode: str,
    boundary_risk: np.ndarray | None,
    partition_id: np.ndarray | None,
    node_volume: np.ndarray,
    batch_size: int,
    config: dict,
) -> np.ndarray:
    return _scoring_pair_cost_batches(
        graph,
        candidate_pairs,
        Z,
        relation_profiles,
        conv_sketch,
        feature_view,
        lambda_spec=lambda_spec,
        lambda_rel=lambda_rel,
        lambda_feat=lambda_feat,
        lambda_conv=lambda_conv,
        lambda_boundary=lambda_boundary,
        relation_profile_distance=relation_profile_distance,
        relation_profile_epsilon=relation_profile_epsilon,
        spec_volume_weighting=spec_volume_weighting,
        spec_volume_epsilon=spec_volume_epsilon,
        boundary_mode=boundary_mode,
        boundary_risk=boundary_risk,
        partition_id=partition_id,
        node_volume=node_volume,
        batch_size=batch_size,
        config=config,
        include_conv=True,
    )


def prepare_pair_scoring_context(
    graph: HeteroGraph,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    features: dict[int, np.ndarray] | None,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> PairScoringContext:
    scoring = config.get("scoring", {})
    lambda_spec = float(scoring.get("lambda_spec", 1.0))
    lambda_rel = float(scoring.get("lambda_rel", 0.2))
    lambda_feat = float(scoring.get("lambda_feat", 0.1))
    lambda_conv = float(scoring.get("lambda_conv", 0.3))
    lambda_boundary = float(scoring.get("lambda_boundary", 0.1))
    relation_profile_distance = str(
        scoring.get("relation_profile_distance", scoring.get("relation_distance", "jsd"))
    ).lower()
    relation_profile_epsilon = float(scoring.get("relation_profile_epsilon", 1.0e-12))
    spec_volume_weighting = bool(scoring.get("spec_volume_weighting", True))
    spec_volume_epsilon = float(scoring.get("spec_volume_epsilon", 1.0e-12))
    boundary_mode = str(scoring.get("boundary_mode", "node_risk")).lower()
    feature_view = (
        _prepare_typewise_feature_view(graph, features, config)
        if features is not None and lambda_feat != 0.0
        else None
    )
    acceleration = config.get("acceleration", {})
    node_volume = _incident_weight_mass(graph)
    boundary_risk = (
        _node_boundary_risk(graph, partition_id, scoring)
        if lambda_boundary != 0.0 and boundary_mode == "node_risk"
        else None
    )
    return PairScoringContext(
        graph=graph,
        Z=Z,
        relation_profiles=relation_profiles,
        conv_sketch=conv_sketch,
        feature_view=feature_view,
        config=config,
        partition_id=partition_id,
        lambda_spec=lambda_spec,
        lambda_rel=lambda_rel,
        lambda_feat=lambda_feat,
        lambda_conv=lambda_conv,
        lambda_boundary=lambda_boundary,
        relation_profile_distance=relation_profile_distance,
        relation_profile_epsilon=relation_profile_epsilon,
        spec_volume_weighting=spec_volume_weighting,
        spec_volume_epsilon=spec_volume_epsilon,
        boundary_mode=boundary_mode,
        boundary_risk=boundary_risk,
        node_volume=node_volume,
        batch_size=int(acceleration.get("scoring_batch_size", 65_536)),
        use_torch=acceleration.get("dense_backend") == "torch",
        acceleration=acceleration,
    )


def score_pair_block(context: PairScoringContext, pairs: np.ndarray) -> np.ndarray:
    scored, _terms = score_pair_block_with_terms(context, pairs)
    return scored


def _weighted_cost_from_terms(
    context: PairScoringContext,
    terms: dict[str, np.ndarray],
) -> np.ndarray:
    return (
        context.lambda_spec * terms["spec"]
        + context.lambda_rel * terms["rel"]
        + context.lambda_feat * terms["feat"]
        + context.lambda_conv * terms["conv"]
        + context.lambda_boundary * terms["boundary"]
    ).astype(np.float32, copy=False)


def score_pair_block_with_terms(
    context: PairScoringContext,
    pairs: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64), {
            name: np.empty(0, dtype=np.float32) for name in SCORE_TERM_NAMES
        }

    graph = context.graph
    candidate_pairs = pairs[:, :2].astype(np.int64, copy=False)
    valid = graph.node_type[candidate_pairs[:, 0]] == graph.node_type[candidate_pairs[:, 1]]
    candidate_pairs = candidate_pairs[valid]
    if candidate_pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64), {
            name: np.empty(0, dtype=np.float32) for name in SCORE_TERM_NAMES
        }

    use_torch = context.use_torch
    if use_torch:
        try:
            from hesf_coarsen.ops.torch_dense import torch_weighted_pairwise_dense_cost

            device = str(context.acceleration.get("device", "auto"))
            max_bytes = context.acceleration.get("max_dense_bytes")
            terms = _scoring_pair_term_batches(
                graph,
                candidate_pairs,
                context.Z,
                context.relation_profiles,
                context.conv_sketch,
                context.feature_view,
                relation_profile_distance=context.relation_profile_distance,
                relation_profile_epsilon=context.relation_profile_epsilon,
                spec_volume_weighting=context.spec_volume_weighting,
                spec_volume_epsilon=context.spec_volume_epsilon,
                boundary_mode=context.boundary_mode,
                boundary_risk=context.boundary_risk,
                partition_id=context.partition_id,
                node_volume=context.node_volume,
                batch_size=context.batch_size,
                config=context.config,
                include_conv=False,
            )
            terms["conv"] = torch_weighted_pairwise_dense_cost(
                [(context.conv_sketch.astype(np.float32, copy=False), 1.0)],
                candidate_pairs,
                device=device,
                batch_size=context.batch_size,
                max_bytes=max_bytes,
                progress_config=context.config,
                progress_desc="score dense batches",
            )
            dense_cost = _weighted_cost_from_terms(context, terms)
        except (ImportError, RuntimeError, MemoryError):
            if not bool(context.acceleration.get("fallback_to_numpy", True)):
                raise
            use_torch = False

    if not use_torch:
        terms = _scoring_pair_term_batches(
            graph,
            candidate_pairs,
            context.Z,
            context.relation_profiles,
            context.conv_sketch,
            context.feature_view,
            relation_profile_distance=context.relation_profile_distance,
            relation_profile_epsilon=context.relation_profile_epsilon,
            spec_volume_weighting=context.spec_volume_weighting,
            spec_volume_epsilon=context.spec_volume_epsilon,
            boundary_mode=context.boundary_mode,
            boundary_risk=context.boundary_risk,
            partition_id=context.partition_id,
            node_volume=context.node_volume,
            batch_size=context.batch_size,
            config=context.config,
            include_conv=True,
        )
        dense_cost = _weighted_cost_from_terms(context, terms)

    scored = np.empty((candidate_pairs.shape[0], 3), dtype=np.float64)
    scored[:, :2] = candidate_pairs
    scored[:, 2] = dense_cost.astype(np.float64, copy=False)
    return scored, terms


def score_candidate_pair_block(
    graph: HeteroGraph,
    pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    features: dict[int, np.ndarray] | None,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    context = prepare_pair_scoring_context(
        graph,
        Z,
        relation_profiles,
        conv_sketch,
        features,
        config,
        partition_id=partition_id,
    )
    return score_pair_block(context, pairs)


def score_candidate_pairs(
    graph: HeteroGraph,
    pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    features: dict[int, np.ndarray] | None,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    context = prepare_pair_scoring_context(
        graph,
        Z,
        relation_profiles,
        conv_sketch,
        features,
        config,
        partition_id=partition_id,
    )
    return score_pair_block(context, pairs)


def _feature_pair_cost_batches(
    graph: HeteroGraph,
    candidate_pairs: np.ndarray,
    feature_view: TypewiseFeatureView,
    *,
    batch_size: int,
    config: dict,
) -> np.ndarray:
    batch_size = max(int(batch_size), 1)
    out = np.empty(candidate_pairs.shape[0], dtype=np.float32)
    starts = range(0, candidate_pairs.shape[0], batch_size)
    for start in progress_iter(
        starts,
        total=ceil(candidate_pairs.shape[0] / batch_size) if len(candidate_pairs) else 0,
        desc="score feature batches",
        config=config,
        unit="batch",
    ):
        stop = min(start + batch_size, candidate_pairs.shape[0])
        out[start:stop] = _typewise_feature_pair_cost(
            graph,
            candidate_pairs[start:stop],
            feature_view,
        )
    return out
