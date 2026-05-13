from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.progress import progress_iter


@dataclass(frozen=True)
class TypewiseFeatureView:
    blocks: dict[int, np.ndarray]
    local_index: np.ndarray


def _projection_dtype(config: dict) -> np.dtype:
    name = str(config.get("features", {}).get("projection_dtype", "float16"))
    dtype = np.dtype(name)
    if dtype not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("features.projection_dtype must be float16 or float32")
    return dtype


def _project_type_feature(
    feature: np.ndarray,
    *,
    type_id: int,
    projected_dim: int,
    seed: int,
    dtype: np.dtype,
    mmap_dir: Path | None = None,
    chunk_size: int = 100_000,
) -> np.ndarray:
    feature = np.asarray(feature)
    if feature.ndim != 2:
        raise ValueError(f"features for type {type_id} must be 2D")
    output_dim = feature.shape[1] if projected_dim <= 0 else min(feature.shape[1], projected_dim)
    if mmap_dir is None and (projected_dim <= 0 or feature.shape[1] <= projected_dim):
        return feature.astype(dtype, copy=False)

    if mmap_dir is None:
        projected = np.empty((feature.shape[0], output_dim), dtype=dtype)
    else:
        mmap_dir.mkdir(parents=True, exist_ok=True)
        projected_path = mmap_dir / f"features_type_{int(type_id)}_projected.npy"
        if projected_path.exists():
            projected = np.lib.format.open_memmap(projected_path, mode="r+")
            if projected.shape != (feature.shape[0], output_dim) or projected.dtype != dtype:
                mmap_handle = getattr(projected, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
                projected = np.lib.format.open_memmap(
                    projected_path,
                    mode="w+",
                    dtype=dtype,
                    shape=(feature.shape[0], output_dim),
                )
        else:
            projected = np.lib.format.open_memmap(
                projected_path,
                mode="w+",
                dtype=dtype,
                shape=(feature.shape[0], output_dim),
            )

    chunk_size = max(int(chunk_size), 1)
    if projected_dim <= 0 or feature.shape[1] <= projected_dim:
        for start in range(0, feature.shape[0], chunk_size):
            stop = min(start + chunk_size, feature.shape[0])
            projected[start:stop] = feature[start:stop].astype(dtype, copy=False)
    else:
        rng = np.random.default_rng(int(seed) + 1009 * int(type_id))
        projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(float(output_dim)),
            size=(feature.shape[1], output_dim),
        ).astype(np.float32)
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
    seed = int(config.get("seed", 12345))
    mmap_dir_value = feature_cfg.get("projection_mmap_dir")
    mmap_dir = None if mmap_dir_value in {None, ""} else Path(mmap_dir_value)
    chunk_size = int(feature_cfg.get("projection_chunk_size", 100_000))
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
    batch_size: int,
    config: dict,
) -> np.ndarray:
    batch_size = max(int(batch_size), 1)
    dense_cost = np.empty(candidate_pairs.shape[0], dtype=np.float32)
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
        left = batch[:, 0]
        right = batch[:, 1]
        spec_values = np.sum(
            (Z[left].astype(np.float32) - Z[right].astype(np.float32)) ** 2,
            axis=1,
        )
        rel_values = np.sum(
            (relation_profiles[left] - relation_profiles[right]) ** 2,
            axis=1,
        )
        conv_values = np.sum(
            (conv_sketch[left] - conv_sketch[right]) ** 2,
            axis=1,
        )
        feat_values = _typewise_feature_pair_cost(graph, batch, feature_view)
        dense_cost[start:stop] = (
            lambda_spec * spec_values
            + lambda_rel * rel_values
            + lambda_feat * feat_values
            + lambda_conv * conv_values
        )
    return dense_cost


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
    scoring = config.get("scoring", {})
    lambda_spec = float(scoring.get("lambda_spec", 1.0))
    lambda_rel = float(scoring.get("lambda_rel", 0.2))
    lambda_feat = float(scoring.get("lambda_feat", 0.1))
    lambda_conv = float(scoring.get("lambda_conv", 0.3))
    lambda_boundary = float(scoring.get("lambda_boundary", 0.1))
    feature_view = (
        _prepare_typewise_feature_view(graph, features, config)
        if features is not None and lambda_feat != 0.0
        else None
    )
    acceleration = config.get("acceleration", {})

    candidate_pairs = pairs[:, :2].astype(np.int64, copy=False)
    valid = graph.node_type[candidate_pairs[:, 0]] == graph.node_type[candidate_pairs[:, 1]]
    candidate_pairs = candidate_pairs[valid]
    if candidate_pairs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    use_torch = acceleration.get("dense_backend") == "torch"
    if use_torch:
        try:
            from hesf_coarsen.ops.torch_dense import torch_weighted_pairwise_dense_cost

            device = str(acceleration.get("device", "auto"))
            max_bytes = acceleration.get("max_dense_bytes")
            dense_blocks = [
                (Z.astype(np.float32, copy=False), lambda_spec),
                (relation_profiles.astype(np.float32, copy=False), lambda_rel),
                (conv_sketch.astype(np.float32, copy=False), lambda_conv),
            ]
            dense_cost = torch_weighted_pairwise_dense_cost(
                dense_blocks,
                candidate_pairs,
                device=device,
                batch_size=int(acceleration.get("scoring_batch_size", 65_536)),
                max_bytes=max_bytes,
                progress_config=config,
                progress_desc="score dense batches",
            )
            if feature_view is not None and lambda_feat != 0.0:
                feature_cost = _feature_pair_cost_batches(
                    graph,
                    candidate_pairs,
                    feature_view,
                    batch_size=int(acceleration.get("scoring_batch_size", 65_536)),
                    config=config,
                )
                dense_cost += np.float32(lambda_feat) * feature_cost
        except (ImportError, RuntimeError, MemoryError):
            if not bool(acceleration.get("fallback_to_numpy", True)):
                raise
            use_torch = False

    if not use_torch:
        dense_cost = _numpy_weighted_pairwise_dense_cost(
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
            batch_size=int(acceleration.get("scoring_batch_size", 65_536)),
            config=config,
        )

    rows: list[tuple[int, int, float]] = []
    row_iter = progress_iter(
        enumerate(candidate_pairs),
        total=len(candidate_pairs),
        desc="score row assembly",
        config=config,
        unit="pair",
    )
    for idx, (i, j) in row_iter:
        i = int(i)
        j = int(j)
        boundary = 0.0 if partition_id is None or partition_id[i] == partition_id[j] else 1.0
        cost = float(dense_cost[idx]) + lambda_boundary * boundary
        rows.append((i, j, cost))
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


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
