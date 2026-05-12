from __future__ import annotations

from math import ceil

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.progress import progress_iter


def _global_feature_matrix(graph: HeteroGraph) -> np.ndarray | None:
    if graph.features is None:
        return None
    max_dim = max(feature.shape[1] for feature in graph.features.values())
    X = np.zeros((graph.num_nodes, max_dim), dtype=np.float32)
    for type_id, feature in graph.features.items():
        nodes = nodes_of_type(graph, type_id)
        X[nodes, : feature.shape[1]] = feature
    return X


def _numpy_weighted_pairwise_dense_cost(
    candidate_pairs: np.ndarray,
    Z: np.ndarray,
    relation_profiles: np.ndarray,
    conv_sketch: np.ndarray,
    X: np.ndarray | None,
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
        feat_values = (
            np.zeros(len(batch), dtype=np.float32)
            if X is None
            else np.sum((X[left] - X[right]) ** 2, axis=1)
        )
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
    X = _global_feature_matrix(graph) if features is not None else None
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
            if X is not None and lambda_feat != 0.0:
                dense_blocks.append((X.astype(np.float32, copy=False), lambda_feat))
            dense_cost = torch_weighted_pairwise_dense_cost(
                dense_blocks,
                candidate_pairs,
                device=device,
                batch_size=int(acceleration.get("scoring_batch_size", 65_536)),
                max_bytes=max_bytes,
                progress_config=config,
                progress_desc="score dense batches",
            )
        except (ImportError, RuntimeError, MemoryError):
            if not bool(acceleration.get("fallback_to_numpy", True)):
                raise
            use_torch = False

    if not use_torch:
        dense_cost = _numpy_weighted_pairwise_dense_cost(
            candidate_pairs,
            Z,
            relation_profiles,
            conv_sketch,
            X,
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
