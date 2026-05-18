"""Experimental proxy diagnostic for deprecated Next17 A4.

Next18 treats this as an appendix-only approximation, not as a validated
task-aligned objective.
"""

from __future__ import annotations

import numpy as np

from hesf_coarsen.eval.hettree_task import build_semantic_tree_features, enumerate_target_paths
from hesf_coarsen.io.schema import HeteroGraph


def target_tree_reconstruction_error(
    original: HeteroGraph,
    hybrid: HeteroGraph,
    *,
    target_node_type: int,
    max_hops: int = 2,
    max_paths: int | None = 16,
) -> dict[str, float | int]:
    paths = enumerate_target_paths(original, target_type=int(target_node_type), max_hops=max_hops, max_paths=max_paths)
    original_features = build_semantic_tree_features(original, target_type=int(target_node_type), paths=paths)
    hybrid_features = build_semantic_tree_features(
        hybrid,
        target_type=int(target_node_type),
        paths=paths,
        feature_width=original_features.feature_width,
        type_ids=original_features.type_ids,
    )
    rows = min(original_features.tensor.shape[0], hybrid_features.tensor.shape[0])
    if rows == 0:
        return {"meta_recon_rmse": 0.0, "meta_recon_relative_error": 0.0, "path_count": int(len(paths))}
    diff = original_features.tensor[:rows] - hybrid_features.tensor[:rows]
    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    denom = float(np.sqrt(np.mean(original_features.tensor[:rows].astype(np.float64) ** 2)))
    return {
        "meta_recon_rmse": rmse,
        "meta_recon_relative_error": float(rmse / max(denom, 1.0e-12)),
        "path_count": int(len(paths)),
    }
