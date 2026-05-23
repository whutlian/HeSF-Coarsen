from __future__ import annotations

import numpy as np

from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache


def select_paths_by_energy(cache: SemanticTreeCache, keep_ratio: float) -> list[int]:
    tensor = np.asarray(cache.tensor, dtype=np.float32)
    path_count = int(tensor.shape[1]) if tensor.ndim == 3 else 0
    if path_count == 0:
        return []
    keep = max(1, min(path_count, int(np.ceil(path_count * float(keep_ratio) - 1.0e-12))))
    energy = np.mean(np.square(tensor), axis=(0, 2))
    order = np.argsort(-energy, kind="mergesort")
    return sorted(int(idx) for idx in order[:keep].tolist())


def prune_cache_paths(cache: SemanticTreeCache, keep_ratio: float) -> SemanticTreeCache:
    keep = select_paths_by_energy(cache, keep_ratio)
    return SemanticTreeCache(
        tensor=np.asarray(cache.tensor[:, keep, :], dtype=np.float32),
        target_nodes=np.asarray(cache.target_nodes, dtype=np.int64),
        paths=[cache.paths[idx] for idx in keep],
        feature_width=int(cache.feature_width),
        type_ids=tuple(cache.type_ids),
    )
