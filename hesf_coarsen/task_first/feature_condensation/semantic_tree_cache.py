from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.eval.hettree_task import PathSpec, build_semantic_tree_features, enumerate_target_paths
from hesf_coarsen.io.schema import HeteroGraph


@dataclass(frozen=True)
class SemanticTreeCache:
    tensor: np.ndarray
    target_nodes: np.ndarray
    paths: list[PathSpec]
    feature_width: int
    type_ids: tuple[int, ...]


def build_semantic_tree_cache(
    graph: HeteroGraph,
    *,
    target_type: int,
    max_hops: int = 2,
    max_paths: int | None = 32,
) -> SemanticTreeCache:
    paths = enumerate_target_paths(graph, target_type=int(target_type), max_hops=int(max_hops), max_paths=max_paths)
    tree = build_semantic_tree_features(graph, target_type=int(target_type), paths=paths)
    return SemanticTreeCache(
        tensor=np.asarray(tree.tensor, dtype=np.float32),
        target_nodes=np.asarray(tree.target_nodes, dtype=np.int64),
        paths=list(tree.paths),
        feature_width=int(tree.feature_width),
        type_ids=tuple(int(value) for value in tree.type_ids),
    )


def save_cache_npz(cache: SemanticTreeCache, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tensor=np.asarray(cache.tensor, dtype=np.float32),
        target_nodes=np.asarray(cache.target_nodes, dtype=np.int64),
        paths=np.asarray(["self" if not path_spec else "->".join(str(x) for x in path_spec) for path_spec in cache.paths], dtype=object),
        feature_width=np.asarray([int(cache.feature_width)], dtype=np.int64),
        type_ids=np.asarray(cache.type_ids, dtype=np.int64),
    )


def cache_metadata(cache: SemanticTreeCache) -> dict[str, Any]:
    return {
        "target_count": int(len(cache.target_nodes)),
        "path_count": int(len(cache.paths)),
        "feature_width": int(cache.feature_width),
        "type_count": int(len(cache.type_ids)),
        "feature_cache_elements": int(np.asarray(cache.tensor).size),
    }
