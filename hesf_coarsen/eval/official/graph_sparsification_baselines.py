from __future__ import annotations

from typing import Iterable

import numpy as np


def sample_relation_edges(edges: Iterable[tuple[int, int]], *, keep_ratio: float, seed: int) -> list[tuple[int, int]]:
    edge_list = [(int(src), int(dst)) for src, dst in edges]
    count = max(0, min(len(edge_list), int(round(len(edge_list) * float(keep_ratio)))))
    if count == len(edge_list):
        return sorted(edge_list)
    if count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(np.arange(len(edge_list)), size=count, replace=False)
    return sorted(edge_list[int(index)] for index in indices.tolist())
