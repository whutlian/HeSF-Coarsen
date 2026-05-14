from __future__ import annotations

from collections import Counter

import numpy as np


class BoundedCandidateStore:
    """Per-node bounded candidate store.

    The v1 implementation uses Python dictionaries for clarity on small and
    medium graphs. The API is intentionally narrow so this can be replaced by
    fixed-size arrays or mmap buffers for the large-scale path.
    """

    def __init__(self, node_type: np.ndarray, K: int, same_type_only: bool = True):
        self.node_type = np.asarray(node_type, dtype=np.int32)
        self.K = int(K)
        self.same_type_only = bool(same_type_only)
        self._per_node: list[dict[int, tuple[float, str]]] = [
            {} for _ in range(len(self.node_type))
        ]
        self._pairs: dict[tuple[int, int], tuple[float, str]] = {}
        self._source_counts: Counter[str] = Counter()

    def _key(self, i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _worst_neighbor(self, node: int) -> int | None:
        candidates = self._per_node[node]
        if not candidates:
            return None
        return max(candidates, key=lambda other: (candidates[other][0], other))

    def _can_accept(self, node: int, other: int, score: float) -> bool:
        current = self._per_node[node]
        if other in current or len(current) < self.K:
            return True
        worst = self._worst_neighbor(node)
        return worst is not None and score < current[worst][0]

    def _remove_pair(self, i: int, j: int) -> None:
        key = self._key(i, j)
        self._pairs.pop(key, None)
        self._per_node[i].pop(j, None)
        self._per_node[j].pop(i, None)

    def _evict_for(self, node: int) -> None:
        if len(self._per_node[node]) < self.K:
            return
        worst = self._worst_neighbor(node)
        if worst is not None:
            self._remove_pair(node, worst)

    def add(self, i: int, j: int, score: float, source: str) -> None:
        i = int(i)
        j = int(j)
        if i == j:
            return
        if i < 0 or j < 0 or i >= len(self.node_type) or j >= len(self.node_type):
            raise IndexError("candidate endpoint out of bounds")
        if self.same_type_only and self.node_type[i] != self.node_type[j]:
            return
        score = float(score)
        key = self._key(i, j)

        if key in self._pairs:
            old_score, old_source = self._pairs[key]
            if score < old_score:
                self._pairs[key] = (score, source)
                self._per_node[i][j] = (score, source)
                self._per_node[j][i] = (score, source)
            else:
                self._per_node[i][j] = (old_score, old_source)
                self._per_node[j][i] = (old_score, old_source)
            return

        if not self._can_accept(i, j, score) or not self._can_accept(j, i, score):
            return
        self._evict_for(i)
        self._evict_for(j)
        self._pairs[key] = (score, source)
        self._per_node[i][j] = (score, source)
        self._per_node[j][i] = (score, source)
        self._source_counts[source] += 1

    def to_pairs(self) -> np.ndarray:
        rows = [
            (i, j, score)
            for (i, j), (score, _source) in sorted(self._pairs.items())
        ]
        if not rows:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def iter_pair_blocks(self, block_size: int = 65_536):
        block_size = max(int(block_size), 1)
        rows: list[tuple[int, int, float]] = []
        for (i, j), (score, _source) in self._pairs.items():
            rows.append((int(i), int(j), float(score)))
            if len(rows) >= block_size:
                yield np.asarray(rows, dtype=np.float64)
                rows = []
        if rows:
            yield np.asarray(rows, dtype=np.float64)

    def pair_count(self) -> int:
        return len(self._pairs)

    def counts(self) -> np.ndarray:
        return np.asarray([len(candidates) for candidates in self._per_node], dtype=np.int32)

    def source_counts(self) -> dict[str, int]:
        return dict(Counter(source for _score, source in self._pairs.values()))

    def source_for_pair(self, i: int, j: int) -> str | None:
        item = self._pairs.get(self._key(int(i), int(j)))
        return None if item is None else str(item[1])
