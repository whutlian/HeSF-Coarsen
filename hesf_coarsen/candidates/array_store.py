from __future__ import annotations

from pathlib import Path

import numpy as np


class ArrayCandidateStore:
    """Fixed-size per-node candidate store with optional mmap backing.

    Each node owns exactly ``K`` slots. Insertion is symmetric: a pair is
    retained only if both endpoints can keep it under their local top-K budget.
    The final pair table is deduplicated by sorting canonical pair keys.
    """

    _SOURCE_CODES = {
        "unknown": 0,
        "onehop": 1,
        "capped_twohop": 2,
        "bucket": 3,
        "fallback": 4,
        "partition_ann": 5,
    }

    def __init__(
        self,
        node_type: np.ndarray,
        K: int,
        same_type_only: bool = True,
        mmap_dir: str | Path | None = None,
    ):
        if K <= 0:
            raise ValueError("K must be positive")
        self.node_type = np.asarray(node_type, dtype=np.int32)
        self.K = int(K)
        self.same_type_only = bool(same_type_only)
        self.mmap_dir = Path(mmap_dir) if mmap_dir is not None else None
        shape = (len(self.node_type), self.K)

        if self.mmap_dir is None:
            self.candidate_ids = np.full(shape, -1, dtype=np.int64)
            self.candidate_scores = np.full(shape, np.inf, dtype=np.float32)
            self.candidate_sources = np.zeros(shape, dtype=np.int16)
            self._counts = np.zeros(len(self.node_type), dtype=np.int32)
        else:
            self.mmap_dir.mkdir(parents=True, exist_ok=True)
            self.candidate_ids = np.lib.format.open_memmap(
                self.mmap_dir / "candidate_ids.npy",
                mode="w+",
                dtype=np.int64,
                shape=shape,
            )
            self.candidate_scores = np.lib.format.open_memmap(
                self.mmap_dir / "candidate_scores.npy",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            self.candidate_sources = np.lib.format.open_memmap(
                self.mmap_dir / "candidate_sources.npy",
                mode="w+",
                dtype=np.int16,
                shape=shape,
            )
            self._counts = np.lib.format.open_memmap(
                self.mmap_dir / "candidate_counts.npy",
                mode="w+",
                dtype=np.int32,
                shape=(len(self.node_type),),
            )
            self.candidate_ids[:] = -1
            self.candidate_scores[:] = np.inf
            self.candidate_sources[:] = 0
            self._counts[:] = 0

    def _source_code(self, source: str) -> int:
        if source not in self._SOURCE_CODES:
            self._SOURCE_CODES[source] = max(self._SOURCE_CODES.values()) + 1
        return int(self._SOURCE_CODES[source])

    def _source_name(self, code: int) -> str:
        for name, value in self._SOURCE_CODES.items():
            if value == int(code):
                return name
        return "unknown"

    def _existing_slot(self, node: int, other: int) -> int | None:
        used = int(self._counts[node])
        if used == 0:
            return None
        matches = np.flatnonzero(self.candidate_ids[node, :used] == other)
        return int(matches[0]) if len(matches) else None

    def _accept_slot(self, node: int, other: int, score: float) -> int | None:
        existing = self._existing_slot(node, other)
        if existing is not None:
            return existing
        used = int(self._counts[node])
        if used < self.K:
            return used
        worst = int(np.argmax(self.candidate_scores[node]))
        if score < float(self.candidate_scores[node, worst]):
            return worst
        return None

    def _delete_slot(self, node: int, slot: int) -> None:
        used = int(self._counts[node])
        if slot < 0 or slot >= used:
            return
        last = used - 1
        if slot != last:
            self.candidate_ids[node, slot] = self.candidate_ids[node, last]
            self.candidate_scores[node, slot] = self.candidate_scores[node, last]
            self.candidate_sources[node, slot] = self.candidate_sources[node, last]
        self.candidate_ids[node, last] = -1
        self.candidate_scores[node, last] = np.inf
        self.candidate_sources[node, last] = 0
        self._counts[node] = last

    def _remove_directed(self, node: int, other: int) -> None:
        slot = self._existing_slot(node, other)
        if slot is not None:
            self._delete_slot(node, slot)

    def _remove_pair(self, i: int, j: int) -> None:
        self._remove_directed(i, j)
        self._remove_directed(j, i)

    def _write_slot(self, node: int, slot: int, other: int, score: float, source_code: int) -> None:
        used = int(self._counts[node])
        if slot < used:
            old_other = int(self.candidate_ids[node, slot])
            if old_other >= 0 and old_other != other:
                self._remove_directed(old_other, node)
        self.candidate_ids[node, slot] = other
        self.candidate_scores[node, slot] = score
        self.candidate_sources[node, slot] = source_code
        if slot >= int(self._counts[node]):
            self._counts[node] = slot + 1

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
        source_code = self._source_code(source)
        existing_i = self._existing_slot(i, j)
        existing_j = self._existing_slot(j, i)
        if existing_i is not None or existing_j is not None:
            if existing_i is None or existing_j is None:
                self._remove_pair(i, j)
            else:
                old_score = min(
                    float(self.candidate_scores[i, existing_i]),
                    float(self.candidate_scores[j, existing_j]),
                )
                if score >= old_score:
                    return
                self._write_slot(i, existing_i, j, score, source_code)
                self._write_slot(j, existing_j, i, score, source_code)
                return

        slot_i = self._accept_slot(i, j, score)
        slot_j = self._accept_slot(j, i, score)
        if slot_i is None or slot_j is None:
            return
        self._write_slot(i, slot_i, j, score, source_code)
        self._write_slot(j, slot_j, i, score, source_code)

    def add_many(
        self,
        left: np.ndarray,
        right: np.ndarray,
        scores: np.ndarray,
        source: str,
    ) -> None:
        for i, j, score in zip(left, right, scores):
            self.add(int(i), int(j), float(score), source)

    def counts(self) -> np.ndarray:
        return np.asarray(self._counts, dtype=np.int32)

    def to_pairs(self) -> np.ndarray:
        rows: list[tuple[int, int, float, int]] = []
        for node in range(len(self.node_type)):
            used = int(self._counts[node])
            for slot in range(used):
                other = int(self.candidate_ids[node, slot])
                if other < 0 or node == other:
                    continue
                i, j = (node, other) if node < other else (other, node)
                rows.append((i, j, float(self.candidate_scores[node, slot]), int(self.candidate_sources[node, slot])))
        if not rows:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.asarray(rows, dtype=np.float64)
        pair_keys = arr[:, 0].astype(np.int64) * np.int64(len(self.node_type)) + arr[:, 1].astype(np.int64)
        order = np.lexsort((arr[:, 2], pair_keys))
        ordered = arr[order]
        ordered_keys = pair_keys[order]
        first = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1]
        return ordered[first, :3]

    def iter_pair_blocks(self, block_size: int = 65_536):
        block_size = max(int(block_size), 1)
        rows: list[tuple[int, int, float]] = []
        for node in range(len(self.node_type)):
            used = int(self._counts[node])
            for slot in range(used):
                other = int(self.candidate_ids[node, slot])
                if other < 0 or node >= other:
                    continue
                rows.append((node, other, float(self.candidate_scores[node, slot])))
                if len(rows) >= block_size:
                    yield np.asarray(rows, dtype=np.float64)
                    rows = []
        if rows:
            yield np.asarray(rows, dtype=np.float64)

    def pair_count(self) -> int:
        count = 0
        for node in range(len(self.node_type)):
            used = int(self._counts[node])
            if used == 0:
                continue
            others = self.candidate_ids[node, :used]
            count += int(np.count_nonzero(others > node))
        return count

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in range(len(self.node_type)):
            used = int(self._counts[node])
            for slot in range(used):
                other = int(self.candidate_ids[node, slot])
                if other < 0 or node >= other:
                    continue
                name = self._source_name(int(self.candidate_sources[node, slot]))
                counts[name] = counts.get(name, 0) + 1
        return counts

    def source_node_coverage(self) -> dict[str, float]:
        nodes_by_source: dict[str, set[int]] = {}
        for node in range(len(self.node_type)):
            used = int(self._counts[node])
            for slot in range(used):
                other = int(self.candidate_ids[node, slot])
                if other < 0:
                    continue
                name = self._source_name(int(self.candidate_sources[node, slot]))
                nodes = nodes_by_source.setdefault(name, set())
                nodes.add(int(node))
                nodes.add(other)
        total_nodes = max(len(self.node_type), 1)
        return {
            source: float(len(nodes) / total_nodes)
            for source, nodes in sorted(nodes_by_source.items())
        }

    def buffer_nbytes(self) -> dict[str, int]:
        payload = {
            "candidate_ids_bytes": int(self.candidate_ids.nbytes),
            "candidate_scores_bytes": int(self.candidate_scores.nbytes),
            "candidate_sources_bytes": int(self.candidate_sources.nbytes),
            "candidate_counts_bytes": int(self._counts.nbytes),
        }
        payload["estimated_total_bytes"] = int(sum(payload.values()))
        return payload

    def source_for_pair(self, i: int, j: int) -> str | None:
        i = int(i)
        j = int(j)
        if i < 0 or j < 0 or i >= len(self.node_type) or j >= len(self.node_type):
            return None
        slot = self._existing_slot(i, j)
        if slot is None:
            slot = self._existing_slot(j, i)
            node = j
        else:
            node = i
        if slot is None:
            return None
        return self._source_name(int(self.candidate_sources[node, slot]))

    def flush(self) -> None:
        for array in [self.candidate_ids, self.candidate_scores, self.candidate_sources, self._counts]:
            if isinstance(array, np.memmap):
                array.flush()
