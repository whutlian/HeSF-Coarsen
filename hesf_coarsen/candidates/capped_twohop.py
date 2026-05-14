from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.progress import progress_iter, progress_message


@dataclass(frozen=True)
class CappedTwoHopIncidentIndex:
    """CSR-style incident lists keyed by ``(middle_node, endpoint_type)``."""

    middle: np.ndarray
    endpoint_type: np.ndarray
    endpoints: np.ndarray
    indptr: np.ndarray
    num_nodes: int
    mmap_dir: Path | None = None

    @classmethod
    def from_graph(
        cls,
        graph: HeteroGraph,
        edge_chunk_size: int = 1_000_000,
        mmap_dir: str | Path | None = None,
        progress_config: dict | None = None,
    ) -> "CappedTwoHopIncidentIndex":
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        if mmap_dir is not None:
            return cls._from_graph_memmap(graph, edge_chunk_size, Path(mmap_dir), progress_config)

        middle_chunks: list[np.ndarray] = []
        endpoint_type_chunks: list[np.ndarray] = []
        endpoint_chunks: list[np.ndarray] = []
        for rel in graph.relations.values():
            ranges = range(0, rel.num_edges, edge_chunk_size)
            total = (rel.num_edges + edge_chunk_size - 1) // edge_chunk_size
            for edge_start in progress_iter(
                ranges,
                total=total,
                desc=f"incident index relation {rel.relation_id}",
                config=progress_config,
                unit="chunk",
            ):
                edge_stop = min(edge_start + edge_chunk_size, rel.num_edges)
                src = rel.src[edge_start:edge_stop]
                dst = rel.dst[edge_start:edge_stop]
                if len(src) == 0:
                    continue
                middle_chunks.extend([dst, src])
                endpoint_chunks.extend([src, dst])
                endpoint_type_chunks.extend(
                    [
                        np.full(len(src), rel.src_type, dtype=np.int32),
                        np.full(len(src), rel.dst_type, dtype=np.int32),
                    ]
                )

        if not middle_chunks:
            return cls._from_arrays(
                middle=np.empty(0, dtype=np.int64),
                endpoint_type=np.empty(0, dtype=np.int32),
                endpoints=np.empty(0, dtype=np.int64),
                indptr=np.array([0], dtype=np.int64),
                num_nodes=graph.num_nodes,
                mmap_dir=mmap_dir,
            )

        middle_all = np.concatenate(middle_chunks).astype(np.int64, copy=False)
        endpoint_type_all = np.concatenate(endpoint_type_chunks).astype(np.int32, copy=False)
        endpoint_all = np.concatenate(endpoint_chunks).astype(np.int64, copy=False)
        order = np.lexsort((endpoint_all, endpoint_type_all, middle_all))
        middle_all = middle_all[order]
        endpoint_type_all = endpoint_type_all[order]
        endpoint_all = endpoint_all[order]

        keep = np.ones(len(endpoint_all), dtype=bool)
        keep[1:] = (
            (middle_all[1:] != middle_all[:-1])
            | (endpoint_type_all[1:] != endpoint_type_all[:-1])
            | (endpoint_all[1:] != endpoint_all[:-1])
        )
        middle_all = middle_all[keep]
        endpoint_type_all = endpoint_type_all[keep]
        endpoint_all = endpoint_all[keep]

        group_start = np.r_[
            0,
            np.flatnonzero(
                (middle_all[1:] != middle_all[:-1])
                | (endpoint_type_all[1:] != endpoint_type_all[:-1])
            )
            + 1,
        ]
        indptr = np.r_[group_start, len(endpoint_all)].astype(np.int64, copy=False)
        return cls._from_arrays(
            middle=middle_all[group_start],
            endpoint_type=endpoint_type_all[group_start],
            endpoints=endpoint_all,
            indptr=indptr,
            num_nodes=graph.num_nodes,
            mmap_dir=mmap_dir,
        )

    @classmethod
    def _from_arrays(
        cls,
        middle: np.ndarray,
        endpoint_type: np.ndarray,
        endpoints: np.ndarray,
        indptr: np.ndarray,
        num_nodes: int,
        mmap_dir: str | Path | None,
    ) -> "CappedTwoHopIncidentIndex":
        if mmap_dir is None:
            return cls(
                middle=middle,
                endpoint_type=endpoint_type,
                endpoints=endpoints,
                indptr=indptr,
                num_nodes=num_nodes,
                mmap_dir=None,
            )

        root = Path(mmap_dir)
        root.mkdir(parents=True, exist_ok=True)
        middle_mmap = np.lib.format.open_memmap(
            root / "incident_middle.npy",
            mode="w+",
            dtype=np.int64,
            shape=middle.shape,
        )
        endpoint_type_mmap = np.lib.format.open_memmap(
            root / "incident_endpoint_type.npy",
            mode="w+",
            dtype=np.int32,
            shape=endpoint_type.shape,
        )
        endpoints_mmap = np.lib.format.open_memmap(
            root / "incident_endpoints.npy",
            mode="w+",
            dtype=np.int64,
            shape=endpoints.shape,
        )
        indptr_mmap = np.lib.format.open_memmap(
            root / "incident_indptr.npy",
            mode="w+",
            dtype=np.int64,
            shape=indptr.shape,
        )
        middle_mmap[:] = middle
        endpoint_type_mmap[:] = endpoint_type
        endpoints_mmap[:] = endpoints
        indptr_mmap[:] = indptr
        for array in [middle_mmap, endpoint_type_mmap, endpoints_mmap, indptr_mmap]:
            array.flush()
        return cls(
            middle=middle_mmap,
            endpoint_type=endpoint_type_mmap,
            endpoints=endpoints_mmap,
            indptr=indptr_mmap,
            num_nodes=num_nodes,
            mmap_dir=root,
        )

    @classmethod
    def _from_graph_memmap(
        cls,
        graph: HeteroGraph,
        edge_chunk_size: int,
        mmap_dir: Path,
        progress_config: dict | None,
    ) -> "CappedTwoHopIncidentIndex":
        mmap_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = mmap_dir / "incident_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for stale in chunk_dir.glob("chunk_*.npy"):
            stale.unlink()

        chunk_paths: list[Path] = []
        chunk_index = 0
        for rel in graph.relations.values():
            ranges = range(0, rel.num_edges, edge_chunk_size)
            total = (rel.num_edges + edge_chunk_size - 1) // edge_chunk_size
            for edge_start in progress_iter(
                ranges,
                total=total,
                desc=f"incident mmap relation {rel.relation_id}",
                config=progress_config,
                unit="chunk",
            ):
                edge_stop = min(edge_start + edge_chunk_size, rel.num_edges)
                src = rel.src[edge_start:edge_stop]
                dst = rel.dst[edge_start:edge_stop]
                if len(src) == 0:
                    continue
                chunk_path = chunk_dir / f"chunk_{chunk_index:08d}.npy"
                cls._write_sorted_incident_chunk(
                    chunk_path,
                    src,
                    dst,
                    rel.src_type,
                    rel.dst_type,
                )
                chunk_paths.append(chunk_path)
                chunk_index += 1

        progress_message(progress_config, "incident mmap merge: counting unique triples")
        total_unique, group_count = cls._count_merged_chunks(chunk_paths)
        middle = np.lib.format.open_memmap(
            mmap_dir / "incident_middle.npy",
            mode="w+",
            dtype=np.int64,
            shape=(group_count,),
        )
        endpoint_type = np.lib.format.open_memmap(
            mmap_dir / "incident_endpoint_type.npy",
            mode="w+",
            dtype=np.int32,
            shape=(group_count,),
        )
        endpoints = np.lib.format.open_memmap(
            mmap_dir / "incident_endpoints.npy",
            mode="w+",
            dtype=np.int64,
            shape=(total_unique,),
        )
        indptr = np.lib.format.open_memmap(
            mmap_dir / "incident_indptr.npy",
            mode="w+",
            dtype=np.int64,
            shape=(group_count + 1,),
        )
        progress_message(progress_config, "incident mmap merge: writing final index")
        cls._write_merged_chunks(chunk_paths, middle, endpoint_type, endpoints, indptr)
        for array in [middle, endpoint_type, endpoints, indptr]:
            array.flush()
        for chunk_path in chunk_paths:
            try:
                chunk_path.unlink()
            except OSError:
                pass
        return cls(
            middle=middle,
            endpoint_type=endpoint_type,
            endpoints=endpoints,
            indptr=indptr,
            num_nodes=graph.num_nodes,
            mmap_dir=mmap_dir,
        )

    @staticmethod
    def _write_sorted_incident_chunk(
        path: Path,
        src: np.ndarray,
        dst: np.ndarray,
        src_type: int,
        dst_type: int,
    ) -> None:
        rows = np.empty((2 * len(src), 3), dtype=np.int64)
        rows[: len(src), 0] = dst
        rows[: len(src), 1] = int(src_type)
        rows[: len(src), 2] = src
        rows[len(src) :, 0] = src
        rows[len(src) :, 1] = int(dst_type)
        rows[len(src) :, 2] = dst
        if len(rows) > 1:
            order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
            keep = np.ones(len(rows), dtype=bool)
            keep[1:] = np.any(rows[1:] != rows[:-1], axis=1)
            rows = rows[keep]
        target = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.int64,
            shape=rows.shape,
        )
        target[:] = rows
        target.flush()

    @staticmethod
    def _iter_merged_chunks(chunk_paths: list[Path]):
        chunks = [np.load(path, mmap_mode="r") for path in chunk_paths]
        heap: list[tuple[int, int, int, int]] = []
        positions = [0 for _ in chunks]
        for chunk_id, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            heapq.heappush(
                heap,
                (
                    int(chunk[0, 0]),
                    int(chunk[0, 1]),
                    int(chunk[0, 2]),
                    chunk_id,
                ),
            )

        previous: tuple[int, int, int] | None = None
        while heap:
            middle, endpoint_type, endpoint, chunk_id = heapq.heappop(heap)
            current = (middle, endpoint_type, endpoint)
            if current != previous:
                yield current
                previous = current
            positions[chunk_id] += 1
            position = positions[chunk_id]
            chunk = chunks[chunk_id]
            if position < len(chunk):
                heapq.heappush(
                    heap,
                    (
                        int(chunk[position, 0]),
                        int(chunk[position, 1]),
                        int(chunk[position, 2]),
                        chunk_id,
                    ),
                )

    @classmethod
    def _count_merged_chunks(cls, chunk_paths: list[Path]) -> tuple[int, int]:
        total_unique = 0
        group_count = 0
        previous_group: tuple[int, int] | None = None
        for middle, endpoint_type, _endpoint in cls._iter_merged_chunks(chunk_paths):
            total_unique += 1
            group = (middle, endpoint_type)
            if group != previous_group:
                group_count += 1
                previous_group = group
        return total_unique, group_count

    @classmethod
    def _write_merged_chunks(
        cls,
        chunk_paths: list[Path],
        middle: np.ndarray,
        endpoint_type: np.ndarray,
        endpoints: np.ndarray,
        indptr: np.ndarray,
    ) -> None:
        endpoint_pos = 0
        group_pos = -1
        previous_group: tuple[int, int] | None = None
        for middle_node, endpoint_type_id, endpoint in cls._iter_merged_chunks(chunk_paths):
            group = (middle_node, endpoint_type_id)
            if group != previous_group:
                group_pos += 1
                middle[group_pos] = middle_node
                endpoint_type[group_pos] = endpoint_type_id
                indptr[group_pos] = endpoint_pos
                previous_group = group
            endpoints[endpoint_pos] = endpoint
            endpoint_pos += 1
        indptr[group_pos + 1] = endpoint_pos

    def collect_middle_range(
        self,
        start_middle: int,
        stop_middle: int,
    ) -> dict[int, dict[int, np.ndarray]]:
        incident: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        if len(self.middle) == 0:
            return incident
        start_group = int(np.searchsorted(self.middle, start_middle, side="left"))
        stop_group = int(np.searchsorted(self.middle, stop_middle, side="left"))
        for group in range(start_group, stop_group):
            start = int(self.indptr[group])
            stop = int(self.indptr[group + 1])
            incident[int(self.middle[group])][int(self.endpoint_type[group])] = self.endpoints[start:stop]
        return incident


def _degree_cap(lengths: list[int], policy: object) -> int | None:
    if policy in (None, "none", "off", False):
        return None
    if isinstance(policy, (int, float)):
        return max(1, int(policy))
    text = str(policy)
    if text.startswith("p"):
        percentile = float(text[1:])
        return max(1, int(np.percentile(np.asarray(lengths, dtype=np.float32), percentile)))
    raise ValueError(f"unsupported middle_degree_cap_policy: {policy}")


def _sample_pairs(nodes: np.ndarray, cap: int, seed: int) -> list[tuple[int, int]]:
    n = len(nodes)
    total = n * (n - 1) // 2
    if total <= cap:
        return [(int(i), int(j)) for i, j in combinations(nodes.tolist(), 2)]
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(cap * 12, 32)
    while len(pairs) < cap and attempts < max_attempts:
        a, b = rng.choice(nodes, size=2, replace=False)
        i, j = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        pairs.add((i, j))
        attempts += 1
    if len(pairs) < cap:
        for offset in range(1, n):
            for idx in range(n):
                i = int(nodes[idx])
                j = int(nodes[(idx + offset) % n])
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                pairs.add(pair)
                if len(pairs) >= cap:
                    return sorted(pairs)
    return sorted(pairs)


def generate_capped_twohop_candidates(
    graph: HeteroGraph,
    Z: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
) -> None:
    candidate_cfg = config.get("candidates", {})
    coarsen_cfg = config.get("coarsening", {})
    per_middle_pair_cap = int(candidate_cfg.get("per_middle_pair_cap", 64))
    quotas = candidate_cfg.get("quotas", {}) or {}
    twohop_max_fraction = float(quotas.get("twohop_max_fraction", 1.0) or 1.0) if isinstance(quotas, dict) else 1.0
    twohop_score_scale = 1.0 + max(0.0, 1.0 - twohop_max_fraction)
    same_partition = bool(coarsen_cfg.get("same_partition_only", True))
    seed = int(config.get("seed", 12345))

    incident: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rel in graph.relations.values():
        for src, dst in zip(rel.src, rel.dst):
            incident[int(dst)][rel.src_type].append(int(src))
            incident[int(src)][rel.dst_type].append(int(dst))

    lengths = [
        len(set(nodes))
        for by_type in incident.values()
        for nodes in by_type.values()
        if len(nodes) > 1
    ]
    cap = _degree_cap(lengths or [1], candidate_cfg.get("middle_degree_cap_policy", "p99"))

    for middle in sorted(incident):
        for endpoint_type in sorted(incident[middle]):
            endpoints = np.asarray(sorted(set(incident[middle][endpoint_type])), dtype=np.int64)
            endpoints = endpoints[endpoints != middle]
            if len(endpoints) < 2:
                continue
            if same_partition:
                partitions = partition_id[endpoints]
                kept: list[np.ndarray] = []
                for partition in np.unique(partitions):
                    group = endpoints[partitions == partition]
                    if len(group) >= 2:
                        kept.append(group)
                groups = kept
            else:
                groups = [endpoints]

            for group in groups:
                if cap is not None and len(group) > cap:
                    rng = np.random.default_rng(seed + middle * 1009 + endpoint_type)
                    group = np.sort(rng.choice(group, size=cap, replace=False))
                pair_seed = seed + middle * 9176 + endpoint_type * 131
                for i, j in _sample_pairs(group, per_middle_pair_cap, pair_seed):
                    diff = Z[i].astype(np.float32) - Z[j].astype(np.float32)
                    store.add(i, j, float(np.dot(diff, diff)) * twohop_score_scale, "capped_twohop")


def _collect_incident_for_middle_range(
    graph: HeteroGraph,
    start_middle: int,
    stop_middle: int,
    edge_chunk_size: int,
) -> dict[int, dict[int, list[int]]]:
    incident: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rel in graph.relations.values():
        for edge_start in range(0, rel.num_edges, edge_chunk_size):
            edge_stop = min(edge_start + edge_chunk_size, rel.num_edges)
            src = rel.src[edge_start:edge_stop]
            dst = rel.dst[edge_start:edge_stop]
            dst_mask = (dst >= start_middle) & (dst < stop_middle)
            for middle, endpoint in zip(dst[dst_mask], src[dst_mask]):
                incident[int(middle)][rel.src_type].append(int(endpoint))
            src_mask = (src >= start_middle) & (src < stop_middle)
            for middle, endpoint in zip(src[src_mask], dst[src_mask]):
                incident[int(middle)][rel.dst_type].append(int(endpoint))
    return incident


def generate_capped_twohop_candidates_chunked(
    graph: HeteroGraph,
    Z: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
    middle_chunk_size: int = 100_000,
    edge_chunk_size: int = 1_000_000,
) -> dict[str, int]:
    if middle_chunk_size <= 0:
        raise ValueError("middle_chunk_size must be positive")
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")
    candidate_cfg = config.get("candidates", {})
    coarsen_cfg = config.get("coarsening", {})
    per_middle_pair_cap = int(candidate_cfg.get("per_middle_pair_cap", 64))
    quotas = candidate_cfg.get("quotas", {}) or {}
    twohop_max_fraction = float(quotas.get("twohop_max_fraction", 1.0) or 1.0) if isinstance(quotas, dict) else 1.0
    twohop_score_scale = 1.0 + max(0.0, 1.0 - twohop_max_fraction)
    same_partition = bool(coarsen_cfg.get("same_partition_only", True))
    seed = int(config.get("seed", 12345))
    global_cap = candidate_cfg.get("middle_degree_cap_policy", "p99")
    cap = None if global_cap in (None, "none", "off", False) else int(global_cap) if isinstance(global_cap, (int, float)) else None

    incident_index = CappedTwoHopIncidentIndex.from_graph(
        graph,
        edge_chunk_size=edge_chunk_size,
        mmap_dir=candidate_cfg.get("incident_index_mmap_dir"),
        progress_config=config,
    )
    total_emitted = 0
    max_per_middle = 0
    middle_count = 0
    ranges = range(0, graph.num_nodes, middle_chunk_size)
    total = (graph.num_nodes + middle_chunk_size - 1) // middle_chunk_size
    for start_middle in progress_iter(
        ranges,
        total=total,
        desc="capped two-hop middle chunks",
        config=config,
        unit="chunk",
    ):
        stop_middle = min(start_middle + middle_chunk_size, graph.num_nodes)
        incident = incident_index.collect_middle_range(start_middle, stop_middle)
        if cap is None and global_cap not in (None, "none", "off", False):
            lengths = [
                len(set(nodes))
                for by_type in incident.values()
                for nodes in by_type.values()
                if len(nodes) > 1
            ]
            local_cap = _degree_cap(lengths or [1], global_cap)
        else:
            local_cap = cap
        for middle in sorted(incident):
            middle_emitted = 0
            middle_count += 1
            for endpoint_type in sorted(incident[middle]):
                endpoints = np.asarray(sorted(set(incident[middle][endpoint_type])), dtype=np.int64)
                endpoints = endpoints[endpoints != middle]
                if len(endpoints) < 2:
                    continue
                if same_partition:
                    partitions = partition_id[endpoints]
                    groups = [
                        endpoints[partitions == partition]
                        for partition in np.unique(partitions)
                        if np.sum(partitions == partition) >= 2
                    ]
                else:
                    groups = [endpoints]
                for group in groups:
                    if local_cap is not None and len(group) > local_cap:
                        rng = np.random.default_rng(seed + middle * 1009 + endpoint_type)
                        group = np.sort(rng.choice(group, size=local_cap, replace=False))
                    remaining = per_middle_pair_cap - middle_emitted
                    if remaining <= 0:
                        break
                    pair_seed = seed + middle * 9176 + endpoint_type * 131
                    pairs = _sample_pairs(group, remaining, pair_seed)
                    for i, j in pairs:
                        diff = Z[i].astype(np.float32) - Z[j].astype(np.float32)
                        store.add(i, j, float(np.dot(diff, diff)) * twohop_score_scale, "capped_twohop")
                    middle_emitted += len(pairs)
                    total_emitted += len(pairs)
                if middle_emitted >= per_middle_pair_cap:
                    break
            max_per_middle = max(max_per_middle, middle_emitted)
    return {
        "middle_nodes_considered": middle_count,
        "pairs_considered": total_emitted,
        "max_pairs_emitted_per_middle": max_per_middle,
    }
