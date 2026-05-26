from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sample_random_edge_indices(*, edge_count: int, budget: int, graph_seed: int, relation_id: int | str) -> np.ndarray:
    edge_count = max(0, int(edge_count))
    budget = max(0, min(int(budget), edge_count))
    if budget >= edge_count:
        return np.arange(edge_count, dtype=np.int64)
    rng = np.random.default_rng(int(graph_seed) * 1009 + int(relation_id) * 9176)
    return np.sort(rng.choice(edge_count, size=budget, replace=False).astype(np.int64))


def max_endpoint_degree(src: np.ndarray, dst: np.ndarray) -> int:
    if len(src) == 0:
        return 0
    counts: dict[tuple[str, int], int] = {}
    for value in np.asarray(src, dtype=np.int64).tolist():
        key = ("s", int(value))
        counts[key] = counts.get(key, 0) + 1
    for value in np.asarray(dst, dtype=np.int64).tolist():
        key = ("d", int(value))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or float(np.sum(arr)) <= 0.0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.sum((np.arange(n) + 1) * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


@dataclass(frozen=True)
class CoverageSampler:
    hub_cap: int | None = 32

    def select(
        self,
        *,
        src: np.ndarray,
        dst: np.ndarray,
        scores: np.ndarray,
        budget: int,
        graph_seed: int,
        relation_id: int | str,
        min_edges: int = 1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        src = np.asarray(src, dtype=np.int64).reshape(-1)
        dst = np.asarray(dst, dtype=np.int64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if src.shape != dst.shape or src.shape != scores.shape:
            raise ValueError("src, dst, and scores must have the same shape")
        edge_count = int(src.size)
        budget = max(0, min(int(budget), edge_count))
        if budget >= edge_count:
            selected = np.arange(edge_count, dtype=np.int64)
            return selected, self._diagnostics(src, dst, selected, 0, 0)
        rng = np.random.default_rng(int(graph_seed) * 1009 + int(relation_id) * 9176)
        jitter = rng.random(edge_count) * 1e-12
        order = np.lexsort((jitter, -scores))
        selected: list[int] = []
        src_degree: dict[int, int] = {}
        dst_degree: dict[int, int] = {}
        hub_cap_active = 0
        for idx in order.tolist():
            if len(selected) >= budget:
                break
            s = int(src[idx])
            d = int(dst[idx])
            if self.hub_cap is not None and (
                src_degree.get(s, 0) >= int(self.hub_cap) or dst_degree.get(d, 0) >= int(self.hub_cap)
            ):
                hub_cap_active += 1
                continue
            selected.append(int(idx))
            src_degree[s] = src_degree.get(s, 0) + 1
            dst_degree[d] = dst_degree.get(d, 0) + 1
        orphan_rescue = 0
        minimum = min(edge_count, max(0, int(min_edges)))
        for idx in order.tolist():
            if len(selected) >= max(minimum, budget):
                break
            if int(idx) not in selected:
                selected.append(int(idx))
                orphan_rescue += 1
        selected_arr = np.asarray(sorted(set(selected[:budget])), dtype=np.int64)
        return selected_arr, self._diagnostics(src, dst, selected_arr, hub_cap_active, orphan_rescue)

    def _diagnostics(self, src: np.ndarray, dst: np.ndarray, selected: np.ndarray, hub_cap_active: int, orphan_rescue: int) -> dict[str, Any]:
        selected_src = src[selected] if selected.size else np.empty(0, dtype=np.int64)
        selected_dst = dst[selected] if selected.size else np.empty(0, dtype=np.int64)
        source_before = len(set(src.tolist()))
        dest_before = len(set(dst.tolist()))
        source_after = len(set(selected_src.tolist()))
        dest_after = len(set(selected_dst.tolist()))
        endpoint_values_before = np.asarray(list(np.bincount(src).tolist()) + list(np.bincount(dst).tolist()), dtype=np.float64)
        endpoint_values_after = (
            np.asarray(list(np.bincount(selected_src).tolist()) + list(np.bincount(selected_dst).tolist()), dtype=np.float64)
            if selected.size
            else np.empty(0, dtype=np.float64)
        )
        degrees = []
        if selected.size:
            for value in selected_src.tolist():
                degrees.append(int(np.sum(selected_src == value)))
            for value in selected_dst.tolist():
                degrees.append(int(np.sum(selected_dst == value)))
        return {
            "candidate_source_node_count": int(source_before),
            "candidate_destination_node_count": int(dest_before),
            "retained_source_node_count": int(source_after),
            "retained_destination_node_count": int(dest_after),
            "source_coverage_ratio": float(source_after / max(source_before, 1)),
            "destination_coverage_ratio": float(dest_after / max(dest_before, 1)),
            "target_author_reachability_before": float(source_before > 0 or dest_before > 0),
            "target_author_reachability_after": float(source_after > 0 or dest_after > 0),
            "paper_coverage_ratio": float((source_after + dest_after) / max(source_before + dest_before, 1)),
            "venue_coverage_ratio": float(dest_after / max(dest_before, 1)),
            "term_coverage_ratio": float(source_after / max(source_before, 1)),
            "max_endpoint_retained_degree": int(max(degrees) if degrees else 0),
            "p95_endpoint_retained_degree": float(np.percentile(degrees, 95)) if degrees else 0.0,
            "hub_cap_active_count": int(hub_cap_active),
            "orphan_rescue_count": int(orphan_rescue),
            "edge_gini_before": _gini(endpoint_values_before),
            "edge_gini_after": _gini(endpoint_values_after),
        }
