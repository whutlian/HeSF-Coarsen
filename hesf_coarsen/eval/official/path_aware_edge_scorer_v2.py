from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hesf_coarsen.eval.official.relation_schema import relation_pair_name
from hesf_coarsen.io.schema import HeteroGraph


EDGE_SCORE_V2_DIAGNOSTIC_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "relation_id",
    "relation_name",
    "relation_pair_name",
    "edge_count",
    "score_min",
    "score_max",
    "score_mean",
    "score_std",
    "score_component_target_reachability_mean",
    "score_component_feature_similarity_mean",
    "score_component_label_proxy_mean",
    "score_component_inverse_frequency_mean",
    "score_component_hub_penalty_mean",
    "score_component_duplicate_penalty_mean",
    "topk_score_threshold",
    "trainval_label_used",
    "test_label_used",
    "no_test_label_usage",
]


def saturate(x: np.ndarray, cap: float) -> np.ndarray:
    return np.minimum(np.asarray(x, dtype=np.float64), float(cap)) / max(float(cap), 1e-12)


@dataclass(frozen=True)
class PathAwareV2Scorer:
    hub_penalty_cap: float = 8.0

    def score_relation(
        self,
        *,
        dataset: str,
        method: str,
        graph_seed: int,
        relation_id: int,
        relation_name: str,
        graph: HeteroGraph,
        train_idx: np.ndarray | None,
        val_idx: np.ndarray | None,
        labels: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rel = graph.relations[int(relation_id)]
        src = np.asarray(rel.src, dtype=np.int64)
        dst = np.asarray(rel.dst, dtype=np.int64)
        if src.size == 0:
            scores = np.empty(0, dtype=np.float64)
            components = {name: np.empty(0, dtype=np.float64) for name in ["target", "feature", "label", "inverse", "hub", "duplicate"]}
        else:
            trainval = set(int(v) for v in np.asarray([] if train_idx is None else train_idx, dtype=np.int64).reshape(-1).tolist())
            trainval.update(int(v) for v in np.asarray([] if val_idx is None else val_idx, dtype=np.int64).reshape(-1).tolist())
            target = saturate(((graph.node_type[src] == 0) | (graph.node_type[dst] == 0)).astype(np.float64), 1.0)
            feature = self._feature_similarity(graph, src, dst)
            label = self._label_proxy(src, dst, trainval, labels)
            inverse = np.full(src.size, 1.0 / max(float(rel.num_edges), 1.0), dtype=np.float64)
            inverse = saturate(inverse, max(float(np.max(inverse)), 1e-12))
            src_degree = np.bincount(src, minlength=graph.num_nodes).astype(np.float64)
            dst_degree = np.bincount(dst, minlength=graph.num_nodes).astype(np.float64)
            hub = saturate(np.sqrt(np.maximum(src_degree[src] * dst_degree[dst], 1.0)), self.hub_penalty_cap)
            duplicate = saturate(src_degree[src] + dst_degree[dst], 2.0 * self.hub_penalty_cap)
            rng = np.random.default_rng(int(graph_seed) * 1009 + int(relation_id) * 9176)
            jitter = rng.random(src.size) * 1e-9
            scores = target + feature + label + inverse - 0.25 * hub - 0.10 * duplicate + jitter
            components = {"target": target, "feature": feature, "label": label, "inverse": inverse, "hub": hub, "duplicate": duplicate}
        diag = {
            "dataset": str(dataset).upper(),
            "method": str(method),
            "graph_seed": int(graph_seed),
            "relation_id": int(relation_id),
            "relation_name": str(relation_name),
            "relation_pair_name": relation_pair_name(str(relation_name)),
            "edge_count": int(src.size),
            "score_min": float(np.min(scores)) if scores.size else 0.0,
            "score_max": float(np.max(scores)) if scores.size else 0.0,
            "score_mean": float(np.mean(scores)) if scores.size else 0.0,
            "score_std": float(np.std(scores)) if scores.size else 0.0,
            "score_component_target_reachability_mean": self._mean(components["target"]),
            "score_component_feature_similarity_mean": self._mean(components["feature"]),
            "score_component_label_proxy_mean": self._mean(components["label"]),
            "score_component_inverse_frequency_mean": self._mean(components["inverse"]),
            "score_component_hub_penalty_mean": self._mean(components["hub"]),
            "score_component_duplicate_penalty_mean": self._mean(components["duplicate"]),
            "topk_score_threshold": "",
            "trainval_label_used": bool(labels is not None and (train_idx is not None or val_idx is not None)),
            "test_label_used": False,
            "no_test_label_usage": True,
        }
        return scores.astype(np.float64, copy=False), {field: diag.get(field, "") for field in EDGE_SCORE_V2_DIAGNOSTIC_FIELDS}

    @staticmethod
    def _mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0

    @staticmethod
    def _feature_similarity(graph: HeteroGraph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if graph.features is None:
            return np.zeros(src.size, dtype=np.float64)
        # Feature matrices are type-local; use feature-bearing endpoint presence as a saturated proxy.
        feature_types = set(int(type_id) for type_id in graph.features)
        has_features = np.asarray(
            [(int(graph.node_type[s]) in feature_types) or (int(graph.node_type[d]) in feature_types) for s, d in zip(src.tolist(), dst.tolist())],
            dtype=np.float64,
        )
        return saturate(has_features, 1.0)

    @staticmethod
    def _label_proxy(src: np.ndarray, dst: np.ndarray, trainval: set[int], labels: np.ndarray | None) -> np.ndarray:
        if labels is None or not trainval:
            return np.zeros(src.size, dtype=np.float64)
        return saturate(np.asarray([(int(s) in trainval) or (int(d) in trainval) for s, d in zip(src.tolist(), dst.tolist())], dtype=np.float64), 1.0)
