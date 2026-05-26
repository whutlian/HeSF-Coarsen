from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


EDGE_SCORE_DIAGNOSTIC_FIELDS = [
    "dataset",
    "seed",
    "method",
    "budget_strategy",
    "edge_score_strategy",
    "relation_id",
    "relation_name",
    "relation_pair_name",
    "edge_count",
    "score_min",
    "score_max",
    "score_mean",
    "score_std",
    "topk_score_threshold",
    "feature_missing",
    "trainval_label_used",
    "test_label_used",
    "no_test_label_usage",
]


@dataclass(frozen=True)
class EdgeScoreDiagnostics:
    dataset: str
    method: str
    seed: int
    relation_id: int | str
    relation_name: str
    score_strategy: str
    edge_count: int
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    topk_score_threshold: float | None
    feature_missing: bool
    trainval_label_used: bool
    test_label_used: bool
    no_test_label_usage: bool

    def to_row(
        self,
        *,
        budget_strategy: str = "",
        edge_score_strategy: str | None = None,
        relation_pair_name: str = "",
    ) -> dict[str, Any]:
        data = asdict(self)
        data["budget_strategy"] = budget_strategy
        data["edge_score_strategy"] = edge_score_strategy or self.score_strategy
        data["relation_pair_name"] = relation_pair_name
        return {field: data.get(field, "") for field in EDGE_SCORE_DIAGNOSTIC_FIELDS}


def _as_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _degree_score(src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    if src_ids.size == 0:
        return np.empty(0, dtype=np.float64)
    max_node = int(max(int(src_ids.max(initial=0)), int(dst_ids.max(initial=0)))) + 1
    src_degree = np.bincount(src_ids, minlength=max_node).astype(np.float64)
    dst_degree = np.bincount(dst_ids, minlength=max_node).astype(np.float64)
    return 1.0 / np.sqrt(np.maximum(src_degree[src_ids] * dst_degree[dst_ids], 1.0))


def _relation_pair(relation_name: str) -> str:
    if relation_name in {"AP", "PA"}:
        return "AP_PA"
    if relation_name in {"PT", "TP"}:
        return "PT_TP"
    if relation_name in {"PV", "VP"}:
        return "PV_VP"
    return relation_name


class PathAwareEdgeScorer:
    def score_edges(
        self,
        *,
        dataset: str,
        relation_id: int | str,
        relation_name: str,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        graph_context: Mapping[str, Any],
        train_idx: np.ndarray | None,
        val_idx: np.ndarray | None,
        labels: np.ndarray | None,
        features_by_type: Mapping[int | str, np.ndarray] | None,
        seed: int,
    ) -> tuple[np.ndarray, EdgeScoreDiagnostics]:
        src = _as_ids(src_ids)
        dst = _as_ids(dst_ids)
        if src.shape != dst.shape:
            raise ValueError("src_ids and dst_ids must have the same shape")
        node_type = np.asarray(graph_context.get("node_type", []), dtype=np.int64)
        train = set(int(v) for v in np.asarray([] if train_idx is None else train_idx, dtype=np.int64).reshape(-1).tolist())
        val = set(int(v) for v in np.asarray([] if val_idx is None else val_idx, dtype=np.int64).reshape(-1).tolist())
        trainval = train | val
        scores = _degree_score(src, dst)
        if scores.size:
            if node_type.size:
                target_endpoint = ((node_type[src] == 0) | (node_type[dst] == 0)).astype(np.float64)
            else:
                target_endpoint = np.zeros(src.size, dtype=np.float64)
            trainval_endpoint = np.asarray([(int(s) in trainval) or (int(d) in trainval) for s, d in zip(src.tolist(), dst.tolist())], dtype=np.float64)
            scores = scores + 1.0 * target_endpoint + 0.8 * trainval_endpoint
            if str(dataset).upper() == "DBLP" and str(relation_name) in {"AP", "PA", "PT", "TP", "PV", "VP"}:
                scores = scores + self._dblp_relation_bonus(str(relation_name), src, dst, node_type, trainval)
            scores = scores + self._feature_endpoint_bonus(str(relation_name), src, dst, features_by_type, node_type)
            scores = scores + self._stable_tiebreaker(src, dst, int(seed))
        feature_missing = features_by_type is None or not bool(features_by_type)
        trainval_label_used = labels is not None and bool(trainval) and str(relation_name) in {"AP", "PA", "PT", "TP", "PV", "VP"}
        diag = EdgeScoreDiagnostics(
            dataset=str(dataset).upper(),
            method="",
            seed=int(seed),
            relation_id=relation_id,
            relation_name=str(relation_name),
            score_strategy="path_aware",
            edge_count=int(src.size),
            score_min=float(np.min(scores)) if scores.size else 0.0,
            score_max=float(np.max(scores)) if scores.size else 0.0,
            score_mean=float(np.mean(scores)) if scores.size else 0.0,
            score_std=float(np.std(scores)) if scores.size else 0.0,
            topk_score_threshold=None,
            feature_missing=bool(feature_missing),
            trainval_label_used=bool(trainval_label_used),
            test_label_used=False,
            no_test_label_usage=True,
        )
        return scores.astype(np.float64, copy=False), diag

    @staticmethod
    def relation_pair_name(relation_name: str) -> str:
        return _relation_pair(str(relation_name))

    @staticmethod
    def _stable_tiebreaker(src: np.ndarray, dst: np.ndarray, seed: int) -> np.ndarray:
        mixed = (src.astype(np.uint64) * np.uint64(1000003)) ^ (dst.astype(np.uint64) * np.uint64(9176)) ^ np.uint64(seed)
        return (mixed % np.uint64(997)).astype(np.float64) * 1e-9

    @staticmethod
    def _dblp_relation_bonus(relation_name: str, src: np.ndarray, dst: np.ndarray, node_type: np.ndarray, trainval: set[int]) -> np.ndarray:
        if src.size == 0:
            return np.empty(0, dtype=np.float64)
        src_trainval = np.asarray([int(v) in trainval for v in src.tolist()], dtype=np.float64)
        dst_trainval = np.asarray([int(v) in trainval for v in dst.tolist()], dtype=np.float64)
        if relation_name in {"AP", "PA"}:
            return 0.8 * (src_trainval + dst_trainval)
        if relation_name in {"PT", "TP"}:
            return 0.5 * (src_trainval + dst_trainval)
        if relation_name in {"PV", "VP"}:
            return 0.3 * (src_trainval + dst_trainval)
        return np.zeros(src.size, dtype=np.float64)

    @staticmethod
    def _feature_endpoint_bonus(
        relation_name: str,
        src: np.ndarray,
        dst: np.ndarray,
        features_by_type: Mapping[int | str, np.ndarray] | None,
        node_type: np.ndarray,
    ) -> np.ndarray:
        if src.size == 0 or features_by_type is None or node_type.size == 0:
            return np.zeros(src.size, dtype=np.float64)
        # Lightweight endpoint importance: feature-bearing low-degree support edges get a small boost.
        present_types = {int(k) for k in features_by_type if str(k).lstrip("-").isdigit()}
        has_features = np.asarray([(int(node_type[s]) in present_types) and (int(node_type[d]) in present_types) for s, d in zip(src.tolist(), dst.tolist())], dtype=np.float64)
        if relation_name in {"AP", "PA"}:
            return 0.2 * has_features
        return 0.1 * has_features
