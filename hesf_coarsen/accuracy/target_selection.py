from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


@dataclass(frozen=True)
class TargetAnchorSelection:
    selected_nodes: np.ndarray
    diagnostics: dict


def support_coverage_sets(
    graph: HeteroGraph,
    *,
    target_node_type: int,
    target_nodes: np.ndarray | None = None,
) -> dict[int, set[tuple[int, int]]]:
    target_type = int(target_node_type)
    if target_nodes is None:
        target_nodes = nodes_of_type(graph, target_type)
    targets = set(int(node) for node in np.asarray(target_nodes, dtype=np.int64).reshape(-1).tolist())
    coverage: dict[int, set[tuple[int, int]]] = {node: set() for node in targets}
    for rel in graph.relations.values():
        if int(rel.src_type) == target_type:
            for src, dst in zip(rel.src.tolist(), rel.dst.tolist()):
                if int(src) in targets:
                    coverage[int(src)].add((int(rel.dst_type), int(dst)))
        if int(rel.dst_type) == target_type:
            for src, dst in zip(rel.src.tolist(), rel.dst.tolist()):
                if int(dst) in targets:
                    coverage[int(dst)].add((int(rel.src_type), int(src)))
    return coverage


def _target_degree_confidence(graph: HeteroGraph, target_nodes: np.ndarray) -> dict[int, float]:
    degree = {int(node): 0.0 for node in target_nodes.tolist()}
    target_set = set(degree)
    for rel in graph.relations.values():
        for src, dst, weight in zip(rel.src.tolist(), rel.dst.tolist(), rel.weight.tolist()):
            if int(src) in target_set:
                degree[int(src)] += float(weight)
            if int(dst) in target_set:
                degree[int(dst)] += float(weight)
    max_degree = max(degree.values(), default=0.0)
    return {node: float(value / max(max_degree, 1.0)) for node, value in degree.items()}


def _pseudo_labels(graph: HeteroGraph, target_nodes: np.ndarray) -> dict[int, int]:
    labels = graph.labels
    if labels is not None:
        raw = np.asarray(labels).reshape(-1)
        known = raw[target_nodes]
        positives = known[known >= 0]
        num_classes = int(positives.max(initial=2)) + 1
        return {
            int(node): int(raw[int(node)]) if raw[int(node)] >= 0 else int(int(node) % max(num_classes, 1))
            for node in target_nodes.tolist()
        }
    return {int(node): int(index % 3) for index, node in enumerate(target_nodes.tolist())}


def select_target_anchors(
    graph: HeteroGraph,
    *,
    target_node_type: int,
    train_nodes: np.ndarray,
    budget: int,
    seed: int = 12345,
    weights: Mapping[str, float] | None = None,
) -> TargetAnchorSelection:
    rng = np.random.default_rng(int(seed))
    target_nodes = nodes_of_type(graph, int(target_node_type))
    target_set = set(int(node) for node in target_nodes.tolist())
    mandatory = [int(node) for node in np.asarray(train_nodes, dtype=np.int64).reshape(-1).tolist() if int(node) in target_set]
    mandatory = list(dict.fromkeys(mandatory))
    effective_budget = max(int(budget), len(mandatory))
    if effective_budget >= len(target_nodes):
        selected = list(dict.fromkeys(mandatory + [int(node) for node in target_nodes.tolist()]))
        return TargetAnchorSelection(
            selected_nodes=np.asarray(selected, dtype=np.int64),
            diagnostics={
                "budget_requested": int(budget),
                "budget_effective": int(len(selected)),
                "mandatory_train_count": int(len(mandatory)),
                "candidate_pool_count": 0,
                "score_terms": {},
            },
        )
    if effective_budget == len(mandatory):
        return TargetAnchorSelection(
            selected_nodes=np.asarray(mandatory, dtype=np.int64),
            diagnostics={
                "budget_requested": int(budget),
                "budget_effective": int(effective_budget),
                "mandatory_train_count": int(len(mandatory)),
                "candidate_pool_count": int(len(target_nodes) - len(mandatory)),
                "score_terms": {},
            },
        )

    w = {
        "confidence": 0.35,
        "margin": 0.10,
        "coverage": 0.25,
        "diversity": 0.20,
        "balance": 0.10,
    }
    if weights:
        w.update({str(key): float(value) for key, value in weights.items()})
    coverage = support_coverage_sets(graph, target_node_type=int(target_node_type), target_nodes=target_nodes)
    confidence = _target_degree_confidence(graph, target_nodes)
    pseudo = _pseudo_labels(graph, target_nodes)
    selected = list(mandatory)
    selected_set = set(selected)
    covered = set().union(*(coverage.get(node, set()) for node in selected)) if selected else set()
    selected_class_counts: dict[int, int] = {}
    for node in selected:
        selected_class_counts[pseudo[node]] = selected_class_counts.get(pseudo[node], 0) + 1
    score_trace: dict[str, float] = {}

    pool = [int(node) for node in target_nodes.tolist() if int(node) not in selected_set]
    max_cover = max((len(coverage.get(node, set())) for node in pool), default=1)
    max_node_gap = max(len(target_nodes) - 1, 1)
    while len(selected) < effective_budget and pool:
        best_node = None
        best_tuple: tuple[float, float] | None = None
        for node in pool:
            cov_gain = len(coverage.get(node, set()) - covered) / max(max_cover, 1)
            if selected:
                diversity = min(abs(node - other) for other in selected) / max_node_gap
            else:
                diversity = float(rng.random() * 1.0e-6)
            conf = confidence.get(node, 0.0)
            margin = 1.0 - abs(conf - 0.5) * 2.0
            cls = pseudo[node]
            balance = 1.0 / (1.0 + selected_class_counts.get(cls, 0))
            score = (
                w["confidence"] * conf
                + w["margin"] * margin
                + w["coverage"] * cov_gain
                + w["diversity"] * diversity
                + w["balance"] * balance
            )
            candidate_tuple = (float(score), float(-node))
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_node = node
                score_trace = {
                    "teacher_confidence": conf,
                    "teacher_margin_or_entropy": margin,
                    "coverage_gain": cov_gain,
                    "diversity_gain": diversity,
                    "class_balance_gain": balance,
                    "score": float(score),
                }
        assert best_node is not None
        selected.append(best_node)
        selected_set.add(best_node)
        covered.update(coverage.get(best_node, set()))
        selected_class_counts[pseudo[best_node]] = selected_class_counts.get(pseudo[best_node], 0) + 1
        pool = [node for node in pool if node != best_node]

    return TargetAnchorSelection(
        selected_nodes=np.asarray(selected, dtype=np.int64),
        diagnostics={
            "budget_requested": int(budget),
            "budget_effective": int(effective_budget),
            "mandatory_train_count": int(len(mandatory)),
            "candidate_pool_count": int(len(target_nodes) - len(mandatory)),
            "selected_count": int(len(selected)),
            "selected_class_counts": {str(k): int(v) for k, v in sorted(selected_class_counts.items())},
            "score_terms": score_trace,
        },
    )
