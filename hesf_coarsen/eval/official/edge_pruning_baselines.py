from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.path_aware_edge_scorer import PathAwareEdgeScorer
from hesf_coarsen.eval.official.relation_budget_allocator import RelationBudgetAllocator, RelationStats
from hesf_coarsen.eval.official.relation_mapping_audit import relation_pair_name
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, validate_schema


@dataclass(frozen=True)
class PrunedGraphResult:
    graph: HeteroGraph
    candidate_edge_counts: dict[int, int]
    retained_edge_counts: dict[int, int]
    requested_relation_budgets: dict[int, int]
    min_edges_constraint_active: dict[int, bool]
    edge_score_diagnostics: list[dict[str, Any]]


def _edge_count(graph: HeteroGraph) -> int:
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _relation_stats(graph: HeteroGraph, *, min_edges_per_relation: int) -> list[RelationStats]:
    stats: list[RelationStats] = []
    for relation_id, rel in sorted(graph.relations.items()):
        spec = graph.relation_specs[int(relation_id)]
        stats.append(
            RelationStats(
                relation_id=int(relation_id),
                relation_name=str(spec.name),
                relation_pair_name=relation_pair_name(str(spec.name)),
                src_type=str(spec.src_type),
                dst_type=str(spec.dst_type),
                full_edge_count=int(rel.num_edges),
                candidate_edge_count=int(rel.num_edges),
                min_edges=min(int(min_edges_per_relation), int(rel.num_edges)),
            )
        )
    return stats


def _degree_scores(rel: RelationAdj, graph: HeteroGraph) -> np.ndarray:
    if rel.num_edges == 0:
        return np.empty(0, dtype=np.float64)
    src_degree = np.bincount(rel.src, minlength=graph.num_nodes).astype(np.float64)
    dst_degree = np.bincount(rel.dst, minlength=graph.num_nodes).astype(np.float64)
    return 1.0 / np.sqrt(np.maximum(src_degree[rel.src] * dst_degree[rel.dst], 1.0))


def _current_scores(rel: RelationAdj, graph: HeteroGraph, target_type: int = 0) -> np.ndarray:
    if rel.num_edges == 0:
        return np.empty(0, dtype=np.float64)
    src_degree = np.bincount(rel.src, minlength=graph.num_nodes).astype(np.float64)
    dst_degree = np.bincount(rel.dst, minlength=graph.num_nodes).astype(np.float64)
    target_bonus = ((graph.node_type[rel.src] == int(target_type)) | (graph.node_type[rel.dst] == int(target_type))).astype(np.float64)
    return target_bonus + 0.25 / np.maximum(src_degree[rel.src], 1.0) + 0.25 / np.maximum(dst_degree[rel.dst], 1.0)


def _random_scores(rel: RelationAdj, *, seed: int, relation_id: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) * 1009 + int(relation_id))
    return rng.random(int(rel.num_edges)).astype(np.float64)


def _required_edge_indices(graph: HeteroGraph) -> dict[int, set[int]]:
    required: dict[int, set[int]] = {int(rid): set() for rid in graph.relations}
    for type_id in sorted(set(int(v) for v in graph.node_type.tolist())):
        type_nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if type_nodes.size == 0:
            continue
        max_node = int(type_nodes[-1])
        for relation_id, rel in sorted(graph.relations.items()):
            hits = np.flatnonzero((rel.src == max_node) | (rel.dst == max_node)).astype(np.int64)
            if hits.size:
                required[int(relation_id)].add(int(hits[0]))
                break
    return required


def prune_relationwise(
    *,
    graph: HeteroGraph,
    dataset: str,
    method: str,
    total_edge_budget: int,
    budget_strategy: str,
    edge_score_strategy: str,
    seed: int,
    train_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    labels: np.ndarray | None,
    features_by_type: Mapping[int | str, np.ndarray] | None,
    min_edges_per_relation: int = 1,
    relation_pair_weights: dict[str, float] | None = None,
    target_type: int = 0,
) -> PrunedGraphResult:
    validate_schema(graph)
    allocator = RelationBudgetAllocator()
    allocations = allocator.allocate(
        relation_stats=_relation_stats(graph, min_edges_per_relation=int(min_edges_per_relation)),
        total_edge_budget=int(total_edge_budget),
        strategy=str(budget_strategy),
        relation_pair_weights=relation_pair_weights,
        min_edges_per_relation=int(min_edges_per_relation),
        seed=int(seed),
    )
    budget_by_relation = {int(row.relation_id): int(row.actual_edges) for row in allocations}
    min_active = {int(row.relation_id): bool(row.min_edges_constraint_active) for row in allocations}
    scorer = PathAwareEdgeScorer()
    required = _required_edge_indices(graph)
    diagnostics: list[dict[str, Any]] = []
    relations: dict[int, RelationAdj] = {}
    for relation_id, rel in sorted(graph.relations.items()):
        spec = graph.relation_specs[int(relation_id)]
        strategy = str(edge_score_strategy)
        if strategy == "path_aware":
            scores, diag = scorer.score_edges(
                dataset=dataset,
                relation_id=int(relation_id),
                relation_name=str(spec.name),
                src_ids=rel.src,
                dst_ids=rel.dst,
                graph_context={"node_type": graph.node_type},
                train_idx=train_idx,
                val_idx=val_idx,
                labels=labels,
                features_by_type=features_by_type,
                seed=int(seed),
            )
            diagnostics.append(
                diag.to_row(
                    budget_strategy=str(budget_strategy),
                    edge_score_strategy=str(edge_score_strategy),
                    relation_pair_name=relation_pair_name(str(spec.name)),
                )
            )
        elif strategy == "random":
            scores = _random_scores(rel, seed=int(seed), relation_id=int(relation_id))
            diagnostics.append(_score_diag(dataset, method, seed, relation_id, str(spec.name), budget_strategy, strategy, scores))
        elif strategy == "degree":
            scores = _degree_scores(rel, graph)
            diagnostics.append(_score_diag(dataset, method, seed, relation_id, str(spec.name), budget_strategy, strategy, scores))
        elif strategy == "current_heuristic":
            scores = _current_scores(rel, graph, int(target_type))
            diagnostics.append(_score_diag(dataset, method, seed, relation_id, str(spec.name), budget_strategy, strategy, scores))
        else:
            raise ValueError(f"unsupported edge score strategy: {edge_score_strategy}")
        keep = _select_edges(rel, scores, budget_by_relation.get(int(relation_id), rel.num_edges), required.get(int(relation_id), set()))
        relations[int(relation_id)] = RelationAdj(rel.src[keep].copy(), rel.dst[keep].copy(), rel.weight[keep].copy(), rel.src_type, rel.dst_type, int(relation_id))
    pruned = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=relations,
        relation_specs=graph.relation_specs,
        features=None if graph.features is None else {int(k): v.copy() for k, v in graph.features.items()},
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
    )
    validate_schema(pruned)
    return PrunedGraphResult(
        graph=pruned,
        candidate_edge_counts={int(rid): int(rel.num_edges) for rid, rel in graph.relations.items()},
        retained_edge_counts={int(rid): int(rel.num_edges) for rid, rel in pruned.relations.items()},
        requested_relation_budgets=budget_by_relation,
        min_edges_constraint_active=min_active,
        edge_score_diagnostics=diagnostics,
    )


def semantic_storage_ratio(graph: HeteroGraph, reference_graph: HeteroGraph) -> float:
    return float((int(graph.num_nodes) + _edge_count(graph)) / max(int(reference_graph.num_nodes) + _edge_count(reference_graph), 1))


def edge_budget_for_storage(reference_graph: HeteroGraph, candidate_graph: HeteroGraph, storage_budget: float) -> int:
    ref_total = int(reference_graph.num_nodes) + _edge_count(reference_graph)
    return max(0, min(_edge_count(candidate_graph), int(np.floor(float(storage_budget) * float(ref_total) - float(candidate_graph.num_nodes)))))


def _select_edges(rel: RelationAdj, scores: np.ndarray, budget: int, required: set[int]) -> np.ndarray:
    if rel.num_edges <= int(budget):
        return np.arange(rel.num_edges, dtype=np.int64)
    budget = max(0, int(budget))
    selected: list[int] = [idx for idx in sorted(required) if 0 <= int(idx) < rel.num_edges]
    remaining = max(0, budget - len(selected))
    order = np.lexsort((rel.dst, rel.src, -np.asarray(scores, dtype=np.float64)))
    selected_set = set(selected)
    for idx in order.tolist():
        if remaining <= 0:
            break
        if int(idx) in selected_set:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        remaining -= 1
    return np.asarray(sorted(selected_set), dtype=np.int64)


def _score_diag(
    dataset: str,
    method: str,
    seed: int,
    relation_id: int,
    relation_name: str,
    budget_strategy: str,
    edge_score_strategy: str,
    scores: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "dataset": str(dataset).upper(),
        "seed": int(seed),
        "method": str(method),
        "budget_strategy": str(budget_strategy),
        "edge_score_strategy": str(edge_score_strategy),
        "relation_id": int(relation_id),
        "relation_name": str(relation_name),
        "relation_pair_name": relation_pair_name(str(relation_name)),
        "edge_count": int(scores.size),
        "score_min": float(np.min(scores)) if scores.size else 0.0,
        "score_max": float(np.max(scores)) if scores.size else 0.0,
        "score_mean": float(np.mean(scores)) if scores.size else 0.0,
        "score_std": float(np.std(scores)) if scores.size else 0.0,
        "topk_score_threshold": "",
        "feature_missing": "",
        "trainval_label_used": False,
        "test_label_used": False,
        "no_test_label_usage": True,
    }
