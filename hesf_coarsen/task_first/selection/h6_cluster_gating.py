from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj


@dataclass(frozen=True)
class H6ClusterUnit:
    cluster_id: int
    cluster_type: int
    member_nodes: np.ndarray

    @property
    def member_count(self) -> int:
        return int(len(self.member_nodes))

    def to_row(self) -> dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "cluster_type": int(self.cluster_type),
            "member_count": int(self.member_count),
            "member_nodes": ",".join(str(int(node)) for node in self.member_nodes.tolist()),
        }


@dataclass(frozen=True)
class H6ClusterSelection:
    selected_cluster_ids: list[int]
    member_count_selected: int
    member_ratio_selected: float
    positive_gain_block_count: int
    neutral_fill_block_count: int
    negative_fill_block_count: int
    proxy_fill_block_count: int
    underfill_ratio: float
    overfill_ratio: float
    budget_penalty_value: float
    validation_drop_from_fill: float

    def to_row(self) -> dict[str, Any]:
        return {
            "selected_cluster_ids": ",".join(str(int(value)) for value in self.selected_cluster_ids),
            "h6_cluster_count_selected": int(len(self.selected_cluster_ids)),
            "h6_cluster_member_count_selected": int(self.member_count_selected),
            "h6_cluster_member_ratio_selected": float(self.member_ratio_selected),
            "positive_gain_block_count": int(self.positive_gain_block_count),
            "neutral_fill_block_count": int(self.neutral_fill_block_count),
            "negative_fill_block_count": int(self.negative_fill_block_count),
            "proxy_fill_block_count": int(self.proxy_fill_block_count),
            "underfill_ratio": float(self.underfill_ratio),
            "overfill_ratio": float(self.overfill_ratio),
            "budget_penalty_value": float(self.budget_penalty_value),
            "validation_drop_from_fill": float(self.validation_drop_from_fill),
        }


def extract_h6_cluster_units(graph: HeteroGraph, h6_assignment: np.ndarray, target_type: int) -> list[H6ClusterUnit]:
    assignment = np.asarray(h6_assignment, dtype=np.int64)
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    units: list[H6ClusterUnit] = []
    for cluster_id in sorted({int(assignment[int(node)]) for node in support_nodes.tolist()}):
        members = np.asarray([int(node) for node in support_nodes.tolist() if int(assignment[int(node)]) == int(cluster_id)], dtype=np.int64)
        if len(members) == 0:
            continue
        cluster_type = int(graph.node_type[int(members[0])])
        units.append(H6ClusterUnit(cluster_id=int(cluster_id), cluster_type=cluster_type, member_nodes=members))
    return units


def _score(unit: H6ClusterUnit, validation_scores: Mapping[int, float]) -> float:
    try:
        return float(validation_scores.get(int(unit.cluster_id), 0.0))
    except (TypeError, ValueError):
        return 0.0


def select_h6_clusters_by_budget(
    units: Sequence[H6ClusterUnit],
    *,
    support_count: int,
    requested_support_ratio: float,
    validation_scores: Mapping[int, float] | None = None,
    min_gain: float = 1.0e-4,
    neutral_fill_max_drop: float = 1.0e-4,
    negative_fill_max_drop: float = 5.0e-4,
    budget_penalty_lambda: float = 0.05,
    underfill_penalty_lambda: float = 0.10,
) -> H6ClusterSelection:
    scores = validation_scores or {}
    target_count = max(0, int(np.ceil(max(0, int(support_count)) * float(requested_support_ratio) - 1.0e-12)))
    selected: list[H6ClusterUnit] = []
    selected_ids: set[int] = set()
    positive = neutral = negative = 0

    def current_count(extra: H6ClusterUnit | None = None) -> int:
        total = sum(unit.member_count for unit in selected)
        if extra is not None:
            total += int(extra.member_count)
        return int(total)

    def penalty(extra: H6ClusterUnit | None = None) -> float:
        ratio = current_count(extra) / max(int(support_count), 1)
        return float(budget_penalty_lambda) * abs(float(ratio) - float(requested_support_ratio))

    ordered = sorted(units, key=lambda unit: (-_score(unit, scores), -unit.member_count, unit.cluster_id))
    for unit in ordered:
        if current_count() >= target_count:
            break
        gain = _score(unit, scores)
        if gain < float(min_gain):
            continue
        selected.append(unit)
        selected_ids.add(int(unit.cluster_id))
        positive += 1

    for unit in ordered:
        if current_count() >= target_count:
            break
        if int(unit.cluster_id) in selected_ids:
            continue
        gain = _score(unit, scores)
        if gain >= -float(neutral_fill_max_drop):
            selected.append(unit)
            selected_ids.add(int(unit.cluster_id))
            neutral += 1

    for unit in ordered:
        if current_count() >= target_count:
            break
        if int(unit.cluster_id) in selected_ids:
            continue
        gain = _score(unit, scores)
        if gain >= -float(negative_fill_max_drop):
            selected.append(unit)
            selected_ids.add(int(unit.cluster_id))
            negative += 1

    count = current_count()
    ratio = count / max(int(support_count), 1)
    underfill_ratio = max(0.0, float(requested_support_ratio) - ratio)
    overfill_ratio = max(0.0, ratio - float(requested_support_ratio))
    validation_drop = sum(max(0.0, -_score(unit, scores)) for unit in selected)
    budget_penalty = penalty(None) + float(underfill_penalty_lambda) * underfill_ratio
    return H6ClusterSelection(
        selected_cluster_ids=[int(unit.cluster_id) for unit in selected],
        member_count_selected=int(count),
        member_ratio_selected=float(ratio),
        positive_gain_block_count=int(positive),
        neutral_fill_block_count=int(neutral),
        negative_fill_block_count=int(negative),
        proxy_fill_block_count=0,
        underfill_ratio=float(underfill_ratio),
        overfill_ratio=float(overfill_ratio),
        budget_penalty_value=float(budget_penalty),
        validation_drop_from_fill=float(validation_drop),
    )


def _induced_coarse_graph(
    coarse: HeteroGraph,
    original_assignment: np.ndarray,
    keep_coarse_nodes: np.ndarray,
) -> tuple[HeteroGraph, np.ndarray]:
    kept = np.asarray(sorted(int(node) for node in np.asarray(keep_coarse_nodes, dtype=np.int64).reshape(-1)), dtype=np.int64)
    local_of = {int(node): idx for idx, node in enumerate(kept.tolist())}
    relations: dict[int, RelationAdj] = {}
    for relation_id, rel in coarse.relations.items():
        mask = np.asarray([int(src) in local_of and int(dst) in local_of for src, dst in zip(rel.src, rel.dst)], dtype=bool)
        src = np.asarray([local_of[int(node)] for node in np.asarray(rel.src)[mask]], dtype=np.int64)
        dst = np.asarray([local_of[int(node)] for node in np.asarray(rel.dst)[mask]], dtype=np.int64)
        weight = np.asarray(rel.weight, dtype=np.float32)[mask] if rel.weight is not None else None
        relations[int(relation_id)] = RelationAdj(
            src=src,
            dst=dst,
            weight=weight,
            src_type=int(rel.src_type),
            dst_type=int(rel.dst_type),
            relation_id=int(relation_id),
        )
    features: dict[int, np.ndarray] = {}
    for type_id, feature in (coarse.features or {}).items():
        type_nodes = np.flatnonzero(coarse.node_type == int(type_id)).astype(np.int64)
        local_lookup = {int(node): idx for idx, node in enumerate(type_nodes.tolist())}
        type_kept = [int(node) for node in kept.tolist() if int(coarse.node_type[int(node)]) == int(type_id)]
        indices = [local_lookup[int(node)] for node in type_kept]
        features[int(type_id)] = np.asarray(feature, dtype=np.float32)[indices].astype(np.float32, copy=False)
    graph = HeteroGraph(
        num_nodes=int(len(kept)),
        node_type=coarse.node_type[kept].astype(np.int32, copy=False),
        relations=relations,
        relation_specs=dict(coarse.relation_specs),
        features=features,
        labels=None if coarse.labels is None else np.asarray(coarse.labels)[kept],
        partitions=None if coarse.partitions is None else np.asarray(coarse.partitions)[kept],
    )
    original_assignment = np.asarray(original_assignment, dtype=np.int64)
    mapped = np.zeros_like(original_assignment, dtype=np.int64)
    for original_node, supernode in enumerate(original_assignment.tolist()):
        if int(supernode) in local_of:
            mapped[int(original_node)] = int(local_of[int(supernode)])
    return graph, mapped


def build_gated_h6_graph(
    *,
    original: HeteroGraph,
    h6_coarse: HeteroGraph,
    h6_assignment: np.ndarray,
    selected_cluster_ids: Sequence[int],
    target_type: int,
) -> tuple[HeteroGraph, np.ndarray, np.ndarray]:
    assignment = np.asarray(h6_assignment, dtype=np.int64)
    target_nodes = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    target_clusters = {int(assignment[int(node)]) for node in target_nodes.tolist()}
    keep = np.asarray(sorted(target_clusters | {int(value) for value in selected_cluster_ids}), dtype=np.int64)
    gated, mapped = _induced_coarse_graph(h6_coarse, assignment, keep)
    return gated, mapped, keep


def h6_fill_support_nodes(
    *,
    graph: HeteroGraph,
    h6_assignment: np.ndarray,
    target_type: int,
    selected_support_nodes: np.ndarray,
    requested_support_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = [int(node) for node in np.asarray(selected_support_nodes, dtype=np.int64).reshape(-1)]
    selected_set = set(selected)
    assignment = np.asarray(h6_assignment, dtype=np.int64)
    units = extract_h6_cluster_units(graph, assignment, int(target_type))
    selected_clusters = {int(assignment[int(node)]) for node in selected if 0 <= int(node) < len(assignment)}
    fill_count = 0
    filled_cluster_ids: set[int] = set()
    ordered_units = sorted(units, key=lambda item: (int(item.cluster_id) in selected_clusters, -item.member_count, item.cluster_id))
    for unit in ordered_units:
        if len(selected) >= int(requested_support_count):
            break
        for node in unit.member_nodes.tolist():
            if int(node) in selected_set:
                continue
            selected.append(int(node))
            selected_set.add(int(node))
            fill_count += 1
            filled_cluster_ids.add(int(unit.cluster_id))
            break
    for unit in ordered_units:
        if len(selected) >= int(requested_support_count):
            break
        for node in unit.member_nodes.tolist():
            if len(selected) >= int(requested_support_count):
                break
            if int(node) in selected_set:
                continue
            selected.append(int(node))
            selected_set.add(int(node))
            fill_count += 1
            filled_cluster_ids.add(int(unit.cluster_id))
        if len(selected) >= int(requested_support_count):
            break
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    for node in support_nodes.tolist():
        if len(selected) >= int(requested_support_count):
            break
        if int(node) in selected_set:
            continue
        selected.append(int(node))
        selected_set.add(int(node))
    filled = np.asarray(sorted(selected[: int(requested_support_count)]), dtype=np.int64)
    overlap = len(set(np.asarray(selected_support_nodes, dtype=np.int64).reshape(-1).tolist()) & set(filled.tolist()))
    diag = {
        "h6_fill_block_count": int(len(filled_cluster_ids)),
        "h6_fill_support_count": int(fill_count),
        "h6_fill_overlap_with_validation_blocks": int(overlap),
        "h6_fill_budget_fraction": float(fill_count / max(int(requested_support_count), 1)),
        "h6_fill_validation_drop": 0.0,
    }
    return filled, diag
