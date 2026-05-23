from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


@dataclass(frozen=True)
class SupportUnit:
    source: str
    unit_id: str
    member_nodes: tuple[int, ...]
    member_count: int
    support_type_distribution: dict[str, int]
    relation_profile: dict[str, float]
    edge_mass: float
    target_anchor_coverage: float
    class_footprint: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnitStructureIndex:
    edge_weight_by_key: dict[tuple[int, int], float]
    node_edge_keys: dict[int, list[tuple[int, int]]]
    node_target_anchors: dict[int, set[int]]
    target_count: int
    labels: np.ndarray | None
    splits: Mapping[str, np.ndarray]


def _json_counter(values: Iterable[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = int(out.get(key, 0) + 1)
    return out


def _weights(rel: Any) -> np.ndarray:
    if rel.weight is None:
        return np.ones(int(rel.num_edges), dtype=np.float32)
    return np.asarray(rel.weight, dtype=np.float32).reshape(-1)


def build_unit_structure_index(
    graph: HeteroGraph,
    *,
    target_type: int,
    labels: np.ndarray | None = None,
    splits: Mapping[str, np.ndarray] | None = None,
) -> UnitStructureIndex:
    edge_weight_by_key: dict[tuple[int, int], float] = {}
    node_edge_keys: dict[int, list[tuple[int, int]]] = {}
    node_target_anchors: dict[int, set[int]] = {}
    for relation_id, rel in sorted(graph.relations.items()):
        src = np.asarray(rel.src, dtype=np.int64)
        dst = np.asarray(rel.dst, dtype=np.int64)
        weight = _weights(rel)
        for edge_idx, (left, right) in enumerate(zip(src.tolist(), dst.tolist())):
            key = (int(relation_id), int(edge_idx))
            edge_weight_by_key[key] = float(weight[int(edge_idx)])
            node_edge_keys.setdefault(int(left), []).append(key)
            node_edge_keys.setdefault(int(right), []).append(key)
            if int(rel.src_type) == int(target_type):
                node_target_anchors.setdefault(int(right), set()).add(int(left))
            elif int(rel.dst_type) == int(target_type):
                node_target_anchors.setdefault(int(left), set()).add(int(right))
    return UnitStructureIndex(
        edge_weight_by_key=edge_weight_by_key,
        node_edge_keys=node_edge_keys,
        node_target_anchors=node_target_anchors,
        target_count=int(np.sum(np.asarray(graph.node_type) == int(target_type))),
        labels=None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1),
        splits=dict(splits or {}),
    )


def unit_structure_from_index(
    graph: HeteroGraph,
    member_nodes: Iterable[int],
    *,
    source: str,
    unit_id: str | int,
    index: UnitStructureIndex,
    metadata: Mapping[str, Any] | None = None,
) -> SupportUnit:
    members = tuple(sorted({int(node) for node in member_nodes if 0 <= int(node) < int(graph.num_nodes)}))
    support_type_distribution = _json_counter(int(graph.node_type[node]) for node in members)
    edge_keys: set[tuple[int, int]] = set()
    target_anchors: set[int] = set()
    for node in members:
        edge_keys.update(index.node_edge_keys.get(int(node), []))
        target_anchors.update(index.node_target_anchors.get(int(node), set()))
    relation_profile: dict[str, float] = {}
    for relation_id, edge_idx in edge_keys:
        key = str(int(relation_id))
        relation_profile[key] = float(relation_profile.get(key, 0.0) + index.edge_weight_by_key.get((int(relation_id), int(edge_idx)), 0.0))
    class_values: list[int] = []
    if index.labels is not None:
        for node in sorted(target_anchors):
            if int(node) < len(index.labels) and int(index.labels[int(node)]) >= 0:
                class_values.append(int(index.labels[int(node)]))
        for node in members:
            if int(node) < len(index.labels) and int(index.labels[int(node)]) >= 0:
                class_values.append(int(index.labels[int(node)]))
    split_counts: dict[str, int] = {}
    for key, nodes in index.splits.items():
        node_set = {int(node) for node in np.asarray(nodes, dtype=np.int64).reshape(-1).tolist()}
        split_counts[f"{key}_target_anchor_count"] = int(len(target_anchors & node_set))
    meta = dict(metadata or {})
    meta.update(
        {
            "target_anchor_count": int(len(target_anchors)),
            "relation_channel_count": int(len(relation_profile)),
            "unit_structure_available": bool(len(relation_profile) > 0 or len(target_anchors) > 0),
            **split_counts,
        }
    )
    return SupportUnit(
        source=str(source),
        unit_id=str(unit_id),
        member_nodes=members,
        member_count=int(len(members)),
        support_type_distribution=support_type_distribution,
        relation_profile=relation_profile,
        edge_mass=float(sum(relation_profile.values())),
        target_anchor_coverage=float(len(target_anchors) / max(1, int(index.target_count))),
        class_footprint=_json_counter(class_values),
        metadata=meta,
    )


def unit_structure_for_members(
    graph: HeteroGraph,
    member_nodes: Iterable[int],
    *,
    source: str,
    unit_id: str | int,
    target_type: int,
    labels: np.ndarray | None = None,
    splits: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SupportUnit:
    index = build_unit_structure_index(graph, target_type=int(target_type), labels=labels, splits=splits)
    return unit_structure_from_index(
        graph,
        member_nodes,
        source=source,
        unit_id=unit_id,
        index=index,
        metadata=metadata,
    )


def with_metadata(unit: SupportUnit, **metadata: Any) -> SupportUnit:
    meta = dict(unit.metadata)
    meta.update(metadata)
    return replace(unit, metadata=meta)
