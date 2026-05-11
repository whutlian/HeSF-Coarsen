from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelationSpec:
    relation_id: int
    name: str
    src_type: int
    dst_type: int

    def to_json(self) -> dict[str, Any]:
        return {
            "relation_id": int(self.relation_id),
            "name": self.name,
            "src_type": int(self.src_type),
            "dst_type": int(self.dst_type),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RelationSpec":
        return cls(
            relation_id=int(data["relation_id"]),
            name=str(data["name"]),
            src_type=int(data["src_type"]),
            dst_type=int(data["dst_type"]),
        )


@dataclass
class RelationAdj:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray | None
    src_type: int
    dst_type: int
    relation_id: int

    def __post_init__(self) -> None:
        if not (isinstance(self.src, np.memmap) and self.src.dtype == np.int64):
            self.src = np.asarray(self.src, dtype=np.int64)
        if not (isinstance(self.dst, np.memmap) and self.dst.dtype == np.int64):
            self.dst = np.asarray(self.dst, dtype=np.int64)
        if self.weight is None:
            self.weight = np.ones(len(self.src), dtype=np.float32)
        else:
            if not (isinstance(self.weight, np.memmap) and self.weight.dtype == np.float32):
                self.weight = np.asarray(self.weight, dtype=np.float32)
        self.src_type = int(self.src_type)
        self.dst_type = int(self.dst_type)
        self.relation_id = int(self.relation_id)

    @property
    def num_edges(self) -> int:
        return int(len(self.src))


@dataclass
class HeteroGraph:
    num_nodes: int
    node_type: np.ndarray
    relations: dict[int, RelationAdj]
    relation_specs: dict[int, RelationSpec] = field(default_factory=dict)
    features: dict[int, np.ndarray] | None = None
    labels: np.ndarray | None = None
    partitions: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.num_nodes = int(self.num_nodes)
        if not (isinstance(self.node_type, np.memmap) and self.node_type.dtype == np.int32):
            self.node_type = np.asarray(self.node_type, dtype=np.int32)
        self.relations = {int(k): v for k, v in self.relations.items()}
        if not self.relation_specs:
            self.relation_specs = {
                relation_id: RelationSpec(
                    relation_id=relation_id,
                    name=f"relation_{relation_id}",
                    src_type=rel.src_type,
                    dst_type=rel.dst_type,
                )
                for relation_id, rel in self.relations.items()
            }
        else:
            self.relation_specs = {int(k): v for k, v in self.relation_specs.items()}
        if self.features is not None:
            self.features = {
                int(type_id): (
                    value
                    if isinstance(value, np.memmap) and value.dtype == np.float32
                    else np.asarray(value, dtype=np.float32)
                )
                for type_id, value in self.features.items()
            }
        if self.labels is not None:
            if not isinstance(self.labels, np.memmap):
                self.labels = np.asarray(self.labels)
        if self.partitions is not None:
            if not (
                isinstance(self.partitions, np.memmap)
                and self.partitions.dtype == np.int32
            ):
                self.partitions = np.asarray(self.partitions, dtype=np.int32)


def nodes_of_type(graph: HeteroGraph, type_id: int) -> np.ndarray:
    return np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)


def type_local_index(graph: HeteroGraph, type_id: int) -> dict[int, int]:
    nodes = nodes_of_type(graph, type_id)
    return {int(node): pos for pos, node in enumerate(nodes)}


def validate_schema(graph: HeteroGraph) -> None:
    if graph.node_type.shape != (graph.num_nodes,):
        raise ValueError("node_type must have shape [num_nodes]")
    if graph.num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if np.any(graph.node_type < 0):
        raise ValueError("every node must have exactly one non-negative node type")

    if len(set(graph.relations)) != len(graph.relations):
        raise ValueError("relation IDs must be unique")
    if set(graph.relations) != set(graph.relation_specs):
        raise ValueError("relation adjacency IDs must match relation specs")

    for relation_id, rel in graph.relations.items():
        spec = graph.relation_specs[relation_id]
        if relation_id != rel.relation_id or relation_id != spec.relation_id:
            raise ValueError("relation ID mismatch")
        if rel.src_type != spec.src_type or rel.dst_type != spec.dst_type:
            raise ValueError(f"relation {relation_id} schema mismatch")
        if rel.src.shape != rel.dst.shape or rel.src.shape != rel.weight.shape:
            raise ValueError(f"relation {relation_id} arrays must have equal length")
        if rel.num_edges == 0:
            continue
        if rel.src.min() < 0 or rel.dst.min() < 0:
            raise ValueError(f"relation {relation_id} contains negative endpoint")
        if rel.src.max() >= graph.num_nodes or rel.dst.max() >= graph.num_nodes:
            raise ValueError(f"relation {relation_id} endpoint out of bounds")
        if not np.all(graph.node_type[rel.src] == rel.src_type):
            raise ValueError(f"relation {relation_id} source endpoint type mismatch")
        if not np.all(graph.node_type[rel.dst] == rel.dst_type):
            raise ValueError(f"relation {relation_id} destination endpoint type mismatch")

    if graph.features is not None:
        for type_id, feature in graph.features.items():
            if feature.ndim != 2:
                raise ValueError(f"features for type {type_id} must be 2D")
            expected = int(np.sum(graph.node_type == int(type_id)))
            if feature.shape[0] != expected:
                raise ValueError(
                    f"features for type {type_id} have {feature.shape[0]} rows, "
                    f"expected {expected}"
                )

    if graph.labels is not None and graph.labels.shape[0] != graph.num_nodes:
        raise ValueError("labels must have shape [num_nodes]")
    if graph.partitions is not None and graph.partitions.shape != (graph.num_nodes,):
        raise ValueError("partitions must have shape [num_nodes]")
