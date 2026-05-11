from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import (
    HeteroGraph,
    RelationAdj,
    RelationSpec,
    validate_schema,
)


NODE_TYPE_NAMES = {0: "user", 1: "item", 2: "tag"}


def _choice_edges(
    rng: np.random.Generator,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    fanout: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    fanout = max(1, min(int(fanout), len(dst_nodes)))
    for src in src_nodes:
        chosen = rng.choice(dst_nodes, size=fanout, replace=False)
        src_parts.append(np.full(fanout, src, dtype=np.int64))
        dst_parts.append(chosen.astype(np.int64))
    return np.concatenate(src_parts), np.concatenate(dst_parts)


def generate_synthetic_graph(
    num_users: int = 1000,
    num_items: int = 500,
    num_tags: int = 100,
    seed: int = 12345,
    feature_dim: int = 8,
) -> HeteroGraph:
    rng = np.random.default_rng(seed)
    user_nodes = np.arange(num_users, dtype=np.int64)
    item_nodes = np.arange(num_users, num_users + num_items, dtype=np.int64)
    tag_nodes = np.arange(num_users + num_items, num_users + num_items + num_tags, dtype=np.int64)
    num_nodes = int(num_users + num_items + num_tags)
    node_type = np.concatenate(
        [
            np.zeros(num_users, dtype=np.int32),
            np.ones(num_items, dtype=np.int32),
            np.full(num_tags, 2, dtype=np.int32),
        ]
    )

    ui_src, ui_dst = _choice_edges(rng, user_nodes, item_nodes, fanout=min(3, num_items))
    it_src, it_dst = _choice_edges(rng, item_nodes, tag_nodes, fanout=min(2, num_tags))
    if num_users > 1:
        uu_src = user_nodes
        uu_dst = np.roll(user_nodes, -1)
    else:
        uu_src = np.array([], dtype=np.int64)
        uu_dst = np.array([], dtype=np.int64)

    relations = {
        0: RelationAdj(ui_src, ui_dst, None, 0, 1, 0),
        1: RelationAdj(ui_dst, ui_src, None, 1, 0, 1),
        2: RelationAdj(it_src, it_dst, None, 1, 2, 2),
        3: RelationAdj(it_dst, it_src, None, 2, 1, 3),
        4: RelationAdj(uu_src, uu_dst, None, 0, 0, 4),
    }
    specs = {
        0: RelationSpec(0, "user_to_item", 0, 1),
        1: RelationSpec(1, "item_to_user", 1, 0),
        2: RelationSpec(2, "item_to_tag", 1, 2),
        3: RelationSpec(3, "tag_to_item", 2, 1),
        4: RelationSpec(4, "user_to_user", 0, 0),
    }
    features = {
        0: rng.normal(size=(num_users, feature_dim)).astype(np.float32),
        1: rng.normal(size=(num_items, feature_dim)).astype(np.float32),
        2: rng.normal(size=(num_tags, feature_dim)).astype(np.float32),
    }
    labels = np.full(num_nodes, -1, dtype=np.int32)
    if num_users:
        labels[user_nodes] = rng.integers(0, 3, size=num_users, dtype=np.int32)

    graph = HeteroGraph(
        num_nodes=num_nodes,
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
        features=features,
        labels=labels,
    )
    validate_schema(graph)
    return graph


def save_graph(graph: HeteroGraph, path: str | Path) -> None:
    validate_schema(graph)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    schema: dict[str, Any] = {
        "num_nodes": graph.num_nodes,
        "node_type_names": NODE_TYPE_NAMES,
        "relations": [
            graph.relation_specs[relation_id].to_json()
            for relation_id in sorted(graph.relation_specs)
        ],
        "features": sorted(graph.features or {}),
        "has_labels": graph.labels is not None,
        "has_partitions": graph.partitions is not None,
    }
    with (root / "schema.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)

    node_payload: dict[str, np.ndarray] = {"node_type": graph.node_type}
    if graph.labels is not None:
        node_payload["labels"] = graph.labels
    if graph.partitions is not None:
        node_payload["partitions"] = graph.partitions
    np.savez_compressed(root / "nodes.npz", **node_payload)

    for relation_id, rel in graph.relations.items():
        np.savez_compressed(
            root / f"relation_{relation_id}.npz",
            src=rel.src,
            dst=rel.dst,
            weight=rel.weight,
            src_type=np.array(rel.src_type, dtype=np.int32),
            dst_type=np.array(rel.dst_type, dtype=np.int32),
            relation_id=np.array(rel.relation_id, dtype=np.int32),
        )

    if graph.features is not None:
        for type_id, feature in graph.features.items():
            np.save(root / f"features_type_{type_id}.npy", feature)


def load_graph(path: str | Path) -> HeteroGraph:
    root = Path(path)
    with (root / "schema.json").open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    node_payload = np.load(root / "nodes.npz")
    node_type = node_payload["node_type"].astype(np.int32)
    labels = node_payload["labels"] if "labels" in node_payload.files else None
    partitions = (
        node_payload["partitions"].astype(np.int32)
        if "partitions" in node_payload.files
        else None
    )

    specs = {
        int(item["relation_id"]): RelationSpec.from_json(item)
        for item in schema["relations"]
    }
    relations: dict[int, RelationAdj] = {}
    for relation_id, spec in specs.items():
        payload = np.load(root / f"relation_{relation_id}.npz")
        relations[relation_id] = RelationAdj(
            src=payload["src"],
            dst=payload["dst"],
            weight=payload["weight"],
            src_type=spec.src_type,
            dst_type=spec.dst_type,
            relation_id=relation_id,
        )

    features: dict[int, np.ndarray] = {}
    for type_id in schema.get("features", []):
        type_int = int(type_id)
        features[type_int] = np.load(root / f"features_type_{type_int}.npy")

    graph = HeteroGraph(
        num_nodes=int(schema["num_nodes"]),
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
        features=features or None,
        labels=labels,
        partitions=partitions,
    )
    validate_schema(graph)
    return graph
