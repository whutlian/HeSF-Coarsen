from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.io.edge_list import save_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema


def _to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def _node_offsets(node_counts: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for node_type, count in node_counts.items():
        offsets[node_type] = cursor
        cursor += int(count)
    return offsets


def _labels_to_scalar(y: np.ndarray, expected_rows: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 0:
        return np.asarray([int(y)], dtype=np.int64)
    if y.ndim == 1:
        return y.astype(np.int64, copy=False)
    if y.shape[0] != expected_rows:
        raise ValueError(
            f"label rows {y.shape[0]} do not match expected node count {expected_rows}"
        )
    if y.shape[1] == 1:
        return y[:, 0].astype(np.int64, copy=False)
    positive = y > 0
    scalar = np.argmax(positive, axis=1).astype(np.int64)
    scalar[~positive.any(axis=1)] = -1
    return scalar


def heterodata_to_hesf_graph(data: Any, dataset_name: str = "hgb") -> HeteroGraph:
    node_types = list(data.node_types)
    node_counts: dict[str, int] = {
        node_type: int(data[node_type].num_nodes) for node_type in node_types
    }
    offsets = _node_offsets(node_counts)
    num_nodes = sum(node_counts.values())

    node_type = np.empty(num_nodes, dtype=np.int32)
    features: dict[int, np.ndarray] = {}
    labels = np.full(num_nodes, -1, dtype=np.int64)

    for type_id, node_name in enumerate(node_types):
        start = offsets[node_name]
        stop = start + node_counts[node_name]
        node_type[start:stop] = type_id
        store = data[node_name]
        if hasattr(store, "x") and store.x is not None:
            features[type_id] = _to_numpy(store.x, np.float32)
        if hasattr(store, "y") and store.y is not None:
            y = _labels_to_scalar(_to_numpy(store.y), node_counts[node_name])
            labels[start : start + len(y)] = y.astype(np.int64, copy=False)

    relations: dict[int, RelationAdj] = {}
    relation_specs: dict[int, RelationSpec] = {}
    for relation_id, edge_type in enumerate(data.edge_types):
        src_name, rel_name, dst_name = edge_type
        edge_index = _to_numpy(data[edge_type].edge_index, np.int64)
        src = edge_index[0] + offsets[src_name]
        dst = edge_index[1] + offsets[dst_name]
        src_type = node_types.index(src_name)
        dst_type = node_types.index(dst_name)
        relation_name = f"{src_name}__{rel_name}__{dst_name}"
        relations[relation_id] = RelationAdj(
            src=src,
            dst=dst,
            weight=None,
            src_type=src_type,
            dst_type=dst_type,
            relation_id=relation_id,
        )
        relation_specs[relation_id] = RelationSpec(
            relation_id=relation_id,
            name=relation_name,
            src_type=src_type,
            dst_type=dst_type,
        )

    graph = HeteroGraph(
        num_nodes=num_nodes,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features=features or None,
        labels=labels,
    )
    validate_schema(graph)
    return graph


def import_hgb_dataset(
    name: str,
    root: str | Path,
    output: str | Path,
    force_reload: bool = False,
) -> HeteroGraph:
    try:
        from torch_geometric.datasets import HGBDataset
    except ImportError as exc:
        raise RuntimeError("torch_geometric is required for HGB import") from exc

    dataset = HGBDataset(root=str(root), name=name.upper(), force_reload=force_reload)
    graph = heterodata_to_hesf_graph(dataset[0], dataset_name=name)
    save_graph(graph, output)
    return graph


def ogb_mag_to_hesf_graph(graph_dict: dict[str, Any], labels: Any | None = None) -> HeteroGraph:
    node_counts = {key: int(value) for key, value in graph_dict["num_nodes_dict"].items()}
    offsets = _node_offsets(node_counts)
    node_names = list(node_counts)
    num_nodes = sum(node_counts.values())
    node_type = np.empty(num_nodes, dtype=np.int32)
    features: dict[int, np.ndarray] = {}
    label_array = np.full(num_nodes, -1, dtype=np.int64)

    node_feat_dict = graph_dict.get("node_feat_dict", {}) or {}
    for type_id, node_name in enumerate(node_names):
        start = offsets[node_name]
        stop = start + node_counts[node_name]
        node_type[start:stop] = type_id
        if node_name in node_feat_dict:
            features[type_id] = _to_numpy(node_feat_dict[node_name], np.float32)

    if labels is not None and "paper" in offsets:
        paper_labels = labels["paper"] if isinstance(labels, dict) else labels
        y = _labels_to_scalar(_to_numpy(paper_labels), node_counts["paper"])
        start = offsets["paper"]
        label_array[start : start + len(y)] = y.astype(np.int64, copy=False)

    relations: dict[int, RelationAdj] = {}
    relation_specs: dict[int, RelationSpec] = {}
    edge_index_dict = graph_dict["edge_index_dict"]
    for relation_id, edge_type in enumerate(edge_index_dict):
        src_name, rel_name, dst_name = edge_type
        edge_index = _to_numpy(edge_index_dict[edge_type], np.int64)
        src = edge_index[0] + offsets[src_name]
        dst = edge_index[1] + offsets[dst_name]
        src_type = node_names.index(src_name)
        dst_type = node_names.index(dst_name)
        relation_name = f"{src_name}__{rel_name}__{dst_name}"
        relations[relation_id] = RelationAdj(
            src=src,
            dst=dst,
            weight=None,
            src_type=src_type,
            dst_type=dst_type,
            relation_id=relation_id,
        )
        relation_specs[relation_id] = RelationSpec(
            relation_id=relation_id,
            name=relation_name,
            src_type=src_type,
            dst_type=dst_type,
        )

    graph = HeteroGraph(
        num_nodes=num_nodes,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features=features or None,
        labels=label_array,
    )
    validate_schema(graph)
    return graph


def import_ogbn_mag_dataset(
    root: str | Path,
    output: str | Path,
) -> HeteroGraph:
    try:
        import torch
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise RuntimeError("ogb is required for ogbn-mag import") from exc

    original_torch_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        dataset = NodePropPredDataset(name="ogbn-mag", root=str(root))
    finally:
        torch.load = original_torch_load
    graph_dict, labels = dataset[0]
    graph = ogb_mag_to_hesf_graph(graph_dict, labels)
    save_graph(graph, output)
    return graph
