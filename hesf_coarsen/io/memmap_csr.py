from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.io.edge_list import NODE_TYPE_NAMES
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema


def _write_memmap(path: Path, array: np.ndarray, dtype: np.dtype, chunk_size: int) -> None:
    source = np.asarray(array, dtype=dtype)
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=source.shape,
    )
    if source.ndim == 0:
        target[...] = source
    else:
        for start in range(0, source.shape[0], chunk_size):
            stop = min(start + chunk_size, source.shape[0])
            target[start:stop] = source[start:stop]
    target.flush()


def save_memmap_graph(
    graph: HeteroGraph,
    path: str | Path,
    chunk_size: int = 1_000_000,
) -> None:
    """Write graph arrays as mmap-loadable `.npy` files.

    This path is intended for explicit large-graph CLI workflows. The default
    medium/small graph path continues to use compressed NPZ files.
    """

    validate_schema(graph)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    _write_memmap(root / "node_type.npy", graph.node_type, np.dtype("int32"), chunk_size)
    node_files: dict[str, str] = {"node_type": "node_type.npy"}
    if graph.labels is not None:
        _write_memmap(root / "labels.npy", graph.labels, graph.labels.dtype, chunk_size)
        node_files["labels"] = "labels.npy"
    if graph.partitions is not None:
        _write_memmap(
            root / "partitions.npy",
            graph.partitions,
            np.dtype("int32"),
            chunk_size,
        )
        node_files["partitions"] = "partitions.npy"

    feature_files: dict[str, str] = {}
    if graph.features is not None:
        for type_id, feature in graph.features.items():
            filename = f"features_type_{type_id}.npy"
            _write_memmap(root / filename, feature, np.dtype("float32"), chunk_size)
            feature_files[str(type_id)] = filename

    relation_files: dict[str, dict[str, str]] = {}
    for relation_id, rel in sorted(graph.relations.items()):
        prefix = f"relation_{relation_id}"
        _write_memmap(root / f"{prefix}_src.npy", rel.src, np.dtype("int64"), chunk_size)
        _write_memmap(root / f"{prefix}_dst.npy", rel.dst, np.dtype("int64"), chunk_size)
        _write_memmap(root / f"{prefix}_weight.npy", rel.weight, np.dtype("float32"), chunk_size)
        relation_files[str(relation_id)] = {
            "src": f"{prefix}_src.npy",
            "dst": f"{prefix}_dst.npy",
            "weight": f"{prefix}_weight.npy",
        }

    schema: dict[str, Any] = {
        "format": "hesf_memmap_v1",
        "chunk_size": int(chunk_size),
        "num_nodes": graph.num_nodes,
        "node_type_names": NODE_TYPE_NAMES,
        "node_files": node_files,
        "feature_files": feature_files,
        "relation_files": relation_files,
        "relations": [
            graph.relation_specs[relation_id].to_json()
            for relation_id in sorted(graph.relation_specs)
        ],
    }
    with (root / "schema.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)


def load_memmap_graph(path: str | Path) -> HeteroGraph:
    root = Path(path)
    with (root / "schema.json").open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if schema.get("format") != "hesf_memmap_v1":
        raise ValueError(f"{root} is not a HeSF memmap graph directory")

    node_files = schema["node_files"]
    node_type = np.load(root / node_files["node_type"], mmap_mode="r")
    labels = (
        np.load(root / node_files["labels"], mmap_mode="r")
        if "labels" in node_files
        else None
    )
    partitions = (
        np.load(root / node_files["partitions"], mmap_mode="r")
        if "partitions" in node_files
        else None
    )

    specs = {
        int(item["relation_id"]): RelationSpec.from_json(item)
        for item in schema["relations"]
    }
    relations: dict[int, RelationAdj] = {}
    for relation_id, spec in specs.items():
        files = schema["relation_files"][str(relation_id)]
        relations[relation_id] = RelationAdj(
            src=np.load(root / files["src"], mmap_mode="r"),
            dst=np.load(root / files["dst"], mmap_mode="r"),
            weight=np.load(root / files["weight"], mmap_mode="r"),
            src_type=spec.src_type,
            dst_type=spec.dst_type,
            relation_id=relation_id,
        )

    features: dict[int, np.ndarray] = {}
    for type_id, filename in schema.get("feature_files", {}).items():
        features[int(type_id)] = np.load(root / filename, mmap_mode="r")

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


def memmap_summary(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    graph = load_memmap_graph(root)
    return {
        "format": "hesf_memmap_v1",
        "path": str(root),
        "num_nodes": graph.num_nodes,
        "relations": {str(k): rel.num_edges for k, rel in graph.relations.items()},
    }
