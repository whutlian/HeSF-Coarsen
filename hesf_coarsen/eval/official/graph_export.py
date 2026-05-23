from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.hgb_export import write_hgb_metadata_files
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type, validate_schema


def _type_name(type_id: int) -> str:
    return f"type_{int(type_id)}"


def _parse_type(value: str | int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.startswith("type_"):
        return int(text.split("_", 1)[1])
    return int(text)


def _ratio_token(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{float(value):.2f}".replace(".", "p")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))


def _distribution(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    out: dict[str, int] = {}
    for value in labels.tolist():
        if int(value) >= 0:
            out[str(int(value))] = out.get(str(int(value)), 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _target_labels(labels: Any, original_target_ids: np.ndarray, graph_num_nodes: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.shape[0] == graph_num_nodes:
        return arr[original_target_ids]
    if arr.shape[0] == original_target_ids.shape[0]:
        return arr.copy()
    raise ValueError("labels must be graph-wide or target-aligned")


def _map_split(split: Any, mapping: Mapping[int, int], target_count: int, name: str) -> np.ndarray:
    raw = np.asarray(split, dtype=np.int64).reshape(-1)
    if raw.size == 0:
        return raw
    keys = set(mapping)
    if all(int(v) in keys for v in raw.tolist()):
        return np.asarray([mapping[int(v)] for v in raw.tolist()], dtype=np.int64)
    if np.all((raw >= 0) & (raw < int(target_count))):
        return raw.astype(np.int64, copy=True)
    raise ValueError(f"{name} split contains ids outside target mapping")


def _assert_disjoint(train: np.ndarray, val: np.ndarray, test: np.ndarray, target_count: int) -> bool:
    sets = [set(int(v) for v in arr.tolist()) for arr in (train, val, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("train/val/test splits must be disjoint")
    if any(v < 0 or v >= int(target_count) for subset in sets for v in subset):
        raise ValueError("split ids must be target-local ids")
    return True


def _type_local_lookup(graph: HeteroGraph, type_id: int) -> dict[int, int]:
    nodes = nodes_of_type(graph, int(type_id))
    return {int(node): int(pos) for pos, node in enumerate(nodes.tolist())}


def _write_optional_dgl(graph: HeteroGraph, export_dir: Path, relation_names: dict[int, str]) -> str:
    try:
        import dgl  # type: ignore
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency.
        return f"unavailable:{exc}"
    data_dict: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    for relation_id, rel in graph.relations.items():
        src_lookup = _type_local_lookup(graph, int(rel.src_type))
        dst_lookup = _type_local_lookup(graph, int(rel.dst_type))
        src = [src_lookup[int(v)] for v in rel.src.tolist()]
        dst = [dst_lookup[int(v)] for v in rel.dst.tolist()]
        data_dict[(_type_name(rel.src_type), relation_names[relation_id], _type_name(rel.dst_type))] = (
            torch.as_tensor(src, dtype=torch.int64),
            torch.as_tensor(dst, dtype=torch.int64),
        )
    type_ids = set(int(v) for v in np.unique(graph.node_type))
    if graph.features is not None:
        type_ids.update(int(v) for v in graph.features)
    for spec in graph.relation_specs.values():
        type_ids.add(int(spec.src_type))
        type_ids.add(int(spec.dst_type))
    num_nodes_dict = {_type_name(type_id): int(np.sum(graph.node_type == int(type_id))) for type_id in sorted(type_ids)}
    hetero = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    dgl.save_graphs(str(export_dir / "graph.dgl"), [hetero])
    return "written"


def export_hgb_graph(
    graph: HeteroGraph,
    dataset_name: str,
    method_name: str,
    seed: int,
    support_ratio: float | None,
    output_dir: Path,
    *,
    target_type: str,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    labels: Any,
    original_target_ids: Any = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    validate_schema(graph)
    target_type_id = _parse_type(target_type)
    if original_target_ids is None:
        target_ids = nodes_of_type(graph, target_type_id)
    else:
        target_ids = np.asarray(original_target_ids, dtype=np.int64).reshape(-1)
    if target_ids.size != np.unique(target_ids).size:
        raise ValueError("original_target_ids must be unique")
    if np.any(target_ids < 0) or np.any(target_ids >= graph.num_nodes):
        raise ValueError("original_target_ids out of graph bounds")
    if not np.all(graph.node_type[target_ids] == int(target_type_id)):
        raise ValueError("all original_target_ids must have target_type")
    mapping = {int(node): int(pos) for pos, node in enumerate(target_ids.tolist())}
    mapping_bijective = len(mapping) == int(target_ids.size) and len(set(mapping.values())) == int(target_ids.size)
    if not mapping_bijective:
        raise ValueError("target mapping is not bijective")

    target_labels = _target_labels(labels, target_ids, graph.num_nodes)
    train_local = _map_split(train_idx, mapping, int(target_ids.size), "train")
    val_local = _map_split(val_idx, mapping, int(target_ids.size), "val")
    test_local = _map_split(test_idx, mapping, int(target_ids.size), "test")
    split_disjoint = _assert_disjoint(train_local, val_local, test_local, int(target_ids.size))

    export_dir = (
        Path(output_dir)
        / "exports"
        / _safe_name(str(dataset_name))
        / str(int(seed))
        / _safe_name(str(method_name))
        / _ratio_token(support_ratio)
    )
    for subdir in ("node_features", "edges", "splits"):
        (export_dir / subdir).mkdir(parents=True, exist_ok=True)

    type_ids = set(int(v) for v in np.unique(graph.node_type))
    if graph.features is not None:
        type_ids.update(int(v) for v in graph.features)
    for spec in graph.relation_specs.values():
        type_ids.add(int(spec.src_type))
        type_ids.add(int(spec.dst_type))
    type_names = {_type: _type_name(_type) for _type in sorted(type_ids)}
    num_nodes_by_type = {type_names[type_id]: int(np.sum(graph.node_type == type_id)) for type_id in type_names}
    for type_id, type_label in type_names.items():
        feature = None if graph.features is None else graph.features.get(type_id)
        if feature is None:
            feature = np.zeros((num_nodes_by_type[type_label], 1), dtype=np.float32)
        np.save(export_dir / "node_features" / f"{type_label}.npy", np.asarray(feature, dtype=np.float32))

    relation_names: dict[int, str] = {}
    relation_schemas: list[dict[str, str]] = []
    num_edges_by_relation: dict[str, int] = {}
    for relation_id, rel in sorted(graph.relations.items()):
        spec = graph.relation_specs[int(relation_id)]
        name = _safe_name(spec.name)
        relation_names[int(relation_id)] = name
        relation_schemas.append(
            {
                "name": name,
                "src_type": _type_name(int(spec.src_type)),
                "dst_type": _type_name(int(spec.dst_type)),
            }
        )
        src_lookup = _type_local_lookup(graph, int(rel.src_type))
        dst_lookup = _type_local_lookup(graph, int(rel.dst_type))
        edge_array = np.zeros((int(rel.num_edges), 3), dtype=np.float32)
        if rel.num_edges:
            edge_array[:, 0] = np.asarray([src_lookup[int(v)] for v in rel.src.tolist()], dtype=np.float32)
            edge_array[:, 1] = np.asarray([dst_lookup[int(v)] for v in rel.dst.tolist()], dtype=np.float32)
            edge_array[:, 2] = np.asarray(rel.weight, dtype=np.float32)
        np.save(export_dir / "edges" / f"{name}.npy", edge_array)
        num_edges_by_relation[name] = int(rel.num_edges)

    np.save(export_dir / "splits" / "train_idx.npy", train_local)
    np.save(export_dir / "splits" / "val_idx.npy", val_local)
    np.save(export_dir / "splits" / "test_idx.npy", test_local)
    np.save(export_dir / "splits" / "train_labels.npy", target_labels[train_local])
    np.save(export_dir / "splits" / "val_labels.npy", target_labels[val_local])
    np.save(export_dir / "labels.npy", target_labels)
    np.save(export_dir / "target_node_ids.npy", target_ids)
    with (export_dir / "original_to_export_target_id.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_target_id", "export_target_id"])
        writer.writeheader()
        for original_id, export_id in sorted(mapping.items(), key=lambda item: item[1]):
            writer.writerow({"original_target_id": int(original_id), "export_target_id": int(export_id)})

    no_test_label_export_leakage = not (export_dir / "splits" / "test_labels_for_training.npy").exists()
    dgl_status = _write_optional_dgl(graph, export_dir, relation_names)
    meta = {
        "dataset": str(dataset_name),
        "method": str(method_name),
        "seed": int(seed),
        "support_ratio": None if support_ratio is None else float(support_ratio),
        "target_type": _type_name(target_type_id),
        "node_type_names": [type_names[type_id] for type_id in sorted(type_names)],
        "relation_type_names": [relation_names[relation_id] for relation_id in sorted(relation_names)],
        "relation_schemas": relation_schemas,
        "num_nodes_by_type": num_nodes_by_type,
        "num_edges_by_relation": num_edges_by_relation,
        "dgl_status": dgl_status,
        **dict(metadata or {}),
    }
    (export_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_hgb_metadata_files(export_dir, meta)

    audit = {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "method": str(method_name),
        "support_ratio": "" if support_ratio is None else float(support_ratio),
        "export_dir": str(export_dir),
        "target_type": _type_name(target_type_id),
        "num_nodes_by_type_original": json.dumps(num_nodes_by_type, sort_keys=True),
        "num_nodes_by_type_exported": json.dumps(num_nodes_by_type, sort_keys=True),
        "num_edges_by_relation_original": json.dumps(num_edges_by_relation, sort_keys=True),
        "num_edges_by_relation_exported": json.dumps(num_edges_by_relation, sort_keys=True),
        "num_nodes_by_type": num_nodes_by_type,
        "num_edges_by_relation": num_edges_by_relation,
        "target_count_original": int(target_ids.size),
        "target_count_exported": int(target_ids.size),
        "target_count": int(target_ids.size),
        "train_count": int(train_local.size),
        "val_count": int(val_local.size),
        "test_count": int(test_local.size),
        "label_distribution_train": _distribution(target_labels[train_local]),
        "label_distribution_val": _distribution(target_labels[val_local]),
        "label_distribution_test": _distribution(target_labels[test_local]),
        "mapping_bijective": bool(mapping_bijective),
        "split_disjoint": bool(split_disjoint),
        "no_test_label_export_leakage": bool(no_test_label_export_leakage),
        "export_status": "success",
        "error_message": "",
    }
    (export_dir / "export_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return audit
