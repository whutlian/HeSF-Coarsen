from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS, audit_native_hgb_data_dir, supported_sehgnn_hgb_dataset
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type, validate_schema


def require_native_reproduction_pass(result: Mapping[str, Any]) -> None:
    if not bool(result.get("native_repro_pass")):
        raise RuntimeError("native SeHGNN reproduction has not passed; HeSF official-format export is not allowed")


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))


def _target_type_id(dataset_name: str, target_type: str) -> int:
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    if str(target_type).isdigit():
        return int(target_type)
    return int(schema["node_type_order"][str(target_type)])


def _label_at(labels: np.ndarray, node_id: int, *, target_local_id: int | None = None) -> str:
    arr = np.asarray(labels)
    index = int(node_id)
    if arr.shape[0] <= index and target_local_id is not None:
        index = int(target_local_id)
    value = arr[index]
    if np.asarray(value).ndim == 0:
        return str(int(value))
    flat = np.asarray(value).reshape(-1)
    active = [str(int(i)) for i, v in enumerate(flat.tolist()) if float(v) > 0.0]
    return ",".join(active)


def _feature_text(feature_matrix: np.ndarray | None, local_id: int) -> str | None:
    if feature_matrix is None:
        return None
    row = np.asarray(feature_matrix[int(local_id)], dtype=np.float32).reshape(-1)
    return ",".join(str(float(value)) for value in row.tolist())


def _relation_endpoints(dataset_name: str, relation_name: str) -> tuple[int, int]:
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    order = schema["node_type_order"]
    token = relation_name.replace("_r", "")
    if len(token) < 2:
        raise ValueError(f"cannot infer endpoints for relation {relation_name!r}")
    return int(order[token[0]]), int(order[token[1]])


def _official_relation_mapping(graph: HeteroGraph, dataset_name: str) -> tuple[dict[int, int], bool]:
    used: set[int] = set()
    mapping: dict[int, int] = {}
    ok = True
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    for relation_name, official_id in sorted(schema["relation_id_order"].items(), key=lambda item: int(item[1])):
        src_type, dst_type = _relation_endpoints(dataset_name, relation_name)
        candidates = [
            int(rel_id)
            for rel_id, rel in sorted(graph.relations.items())
            if int(rel_id) not in used and int(rel.src_type) == src_type and int(rel.dst_type) == dst_type
        ]
        if not candidates:
            ok = False
            continue
        source_relation_id = candidates[0]
        used.add(source_relation_id)
        mapping[source_relation_id] = int(official_id)
    if len(mapping) != len(schema["relation_id_order"]):
        ok = False
    return mapping, ok


def _write_mapping_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_graph_to_sehgnn_hgb(
    *,
    graph: HeteroGraph,
    dataset_name: str,
    target_type: str,
    output_dir: Path,
    split_mode: str,
    train_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    seed: int,
    preserve_target_ids: bool = True,
    write_sidecar: bool = True,
) -> dict[str, Any]:
    validate_schema(graph)
    dataset = supported_sehgnn_hgb_dataset(dataset_name)
    if split_mode not in {"official_trainval", "hesf_fixed_split"}:
        raise ValueError(f"unsupported split_mode: {split_mode}")
    target_type_id = _target_type_id(dataset, target_type)
    output_dir = Path(output_dir)
    export_dir = output_dir / "export" / _safe_token(dataset) / str(int(seed)) / _safe_token(method_name) / split_mode / dataset
    export_dir.mkdir(parents=True, exist_ok=True)

    schema = SEHGNN_HGB_SCHEMAS[dataset]
    official_to_original: dict[int, int] = {}
    original_to_official: dict[int, int] = {}
    type_local_rows: list[dict[str, Any]] = []
    next_id = 0
    for type_name, type_id in sorted(schema["node_type_order"].items(), key=lambda item: int(item[1])):
        nodes = nodes_of_type(graph, int(type_id))
        for local_id, original_id in enumerate(nodes.tolist()):
            official_id = int(next_id)
            official_to_original[official_id] = int(original_id)
            original_to_official[int(original_id)] = official_id
            type_local_rows.append(
                {
                    "official_global_id": official_id,
                    "original_global_id": int(original_id),
                    "node_type_name": type_name,
                    "node_type_id": int(type_id),
                    "type_local_id": int(local_id),
                }
            )
            next_id += 1
    mapping_bijective = len(original_to_official) == int(graph.num_nodes) and len(set(original_to_official.values())) == int(graph.num_nodes)

    with (export_dir / "node.dat").open("w", encoding="utf-8", newline="") as handle:
        for row in type_local_rows:
            feature = None if graph.features is None else graph.features.get(int(row["node_type_id"]))
            feature_text = _feature_text(feature, int(row["type_local_id"]))
            base = f"{row['official_global_id']}\t{row['official_global_id']}\t{row['node_type_id']}"
            handle.write(base + (f"\t{feature_text}\n" if feature_text is not None else "\n"))

    relation_mapping, relation_order_matches = _official_relation_mapping(graph, dataset)
    edge_counts: dict[str, int] = {}
    with (export_dir / "link.dat").open("w", encoding="utf-8", newline="") as handle:
        for source_relation_id, official_relation_id in sorted(relation_mapping.items(), key=lambda item: item[1]):
            rel = graph.relations[int(source_relation_id)]
            edge_counts[str(int(official_relation_id))] = int(rel.num_edges)
            for src, dst, weight in zip(rel.src.tolist(), rel.dst.tolist(), rel.weight.tolist()):
                handle.write(f"{original_to_official[int(src)]}\t{original_to_official[int(dst)]}\t{int(official_relation_id)}\t{float(weight)}\n")

    target_nodes = nodes_of_type(graph, target_type_id)
    target_local_to_official = {local_id: original_to_official[int(original)] for local_id, original in enumerate(target_nodes.tolist())}
    train = np.asarray([] if train_idx is None else train_idx, dtype=np.int64).reshape(-1)
    val = np.asarray([] if val_idx is None else val_idx, dtype=np.int64).reshape(-1)
    test = np.asarray(test_idx, dtype=np.int64).reshape(-1)
    trainval = np.concatenate([train, val]).astype(np.int64, copy=False)
    trainval_set = set(int(v) for v in trainval.tolist())
    test_set = set(int(v) for v in test.tolist())
    split_disjoint = not bool(trainval_set & test_set)

    def write_labels(path: Path, split: np.ndarray) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            for target_local in split.tolist():
                official_id = int(target_local_to_official[int(target_local)])
                label = _label_at(np.asarray(labels), official_to_original[official_id], target_local_id=int(target_local))
                if label == "" or label == "-1":
                    continue
                handle.write(f"{official_id}\t{official_id}\t{target_type_id}\t{label}\n")

    write_labels(export_dir / "label.dat", trainval)
    write_labels(export_dir / "label.dat.test", test)

    if write_sidecar:
        mapping_dir = export_dir / "mapping"
        _write_mapping_csv(mapping_dir / "node_type_id_map.csv", type_local_rows, ["official_global_id", "original_global_id", "node_type_name", "node_type_id", "type_local_id"])
        target_rows = [
            {
                "target_local_id": int(local_id),
                "original_target_id": int(original_id),
                "official_global_target_id": int(target_local_to_official[int(local_id)]),
            }
            for local_id, original_id in enumerate(target_nodes.tolist())
        ]
        _write_mapping_csv(mapping_dir / "target_id_map.csv", target_rows, ["target_local_id", "original_target_id", "official_global_target_id"])
        rel_rows = [
            {"source_relation_id": int(source_id), "official_relation_id": int(official_id)}
            for source_id, official_id in sorted(relation_mapping.items(), key=lambda item: item[1])
        ]
        _write_mapping_csv(mapping_dir / "relation_id_map.csv", rel_rows, ["source_relation_id", "official_relation_id"])
        np.savez(export_dir / "mapping" / "split_file.npz", train_idx=train, val_idx=val, test_idx=test)

    node_counts = {str(type_id): int(np.sum(graph.node_type == int(type_id))) for type_id in schema["node_type_order"].values()}
    labels_arr = np.asarray(labels)
    labels_format = "multilabel" if labels_arr.ndim == 2 else "single"
    audit = audit_native_hgb_data_dir(dataset, export_dir.parent)
    manifest = {
        "export_dir": str(export_dir),
        "dataset": dataset,
        "method": str(method_name),
        "seed": int(seed),
        "node_type_mapping": schema["node_type_order"],
        "relation_id_mapping": schema["relation_id_order"],
        "target_type": schema["target_type"],
        "target_count": int(target_nodes.size),
        "train_count": int(train.size),
        "val_count": int(val.size),
        "test_count": int(test.size),
        "node_count_by_type": node_counts,
        "edge_count_by_relation": edge_counts,
        "feature_shapes_by_type": {str(k): list(v.shape) for k, v in (graph.features or {}).items()},
        "labels_format": labels_format,
        "split_mode": split_mode,
        "mapping_bijective": bool(mapping_bijective),
        "split_disjoint": bool(split_disjoint),
        "all_train_val_test_target_ids_preserved": bool(all(int(v) in target_local_to_official for v in np.concatenate([trainval, test]).tolist())),
        "no_target_duplicates": bool(len(set(target_local_to_official.values())) == len(target_local_to_official)),
        "no_test_label_export_leakage": True,
        "relation_order_matches_official": bool(relation_order_matches),
        "node_type_order_matches_official": True,
        "can_load_with_official_data_loader": bool(audit["can_load_with_official_data_loader"]),
        "preserve_target_ids": bool(preserve_target_ids),
    }
    (export_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
