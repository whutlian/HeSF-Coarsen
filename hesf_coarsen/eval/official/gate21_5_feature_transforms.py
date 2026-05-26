from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix
from hesf_coarsen.io.schema import HeteroGraph


DBLP_TYPE_BY_NAME = {"author": 0, "paper": 1, "term": 2, "venue": 3}
DBLP_NAME_BY_TYPE = {value: key for key, value in DBLP_TYPE_BY_NAME.items()}
TARGET_TYPES = {"author"}
SUPPORT_TYPES = {"paper", "term", "venue"}


def transform_gate21_5_graph_features(
    graph: HeteroGraph,
    transform_name: str,
    *,
    dataset: str = "DBLP",
    seed: int = 1,
) -> HeteroGraph:
    """Apply Gate21.5 feature ablation/compression transforms by DBLP node type."""

    if str(dataset).upper() != "DBLP":
        raise ValueError("Gate21.5 feature transforms are currently DBLP-specific")
    before = _copy_features(graph.features)
    features = _copy_features(graph.features)
    name = str(transform_name)
    modified: set[int] = set()
    family = "raw"
    audit_extra: dict[str, Any] = {}

    if name in {"raw", "raw-paper", "raw_features_adapter_control"}:
        pass
    elif name.startswith("zero-"):
        family = "zero"
        modified.update(_zero_transform(features, graph, name))
    elif name.endswith("-only-features"):
        family = "type_only"
        keep_name = name.replace("-only-features", "")
        keep_type = DBLP_TYPE_BY_NAME.get(keep_name)
        if keep_type is None:
            raise ValueError(f"unsupported Gate21.5 type-only transform: {name!r}")
        for type_id in list(_all_dblp_type_ids()):
            if int(type_id) != int(keep_type):
                _ensure_feature(features, graph, int(type_id), dim=_default_dim(features))
                features[int(type_id)] = np.zeros_like(features[int(type_id)], dtype=np.float32)
                modified.add(int(type_id))
    elif name == "type-constant-support-features":
        family = "type_constant"
        for type_name in SUPPORT_TYPES:
            type_id = DBLP_TYPE_BY_NAME[type_name]
            _ensure_feature(features, graph, type_id, dim=_default_dim(features))
            features[type_id] = np.full_like(features[type_id], _constant_for_type(type_id), dtype=np.float32)
            modified.add(type_id)
    elif name == "all-type-constant-features":
        family = "type_constant"
        for type_id in _all_dblp_type_ids():
            _ensure_feature(features, graph, type_id, dim=_default_dim(features))
            features[type_id] = np.full_like(features[type_id], _constant_for_type(type_id), dtype=np.float32)
            modified.add(type_id)
    elif match := re.fullmatch(r"pca-all-types-(\d+)", name):
        family = "pca_all_types"
        dim = int(match.group(1))
        for type_id in _all_dblp_type_ids():
            _ensure_feature(features, graph, type_id, dim=dim)
            transformed, audit = transform_feature_matrix(features[type_id], f"pca-paper-{dim}", seed=int(seed))
            features[type_id] = transformed
            audit_extra[f"{DBLP_NAME_BY_TYPE[type_id]}_pca_dim"] = int(audit.get("feature_dim", dim))
            modified.add(type_id)
    elif name in {
        "pca-paper-128",
        "pca-paper-64",
        "pca-paper-256",
        "random_projection_dim128",
        "random_projection_dim64",
        "random-projection-paper-128",
        "random-projection-paper-64",
        "fp16-paper",
        "int8-paper",
    }:
        family = "paper_projection" if ("pca" in name or "projection" in name) else "paper_quantization"
        _ensure_feature(features, graph, DBLP_TYPE_BY_NAME["paper"], dim=_default_dim(features))
        transformed, audit = transform_feature_matrix(features[DBLP_TYPE_BY_NAME["paper"]], _paper_transform_name(name), seed=int(seed))
        features[DBLP_TYPE_BY_NAME["paper"]] = transformed
        audit_extra.update(audit)
        modified.add(DBLP_TYPE_BY_NAME["paper"])
    else:
        raise ValueError(f"unsupported Gate21.5 feature transform: {name!r}")

    out = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=graph.relations,
        relation_specs=graph.relation_specs,
        features=features,
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
        partitions=None if graph.partitions is None else np.asarray(graph.partitions).copy(),
    )
    audit = {
        "transform_name": name,
        "feature_transform_family": family,
        "node_types_modified": ",".join(DBLP_NAME_BY_TYPE.get(int(t), str(t)) for t in sorted(modified)),
        "fit_uses_labels": False,
        "fit_uses_test_labels": False,
        "fit_uses_validation_labels": False,
        "source_storage_dtype": _source_storage_dtype(name),
        "model_input_dtype": "fp32",
        "feature_dim": _paper_dim(features),
        "sidecar_metadata_bytes": int(audit_extra.get("sidecar_metadata_bytes", 0) or 0),
        "metadata_keys": str(audit_extra.get("metadata_keys", "")),
        "input_shape": _shape_map(before),
        "output_shape": _shape_map(features),
        "feature_dim_by_type_after_loader": _dim_map(features),
    }
    object.__setattr__(out, "_gate21_5_features_before", before)
    object.__setattr__(out, "_gate21_5_features_after", features)
    object.__setattr__(out, "_gate21_5_transform_audit", audit)
    object.__setattr__(out, "_gate21_4_transform_audit", audit)
    return out


def _zero_transform(features: dict[int, np.ndarray], graph: HeteroGraph, name: str) -> set[int]:
    modified: set[int] = set()
    if name == "zero-all-support-features":
        target_names = SUPPORT_TYPES
    elif name == "zero-target-author-only":
        target_names = {"author"}
    elif name == "zero-support-author-only":
        target_names = {"author"}
    else:
        type_name = name.replace("zero-", "")
        target_names = {type_name}
    for type_name in target_names:
        type_id = DBLP_TYPE_BY_NAME.get(type_name)
        if type_id is None:
            raise ValueError(f"unsupported zero transform: {name!r}")
        _ensure_feature(features, graph, type_id, dim=_default_dim(features))
        features[type_id] = np.zeros_like(features[type_id], dtype=np.float32)
        modified.add(type_id)
    return modified


def _copy_features(features: Mapping[int, np.ndarray] | None) -> dict[int, np.ndarray]:
    return {} if features is None else {int(k): np.asarray(v, dtype=np.float32).copy() for k, v in features.items()}


def _ensure_feature(features: dict[int, np.ndarray], graph: HeteroGraph, type_id: int, *, dim: int) -> None:
    if int(type_id) in features:
        return
    count = int(np.sum(graph.node_type == int(type_id)))
    features[int(type_id)] = np.zeros((count, max(int(dim), 1)), dtype=np.float32)


def _all_dblp_type_ids() -> tuple[int, int, int, int]:
    return (0, 1, 2, 3)


def _default_dim(features: Mapping[int, np.ndarray]) -> int:
    if 1 in features and features[1].ndim == 2:
        return int(features[1].shape[1])
    for value in features.values():
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] > 0:
            return int(arr.shape[1])
    return 1


def _constant_for_type(type_id: int) -> float:
    return float((int(type_id) + 1) / 10.0)


def _paper_dim(features: Mapping[int, np.ndarray]) -> int:
    paper = features.get(DBLP_TYPE_BY_NAME["paper"])
    return int(paper.shape[1]) if paper is not None and paper.ndim == 2 else 0


def _shape_map(features: Mapping[int, np.ndarray]) -> dict[str, list[int]]:
    return {DBLP_NAME_BY_TYPE.get(int(k), str(k)): list(np.asarray(v).shape) for k, v in sorted(features.items())}


def _dim_map(features: Mapping[int, np.ndarray]) -> dict[str, int]:
    return {DBLP_NAME_BY_TYPE.get(int(k), str(k)): int(np.asarray(v).shape[1]) if np.asarray(v).ndim == 2 else 0 for k, v in sorted(features.items())}


def _source_storage_dtype(name: str) -> str:
    if name in {"fp16-paper", "fp16_node_features"}:
        return "fp16"
    if name in {"int8-paper", "int8_per_feature"}:
        return "int8"
    return "fp32"


def _paper_transform_name(name: str) -> str:
    if name == "random-projection-paper-128":
        return "random_projection_dim128"
    if name == "random-projection-paper-64":
        return "random_projection_dim64"
    return str(name)
