from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix
from hesf_coarsen.io.schema import HeteroGraph


DBLP_TYPE_BY_NAME = {"author": 0, "paper": 1, "term": 2, "venue": 3}
DBLP_NAME_BY_TYPE = {value: key for key, value in DBLP_TYPE_BY_NAME.items()}
SUPPORT_TYPES = {"paper", "term", "venue"}


def transform_gate21_6_graph_features(
    graph: HeteroGraph,
    transform_name: str,
    *,
    dataset: str = "DBLP",
    seed: int = 1,
) -> HeteroGraph:
    """Apply Gate21.6 feature transforms with preserve-dim zero semantics."""

    if str(dataset).upper() != "DBLP":
        raise ValueError("Gate21.6 feature transforms are currently DBLP-specific")

    before = _feature_map_with_all_dblp_types(graph)
    features = _copy_features(before)
    name = str(transform_name)
    modified: set[int] = set()
    family = "raw"
    audit_extra: dict[str, Any] = {}

    if name in {"raw", "full-native", "export-full"}:
        pass
    elif name.startswith("inject-"):
        family = "inject"
        modified.update(_inject_transform(features, graph, name))
    elif name.startswith("zero-"):
        family = "zero"
        _reject_implicit_dimension_injection(name)
        modified.update(_zero_preserve_dim_transform(features, name))
    elif name in {"paper-PCA64", "paper-random-projection64", "paper-int8", "paper-fp16"}:
        family = "paper_projection" if ("PCA" in name or "projection" in name) else "paper_quantization"
        paper_type = DBLP_TYPE_BY_NAME["paper"]
        transformed, audit = transform_feature_matrix(features[paper_type], _paper_transform_name(name), seed=int(seed))
        features[paper_type] = transformed
        audit_extra.update(audit)
        modified.add(paper_type)
    else:
        raise ValueError(f"unsupported Gate21.6 feature transform: {name!r}")

    out = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=graph.relations,
        relation_specs=graph.relation_specs,
        features=features,
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
        partitions=None if graph.partitions is None else np.asarray(graph.partitions).copy(),
    )
    audit = _transform_audit(
        name=name,
        family=family,
        before=before,
        after=features,
        modified=modified,
        audit_extra=audit_extra,
    )
    object.__setattr__(out, "_gate21_6_features_before", before)
    object.__setattr__(out, "_gate21_6_features_after", features)
    object.__setattr__(out, "_gate21_6_transform_audit", audit)
    return out


def _zero_preserve_dim_transform(features: dict[int, np.ndarray], name: str) -> set[int]:
    modified: set[int] = set()
    if name == "zero-all-support-preserve-dim":
        target_names = SUPPORT_TYPES
    elif name == "zero-all-features-preserve-dim":
        target_names = set(DBLP_TYPE_BY_NAME)
    else:
        match = re.fullmatch(r"zero-(author|paper|term|venue)-preserve-dim", name)
        if not match:
            raise ValueError(f"unsupported zero preserve-dim transform: {name!r}")
        target_names = {match.group(1)}

    for type_name in target_names:
        type_id = DBLP_TYPE_BY_NAME[type_name]
        features[type_id] = np.zeros_like(features[type_id], dtype=np.float32)
        modified.add(type_id)
    return modified


def _inject_transform(features: dict[int, np.ndarray], graph: HeteroGraph, name: str) -> set[int]:
    zero_match = re.fullmatch(r"inject-zero-(author|paper|term|venue)-dim(\d+)", name)
    constant_match = re.fullmatch(r"inject-type-constant-(author|paper|term|venue)-dim(\d+)", name)
    match = zero_match or constant_match
    if not match:
        raise ValueError(f"unsupported inject transform: {name!r}")

    type_name = match.group(1)
    dim = int(match.group(2))
    if dim < 0:
        raise ValueError("inject transform dimension must be non-negative")
    type_id = DBLP_TYPE_BY_NAME[type_name]
    count = int(np.sum(graph.node_type == type_id))
    if zero_match:
        features[type_id] = np.zeros((count, dim), dtype=np.float32)
    else:
        features[type_id] = np.full((count, dim), _constant_for_type(type_id), dtype=np.float32)
    return {type_id}


def _reject_implicit_dimension_injection(name: str) -> None:
    if re.search(r"-dim\d+$", name):
        raise ValueError(f"dimension-changing feature injection must use an inject- prefix: {name!r}")


def _transform_audit(
    *,
    name: str,
    family: str,
    before: Mapping[int, np.ndarray],
    after: Mapping[int, np.ndarray],
    modified: set[int],
    audit_extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "transform_name": name,
        "feature_transform_family": family,
        "node_types_modified": ",".join(DBLP_NAME_BY_TYPE.get(int(t), str(t)) for t in sorted(modified)),
        "original_feature_shape_by_type": _shape_map(before),
        "transformed_feature_shape_by_type": _shape_map(after),
        "shape_preserved_by_type": _shape_preserved_map(before, after),
        "feature_transform_leakage_flag": False,
        "all_zero_feature_fraction_by_type": _all_zero_fraction_map(after),
        "feature_transform_fit_split": "train_only_or_unsupervised",
        "uses_test_data_for_transform": False,
        "fit_uses_labels": False,
        "fit_uses_test_labels": False,
        "fit_uses_validation_labels": False,
        "source_storage_dtype": str(audit_extra.get("feature_dtype", "fp32")),
        "model_input_dtype": "fp32",
        "feature_dim": _paper_dim(after),
        "sidecar_metadata_bytes": int(audit_extra.get("sidecar_metadata_bytes", 0) or 0),
        "metadata_keys": str(audit_extra.get("metadata_keys", "")),
    }


def _feature_map_with_all_dblp_types(graph: HeteroGraph) -> dict[int, np.ndarray]:
    features = _copy_features({} if graph.features is None else graph.features)
    for type_id in DBLP_NAME_BY_TYPE:
        if type_id not in features:
            count = int(np.sum(graph.node_type == int(type_id)))
            features[type_id] = np.zeros((count, 0), dtype=np.float32)
    return features


def _copy_features(features: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {int(k): np.asarray(v, dtype=np.float32).copy() for k, v in features.items()}


def _shape_map(features: Mapping[int, np.ndarray]) -> dict[str, list[int]]:
    return {DBLP_NAME_BY_TYPE.get(int(k), str(k)): list(np.asarray(v).shape) for k, v in sorted(features.items())}


def _shape_preserved_map(before: Mapping[int, np.ndarray], after: Mapping[int, np.ndarray]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for type_id in sorted(set(before) | set(after)):
        out[DBLP_NAME_BY_TYPE.get(int(type_id), str(type_id))] = tuple(np.asarray(before[type_id]).shape) == tuple(
            np.asarray(after[type_id]).shape
        )
    return out


def _all_zero_fraction_map(features: Mapping[int, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for type_id, values in sorted(features.items()):
        arr = np.asarray(values)
        out[DBLP_NAME_BY_TYPE.get(int(type_id), str(type_id))] = 1.0 if arr.size == 0 else float(np.mean(arr == 0))
    return out


def _paper_dim(features: Mapping[int, np.ndarray]) -> int:
    paper = features.get(DBLP_TYPE_BY_NAME["paper"])
    return int(paper.shape[1]) if paper is not None and paper.ndim == 2 else 0


def _constant_for_type(type_id: int) -> float:
    return float((int(type_id) + 1) / 10.0)


def _paper_transform_name(name: str) -> str:
    if name == "paper-PCA64":
        return "pca-paper-64"
    if name == "paper-random-projection64":
        return "random_projection_dim64"
    if name == "paper-int8":
        return "int8-paper"
    if name == "paper-fp16":
        return "fp16-paper"
    return str(name)
