from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix
from hesf_coarsen.io.schema import HeteroGraph


DBLP_TYPE_BY_NAME = {"author": 0, "paper": 1, "term": 2, "venue": 3}
DBLP_NAME_BY_TYPE = {value: key for key, value in DBLP_TYPE_BY_NAME.items()}
SUPPORT_TYPE_NAMES = {"paper", "term", "venue"}

REQUIRED_FEATURE_TRANSFORMS = [
    "raw",
    "zero-author-preserve-dim",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-venue-preserve-dim",
    "zero-all-support-preserve-dim",
    "zero-all-features-preserve-dim",
    "paper-only-preserve-original-dims",
    "term-only-preserve-original-dims",
    "venue-only-preserve-original-dims",
    "paper-random-projection64",
    "paper-pca64",
    "inject-zero-venue-dim4231",
]


def apply_repaired_feature_transform(
    graph: HeteroGraph,
    transform_name: str,
    *,
    dataset: str = "DBLP",
    seed: int = 1,
) -> HeteroGraph:
    """Apply Gate21.7 repaired feature ablations without non-inject shape drift."""

    if str(dataset).upper() != "DBLP":
        raise ValueError("Gate21.7 repaired feature ablations are currently DBLP-specific")

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
        audit_extra["diagnostic_only"] = True
    elif name.startswith("zero-"):
        family = "zero"
        _reject_noninject_dimension_token(name)
        modified.update(_zero_preserve_dim_transform(features, name))
    elif name in {
        "paper-only-preserve-original-dims",
        "term-only-preserve-original-dims",
        "venue-only-preserve-original-dims",
    }:
        family = "only_preserve_original_dims"
        modified.update(_only_preserve_original_dims(features, name))
    elif name in {"paper-random-projection64", "paper-pca64"}:
        family = "shape_safe_adapter_projection"
        modified.update(_shape_safe_paper_projection(features, name, seed=int(seed), audit_extra=audit_extra))
    else:
        _reject_noninject_dimension_token(name)
        raise ValueError(f"unsupported Gate21.7 repaired feature transform: {name!r}")

    assertions = feature_shape_assertion_rows(name, before=before, after=features)
    if not feature_ablation_shape_safe_pass(assertions):
        failing = [str(row["node_type_name"]) for row in assertions if not bool(row["pass"])]
        raise ValueError(f"feature transform changed forbidden dimensions for {name!r}: {','.join(failing)}")

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
        assertion_rows=assertions,
        audit_extra=audit_extra,
    )
    object.__setattr__(out, "_gate21_7_features_before", before)
    object.__setattr__(out, "_gate21_7_features_after", features)
    object.__setattr__(out, "_gate21_7_transform_audit", audit)
    return out


def feature_shape_assertion_rows(
    transform_name: str,
    *,
    before: Mapping[int, np.ndarray],
    after: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    name = str(transform_name)
    allow_dim_change = name.startswith("inject-")
    rows: list[dict[str, Any]] = []
    for type_id in sorted(set(before) | set(after)):
        before_arr = np.asarray(before[type_id])
        after_arr = np.asarray(after[type_id])
        before_rows = int(before_arr.shape[0]) if before_arr.ndim >= 1 else 0
        after_rows = int(after_arr.shape[0]) if after_arr.ndim >= 1 else 0
        before_dim = int(before_arr.shape[1]) if before_arr.ndim == 2 else 0
        after_dim = int(after_arr.shape[1]) if after_arr.ndim == 2 else 0
        rows_match = before_rows == after_rows
        dims_match = before_dim == after_dim
        pass_flag = bool(rows_match and (dims_match or allow_dim_change))
        rows.append(
            {
                "transform_name": name,
                "node_type": int(type_id),
                "node_type_name": DBLP_NAME_BY_TYPE.get(int(type_id), str(type_id)),
                "before_rows": before_rows,
                "before_dim": before_dim,
                "after_rows": after_rows,
                "after_dim": after_dim,
                "shape_preserved": bool(rows_match and dims_match),
                "dimension_change_allowed": bool(allow_dim_change),
                "pass": pass_flag,
                "failure_message": "" if pass_flag else "non-inject transforms must preserve row count and feature dimension",
            }
        )
    return rows


def feature_ablation_shape_safe_pass(assertion_rows: list[Mapping[str, Any]]) -> bool:
    return bool(assertion_rows) and all(bool(row.get("pass", False)) for row in assertion_rows)


def _zero_preserve_dim_transform(features: dict[int, np.ndarray], name: str) -> set[int]:
    if name == "zero-all-support-preserve-dim":
        target_names = SUPPORT_TYPE_NAMES
    elif name == "zero-all-features-preserve-dim":
        target_names = set(DBLP_TYPE_BY_NAME)
    else:
        match = re.fullmatch(r"zero-(author|paper|term|venue)-preserve-dim", name)
        if not match:
            raise ValueError(f"unsupported zero preserve-dim transform: {name!r}")
        target_names = {match.group(1)}
    return _zero_type_names(features, target_names)


def _only_preserve_original_dims(features: dict[int, np.ndarray], name: str) -> set[int]:
    match = re.fullmatch(r"(paper|term|venue)-only-preserve-original-dims", name)
    if not match:
        raise ValueError(f"unsupported preserve-original-dims transform: {name!r}")
    keep_name = match.group(1)
    zero_names = set(DBLP_TYPE_BY_NAME) - {keep_name}
    return _zero_type_names(features, zero_names)


def _zero_type_names(features: dict[int, np.ndarray], target_names: set[str]) -> set[int]:
    modified: set[int] = set()
    for type_name in target_names:
        type_id = DBLP_TYPE_BY_NAME[type_name]
        features[type_id] = np.zeros_like(features[type_id], dtype=np.float32)
        modified.add(type_id)
    return modified


def _shape_safe_paper_projection(
    features: dict[int, np.ndarray],
    name: str,
    *,
    seed: int,
    audit_extra: dict[str, Any],
) -> set[int]:
    paper_type = DBLP_TYPE_BY_NAME["paper"]
    original = np.asarray(features[paper_type], dtype=np.float32)
    transform = "random_projection_dim64" if name == "paper-random-projection64" else "pca-paper-64"
    projected, audit = transform_feature_matrix(original, transform, seed=int(seed))
    restored = _restore_original_dim(projected, original_dim=int(original.shape[1]))
    features[paper_type] = restored.astype(np.float32, copy=False)
    audit_extra.update(audit)
    audit_extra.update(
        {
            "internal_projection_dim": 64,
            "shape_safe_adapter_diagnostic": True,
            "compressed_feature_shape": list(projected.shape),
            "model_input_shape": list(restored.shape),
        }
    )
    return {paper_type}


def _restore_original_dim(values: np.ndarray, *, original_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("shape-safe projection expects a 2D matrix")
    current_dim = int(arr.shape[1])
    target_dim = int(original_dim)
    if current_dim == target_dim:
        return arr.copy()
    if current_dim > target_dim:
        return arr[:, :target_dim].copy()
    pad = np.zeros((arr.shape[0], target_dim - current_dim), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


def _inject_transform(features: dict[int, np.ndarray], graph: HeteroGraph, name: str) -> set[int]:
    match = re.fullmatch(r"inject-zero-(author|paper|term|venue)-dim(\d+)", name)
    if not match:
        raise ValueError(f"unsupported inject transform: {name!r}")

    type_name = match.group(1)
    dim = int(match.group(2))
    if dim < 0:
        raise ValueError("inject transform dimension must be non-negative")
    type_id = DBLP_TYPE_BY_NAME[type_name]
    count = int(np.sum(graph.node_type == type_id))
    features[type_id] = np.zeros((count, dim), dtype=np.float32)
    return {type_id}


def _reject_noninject_dimension_token(name: str) -> None:
    if re.search(r"-dim\d+$", str(name)):
        raise ValueError(f"dimension-changing feature injection must use an inject- prefix: {name!r}")


def _transform_audit(
    *,
    name: str,
    family: str,
    before: Mapping[int, np.ndarray],
    after: Mapping[int, np.ndarray],
    modified: set[int],
    assertion_rows: list[Mapping[str, Any]],
    audit_extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "transform_name": name,
        "feature_transform_family": family,
        "node_types_modified": ",".join(DBLP_NAME_BY_TYPE.get(int(t), str(t)) for t in sorted(modified)),
        "original_feature_shape_by_type": _shape_map(before),
        "transformed_feature_shape_by_type": _shape_map(after),
        "shape_assertion_rows": [dict(row) for row in assertion_rows],
        "FEATURE_ABLATION_SHAPE_SAFE_PASS": feature_ablation_shape_safe_pass(assertion_rows),
        "feature_transform_leakage_flag": False,
        "feature_transform_fit_split": "train_only_or_unsupervised",
        "uses_test_data_for_transform": False,
        "fit_uses_labels": False,
        "fit_uses_validation_labels": False,
        "fit_uses_test_labels": False,
        "internal_projection_dim": int(audit_extra.get("internal_projection_dim", 0) or 0),
        "shape_safe_adapter_diagnostic": bool(audit_extra.get("shape_safe_adapter_diagnostic", False)),
        "diagnostic_only": bool(audit_extra.get("diagnostic_only", False)),
        "compressed_feature_shape": list(audit_extra.get("compressed_feature_shape", [])),
        "model_input_shape": list(audit_extra.get("model_input_shape", [])),
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
