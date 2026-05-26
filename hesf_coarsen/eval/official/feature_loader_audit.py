from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


FEATURE_LOADER_AUDIT_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "feature_transform_name",
    "feature_transform_written_hash",
    "feature_transform_loaded_hash",
    "feature_transform_applied_flag",
    "feature_transform_family",
    "node_types_modified",
    "node_type",
    "node_type_name",
    "feature_shape_before_transform",
    "feature_shape_after_transform",
    "feature_shape_after_loader",
    "feature_dtype_after_loader",
    "feature_dim_after_loader",
    "feature_l2_norm_after_loader",
    "feature_l1_sum_after_loader",
    "feature_mean_after_loader",
    "feature_std_after_loader",
    "feature_zero_fraction_after_loader",
    "feature_min_after_loader",
    "feature_max_after_loader",
    "source_storage_dtype",
    "model_input_dtype",
    "fit_uses_labels",
    "fit_uses_test_labels",
    "loader_uses_sidecar_flag",
    "loader_uses_text_node_dat_flag",
    "feature_loader_audit_pass",
]

DBLP_TYPE_NAMES = {0: "author", 1: "paper", 2: "term", 3: "venue"}


def feature_loader_audit_rows(
    *,
    dataset: str,
    method: str,
    canonical_method: str,
    graph_seed: int,
    training_seed: int,
    feature_transform_name: str,
    before_features: Mapping[int, np.ndarray] | None,
    after_features: Mapping[int, np.ndarray] | None,
    loaded_features: Mapping[int, np.ndarray] | None = None,
    fit_uses_labels: bool = False,
    fit_uses_test_labels: bool = False,
    loader_uses_sidecar_flag: bool = False,
    loader_uses_text_node_dat_flag: bool = True,
    feature_transform_family: str = "",
    node_types_modified: str = "",
    source_storage_dtype: str = "fp32",
    model_input_dtype: str = "fp32",
) -> list[dict[str, Any]]:
    before_features = before_features or {}
    after_features = after_features or {}
    loaded_features = loaded_features or after_features
    rows: list[dict[str, Any]] = []
    for node_type in sorted(set(before_features) | set(after_features) | set(loaded_features)):
        before = _array_or_empty(before_features.get(int(node_type)))
        after = _array_or_empty(after_features.get(int(node_type)))
        loaded = _array_or_empty(loaded_features.get(int(node_type), after))
        expected_hash = _array_hash(after)
        loaded_hash = _array_hash(loaded)
        stats = _feature_stats(loaded)
        row = {
            "dataset": str(dataset).upper(),
            "method": method,
            "canonical_method": canonical_method,
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "feature_transform_name": feature_transform_name,
            "feature_transform_written_hash": expected_hash,
            "feature_transform_loaded_hash": loaded_hash,
            "feature_transform_applied_flag": expected_hash == loaded_hash,
            "node_type": int(node_type),
            "node_type_name": _type_name(dataset, int(node_type)),
            "feature_transform_family": feature_transform_family,
            "node_types_modified": node_types_modified,
            "feature_shape_before_transform": json.dumps(list(before.shape)),
            "feature_shape_after_transform": json.dumps(list(after.shape)),
            "feature_shape_after_loader": json.dumps(list(loaded.shape)),
            "feature_dtype_after_loader": str(loaded.dtype),
            "feature_dim_after_loader": int(loaded.shape[1]) if loaded.ndim == 2 else 0,
            "source_storage_dtype": source_storage_dtype,
            "model_input_dtype": model_input_dtype,
            "fit_uses_labels": bool(fit_uses_labels),
            "fit_uses_test_labels": bool(fit_uses_test_labels),
            "loader_uses_sidecar_flag": bool(loader_uses_sidecar_flag),
            "loader_uses_text_node_dat_flag": bool(loader_uses_text_node_dat_flag),
            **stats,
        }
        row["feature_loader_audit_pass"] = _passes_transform_assertions(str(feature_transform_name), int(node_type), row)
        rows.append({field: row.get(field, "") for field in FEATURE_LOADER_AUDIT_FIELDS})
    return rows


def _array_or_empty(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _feature_stats(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return {
            "feature_l2_norm_after_loader": 0.0,
            "feature_l1_sum_after_loader": 0.0,
            "feature_mean_after_loader": 0.0,
            "feature_std_after_loader": 0.0,
            "feature_zero_fraction_after_loader": 1.0,
            "feature_min_after_loader": 0.0,
            "feature_max_after_loader": 0.0,
        }
    return {
        "feature_l2_norm_after_loader": float(np.linalg.norm(arr.ravel(), ord=2)),
        "feature_l1_sum_after_loader": float(np.abs(arr).sum()),
        "feature_mean_after_loader": float(arr.mean()),
        "feature_std_after_loader": float(arr.std()),
        "feature_zero_fraction_after_loader": float(np.mean(arr == 0.0)),
        "feature_min_after_loader": float(arr.min()),
        "feature_max_after_loader": float(arr.max()),
    }


def _passes_transform_assertions(transform: str, node_type: int, row: Mapping[str, Any]) -> bool:
    if transform == "zero-paper" and int(node_type) == 1:
        return float(row.get("feature_l2_norm_after_loader", 1.0)) <= 1e-8 and float(row.get("feature_zero_fraction_after_loader", 0.0)) >= 0.999999
    if transform == "pca-paper-128" and int(node_type) == 1:
        return int(row.get("feature_dim_after_loader", 0)) == 128
    if transform == "pca-paper-64" and int(node_type) == 1:
        return int(row.get("feature_dim_after_loader", 0)) == 64
    return bool(row.get("feature_transform_applied_flag", False))


def _type_name(dataset: str, node_type: int) -> str:
    if str(dataset).upper() == "DBLP":
        return DBLP_TYPE_NAMES.get(int(node_type), str(node_type))
    return str(node_type)
