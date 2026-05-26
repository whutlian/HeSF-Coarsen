from __future__ import annotations

from typing import Any, Mapping


FEATURE_CACHE_PROBE_FIELDS = [
    "dataset",
    "method",
    "base_graph_method",
    "graph_seed",
    "training_seed",
    "feature_compression_method",
    "feature_dtype",
    "feature_dim",
    "feature_storage_ratio",
    "raw_hgb_byte_ratio",
    "binary_feature_sidecar_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "node_dat_bytes",
    "link_dat_bytes",
    "feature_sidecar_bytes",
    "export_total_bytes",
    "native_full_total_bytes",
    "train_time_seconds",
    "train_time_ratio",
    "peak_memory_mb",
    "peak_memory_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "adapter_family",
]


def planned_feature_cache_probe_rows(
    *,
    dataset: str,
    base_graph_method: str,
    graph_seed: int,
    training_seed: int,
    storage_row: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    storage_row = storage_row or {}
    methods = [
        ("raw_features", "fp32", ""),
        ("fp16_node_features", "fp16", ""),
        ("int8_per_feature", "int8", ""),
        ("pca_svd_dim256", "fp32", 256),
        ("pca_svd_dim128", "fp32", 128),
        ("pca_svd_dim64", "fp32", 64),
        ("propagated_cache_fp16", "fp16", ""),
        ("propagated_cache_int8", "int8", ""),
    ]
    rows: list[dict[str, Any]] = []
    raw_ratio = _float(storage_row.get("hgb_raw_file_byte_ratio"))
    node_bytes = _int(storage_row.get("node_dat_bytes"))
    native_total = _int(storage_row.get("native_full_total_bytes"))
    for name, dtype, dim in methods:
        if name == "raw_features":
            feature_ratio = 1.0
        elif dtype == "fp16":
            feature_ratio = 0.5
        elif dtype == "int8":
            feature_ratio = 0.25
        elif dim:
            feature_ratio = min(1.0, float(dim) / 334.0)
        else:
            feature_ratio = ""
        sidecar = int(node_bytes * feature_ratio) if isinstance(feature_ratio, float) else ""
        rows.append(
            {
                "dataset": str(dataset).upper(),
                "method": "SeHGNN-feature-compressed-adapter",
                "base_graph_method": str(base_graph_method),
                "graph_seed": int(graph_seed),
                "training_seed": int(training_seed),
                "feature_compression_method": name,
                "feature_dtype": dtype,
                "feature_dim": dim,
                "feature_storage_ratio": feature_ratio,
                "raw_hgb_byte_ratio": raw_ratio if raw_ratio is not None else "",
                "binary_feature_sidecar_byte_ratio": "" if native_total <= 0 or sidecar == "" else float(sidecar / native_total),
                "preprocessed_cache_byte_ratio": "",
                "node_dat_bytes": node_bytes,
                "link_dat_bytes": _int(storage_row.get("link_dat_bytes")),
                "feature_sidecar_bytes": sidecar,
                "export_total_bytes": _int(storage_row.get("export_total_bytes")),
                "native_full_total_bytes": native_total,
                "train_time_seconds": "",
                "train_time_ratio": "",
                "peak_memory_mb": "",
                "peak_memory_ratio": "",
                "test_micro_f1": "",
                "test_macro_f1": "",
                "validation_micro_f1": "",
                "validation_macro_f1": "",
                "official_sehgnn_unmodified": False,
                "eligible_for_main_decision": False,
                "adapter_family": "feature_cache_compression",
            }
        )
    return rows


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    if value in {"", None}:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
