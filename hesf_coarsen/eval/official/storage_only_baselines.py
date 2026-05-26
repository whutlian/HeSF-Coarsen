from __future__ import annotations

from typing import Any


STORAGE_ONLY_FIELDS = [
    "dataset",
    "artifact_name",
    "artifact_family",
    "changes_training_semantics",
    "requires_loader_adapter",
    "raw_hgb_text_bytes",
    "gzip_bytes",
    "zstd_bytes",
    "binary_relation_bytes",
    "binary_feature_bytes",
    "metadata_bytes",
    "total_artifact_bytes",
    "artifact_ratio_vs_native_full_text",
    "read_time_seconds",
    "write_time_seconds",
    "loader_supported",
    "notes",
]


def build_storage_only_row(
    *,
    dataset: str,
    artifact_name: str,
    native_full_text_bytes: int,
    total_artifact_bytes: int,
    changes_training_semantics: bool,
    requires_loader_adapter: bool,
    raw_hgb_text_bytes: int | None = None,
    gzip_bytes: int | None = None,
    zstd_bytes: int | None = None,
    binary_relation_bytes: int | None = None,
    binary_feature_bytes: int | None = None,
    metadata_bytes: int | None = None,
    read_time_seconds: float | None = None,
    write_time_seconds: float | None = None,
    loader_supported: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    native = int(native_full_text_bytes)
    if native <= 0:
        raise ValueError("native_full_text_bytes must be positive")
    row = {
        "dataset": str(dataset),
        "artifact_name": str(artifact_name),
        "artifact_family": "storage_only",
        "changes_training_semantics": bool(changes_training_semantics),
        "requires_loader_adapter": bool(requires_loader_adapter),
        "raw_hgb_text_bytes": raw_hgb_text_bytes,
        "gzip_bytes": gzip_bytes,
        "zstd_bytes": zstd_bytes,
        "binary_relation_bytes": binary_relation_bytes,
        "binary_feature_bytes": binary_feature_bytes,
        "metadata_bytes": metadata_bytes,
        "total_artifact_bytes": int(total_artifact_bytes),
        "artifact_ratio_vs_native_full_text": float(int(total_artifact_bytes) / native),
        "read_time_seconds": read_time_seconds,
        "write_time_seconds": write_time_seconds,
        "loader_supported": loader_supported,
        "notes": str(notes),
    }
    return {field: row.get(field) for field in STORAGE_ONLY_FIELDS}
