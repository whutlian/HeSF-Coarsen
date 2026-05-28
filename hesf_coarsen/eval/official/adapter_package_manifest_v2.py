from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


GATE21_7_ADAPTER_MANIFEST_REQUIRED_FIELDS = (
    "method",
    "feature_adapter",
    "package_type",
    "static_snapshot_package_total_bytes",
    "reproducible_transform_package_total_bytes",
    "native_full_text_total_bytes",
    "static_snapshot_package_ratio",
    "reproducible_transform_package_ratio",
    "link_dat_bytes",
    "node_id_mapping_bytes",
    "type_schema_bytes",
    "relation_schema_bytes",
    "label_split_bytes",
    "loader_config_bytes",
    "sidecar_feature_bytes_total",
    "sidecar_feature_bytes_by_node_type",
    "projection_seed_bytes",
    "projection_generator_name",
    "projection_generator_version",
    "projection_dtype",
    "projection_input_dim",
    "projection_output_dim",
    "projection_matrix_bytes",
    "pca_basis_bytes",
    "pca_mean_bytes",
    "pca_dtype",
    "quantization_scale_bytes",
    "quantization_zero_point_bytes",
    "quantization_metadata_bytes",
    "adapter_manifest_complete",
    "static_snapshot_package_complete",
    "reproducible_transform_package_complete",
    "eligible_for_official_main_table",
    "eligible_for_adapter_table",
)

STATIC_SNAPSHOT_REQUIRED_FIELDS = (
    "static_snapshot_package_total_bytes",
    "native_full_text_total_bytes",
    "link_dat_bytes",
    "node_id_mapping_bytes",
    "type_schema_bytes",
    "relation_schema_bytes",
    "label_split_bytes",
    "loader_config_bytes",
    "sidecar_feature_bytes_total",
)

RANDOM_PROJECTION_REPRODUCIBLE_FIELDS = (
    "projection_seed_bytes",
    "projection_generator_name",
    "projection_generator_version",
    "projection_dtype",
    "projection_input_dim",
    "projection_output_dim",
    "projection_matrix_bytes",
)

PCA_REPRODUCIBLE_FIELDS = (
    "pca_basis_bytes",
    "pca_mean_bytes",
    "pca_dtype",
)

INT8_REPRODUCIBLE_FIELDS = (
    "quantization_scale_bytes",
    "quantization_zero_point_bytes",
    "quantization_metadata_bytes",
)


def build_adapter_manifest_v2(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and evaluate a Gate21.7 adapter manifest mapping."""

    return evaluate_adapter_manifest_v2(manifest)


def evaluate_adapter_manifest_v2(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fill computed Gate21.7 adapter package completeness flags."""

    result = _with_defaults(manifest)
    adapter = _adapter_name(result)

    static_missing = [field for field in STATIC_SNAPSHOT_REQUIRED_FIELDS if not _present(result, field)]
    result["missing_static_fields"] = static_missing
    result["static_snapshot_package_complete"] = not static_missing

    reproducible_missing = _adapter_reproducible_missing_fields(result, adapter)
    if _int(result.get("reproducible_transform_package_total_bytes")) <= 0:
        reproducible_missing.append("reproducible_transform_package_total_bytes")
    if static_missing:
        reproducible_missing.extend(field for field in static_missing if field not in reproducible_missing)
    result["missing_reproducible_fields"] = reproducible_missing
    result["reproducible_transform_package_complete"] = not reproducible_missing

    native_bytes = _int(result.get("native_full_text_total_bytes"))
    if native_bytes > 0:
        result["static_snapshot_package_ratio"] = _ratio_if_missing(
            result.get("static_snapshot_package_ratio"),
            _int(result.get("static_snapshot_package_total_bytes")),
            native_bytes,
        )
        result["reproducible_transform_package_ratio"] = _ratio_if_missing(
            result.get("reproducible_transform_package_ratio"),
            _int(result.get("reproducible_transform_package_total_bytes")),
            native_bytes,
        )

    result["manifest_required_fields_present"] = all(field in result for field in GATE21_7_ADAPTER_MANIFEST_REQUIRED_FIELDS)
    result["adapter_manifest_complete"] = bool(
        result["static_snapshot_package_complete"]
        and (result["reproducible_transform_package_complete"] or bool(str(result.get("missing_reason", "")).strip()))
    )

    if adapter not in {"", "raw", "none"}:
        result["eligible_for_official_main_table"] = False
    result["eligible_for_adapter_table"] = _bool(result.get("eligible_for_adapter_table", True))
    return result


def write_adapter_manifest_v2(manifest: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _adapter_reproducible_missing_fields(manifest: Mapping[str, Any], adapter: str) -> list[str]:
    if "random_projection" in adapter:
        return [field for field in RANDOM_PROJECTION_REPRODUCIBLE_FIELDS if not _present(manifest, field)]
    if "pca" in adapter or "svd" in adapter:
        return [field for field in PCA_REPRODUCIBLE_FIELDS if not _present(manifest, field)]
    if "int8" in adapter:
        return [field for field in INT8_REPRODUCIBLE_FIELDS if not _present(manifest, field)]
    if "fp16" in adapter:
        return ["projection_dtype"] if not _present(manifest, "projection_dtype") else []
    return []


def _with_defaults(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    for field in GATE21_7_ADAPTER_MANIFEST_REQUIRED_FIELDS:
        if field in result:
            continue
        if field.endswith("_bytes") or field.endswith("_dim") or field.endswith("_total_bytes"):
            result[field] = 0
        elif field.endswith("_ratio"):
            result[field] = None
        elif field.endswith("_complete") or field.startswith("eligible_for_"):
            result[field] = False
        elif field == "sidecar_feature_bytes_by_node_type":
            result[field] = {}
        else:
            result[field] = ""
    return result


def _adapter_name(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("feature_adapter", manifest.get("adapter", manifest.get("method", "")))).strip().lower()


def _present(manifest: Mapping[str, Any], field: str) -> bool:
    value = manifest.get(field)
    if field.endswith("_bytes") or field.endswith("_dim") or field.endswith("_total_bytes"):
        return _int(value) > 0
    if field.endswith("_by_node_type"):
        return isinstance(value, Mapping) and bool(value)
    return bool(str(value).strip())


def _ratio_if_missing(current: Any, numerator: int, denominator: int) -> float:
    try:
        if current not in {"", None}:
            return float(current)
    except (TypeError, ValueError):
        pass
    return float(numerator) / float(denominator)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _int(value: Any) -> int:
    if value in {"", None}:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
