from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS = (
    "dataset",
    "method",
    "graph_export_hash",
    "adapter_package_sha256",
    "adapter_package_total_bytes",
    "sidecar_feature_bytes_total",
    "sidecar_feature_bytes_by_node_type",
    "projection_matrix_bytes",
    "projection_seed_bytes",
    "pca_basis_bytes",
    "pca_mean_bytes",
    "quantization_metadata_bytes",
    "node_id_mapping_bytes",
    "type_schema_bytes",
    "relation_schema_bytes",
    "label_split_bytes",
    "link_dat_bytes",
    "node_table_bytes_required_for_loader",
    "loader_config_bytes",
    "model_config_bytes",
    "readme_or_manifest_bytes",
    "excluded_bytes_with_reason",
)

_ARTIFACT_BYTE_FIELD_BY_NAME = {
    "projection_matrix": "projection_matrix_bytes",
    "projection_seed": "projection_seed_bytes",
    "pca_basis": "pca_basis_bytes",
    "pca_mean": "pca_mean_bytes",
    "quantization_metadata": "quantization_metadata_bytes",
    "node_id_mapping": "node_id_mapping_bytes",
    "type_schema": "type_schema_bytes",
    "relation_schema": "relation_schema_bytes",
    "label_split": "label_split_bytes",
    "link_dat": "link_dat_bytes",
    "node_table": "node_table_bytes_required_for_loader",
    "node_table_required_for_loader": "node_table_bytes_required_for_loader",
    "loader_config": "loader_config_bytes",
    "model_config": "model_config_bytes",
    "readme": "readme_or_manifest_bytes",
    "manifest": "readme_or_manifest_bytes",
    "adapter_manifest": "readme_or_manifest_bytes",
}


def build_adapter_manifest(
    *,
    dataset: str,
    method: str,
    graph_export_hash: str,
    included_artifacts: Mapping[str, str | Path],
    excluded_bytes_with_reason: Sequence[Mapping[str, Any]] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Gate21.6 adapter package manifest from concrete package artifacts."""

    manifest = _empty_manifest(
        dataset=dataset,
        method=method,
        graph_export_hash=graph_export_hash,
        excluded_bytes_with_reason=excluded_bytes_with_reason,
    )
    package_hash = hashlib.sha256()
    package_total = 0
    sidecar_total = 0
    sidecar_by_type: dict[str, int] = {}

    for artifact_name, path_like in sorted(included_artifacts.items()):
        path = Path(path_like)
        size = _file_size(path)
        package_total += size
        _hash_artifact(package_hash, artifact_name, path)

        if artifact_name.startswith("sidecar_feature:"):
            node_type = artifact_name.split(":", 1)[1]
            sidecar_by_type[node_type] = sidecar_by_type.get(node_type, 0) + size
            sidecar_total += size
            continue

        byte_field = _ARTIFACT_BYTE_FIELD_BY_NAME.get(artifact_name)
        if byte_field is None:
            raise ValueError(f"unsupported adapter artifact name: {artifact_name!r}")
        manifest[byte_field] = int(manifest[byte_field]) + size

    manifest["adapter_package_total_bytes"] = int(package_total)
    manifest["adapter_package_sha256"] = package_hash.hexdigest()
    manifest["sidecar_feature_bytes_total"] = int(sidecar_total)
    manifest["sidecar_feature_bytes_by_node_type"] = dict(sorted(sidecar_by_type.items()))
    if extra_fields:
        manifest.update(dict(extra_fields))
    manifest["manifest_required_fields_present"] = _required_fields_present(manifest)
    manifest["adapter_manifest_complete"] = bool(
        manifest["manifest_required_fields_present"] and isinstance(manifest["excluded_bytes_with_reason"], list)
    )
    return manifest


def write_adapter_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _empty_manifest(
    *,
    dataset: str,
    method: str,
    graph_export_hash: str,
    excluded_bytes_with_reason: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "dataset": str(dataset),
        "method": str(method),
        "graph_export_hash": str(graph_export_hash),
        "adapter_package_sha256": "",
        "adapter_package_total_bytes": 0,
        "sidecar_feature_bytes_total": 0,
        "sidecar_feature_bytes_by_node_type": {},
        "excluded_bytes_with_reason": [dict(item) for item in (excluded_bytes_with_reason or [])],
    }
    for field in GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS:
        if "bytes" in field and field not in manifest:
            manifest[field] = 0
    return manifest


def _file_size(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(path.stat().st_size)


def _hash_artifact(package_hash: "hashlib._Hash", artifact_name: str, path: Path) -> None:
    package_hash.update(str(artifact_name).encode("utf-8"))
    package_hash.update(b"\0")
    package_hash.update(path.read_bytes())
    package_hash.update(b"\0")


def _required_fields_present(manifest: Mapping[str, Any]) -> bool:
    return all(field in manifest for field in GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS)


def aggregate_adapter_by_method_gate21_10(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("base_method", row.get("base_graph_method", ""))), str(row.get("adapter_method", row.get("feature_adapter", row.get("adapter_name", "")))))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (base_method, adapter_method), group_rows in sorted(grouped.items()):
        valid = [row for row in group_rows if _gate21_10_row_success(row)]
        out.append(
            {
                "base_method": base_method,
                "adapter_method": adapter_method,
                "method": f"{base_method}+{adapter_method}",
                "success_count": len(valid),
                "row_count": len(group_rows),
                "failed_rows_excluded": len(group_rows) - len(valid),
                "test_micro_f1_mean": _gate21_10_mean(valid, "test_micro_f1"),
                "test_macro_f1_mean": _gate21_10_mean(valid, "test_macro_f1"),
                "static_inference_package_ratio_mean": _gate21_10_mean(valid, "static_inference_package_ratio"),
                "transform_recipe_package_ratio_mean": _gate21_10_mean(valid, "transform_recipe_package_ratio"),
                "reconstructable_package_ratio_mean": _gate21_10_mean(valid, "reconstructable_package_ratio"),
                "eligible_for_adapter_table": True,
                "eligible_for_official_main_table": False,
            }
        )
    return out


def _gate21_10_row_success(row: Mapping[str, Any]) -> bool:
    return _gate21_10_bool(row.get("success", True)) and _gate21_10_bool(row.get("training_executed", True)) and _gate21_10_valid_number(row.get("static_inference_package_ratio"))


def _gate21_10_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_gate21_10_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "" if not finite else sum(finite) / len(finite)


def _gate21_10_valid_number(value: Any) -> bool:
    parsed = _gate21_10_float(value)
    return parsed is not None and parsed >= 0 and parsed not in {10240.0, 1000000.0}


def _gate21_10_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _gate21_10_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
