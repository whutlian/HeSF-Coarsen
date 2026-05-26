from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


METAPATH_CACHE_AUDIT_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "export_hash",
    "cache_namespace",
    "cache_reused_flag",
    "force_reprocess_flag",
    "metapath_key",
    "relation_sequence",
    "input_relation_ids",
    "input_relation_names",
    "feature_tensor_shape",
    "feature_tensor_nonzero_count",
    "feature_tensor_density",
    "feature_tensor_bytes",
    "feature_tensor_hash",
    "label_feature_key",
    "label_feature_shape",
    "label_feature_nonzero_count",
    "label_feature_bytes",
    "label_feature_hash",
    "cache_file_path",
    "cache_file_bytes",
    "introspection_supported",
    "introspection_failure_reason",
    "fallback_loaded_relation_audit_used",
]


def fallback_metapath_cache_row(cache_row: Mapping[str, Any], *, reason: str = "official_sehgnn_intermediate_tensors_not_exposed") -> dict[str, Any]:
    cache_dir = Path(str(cache_row.get("preprocess_cache_dir", "")))
    return {
        "dataset": cache_row.get("dataset", "DBLP"),
        "method": cache_row.get("method", ""),
        "graph_seed": cache_row.get("graph_seed", ""),
        "training_seed": cache_row.get("training_seed", ""),
        "export_hash": cache_row.get("export_file_list_hash", ""),
        "cache_namespace": str(cache_dir),
        "cache_reused_flag": cache_row.get("cache_reused_flag", False),
        "force_reprocess_flag": cache_row.get("force_reprocess_flag", False),
        "metapath_key": "",
        "relation_sequence": "",
        "input_relation_ids": "",
        "input_relation_names": "",
        "feature_tensor_shape": "",
        "feature_tensor_nonzero_count": "",
        "feature_tensor_density": "",
        "feature_tensor_bytes": "",
        "feature_tensor_hash": "",
        "label_feature_key": "",
        "label_feature_shape": "",
        "label_feature_nonzero_count": "",
        "label_feature_bytes": "",
        "label_feature_hash": "",
        "cache_file_path": "",
        "cache_file_bytes": "",
        "introspection_supported": False,
        "introspection_failure_reason": reason,
        "fallback_loaded_relation_audit_used": True,
    }


def cache_hash_comparison_row(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "left_method": left.get("method", ""),
        "right_method": right.get("method", ""),
        "left_link_dat_hash": left.get("link_dat_hash", ""),
        "right_link_dat_hash": right.get("link_dat_hash", ""),
        "link_hash_differs": left.get("link_dat_hash", "") != right.get("link_dat_hash", ""),
        "left_cache_hash": left.get("cache_hash_after", ""),
        "right_cache_hash": right.get("cache_hash_after", ""),
        "cache_hash_differs": left.get("cache_hash_after", "") != right.get("cache_hash_after", ""),
    }
