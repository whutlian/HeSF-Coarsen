from __future__ import annotations

from typing import Any

from hesf_coarsen.eval.official.gate21_9_decision import EMPTY_SHA256


def unsupported_tensor_dump_row(*, dataset: str, method: str, metapath_key: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "metapath_key": metapath_key,
        "feature_tensor_hash": EMPTY_SHA256,
        "feature_tensor_bytes": 0,
        "real_tensor_dumped": False,
        "introspection_supported": False,
        "failure_type": "official_sehgnn_intermediate_tensors_not_exposed",
        "failure_message": "No patched SeHGNN tensor dump was available for this run.",
    }


def metapath_tensor_pass(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        bool(str(row.get("feature_tensor_hash", "")).strip())
        and str(row.get("feature_tensor_hash", "")).lower() != EMPTY_SHA256
        and float(row.get("feature_tensor_bytes", 0) or 0) > 0
        and str(row.get("introspection_supported", "")).lower() in {"1", "true", "yes", "y"}
        for row in rows
    )


def cache_hash_pass(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        str(row.get("assertion_pass", "")).lower() in {"1", "true", "yes", "y"}
        and str(row.get("cache_hash_non_empty", "")).lower() in {"1", "true", "yes", "y"}
        for row in rows
    )
