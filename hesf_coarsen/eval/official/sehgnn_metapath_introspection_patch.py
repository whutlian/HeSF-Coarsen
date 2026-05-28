from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_CACHE_HASH_KEYS = ("preprocess_cache_hash_after", "cache_hash_after", "cache_hash", "tensor_hash")


def cache_hash_is_real(value: str | Mapping[str, Any]) -> bool:
    cache_hash = _extract_cache_hash(value) if isinstance(value, Mapping) else str(value)
    return bool(cache_hash) and cache_hash != EMPTY_SHA256


def compare_cache_hashes_for_perturbation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison_name: str,
    perturbation_expected_to_change: bool = True,
) -> dict[str, Any]:
    left_cache_hash = _extract_cache_hash(left)
    right_cache_hash = _extract_cache_hash(right)
    left_link_hash = str(left.get("link_dat_hash", ""))
    right_link_hash = str(right.get("link_dat_hash", ""))
    link_hash_differs = bool(left_link_hash and right_link_hash and left_link_hash != right_link_hash)
    cache_hash_differs = bool(left_cache_hash and right_cache_hash and left_cache_hash != right_cache_hash)

    failure_reasons: list[str] = []
    _append_hash_failures(failure_reasons, "left", left_cache_hash)
    _append_hash_failures(failure_reasons, "right", right_cache_hash)
    if _bool(left.get("fallback_loaded_relation_audit_used", False)) or _bool(right.get("fallback_loaded_relation_audit_used", False)):
        failure_reasons.append("fallback_loaded_relation_audit_used")
    if _bool(left.get("cache_hash_from_fallback", False)) or _bool(right.get("cache_hash_from_fallback", False)):
        failure_reasons.append("fallback_cache_hash_used")
    if perturbation_expected_to_change and not cache_hash_differs:
        failure_reasons.append("perturbation_cache_hash_unchanged")

    pass_flag = not failure_reasons
    return {
        "comparison_name": str(comparison_name),
        "left_method": left.get("method", ""),
        "right_method": right.get("method", ""),
        "left_link_dat_hash": left_link_hash,
        "right_link_dat_hash": right_link_hash,
        "link_hash_differs": link_hash_differs,
        "left_preprocess_cache_hash_after": left_cache_hash,
        "right_preprocess_cache_hash_after": right_cache_hash,
        "left_cache_hash_real": cache_hash_is_real(left_cache_hash),
        "right_cache_hash_real": cache_hash_is_real(right_cache_hash),
        "cache_hash_differs": cache_hash_differs,
        "perturbation_expected_to_change": bool(perturbation_expected_to_change),
        "CACHE_HASH_REAL_PASS": pass_flag,
        "cache_hash_real_pass": pass_flag,
        "failure_reasons": ";".join(failure_reasons),
    }


def cache_hash_comparison_pass(row: Mapping[str, Any]) -> bool:
    return bool(row.get("CACHE_HASH_REAL_PASS", row.get("cache_hash_real_pass", False)))


def metapath_introspection_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if _bool(row.get("fallback_loaded_relation_audit_used", False)):
            continue
        if _bool(row.get("introspection_supported", True)) is False:
            continue
        if _has_real_tensor_or_cache_key(row):
            return True
    return False


def _has_real_tensor_or_cache_key(row: Mapping[str, Any]) -> bool:
    key_fields = (
        "metapath_key",
        "label_feature_key",
        "cache_file_path",
        "sehgnn_generated_feature_cache_keys",
        "sehgnn_generated_label_cache_keys",
        "sehgnn_generated_metapath_keys",
    )
    if any(str(row.get(key, "")).strip() for key in key_fields):
        return True
    return cache_hash_is_real(row)


def _extract_cache_hash(row: Mapping[str, Any]) -> str:
    for key in _CACHE_HASH_KEYS:
        value = row.get(key, "")
        if value not in {"", None}:
            return str(value)
    return ""


def _append_hash_failures(failure_reasons: list[str], side: str, cache_hash: str) -> None:
    if not cache_hash:
        failure_reasons.append(f"{side}_cache_hash_empty")
    elif cache_hash == EMPTY_SHA256:
        failure_reasons.append(f"{side}_cache_hash_empty_sha256")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
