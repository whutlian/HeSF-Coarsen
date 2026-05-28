from __future__ import annotations

import hashlib


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_empty_sha256_cache_hash_cannot_pass_cache_hash_real_gate() -> None:
    from hesf_coarsen.eval.official.sehgnn_metapath_introspection_patch import (
        EMPTY_SHA256,
        compare_cache_hashes_for_perturbation,
    )

    row = compare_cache_hashes_for_perturbation(
        {
            "method": "base",
            "link_dat_hash": _sha(b"link-a"),
            "preprocess_cache_hash_after": EMPTY_SHA256,
        },
        {
            "method": "perturbed",
            "link_dat_hash": _sha(b"link-b"),
            "preprocess_cache_hash_after": _sha(b"cache-b"),
        },
        comparison_name="same_node_different_link",
    )

    assert row["cache_hash_differs"] is True
    assert row["CACHE_HASH_REAL_PASS"] is False
    assert "left_cache_hash_empty_sha256" in row["failure_reasons"]


def test_fallback_loaded_relation_cache_audit_cannot_pass() -> None:
    from hesf_coarsen.eval.official.sehgnn_metapath_introspection_patch import compare_cache_hashes_for_perturbation

    row = compare_cache_hashes_for_perturbation(
        {
            "method": "fallback-base",
            "link_dat_hash": _sha(b"link-a"),
            "cache_hash_after": _sha(b"cache-a"),
            "fallback_loaded_relation_audit_used": True,
        },
        {
            "method": "fallback-perturbed",
            "link_dat_hash": _sha(b"link-b"),
            "cache_hash_after": _sha(b"cache-b"),
            "fallback_loaded_relation_audit_used": False,
        },
        comparison_name="fallback_is_not_real_cache",
    )

    assert row["CACHE_HASH_REAL_PASS"] is False
    assert "fallback_loaded_relation_audit_used" in row["failure_reasons"]


def test_link_perturbation_with_unchanged_cache_hash_cannot_pass() -> None:
    from hesf_coarsen.eval.official.sehgnn_metapath_introspection_patch import compare_cache_hashes_for_perturbation

    cache_hash = _sha(b"unchanged-real-cache")
    row = compare_cache_hashes_for_perturbation(
        {
            "method": "base",
            "link_dat_hash": _sha(b"link-a"),
            "cache_hash_after": cache_hash,
        },
        {
            "method": "perturbed",
            "link_dat_hash": _sha(b"link-b"),
            "cache_hash_after": cache_hash,
        },
        comparison_name="same_node_different_link",
    )

    assert row["link_hash_differs"] is True
    assert row["cache_hash_differs"] is False
    assert row["CACHE_HASH_REAL_PASS"] is False
    assert "perturbation_cache_hash_unchanged" in row["failure_reasons"]
