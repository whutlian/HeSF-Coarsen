from __future__ import annotations


def test_metapath_tensor_pass_requires_real_nonempty_hashes() -> None:
    from hesf_coarsen.eval.official.sehgnn_metapath_tensor_dump import cache_hash_pass, metapath_tensor_pass

    assert metapath_tensor_pass([{"method": "APV12", "feature_tensor_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "feature_tensor_bytes": 10}]) is False
    assert metapath_tensor_pass([{"method": "APV12", "feature_tensor_hash": "abc123", "feature_tensor_bytes": 10, "introspection_supported": True}]) is True
    assert cache_hash_pass([{"cache_hash_non_empty": False, "assertion_pass": True}]) is False
    assert cache_hash_pass([{"cache_hash_non_empty": True, "cache_hash_differs": True, "assertion_pass": True}]) is True
