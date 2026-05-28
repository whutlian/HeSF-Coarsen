from __future__ import annotations

import importlib


def _evaluate_adapter_manifest_v2():
    spec = importlib.util.find_spec("hesf_coarsen.eval.official.adapter_package_manifest_v2")
    assert spec is not None, "adapter_package_manifest_v2 module must exist"
    module = importlib.import_module("hesf_coarsen.eval.official.adapter_package_manifest_v2")
    fn = getattr(module, "evaluate_adapter_manifest_v2", None)
    assert callable(fn), "evaluate_adapter_manifest_v2 must be reusable by runners/summarizers"
    return fn


def _pca_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "method": "HeSF-RCS-APV12+pca_svd_dim64",
        "feature_adapter": "pca_svd_dim64",
        "package_type": "reproducible_transform_package",
        "static_snapshot_package_total_bytes": 4096,
        "reproducible_transform_package_total_bytes": 8192,
        "native_full_text_total_bytes": 16384,
        "link_dat_bytes": 1024,
        "node_id_mapping_bytes": 128,
        "type_schema_bytes": 64,
        "relation_schema_bytes": 64,
        "label_split_bytes": 128,
        "loader_config_bytes": 128,
        "sidecar_feature_bytes_total": 2048,
        "sidecar_feature_bytes_by_node_type": {"paper": 2048},
        "pca_basis_bytes": 4096,
        "pca_mean_bytes": 256,
        "pca_dtype": "float32",
        "eligible_for_official_main_table": False,
        "eligible_for_adapter_table": True,
    }
    manifest.update(overrides)
    return manifest


def test_pca_reproducible_package_is_incomplete_without_basis_bytes() -> None:
    evaluate_adapter_manifest_v2 = _evaluate_adapter_manifest_v2()

    evaluated = evaluate_adapter_manifest_v2(_pca_manifest(pca_basis_bytes=0))

    assert evaluated["static_snapshot_package_complete"] is True
    assert evaluated["reproducible_transform_package_complete"] is False
    assert "pca_basis_bytes" in evaluated["missing_reproducible_fields"]


def test_pca_reproducible_package_is_incomplete_without_mean_bytes() -> None:
    evaluate_adapter_manifest_v2 = _evaluate_adapter_manifest_v2()

    evaluated = evaluate_adapter_manifest_v2(_pca_manifest(pca_mean_bytes=0))

    assert evaluated["static_snapshot_package_complete"] is True
    assert evaluated["reproducible_transform_package_complete"] is False
    assert "pca_mean_bytes" in evaluated["missing_reproducible_fields"]
