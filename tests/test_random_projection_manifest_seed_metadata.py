from __future__ import annotations

import importlib

import pytest


def _evaluate_adapter_manifest_v2():
    spec = importlib.util.find_spec("hesf_coarsen.eval.official.adapter_package_manifest_v2")
    assert spec is not None, "adapter_package_manifest_v2 module must exist"
    module = importlib.import_module("hesf_coarsen.eval.official.adapter_package_manifest_v2")
    fn = getattr(module, "evaluate_adapter_manifest_v2", None)
    assert callable(fn), "evaluate_adapter_manifest_v2 must be reusable by runners/summarizers"
    return fn


def _random_projection_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "method": "HeSF-RCS-APV12+random_projection_dim64",
        "feature_adapter": "random_projection_dim64",
        "package_type": "reproducible_transform_package",
        "static_snapshot_package_total_bytes": 4096,
        "reproducible_transform_package_total_bytes": 4608,
        "native_full_text_total_bytes": 16384,
        "link_dat_bytes": 1024,
        "node_id_mapping_bytes": 128,
        "type_schema_bytes": 64,
        "relation_schema_bytes": 64,
        "label_split_bytes": 128,
        "loader_config_bytes": 128,
        "sidecar_feature_bytes_total": 2048,
        "sidecar_feature_bytes_by_node_type": {"paper": 2048},
        "projection_seed_bytes": 8,
        "projection_generator_name": "numpy.random.default_rng",
        "projection_generator_version": "1.26.4",
        "projection_dtype": "float32",
        "projection_input_dim": 4231,
        "projection_output_dim": 64,
        "projection_matrix_bytes": 1083136,
        "eligible_for_official_main_table": False,
        "eligible_for_adapter_table": True,
    }
    manifest.update(overrides)
    return manifest


def test_random_projection_reproducible_package_requires_seed_generator_dtype_shape_and_matrix() -> None:
    evaluate_adapter_manifest_v2 = _evaluate_adapter_manifest_v2()

    evaluated = evaluate_adapter_manifest_v2(_random_projection_manifest())

    assert evaluated["reproducible_transform_package_complete"] is True
    assert evaluated["missing_reproducible_fields"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("projection_seed_bytes", 0),
        ("projection_generator_name", ""),
        ("projection_generator_version", ""),
        ("projection_dtype", ""),
        ("projection_input_dim", 0),
        ("projection_output_dim", 0),
        ("projection_matrix_bytes", 0),
    ],
)
def test_random_projection_reproducible_package_is_incomplete_when_metadata_is_missing(
    field: str, value: object
) -> None:
    evaluate_adapter_manifest_v2 = _evaluate_adapter_manifest_v2()

    evaluated = evaluate_adapter_manifest_v2(_random_projection_manifest(**{field: value}))

    assert evaluated["reproducible_transform_package_complete"] is False
    assert field in evaluated["missing_reproducible_fields"]
