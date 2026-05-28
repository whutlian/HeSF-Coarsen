from __future__ import annotations


def test_adapter_aggregate_excludes_failed_placeholder_rows() -> None:
    from hesf_coarsen.eval.official.adapter_package_manifest import aggregate_adapter_by_method_gate21_10

    rows = [
        {"base_method": "APV12", "adapter_method": "random_projection_dim64", "success": True, "training_executed": True, "test_micro_f1": 0.94, "static_inference_package_ratio": 0.12, "transform_recipe_package_ratio": 0.01, "reconstructable_package_ratio": 0.13},
        {"base_method": "APV12", "adapter_method": "random_projection_dim64", "success": False, "training_executed": False, "test_micro_f1": "", "static_inference_package_ratio": 1000000, "transform_recipe_package_ratio": 10240, "reconstructable_package_ratio": 1000000},
    ]

    summary = aggregate_adapter_by_method_gate21_10(rows)[0]

    assert summary["success_count"] == 1
    assert summary["failed_rows_excluded"] == 1
    assert summary["static_inference_package_ratio_mean"] == 0.12
    assert summary["eligible_for_official_main_table"] is False
