from __future__ import annotations


def test_gate21_11_failed_adapter_rows_have_nan_ratios_and_do_not_aggregate() -> None:
    from hesf_coarsen.eval.official.adapter_package_manifest import clean_gate21_11_adapter_rows, summarize_gate21_11_adapters

    rows = clean_gate21_11_adapter_rows(
        [
            {
                "base_method": "HeSF-RCS-APV16",
                "adapter_method": "random_projection_dim64",
                "success": False,
                "failure_type": "not_executed",
                "failure_reason": "metric missing",
                "static_inference_package_ratio": 10240,
                "transform_recipe_package_ratio": 1000000,
                "reconstructable_package_ratio": "inf",
            },
            {
                "base_method": "HeSF-RCS-APV12",
                "adapter_method": "random_projection_dim64",
                "success": True,
                "training_executed": True,
                "test_micro_f1": 0.94,
                "test_macro_f1": 0.93,
                "static_inference_package_ratio": 0.08,
                "transform_recipe_package_ratio": 0.01,
                "reconstructable_package_ratio": 0.09,
            },
        ]
    )
    summary = summarize_gate21_11_adapters(rows)

    failed = [row for row in rows if row["success"] is False][0]
    assert failed["static_inference_package_ratio"] == "NaN"
    assert failed["transform_recipe_package_ratio"] == "NaN"
    assert failed["reconstructable_package_ratio"] == "NaN"
    assert any(row["base_method"] == "HeSF-RCS-APV16" and row["success_count"] == 0 and row["static_inference_package_ratio_mean"] == "NaN" for row in summary)


def test_gate21_11_freehgc_by_method_drops_unverified_imported_metrics_when_success_count_zero() -> None:
    from hesf_coarsen.eval.official.freehgc_standard_runner import summarize_gate21_11_freehgc_standard

    summary = summarize_gate21_11_freehgc_standard(
        [
            {
                "ratio": 0.012,
                "seed": 1,
                "success": False,
                "training_executed": False,
                "test_micro_f1": 0.7,
                "test_macro_f1": 0.68,
                "imported_unverified_metric": True,
            }
        ],
        expected_seed_count=5,
    )

    assert summary[0]["success_count"] == 0
    assert summary[0]["test_micro_f1_mean"] == "NaN"
    assert summary[0]["test_macro_f1_mean"] == "NaN"
    assert summary[0]["imported_unverified_metric"] is True
    assert summary[0]["eligible_for_decision"] is False
