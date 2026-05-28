from __future__ import annotations


def test_dblp_selector_prefers_ap_pv_and_suppresses_pt_tp_without_test_metrics() -> None:
    from hesf_coarsen.eval.official.auto_relation_channel_selector_v2 import select_relation_channels_v2

    result = select_relation_channels_v2(dataset="DBLP")
    plan = result["plan"]
    trace = {row["channel_key"]: row for row in result["channel_utility_rows"]}

    assert plan["AP_keep"] >= 0.90
    assert plan["PV_keep"] >= 0.90
    assert plan["PT_keep"] <= 0.05
    assert plan["TP_keep"] <= 0.05
    assert trace["PT"]["feature_redundancy_score"] > trace["AP"]["feature_redundancy_score"]
    assert all(row["uses_test_metrics"] is False for row in result["channel_utility_rows"])


def test_channel_removal_probe_rows_are_validation_only() -> None:
    from hesf_coarsen.eval.official.channel_removal_probe import build_dblp_channel_removal_probes

    rows = build_dblp_channel_removal_probes()

    assert {row["probe_name"] for row in rows} >= {"drop_PTTP", "drop_AP", "AP100-PV100-PA00-VP00-PTTP00"}
    assert all(row["metric_split"] == "validation" for row in rows)
    assert all(row["uses_test_metrics"] is False for row in rows)


def test_ratio_denominator_v2_uses_explicit_denominator_names() -> None:
    from hesf_coarsen.eval.official.ratio_denominator_audit_v2 import ratio_denominator_audit_v2

    row = ratio_denominator_audit_v2(
        {
            "method": "APV12",
            "artifact_bytes": 25,
            "original_native_full_hgb_text_bytes": 100,
            "current_export_full_text_bytes": 50,
            "current_compressed_control_text_bytes": 200,
        }
    )

    assert row["ratio_denominator_audit_v2_pass"] is True
    assert row["ratio_vs_original_native_full_hgb_text"] == 0.25
    assert row["ratio_vs_current_export_full_text"] == 0.5
    assert row["ratio_vs_current_compressed_control_text"] == 0.125
    assert "ratio_vs_native_full_text" not in row


def test_static_inference_ratio_not_replaced_by_transform_recipe_ratio() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    rows = [
        {
            "method": "HeSF-RCS-APV12+random_projection_dim64",
            "base_graph_method": "HeSF-RCS-APV12",
            "feature_adapter": "random_projection_dim64",
            "test_micro_f1": 0.948,
            "static_inference_package_ratio": 0.12,
            "transform_recipe_package_ratio": 0.004,
            "reconstructable_package_ratio": 0.13,
            "projection_generator_name": "PCG64",
            "projection_generator_version": "numpy-1.26",
            "projection_matrix_shape": "4231x64",
            "projection_matrix_dtype": "float32",
            "projection_distribution": "normal",
        },
        {
            "method": "HeSF-RCS-APV16+random_projection_dim64",
            "base_graph_method": "HeSF-RCS-APV16",
            "feature_adapter": "random_projection_dim64",
            "test_micro_f1": 0.949,
            "static_inference_package_ratio": 0.13,
            "transform_recipe_package_ratio": 0.004,
            "reconstructable_package_ratio": 0.14,
            "projection_generator_name": "PCG64",
            "projection_generator_version": "numpy-1.26",
            "projection_matrix_shape": "4231x64",
            "projection_matrix_dtype": "float32",
            "projection_distribution": "normal",
        },
    ]
    decision = gate21_9_decision(adapter_rows=rows)

    assert decision["flags"]["ADAPTER_APV12_RP64_READY"] is True
    assert decision["flags"]["ADAPTER_APV16_RP64_READY"] is True
    assert decision["flags"]["ADAPTER_PACKAGE_SEMANTICS_PASS"] is True


def test_pca_reproducible_package_requires_basis_mean_config_and_training_hash() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import pca_reproducible_package_complete

    complete = {
        "feature_adapter": "pca_svd_dim64",
        "pca_basis_bytes": 10,
        "pca_mean_bytes": 10,
        "pca_fit_config": "svd_solver=randomized",
        "pca_training_node_ids_hash": "abc",
    }

    assert pca_reproducible_package_complete(complete) is True
    assert pca_reproducible_package_complete({**complete, "pca_basis_bytes": 0}) is False
