from __future__ import annotations


def test_gate21_6_official_flags_keep_raw_and_structural_ratios_separate() -> None:
    from hesf_coarsen.eval.official.gate21_6_decision import gate21_6_method_flags

    flags = gate21_6_method_flags(
        {
            "method": "HeSF-RCS-APV12",
            "method_family": "schema_preserving_rcs",
            "schema_compatible": True,
            "official_sehgnn_unmodified": True,
            "uses_feature_adapter": False,
            "uses_weighted_superedges": False,
            "keeps_all_target_nodes": True,
            "eligible_for_official_main_table": True,
            "structural_storage_ratio": 0.1195,
            "raw_hgb_text_byte_ratio": 0.5300,
            "test_micro_mean": 0.9448,
            "test_macro_mean": 0.9405,
        },
        full_micro=0.9533802,
        full_macro=0.9498198,
    )

    assert flags["OFFICIAL_STRUCTURAL_APV12_PASS"] is True
    assert flags["RAW_HGB_BYTE50_PASS"] is False
    assert flags["raw_hgb_text_byte_ratio"] != flags["structural_storage_ratio"]


def test_gate21_6_adapter_cannot_be_official_main_even_if_accurate() -> None:
    from hesf_coarsen.eval.official.gate21_6_decision import gate21_6_method_flags

    flags = gate21_6_method_flags(
        {
            "method": "HeSF-RCS-APV12+random_projection_dim64",
            "method_family": "feature_compressed_adapter",
            "schema_compatible": True,
            "official_sehgnn_unmodified": False,
            "uses_feature_adapter": True,
            "uses_weighted_superedges": False,
            "keeps_all_target_nodes": True,
            "eligible_for_adapter_table": True,
            "eligible_for_official_main_table": True,
            "adapter_package_ratio": 0.045,
            "adapter_manifest_complete": True,
            "test_micro_mean": 0.9484,
            "test_macro_mean": 0.9443,
        },
        full_micro=0.9533802,
        full_macro=0.9498198,
    )

    assert flags["eligible_for_official_main_table"] is False
    assert flags["eligible_for_adapter_table"] is True
    assert flags["ADAPTER_PACKAGE05_PASS"] is True


def test_gate21_6_stability_flags_distinguish_deterministic_and_underseeded_stochastic() -> None:
    from hesf_coarsen.eval.official.gate21_6_decision import graph_seed_stability_flags

    deterministic = graph_seed_stability_flags(
        deterministic_graph_method=True,
        graph_seed_count=1,
        actual_export_hash_unique_count=1,
    )
    stochastic = graph_seed_stability_flags(
        deterministic_graph_method=False,
        graph_seed_count=3,
        actual_export_hash_unique_count=3,
    )

    assert deterministic["graph_sampling_stability_pass"] is True
    assert deterministic["expected_export_hash_unique_count"] == 1
    assert stochastic["graph_sampling_stability_pass"] is False
    assert "graph_seed_count<5" in stochastic["graph_sampling_warning"]
