from __future__ import annotations


def test_gate21_5_decision_marks_apv_as_official_structural20_pass() -> None:
    from hesf_coarsen.eval.official.gate21_5_decision import gate21_5_method_flags

    flags = gate21_5_method_flags(
        row={
            "method": "H6-APV-skeleton",
            "mean_semantic_structural_storage_ratio": 0.198819,
            "mean_hgb_raw_file_byte_ratio": 0.53255,
            "mean_test_micro_f1": 0.948169,
            "mean_test_macro_f1": 0.944491,
            "cache_hygiene_pass_all": True,
            "relation_mapping_audit_pass_all": True,
            "relation_retention_audit_pass_all": True,
            "official_sehgnn_unmodified_all": True,
            "eligible_for_main_decision": True,
            "deterministic_graph_method": True,
        },
        native_full_micro=0.9533802,
        native_full_macro=0.9498198,
    )

    assert flags["official_structural20_pass"] is True
    assert flags["official_text_hgb_byte50_pass"] is False
    assert flags["graph_seed_independence_status"] == "not_applicable_deterministic"


def test_gate21_5_decision_keeps_adapter_out_of_main_table() -> None:
    from hesf_coarsen.eval.official.gate21_5_decision import gate21_5_adapter_flags

    flags = gate21_5_adapter_flags(
        row={
            "method": "SeHGNN-feature-compressed-adapter",
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
            "eligible_for_adapter_table": True,
            "adapter_effective_deployment_byte_ratio": 0.009633,
            "test_micro_f1": 0.949531,
        },
        native_full_micro=0.9533802,
    )

    assert flags["adapter_effective_byte10_pass"] is True
    assert flags["adapter_official_unmodified_false"] is True
    assert flags["adapter_not_eligible_for_main_table"] is True
    assert flags["adapter_eligible_for_adapter_table"] is True


def test_gate21_5_decision_best_official_requires_accuracy_pass() -> None:
    from hesf_coarsen.eval.official.gate21_5_decision import gate21_5_decision

    decision = gate21_5_decision(
        official_rows=[
            {
                "method": "too-small",
                "mean_semantic_structural_storage_ratio": 0.08,
                "mean_test_micro_f1": 0.87,
                "mean_test_macro_f1": 0.86,
                "cache_hygiene_pass_all": True,
                "relation_mapping_audit_pass_all": True,
                "relation_retention_audit_pass_all": True,
                "official_sehgnn_unmodified_all": True,
                "eligible_for_main_decision": True,
            },
            {
                "method": "accurate-directed",
                "mean_semantic_structural_storage_ratio": 0.12,
                "mean_test_micro_f1": 0.945,
                "mean_test_macro_f1": 0.941,
                "cache_hygiene_pass_all": True,
                "relation_mapping_audit_pass_all": True,
                "relation_retention_audit_pass_all": True,
                "official_sehgnn_unmodified_all": True,
                "eligible_for_main_decision": True,
            },
        ],
        adapter_rows=[],
    )

    assert decision["best_official_structural_method"] == "accurate-directed"
