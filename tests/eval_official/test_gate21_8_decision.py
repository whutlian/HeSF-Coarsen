from __future__ import annotations

import hashlib

from hesf_coarsen.eval.official.gate21_8_decision import (
    EMPTY_SHA256,
    apv16_graph_seed_stability_status,
    budget_alignment_status,
    external_tp_5x5_method_status,
    gate21_8_decision,
    ratio_denominator_status,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_budget_alignment_requires_requested_budget_to_match_budget_value() -> None:
    assert budget_alignment_status({"requested_budget": 0.3, "budget_value": 0.3})["budget_alignment_pass"] is True

    failed = budget_alignment_status({"requested_budget": 0.3, "budget_value": 0.5})

    assert failed["budget_alignment_pass"] is False
    assert "differs" in failed["budget_alignment_error"]


def test_ratio_denominator_audit_recomputes_all_denominators() -> None:
    row = {
        "method_text_bytes": 25,
        "native_full_text_bytes": 100,
        "export_full_text_bytes": 50,
        "current_control_text_bytes": 200,
        "ratio_vs_native_full_text": 0.25,
        "ratio_vs_export_full_text": 0.5,
        "ratio_vs_current_control_text": 0.125,
    }

    assert ratio_denominator_status(row)["ratio_consistency_pass"] is True
    assert ratio_denominator_status({**row, "ratio_vs_current_control_text": 0.25})["ratio_consistency_pass"] is False


def test_apv16_deterministic_proof_passes_when_graph_seed_is_ignored_and_hash_is_unique() -> None:
    row = {
        "method": "HeSF-RCS-APV16",
        "graph_seed_count": 1,
        "training_seed_count": 5,
        "sampler_deterministic": True,
        "graph_seed_ignored_by_sampler": True,
        "export_hash_unique_count": 1,
        "mean_test_micro_f1": 0.950,
        "std_test_micro_f1": 0.001,
        "structural_storage_ratio": 0.16,
    }

    assert apv16_graph_seed_stability_status(row)["graph_seed_stability_pass"] is True


def test_apv16_stochastic_row_requires_five_graph_seeds() -> None:
    row = {
        "method": "HeSF-RCS-APV16",
        "graph_seed_count": 1,
        "training_seed_count": 5,
        "sampler_deterministic": False,
        "export_hash_unique_count": 1,
        "mean_test_micro_f1": 0.950,
        "std_test_micro_f1": 0.001,
        "structural_storage_ratio": 0.16,
    }

    status = apv16_graph_seed_stability_status(row)

    assert status["graph_seed_stability_pass"] is False
    assert status["stability_failure_reason"] == "graph_seed_count_lt_5_without_deterministic_proof"


def test_external_tp_5x5_not_ready_with_single_seed_smoke_rows() -> None:
    rows = [
        {
            "method": "Random-HG-TP",
            "graph_seed": 1,
            "training_seed": 1,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "training_executed": True,
            "test_micro_f1": 0.8,
            "test_macro_f1": 0.7,
            "budget_alignment_pass": True,
        }
    ]

    status = external_tp_5x5_method_status(rows, "Random-HG-TP")

    assert status["ready"] is False
    assert "graph_seed_count_lt_5" in status["missing_requirements"]
    assert "training_seed_count_lt_5" in status["missing_requirements"]


def test_empty_cache_hash_cannot_pass_gate21_8_cache_hash_real_gate() -> None:
    decision = gate21_8_decision(cache_assertion_rows=[{"assertion_pass": True, "cache_hash": EMPTY_SHA256}])

    assert decision["flags"]["CACHE_HASH_REAL_PASS"] is False
    assert decision["flags"]["CACHE_HASH_EMPTY_ONLY_FAIL"] is False


def test_feature_ablation_shape_safe_is_not_same_as_task_results_ready() -> None:
    decision = gate21_8_decision(
        feature_ablation_rows=[
            {
                "shape_safe_pass": True,
                "training_executed": False,
                "test_micro_f1": "",
                "failure_type": "shape_audit_only",
            }
        ]
    )

    assert decision["flags"]["FEATURE_ABLATION_SHAPE_SAFE_PASS"] is True
    assert decision["flags"]["FEATURE_ABLATION_TASK_RESULTS_READY"] is False


def test_freehgc_tp_hard_incompatibility_report_satisfies_tp_audit_but_not_5x5() -> None:
    decision = gate21_8_decision(
        freehgc_tp_rows=[
            {
                "method": "FreeHGC-TP",
                "protocol": "schema_preserving_tp",
                "failure_type": "hard_incompatibility",
                "training_executed": False,
            }
        ]
    )

    assert decision["flags"]["EXTERNAL_TP_FREEHGC_TP_READY"] is True
    assert decision["flags"]["EXTERNAL_TP_ALL_REQUIRED_READY"] is False


def test_adapter_pca_requires_complete_reproducible_package() -> None:
    incomplete = gate21_8_decision(
        adapter_rows=[
            {
                "method": "HeSF-RCS-APV12+pca_svd_dim64",
                "adapter_name": "pca_svd_dim64",
                "test_micro_f1": 0.94,
                "reproducible_transform_package_complete": False,
            }
        ]
    )
    complete = gate21_8_decision(
        adapter_rows=[
            {
                "method": "HeSF-RCS-APV12+pca_svd_dim64",
                "adapter_name": "pca_svd_dim64",
                "test_micro_f1": 0.94,
                "reproducible_transform_package_complete": True,
            }
        ]
    )

    assert incomplete["flags"]["ADAPTER_PCA_REPRODUCIBLE_READY"] is False
    assert complete["flags"]["ADAPTER_PCA_REPRODUCIBLE_READY"] is True


def test_adapter_random_projection_reproducibility_is_required_for_apv12_rp64() -> None:
    decision = gate21_8_decision(
        adapter_rows=[
            {
                "method": "HeSF-RCS-APV12+random_projection_dim64",
                "adapter_name": "random_projection_dim64",
                "test_micro_f1": 0.948,
                "reproducible_transform_package_complete": True,
                "projection_reproducibility_test_pass": False,
            }
        ]
    )

    assert decision["flags"]["ADAPTER_APV12_RP64_REPRODUCIBLE_READY"] is False


def test_standard_condensation_protocol_rows_must_not_include_tp_rows() -> None:
    separated = gate21_8_decision(
        freehgc_standard_rows=[{"method": "FreeHGC", "protocol": "standard_condensation", "success": True, "seed": 1, "test_micro_f1": 0.87}]
    )
    mixed = gate21_8_decision(
        freehgc_standard_rows=[
            {"method": "FreeHGC", "protocol": "standard_condensation", "success": True, "seed": 1, "test_micro_f1": 0.87},
            {"method": "FreeHGC-TP", "protocol": "schema_preserving_tp", "success": False, "seed": 1, "test_micro_f1": ""},
        ]
    )

    assert separated["flags"]["FREEHGC_STANDARD_PROTOCOL_VERIFIED"] is True
    assert mixed["flags"]["FREEHGC_STANDARD_PROTOCOL_VERIFIED"] is False


def test_full_ready_decision_requires_cross_dataset_task_results_even_with_other_evidence() -> None:
    decision = gate21_8_decision(
        official_rows=[
            {
                "method": "HeSF-RCS-APV12",
                "structural_storage_ratio": 0.12,
                "test_micro_f1_mean": 0.95,
                "official_sehgnn_unmodified": True,
                "training_executed": True,
            }
        ],
        apv16_stability_rows=[
            {
                "method": "HeSF-RCS-APV16",
                "graph_seed_count": 1,
                "training_seed_count": 5,
                "sampler_deterministic": True,
                "graph_seed_ignored_by_sampler": True,
                "export_hash_unique_count": 1,
                "mean_test_micro_f1": 0.950,
                "std_test_micro_f1": 0.001,
                "structural_storage_ratio": 0.16,
            }
        ],
        cache_assertion_rows=[{"assertion_pass": True, "cache_hash": _sha(b"real-cache")}],
        ratio_audit_rows=[{"ratio_consistency_pass": True}],
    )

    assert decision["paper_ready_status"] == "ICDE_EVIDENCE_PARTIAL"
    assert "CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY" in decision["blocking_issues"]
