from __future__ import annotations


def test_apv_confirmed_requires_cache_hygiene() -> None:
    from hesf_coarsen.eval.official.gate21_4_decision import gate21_4_decision_flags

    flags = gate21_4_decision_flags(
        apv_success_count=25,
        apv_mean_structural_ratio=0.198,
        apv_mean_micro=0.949,
        apv_mean_macro=0.946,
        apv_std_micro=0.001,
        relation_mapping_pass=True,
        relation_retention_pass=True,
        cache_hygiene_pass=True,
        official_unmodified=True,
        eligible_for_main_decision=True,
        pathaware_success_count=1,
        feature_adapter_has_metrics=False,
    )

    assert "APV_SKELETON_5X5_CONFIRMED" in flags
    assert "STRUCTURAL_STORAGE20_PASS" in flags
    assert "PATHAWARE_V2_GAIN_NOT_VALIDATED" in flags
    assert "FEATURE_ADAPTER_ACCURACY_NOT_VALIDATED" in flags


def test_cache_hygiene_failure_blocks_apv_confirmation() -> None:
    from hesf_coarsen.eval.official.gate21_4_decision import gate21_4_decision_flags

    flags = gate21_4_decision_flags(
        apv_success_count=25,
        apv_mean_structural_ratio=0.198,
        apv_mean_micro=0.949,
        apv_mean_macro=0.946,
        apv_std_micro=0.001,
        relation_mapping_pass=True,
        relation_retention_pass=True,
        cache_hygiene_pass=False,
        official_unmodified=True,
        eligible_for_main_decision=True,
        pathaware_success_count=9,
        feature_adapter_has_metrics=True,
    )

    assert "APV_SKELETON_CACHE_CLEAN_FAIL" in flags
    assert "APV_SKELETON_5X5_NOT_CONFIRMED" in flags
