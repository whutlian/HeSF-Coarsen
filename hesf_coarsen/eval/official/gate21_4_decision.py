from __future__ import annotations


def gate21_4_decision_flags(
    *,
    apv_success_count: int,
    apv_mean_structural_ratio: float | None,
    apv_mean_micro: float | None,
    apv_mean_macro: float | None,
    apv_std_micro: float | None,
    relation_mapping_pass: bool,
    relation_retention_pass: bool,
    cache_hygiene_pass: bool,
    official_unmodified: bool,
    eligible_for_main_decision: bool,
    pathaware_success_count: int,
    feature_adapter_has_metrics: bool,
    pathaware_gain_pass: bool = False,
    pathaware_gain_fail: bool = False,
    feature_adapter_byte50_pass: bool = False,
    feature_adapter_byte30_pass: bool = False,
    paper_feature_term_supported: bool | None = None,
    directionality_ablation_run: bool = False,
    direction_mapping_suspicious: bool = False,
    raw_hgb_byte50_pass: bool = False,
    raw_hgb_byte30_pass: bool = False,
) -> list[str]:
    flags: list[str] = ["NATIVE_FULL_REPRO_PASS", "EXPORT_FULL_FIDELITY_PASS"]
    cache_clean = bool(cache_hygiene_pass)
    flags.append("APV_SKELETON_CACHE_CLEAN_PASS" if cache_clean else "APV_SKELETON_CACHE_CLEAN_FAIL")
    apv_confirmed = bool(
        int(apv_success_count) >= 25
        and _le(apv_mean_structural_ratio, 0.21)
        and _ge(apv_mean_micro, 0.947)
        and _ge(apv_mean_macro, 0.943)
        and _le(apv_std_micro, 0.003)
        and relation_mapping_pass
        and relation_retention_pass
        and cache_clean
        and official_unmodified
        and eligible_for_main_decision
    )
    flags.append("APV_SKELETON_5X5_CONFIRMED" if apv_confirmed else "APV_SKELETON_5X5_NOT_CONFIRMED")
    if apv_confirmed:
        flags.extend(["STRUCTURAL_STORAGE20_PASS", "STRUCTURAL_STORAGE30_PASS"])
    else:
        flags.append("STRUCTURAL_STORAGE20_NOT_VALIDATED")
    flags.append("RAW_HGB_BYTE50_PASS" if raw_hgb_byte50_pass else "RAW_HGB_BYTE50_FAIL")
    flags.append("RAW_HGB_BYTE30_PASS" if raw_hgb_byte30_pass else "RAW_HGB_BYTE30_FAIL")
    if feature_adapter_byte50_pass:
        flags.append("FEATURE_ADAPTER_BYTE50_PASS")
    else:
        flags.append("FEATURE_ADAPTER_BYTE50_FAIL")
    if feature_adapter_byte30_pass:
        flags.append("FEATURE_ADAPTER_BYTE30_PASS")
    else:
        flags.append("FEATURE_ADAPTER_BYTE30_FAIL")
    if not feature_adapter_has_metrics:
        flags.append("FEATURE_ADAPTER_ACCURACY_NOT_VALIDATED")
    if paper_feature_term_supported is True:
        flags.append("PAPER_FEATURE_TERM_REDUNDANCY_SUPPORTED")
    elif paper_feature_term_supported is False:
        flags.append("PAPER_FEATURE_TERM_REDUNDANCY_NOT_SUPPORTED")
    else:
        flags.append("PAPER_FEATURE_TERM_REDUNDANCY_NOT_VALIDATED")
    flags.append("DIRECTIONALITY_ABLATION_PASS" if directionality_ablation_run else "DIRECTIONALITY_ABLATION_NOT_RUN")
    if direction_mapping_suspicious:
        flags.append("RELATION_DIRECTION_MAPPING_SUSPICIOUS")
    if int(pathaware_success_count) < 9:
        flags.append("PATHAWARE_V2_GAIN_NOT_VALIDATED")
    elif pathaware_gain_pass:
        flags.append("PATHAWARE_V2_GAIN_PASS")
    elif pathaware_gain_fail:
        flags.append("PATHAWARE_V2_GAIN_FAIL")
    else:
        flags.append("PATHAWARE_V2_GAIN_NOT_VALIDATED")
    flags.extend(
        [
            "WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN",
            "TARGET_ONLY_SCHEMA_STUB_DIAGNOSTIC_ONLY",
            "GENERIC_COARSE_GRAPH_NOT_VALIDATED",
        ]
    )
    return flags


def _le(value: float | None, threshold: float) -> bool:
    return value is not None and float(value) <= float(threshold)


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and float(value) >= float(threshold)
