from __future__ import annotations

from typing import Any, Mapping, Sequence


def gate21_5_method_flags(*, row: Mapping[str, Any], native_full_micro: float, native_full_macro: float) -> dict[str, Any]:
    structural = _float(row.get("mean_semantic_structural_storage_ratio", row.get("semantic_structural_storage_ratio")))
    raw = _float(row.get("mean_official_text_hgb_byte_ratio", row.get("mean_hgb_raw_file_byte_ratio", row.get("hgb_raw_file_byte_ratio"))))
    micro = _float(row.get("mean_test_micro_f1", row.get("test_micro_f1")))
    macro = _float(row.get("mean_test_macro_f1", row.get("test_macro_f1")))
    deterministic = _bool(row.get("deterministic_graph_method", False))
    official_ok = (
        _bool(row.get("official_sehgnn_unmodified_all", row.get("official_sehgnn_unmodified", False)))
        and _bool(row.get("eligible_for_main_decision", False))
        and _bool(row.get("cache_hygiene_pass_all", row.get("cache_hygiene_pass", True)))
        and _bool(row.get("relation_mapping_audit_pass_all", True))
        and _bool(row.get("relation_retention_audit_pass_all", True))
        and _ge(micro, float(native_full_micro) - 0.03)
        and _ge(macro, float(native_full_macro) - 0.03)
    )
    return {
        "official_structural20_pass": bool(official_ok and _le(structural, 0.20)),
        "official_structural15_pass": bool(official_ok and _le(structural, 0.15)),
        "official_structural12_pass": bool(official_ok and _le(structural, 0.12)),
        "official_structural10_pass": bool(official_ok and _le(structural, 0.10)),
        "official_text_hgb_byte50_pass": bool(raw is not None and raw <= 0.50),
        "official_text_hgb_byte30_pass": bool(raw is not None and raw <= 0.30),
        "raw_hgb_byte50_pass": bool(raw is not None and raw <= 0.50),
        "raw_hgb_byte30_pass": bool(raw is not None and raw <= 0.30),
        "deterministic_graph_method": deterministic,
        "graph_seed_independence_required": not deterministic,
        "graph_seed_independence_status": "not_applicable_deterministic" if deterministic else str(row.get("graph_seed_independence_status", "required")),
        "num_effective_graph_variants": 1 if deterministic else int(_float(row.get("graph_seed_count")) or 0),
    }


def gate21_5_adapter_flags(*, row: Mapping[str, Any], native_full_micro: float) -> dict[str, Any]:
    effective = _float(row.get("adapter_effective_deployment_byte_ratio", row.get("effective_total_byte_ratio")))
    micro = _float(row.get("mean_test_micro_f1", row.get("test_micro_f1")))
    return {
        "adapter_effective_byte50_pass": bool(_le(effective, 0.50)),
        "adapter_effective_byte30_pass": bool(_le(effective, 0.30)),
        "adapter_effective_byte20_pass": bool(_le(effective, 0.20)),
        "adapter_effective_byte15_pass": bool(_le(effective, 0.15)),
        "adapter_effective_byte10_pass": bool(_le(effective, 0.10)),
        "adapter_accuracy_seed3_prelim_pass": bool(micro is not None and micro >= float(native_full_micro) - 0.01),
        "adapter_accuracy_seed5_pass": bool(micro is not None and micro >= float(native_full_micro) - 0.01 and int(_float(row.get("training_seed_count")) or 1) >= 5),
        "adapter_official_unmodified_false": not _bool(row.get("official_sehgnn_unmodified", row.get("official_sehgnn_unmodified_all", True))),
        "adapter_not_eligible_for_main_table": not _bool(row.get("eligible_for_main_decision", True)),
        "adapter_eligible_for_adapter_table": _bool(row.get("eligible_for_adapter_table", False)),
    }


def gate21_5_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]],
    adapter_rows: Sequence[Mapping[str, Any]],
    native_full_micro: float = 0.9533802,
    native_full_macro: float = 0.9498198,
) -> dict[str, Any]:
    official_scored = [
        {**row, **gate21_5_method_flags(row=row, native_full_micro=native_full_micro, native_full_macro=native_full_macro)}
        for row in official_rows
    ]
    adapter_scored = [{**row, **gate21_5_adapter_flags(row=row, native_full_micro=native_full_micro)} for row in adapter_rows]
    official_candidates = [row for row in official_scored if _bool(row.get("official_structural20_pass", False))]
    adapter_candidates = [row for row in adapter_scored if _bool(row.get("adapter_eligible_for_adapter_table", False))]
    best_official = min(
        official_candidates,
        key=lambda row: (
            _float(row.get("mean_semantic_structural_storage_ratio", row.get("semantic_structural_storage_ratio"))) or 999.0,
            -(_float(row.get("mean_test_micro_f1", row.get("test_micro_f1"))) or 0.0),
        ),
        default={},
    )
    best_adapter = min(
        adapter_candidates,
        key=lambda row: (_float(row.get("adapter_effective_deployment_byte_ratio", row.get("effective_total_byte_ratio"))) or 999.0),
        default={},
    )
    decisions: list[str] = []
    if any(_bool(row.get("official_structural10_pass", False)) for row in official_scored):
        decisions.append("OFFICIAL_STRUCTURAL10_PASS")
    elif any(_bool(row.get("official_structural12_pass", False)) for row in official_scored):
        decisions.append("OFFICIAL_STRUCTURAL12_PASS")
    elif any(_bool(row.get("official_structural20_pass", False)) for row in official_scored):
        decisions.append("OFFICIAL_STRUCTURAL20_PASS")
    else:
        decisions.append("OFFICIAL_STRUCTURAL_NOT_VALIDATED")
    if any(_bool(row.get("adapter_effective_byte10_pass", False)) for row in adapter_scored):
        decisions.append("ADAPTER_EFFECTIVE_BYTE10_PASS")
    elif any(_bool(row.get("adapter_effective_byte15_pass", False)) for row in adapter_scored):
        decisions.append("ADAPTER_EFFECTIVE_BYTE15_PASS")
    elif any(_bool(row.get("adapter_effective_byte30_pass", False)) for row in adapter_scored):
        decisions.append("ADAPTER_EFFECTIVE_BYTE30_PASS")
    else:
        decisions.append("ADAPTER_EFFECTIVE_BYTE_NOT_VALIDATED")
    decisions.extend(
        [
            "FEATURE_ADAPTER_NOT_OFFICIAL_UNMODIFIED",
            "WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN",
            "GENERIC_COARSE_GRAPH_NOT_CLAIMED",
            "STRUCTURAL_RATIO_IS_NOT_RAW_HGB_BYTE_RATIO",
        ]
    )
    return {
        "native_full_micro": float(native_full_micro),
        "native_full_macro": float(native_full_macro),
        "best_official_structural_method": best_official.get("method", ""),
        "best_official_structural_method_micro": _float(best_official.get("mean_test_micro_f1", best_official.get("test_micro_f1"))) or 0.0,
        "best_official_structural_method_macro": _float(best_official.get("mean_test_macro_f1", best_official.get("test_macro_f1"))) or 0.0,
        "best_official_structural_method_structural_ratio": _float(best_official.get("mean_semantic_structural_storage_ratio", best_official.get("semantic_structural_storage_ratio"))) or 0.0,
        "best_adapter_method": best_adapter.get("feature_compression_method", best_adapter.get("method", "")),
        "best_adapter_method_micro": _float(best_adapter.get("mean_test_micro_f1", best_adapter.get("test_micro_f1"))) or 0.0,
        "best_adapter_method_effective_byte_ratio": _float(best_adapter.get("adapter_effective_deployment_byte_ratio", best_adapter.get("effective_total_byte_ratio"))) or 0.0,
        "decisions": decisions,
    }


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _le(value: float | None, threshold: float) -> bool:
    return value is not None and float(value) <= float(threshold)


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and float(value) >= float(threshold)
