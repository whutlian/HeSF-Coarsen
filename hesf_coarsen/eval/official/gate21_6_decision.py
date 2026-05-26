from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def gate21_6_method_flags(row: Mapping[str, Any], *, full_micro: float, full_macro: float) -> dict[str, Any]:
    structural = _float(row.get("structural_storage_ratio", row.get("semantic_structural_storage_ratio")))
    raw = _float(row.get("raw_hgb_text_byte_ratio", row.get("official_text_hgb_byte_ratio", row.get("hgb_raw_file_byte_ratio"))))
    micro = _float(row.get("test_micro_mean", row.get("mean_test_micro_f1", row.get("test_micro_f1"))))
    macro = _float(row.get("test_macro_mean", row.get("mean_test_macro_f1", row.get("test_macro_f1"))))
    adapter_package = _float(row.get("adapter_package_ratio"))
    uses_adapter = _bool(row.get("uses_feature_adapter", False))
    official_eligible = _official_main_eligible(row)
    adapter_eligible = _bool(row.get("eligible_for_adapter_table", False)) and uses_adapter and _bool(row.get("schema_compatible", True)) and _bool(row.get("keeps_all_target_nodes", True))
    method = str(row.get("method", ""))
    return {
        "method": method,
        "structural_storage_ratio": structural,
        "raw_hgb_text_byte_ratio": raw,
        "adapter_package_ratio": adapter_package,
        "eligible_for_official_main_table": official_eligible,
        "eligible_for_adapter_table": adapter_eligible,
        "OFFICIAL_STRUCTURAL_APV12_PASS": bool(
            official_eligible
            and ("APV12" in method or method.endswith("AP100-PA00-PV100-VP00-PTTP00"))
            and _le(structural, 0.125)
            and _ge(micro, float(full_micro) - 0.010)
            and _ge(macro, float(full_macro) - 0.012)
        ),
        "OFFICIAL_STRUCTURAL_APV16_PASS": bool(
            official_eligible
            and ("APV16" in method or method.endswith("AP100-PA50-PV100-VP50-PTTP00"))
            and _le(structural, 0.170)
            and _ge(micro, float(full_micro) - 0.005)
            and _ge(macro, float(full_macro) - 0.006)
        ),
        "OFFICIAL_STRUCTURAL30_PASS": bool(official_eligible and _le(structural, 0.30) and _ge(micro, float(full_micro) - 0.03)),
        "OFFICIAL_STRUCTURAL20_PASS": bool(official_eligible and _le(structural, 0.20) and _ge(micro, float(full_micro) - 0.03)),
        "OFFICIAL_STRUCTURAL12_PASS": bool(official_eligible and _le(structural, 0.12) and _ge(micro, float(full_micro) - 0.03)),
        "RAW_HGB_BYTE50_PASS": bool(_le(raw, 0.50)),
        "RAW_HGB_BYTE30_PASS": bool(_le(raw, 0.30)),
        "ADAPTER_PACKAGE10_PASS": bool(adapter_eligible and _le(adapter_package, 0.10) and _ge(micro, float(full_micro) - 0.010) and _bool(row.get("adapter_manifest_complete", False))),
        "ADAPTER_PACKAGE05_PASS": bool(adapter_eligible and _le(adapter_package, 0.05) and _ge(micro, float(full_micro) - 0.010) and _bool(row.get("adapter_manifest_complete", False))),
    }


def graph_seed_stability_flags(
    *,
    deterministic_graph_method: bool,
    graph_seed_count: int,
    actual_export_hash_unique_count: int,
) -> dict[str, Any]:
    deterministic = bool(deterministic_graph_method)
    graph_seeds = int(graph_seed_count)
    unique_hashes = int(actual_export_hash_unique_count)
    expected = 1 if deterministic else max(5, graph_seeds)
    warnings: list[str] = []
    if deterministic and unique_hashes != 1:
        warnings.append("deterministic_export_hash_unique_count!=1")
    if not deterministic and graph_seeds < 5:
        warnings.append("graph_seed_count<5")
    if not deterministic and unique_hashes < min(graph_seeds, 5):
        warnings.append("stochastic_export_hash_unique_count_low")
    return {
        "deterministic_graph_method": deterministic,
        "expected_export_hash_unique_count": expected,
        "actual_export_hash_unique_count": unique_hashes,
        "graph_sampling_stability_pass": not warnings,
        "graph_sampling_warning": ";".join(warnings),
    }


def gate21_6_decision(
    *,
    official_rows: Sequence[Mapping[str, Any]],
    adapter_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
    feature_ablation_rows: Sequence[Mapping[str, Any]],
    metapath_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    full_micro: float = 0.9533802,
    full_macro: float = 0.9498198,
) -> dict[str, Any]:
    official_scored = [gate21_6_method_flags(row, full_micro=full_micro, full_macro=full_macro) for row in official_rows]
    adapter_scored = [gate21_6_method_flags(row, full_micro=full_micro, full_macro=full_macro) for row in adapter_rows]
    external_success = [row for row in external_rows if str(row.get("success", "")).lower() == "true"]
    flags = {
        "NATIVE_EXPORT_FIDELITY_PASS": _native_export_fidelity_pass(official_rows),
        "OFFICIAL_STRUCTURAL_APV12_PASS": any(_bool(row.get("OFFICIAL_STRUCTURAL_APV12_PASS")) for row in official_scored),
        "OFFICIAL_STRUCTURAL_APV16_PASS": any(_bool(row.get("OFFICIAL_STRUCTURAL_APV16_PASS")) for row in official_scored),
        "OFFICIAL_STRUCTURAL30_PASS": any(_bool(row.get("OFFICIAL_STRUCTURAL30_PASS")) for row in official_scored),
        "OFFICIAL_STRUCTURAL20_PASS": any(_bool(row.get("OFFICIAL_STRUCTURAL20_PASS")) for row in official_scored),
        "OFFICIAL_STRUCTURAL12_PASS": any(_bool(row.get("OFFICIAL_STRUCTURAL12_PASS")) for row in official_scored),
        "RAW_HGB_BYTE50_PASS": any(_bool(row.get("RAW_HGB_BYTE50_PASS")) for row in official_scored),
        "RAW_HGB_BYTE30_PASS": any(_bool(row.get("RAW_HGB_BYTE30_PASS")) for row in official_scored),
        "ADAPTER_PACKAGE10_PASS": any(_bool(row.get("ADAPTER_PACKAGE10_PASS")) for row in adapter_scored),
        "ADAPTER_PACKAGE05_PASS": any(_bool(row.get("ADAPTER_PACKAGE05_PASS")) for row in adapter_scored),
        "FEATURE_ABLATION_SHAPE_SAFE_PASS": bool(feature_ablation_rows) and all(_bool(row.get("shape_safe_pass", row.get("feature_ablation_shape_safe_pass", False))) for row in feature_ablation_rows),
        "CACHE_HYGIENE_PASS": any(_bool(row.get("cache_hygiene_pass", False)) for row in official_rows + adapter_rows),
        "METAPATH_INTROSPECTION_PASS": bool(metapath_rows) and any(_bool(row.get("introspection_supported", False)) for row in metapath_rows),
        "COVERAGE_DIAGNOSTICS_PASS": bool(coverage_rows),
        "EXTERNAL_TP_BASELINES_READY": _external_tp_ready(external_success),
        "FREEHGC_TP_READY": any(str(row.get("baseline_name", "")) == "FreeHGC-TP" and str(row.get("success", "")).lower() == "true" for row in external_rows),
        "STANDARD_CONDENSATION_BASELINES_READY": False,
        "CROSS_DATASET_AUTO_CHANNEL_READY": False,
    }
    flags["ICDE_MAIN_TABLE_READY"] = bool(
        flags["OFFICIAL_STRUCTURAL_APV12_PASS"]
        and flags["OFFICIAL_STRUCTURAL_APV16_PASS"]
        and flags["FEATURE_ABLATION_SHAPE_SAFE_PASS"]
        and flags["EXTERNAL_TP_BASELINES_READY"]
    )
    return {
        "native_full_micro": float(full_micro),
        "native_full_macro": float(full_macro),
        "flags": flags,
        "decisions": [name for name, value in flags.items() if value],
        "failures": [name for name, value in flags.items() if not value],
        "counts": {
            "official_rows": len(official_rows),
            "adapter_rows": len(adapter_rows),
            "external_tp_rows": len(external_rows),
            "feature_ablation_rows": len(feature_ablation_rows),
            "metapath_rows": len(metapath_rows),
            "coverage_rows": len(coverage_rows),
        },
    }


def decision_md(decision: Mapping[str, Any]) -> str:
    flags = decision.get("flags", {})
    lines = [
        "# Gate21.6 ICDE-Ready Decision",
        "",
        f"- native_full_micro: `{decision.get('native_full_micro', '')}`",
        f"- native_full_macro: `{decision.get('native_full_macro', '')}`",
        "",
        "## Pass Flags",
        *[f"- `{name}`" for name, value in flags.items() if value],
        "",
        "## Fail Or Not Ready Flags",
        *[f"- `{name}`" for name, value in flags.items() if not value],
        "",
        "## Counts",
        f"```json\n{json.dumps(decision.get('counts', {}), indent=2, sort_keys=True)}\n```",
        "",
        "## Claim Boundaries",
        "- Feature adapters are not official-unmodified SeHGNN main-table methods.",
        "- Structural storage ratio, raw HGB text byte ratio, cache ratio, and adapter package ratio are separate metrics.",
        "- Missing FreeHGC/HGCond dependencies are recorded as failure rows rather than silently skipped.",
        "- Standard condensation and schema-preserving TP workload protocols are reported separately.",
    ]
    return "\n".join(lines) + "\n"


def _official_main_eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        _bool(row.get("schema_compatible", True))
        and _bool(row.get("official_sehgnn_unmodified", False))
        and not _bool(row.get("uses_feature_adapter", False))
        and not _bool(row.get("uses_weighted_superedges", False))
        and not _bool(row.get("uses_synthetic_nodes", False))
        and _bool(row.get("keeps_all_target_nodes", True))
        and not _bool(row.get("used_test_data", False))
        and not _bool(row.get("uses_test_data_for_transform", False))
        and _bool(row.get("eligible_for_official_main_table", row.get("eligible_for_main_decision", False)))
    )


def _native_export_fidelity_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    native = _first_float(rows, "full-native-SeHGNN", "test_micro_mean")
    export = _first_float(rows, "export-full-SeHGNN", "test_micro_mean")
    if native is None or export is None:
        return False
    return abs(native - export) <= 1e-6


def _external_tp_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {"Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP"}
    present = {str(row.get("baseline_name", row.get("method", ""))) for row in rows}
    return required.issubset(present)


def _first_float(rows: Sequence[Mapping[str, Any]], method: str, field: str) -> float | None:
    for row in rows:
        if str(row.get("method", "")) == str(method):
            return _float(row.get(field, row.get("mean_test_micro_f1")))
    return None


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
