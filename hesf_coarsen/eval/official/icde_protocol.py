from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PROTOCOL_STANDARD_CONDENSATION = "standard_condensation"
PROTOCOL_SCHEMA_PRESERVING_TP = "schema_preserving_tp_workload"

PROTOCOL_FIELDS = [
    "baseline_name",
    "protocol",
    "method_family",
    "schema_compatible",
    "official_sehgnn_unmodified",
    "uses_feature_adapter",
    "uses_weighted_superedges",
    "uses_synthetic_nodes",
    "keeps_all_target_nodes",
    "used_test_data",
    "eligible_for_official_main_table",
    "eligible_for_adapter_table",
    "eligible_for_standard_condensation_table",
    "eligible_for_tp_workload_table",
    "eligibility_failure_reasons",
    "support_node_ratio",
    "support_edge_ratio",
    "structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "official_text_hgb_byte_ratio",
    "binary_feature_sidecar_ratio",
    "adapter_package_ratio",
    "preprocessed_cache_byte_ratio",
    "train_time_seconds",
    "preprocess_time_seconds",
    "compress_time_seconds",
    "peak_cpu_memory_mb",
    "peak_gpu_memory_mb",
]


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    description: str
    keeps_all_target_nodes_required: bool
    official_schema_required: bool


PROTOCOLS = {
    PROTOCOL_STANDARD_CONDENSATION: ProtocolSpec(
        name=PROTOCOL_STANDARD_CONDENSATION,
        description="Condensed graph is used for training under the standard condensation protocol.",
        keeps_all_target_nodes_required=False,
        official_schema_required=False,
    ),
    PROTOCOL_SCHEMA_PRESERVING_TP: ProtocolSpec(
        name=PROTOCOL_SCHEMA_PRESERVING_TP,
        description="All target nodes and the official HGB schema are preserved for TP workload evaluation.",
        keeps_all_target_nodes_required=True,
        official_schema_required=True,
    ),
}


def _as_bool(value: Any) -> bool:
    return bool(value)


def _ratio(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def build_protocol_row(
    *,
    baseline_name: str,
    protocol: str,
    method_family: str = "",
    schema_compatible: bool = False,
    official_sehgnn_unmodified: bool = False,
    uses_feature_adapter: bool = False,
    uses_weighted_superedges: bool = False,
    uses_synthetic_nodes: bool = False,
    keeps_all_target_nodes: bool = False,
    used_test_data: bool = False,
    support_node_ratio: float | None = None,
    support_edge_ratio: float | None = None,
    structural_storage_ratio: float | None = None,
    raw_hgb_text_byte_ratio: float | None = None,
    official_text_hgb_byte_ratio: float | None = None,
    binary_feature_sidecar_ratio: float | None = None,
    adapter_package_ratio: float | None = None,
    preprocessed_cache_byte_ratio: float | None = None,
    train_time_seconds: float | None = None,
    preprocess_time_seconds: float | None = None,
    compress_time_seconds: float | None = None,
    peak_cpu_memory_mb: float | None = None,
    peak_gpu_memory_mb: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported ICDE protocol: {protocol}")

    schema_compatible = _as_bool(schema_compatible)
    official_sehgnn_unmodified = _as_bool(official_sehgnn_unmodified)
    uses_feature_adapter = _as_bool(uses_feature_adapter)
    uses_weighted_superedges = _as_bool(uses_weighted_superedges)
    uses_synthetic_nodes = _as_bool(uses_synthetic_nodes)
    keeps_all_target_nodes = _as_bool(keeps_all_target_nodes)
    used_test_data = _as_bool(used_test_data)

    failures: list[str] = []
    if not official_sehgnn_unmodified:
        failures.append("official_sehgnn_unmodified_false")
    if not schema_compatible:
        failures.append("schema_incompatible")
    if not keeps_all_target_nodes:
        failures.append("drops_target_nodes")
    if uses_feature_adapter:
        failures.append("uses_feature_adapter")
    if uses_weighted_superedges:
        failures.append("uses_weighted_superedges")
    if uses_synthetic_nodes:
        failures.append("uses_synthetic_nodes")
    if used_test_data:
        failures.append("used_test_data")

    eligible_for_official_main = len(failures) == 0
    eligible_for_adapter = (
        schema_compatible
        and keeps_all_target_nodes
        and uses_feature_adapter
        and not uses_weighted_superedges
        and not used_test_data
    )
    eligible_for_standard = protocol == PROTOCOL_STANDARD_CONDENSATION and not used_test_data
    eligible_for_tp = (
        protocol == PROTOCOL_SCHEMA_PRESERVING_TP
        and schema_compatible
        and keeps_all_target_nodes
        and not used_test_data
    )

    row: dict[str, Any] = {
        "baseline_name": str(baseline_name),
        "protocol": str(protocol),
        "method_family": str(method_family),
        "schema_compatible": schema_compatible,
        "official_sehgnn_unmodified": official_sehgnn_unmodified,
        "uses_feature_adapter": uses_feature_adapter,
        "uses_weighted_superedges": uses_weighted_superedges,
        "uses_synthetic_nodes": uses_synthetic_nodes,
        "keeps_all_target_nodes": keeps_all_target_nodes,
        "used_test_data": used_test_data,
        "eligible_for_official_main_table": eligible_for_official_main,
        "eligible_for_adapter_table": eligible_for_adapter,
        "eligible_for_standard_condensation_table": eligible_for_standard,
        "eligible_for_tp_workload_table": eligible_for_tp,
        "eligibility_failure_reasons": ";".join(failures),
        "support_node_ratio": _ratio(support_node_ratio),
        "support_edge_ratio": _ratio(support_edge_ratio),
        "structural_storage_ratio": _ratio(structural_storage_ratio),
        "raw_hgb_text_byte_ratio": _ratio(raw_hgb_text_byte_ratio),
        "official_text_hgb_byte_ratio": _ratio(official_text_hgb_byte_ratio),
        "binary_feature_sidecar_ratio": _ratio(binary_feature_sidecar_ratio),
        "adapter_package_ratio": _ratio(adapter_package_ratio),
        "preprocessed_cache_byte_ratio": _ratio(preprocessed_cache_byte_ratio),
        "train_time_seconds": _ratio(train_time_seconds),
        "preprocess_time_seconds": _ratio(preprocess_time_seconds),
        "compress_time_seconds": _ratio(compress_time_seconds),
        "peak_cpu_memory_mb": _ratio(peak_cpu_memory_mb),
        "peak_gpu_memory_mb": _ratio(peak_gpu_memory_mb),
    }
    row.update(extra)
    return row


def assess_graph_seed_stability(
    *,
    deterministic_graph_method: bool,
    graph_seeds: Iterable[int],
    export_hashes: Iterable[str],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in graph_seeds]
    hashes = [str(value) for value in export_hashes if value is not None and str(value) != ""]
    unique_count = len(set(hashes))
    expected = 1 if deterministic_graph_method else 5
    warnings: list[str] = []
    if not deterministic_graph_method and len(set(seeds)) < 5:
        warnings.append("graph_seed_count_lt_5")
    if deterministic_graph_method and unique_count != 1:
        warnings.append("deterministic_export_hash_not_unique")
    if not deterministic_graph_method and unique_count < 2 and len(set(seeds)) >= 5:
        warnings.append("stochastic_export_hash_not_varying")
    return {
        "deterministic_graph_method": bool(deterministic_graph_method),
        "expected_export_hash_unique_count": int(expected),
        "actual_export_hash_unique_count": int(unique_count),
        "graph_seed_count": len(set(seeds)),
        "graph_sampling_stability_pass": len(warnings) == 0,
        "graph_sampling_stability_warnings": ";".join(warnings),
    }


def protocol_csv_fields(extra_fields: Iterable[str] = ()) -> list[str]:
    fields = list(PROTOCOL_FIELDS)
    for field in extra_fields:
        if field not in fields:
            fields.append(str(field))
    return fields
