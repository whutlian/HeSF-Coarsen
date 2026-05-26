from __future__ import annotations

from typing import Any


def compute_effective_total_bytes(
    *,
    link_dat_bytes: int,
    label_dat_bytes: int,
    label_test_dat_bytes: int,
    info_dat_bytes: int,
    sidecar_feature_bytes: int,
    sidecar_metadata_bytes: int = 0,
    preprocessed_cache_bytes: int = 0,
    adapter_config_bytes: int = 0,
) -> int:
    return int(
        link_dat_bytes
        + label_dat_bytes
        + label_test_dat_bytes
        + info_dat_bytes
        + sidecar_feature_bytes
        + sidecar_metadata_bytes
        + preprocessed_cache_bytes
        + adapter_config_bytes
    )


def adapter_storage_row(
    *,
    dataset: str,
    method: str,
    base_graph_method: str,
    graph_seed: int,
    training_seed: int,
    native_full_total_bytes: int,
    export_total_bytes: int,
    node_dat_bytes: int,
    link_dat_bytes: int,
    label_dat_bytes: int,
    label_test_dat_bytes: int,
    info_dat_bytes: int,
    sidecar_feature_bytes: int,
    sidecar_metadata_bytes: int = 0,
    preprocessed_cache_bytes: int = 0,
    adapter_config_bytes: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    effective = compute_effective_total_bytes(
        link_dat_bytes=link_dat_bytes,
        label_dat_bytes=label_dat_bytes,
        label_test_dat_bytes=label_test_dat_bytes,
        info_dat_bytes=info_dat_bytes,
        sidecar_feature_bytes=sidecar_feature_bytes,
        sidecar_metadata_bytes=sidecar_metadata_bytes,
        preprocessed_cache_bytes=preprocessed_cache_bytes,
        adapter_config_bytes=adapter_config_bytes,
    )
    native = max(int(native_full_total_bytes), 1)
    row = {
        "dataset": str(dataset).upper(),
        "method": method,
        "base_graph_method": base_graph_method,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "raw_hgb_byte_ratio": float(int(export_total_bytes) / native),
        "effective_total_byte_ratio": float(effective / native),
        "binary_feature_sidecar_byte_ratio": float(int(sidecar_feature_bytes) / native),
        "sidecar_feature_bytes": int(sidecar_feature_bytes),
        "sidecar_metadata_bytes": int(sidecar_metadata_bytes),
        "node_dat_bytes": int(node_dat_bytes),
        "link_dat_bytes": int(link_dat_bytes),
        "label_dat_bytes": int(label_dat_bytes),
        "label_test_dat_bytes": int(label_test_dat_bytes),
        "info_dat_bytes": int(info_dat_bytes),
        "export_total_bytes": int(export_total_bytes),
        "native_full_total_bytes": int(native_full_total_bytes),
        "preprocessed_cache_bytes": int(preprocessed_cache_bytes),
        "official_sehgnn_unmodified": False,
        "eligible_for_main_decision": False,
        "adapter_family": "feature_cache_compression",
    }
    row.update(extra)
    return row
