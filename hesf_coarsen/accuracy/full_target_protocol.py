from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


PROTOCOL_DOMAINS = {
    "coarse_transfer": {
        "target_domain": "coarse_target_nodes",
        "support_domain": "compressed_support_nodes",
        "inference_domain": "projected_original_targets",
        "metric_source": "projected_original",
    },
    "approx_full_target_adapter": {
        "target_domain": "adapter_projected_targets",
        "support_domain": "compressed_support_nodes",
        "inference_domain": "projected_original_targets",
        "metric_source": "projected_original",
    },
    "real_full_target_inference": {
        "target_domain": "original_target_nodes",
        "support_domain": "compressed_support_nodes",
        "inference_domain": "full_original_target_set",
        "metric_source": "hybrid_target_original",
    },
}


def target_preserve_protocol_report(
    original: HeteroGraph,
    hybrid: HeteroGraph,
    original_to_hybrid: np.ndarray,
    *,
    target_node_type: int,
) -> dict[str, Any]:
    mapping = np.asarray(original_to_hybrid, dtype=np.int64).reshape(-1)
    target_nodes = nodes_of_type(original, int(target_node_type))
    support_nodes = np.flatnonzero(original.node_type != int(target_node_type))
    if mapping.shape != (original.num_nodes,):
        return {
            "target_mapping_valid": False,
            "target_mapping_one_to_one": False,
            "target_domain": "invalid",
            "support_domain": "invalid",
            "inference_domain": "invalid",
            "reason": "mapping_shape_mismatch",
        }
    mapped_target = mapping[target_nodes]
    valid = bool(np.all((mapped_target >= 0) & (mapped_target < hybrid.num_nodes)))
    same_type = bool(valid and np.all(hybrid.node_type[mapped_target] == int(target_node_type)))
    one_to_one = bool(valid and len(np.unique(mapped_target)) == len(mapped_target))
    support_mapped = mapping[support_nodes] if len(support_nodes) else np.array([], dtype=np.int64)
    support_compressed = bool(len(np.unique(support_mapped)) < len(support_nodes)) if len(support_nodes) else False
    return {
        "target_mapping_valid": bool(valid and same_type),
        "target_mapping_same_type": same_type,
        "target_mapping_one_to_one": one_to_one,
        "target_original_nodes": int(len(target_nodes)),
        "target_hybrid_nodes": int(len(np.unique(mapped_target))) if valid else 0,
        "support_original_nodes": int(len(support_nodes)),
        "support_hybrid_nodes": int(len(np.unique(support_mapped))) if len(support_nodes) else 0,
        "support_compressed": support_compressed,
        "target_domain": "original_target_nodes" if one_to_one and same_type else "coarse_or_merged_targets",
        "support_domain": "compressed_support_nodes" if support_compressed else "support_nodes",
        "inference_domain": "full_original_target_set" if one_to_one and same_type else "projected_original_targets",
    }


def make_protocol_row(
    metrics: Mapping[str, Any],
    *,
    eval_mode: str,
    model_name: str,
    model_fidelity: str,
    official_repo: str,
    official_preprocess: str,
    adapter_mode: str,
    path_set: str,
    split_policy: str,
    max_hops: int | str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if eval_mode not in PROTOCOL_DOMAINS:
        raise ValueError(f"unsupported eval_mode: {eval_mode}")
    domains = dict(PROTOCOL_DOMAINS[eval_mode])
    metric_source = str(domains["metric_source"])
    prefix = {
        "projected_original": "projected_original",
        "hybrid_target_original": "hybrid_target_original",
    }[metric_source]
    row = {
        "model_name": str(model_name),
        "model_fidelity": str(model_fidelity),
        "eval_mode": str(eval_mode),
        "official_repo": str(official_repo),
        "official_preprocess": str(official_preprocess),
        "adapter_mode": str(adapter_mode),
        "split_policy": str(split_policy),
        "path_set": str(path_set),
        "max_hops": max_hops,
        **domains,
        "micro_f1": metrics.get(f"{prefix}_micro_f1", ""),
        "macro_f1": metrics.get(f"{prefix}_macro_f1", ""),
        "accuracy": metrics.get(f"{prefix}_accuracy", metrics.get(f"{prefix}_micro_f1", "")),
    }
    if extra:
        row.update(dict(extra))
    return row


def required_provenance_fields() -> tuple[str, ...]:
    return (
        "eval_mode",
        "official_repo",
        "official_preprocess",
        "adapter_mode",
        "path_set",
        "target_domain",
        "support_domain",
        "inference_domain",
        "model_fidelity",
    )
