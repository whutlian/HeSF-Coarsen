from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompressionCost:
    method: str
    dataset: str
    seed: int
    requested_budget: float

    support_node_count: int = 0
    support_edge_count: int = 0
    unit_count: int = 0
    path_channel_count: int = 0
    feature_cache_elements: int = 0
    feature_cache_bytes: int = 0
    logit_cache_bytes: int = 0
    model_param_bytes: int = 0

    full_support_node_count: int = 0
    full_support_edge_count: int = 0
    full_unit_count: int = 0
    full_path_channel_count: int = 0
    full_feature_cache_elements: int = 0
    full_feature_cache_bytes: int = 0
    full_logit_cache_bytes: int = 0
    full_model_param_bytes: int = 0

    support_node_ratio: float = 0.0
    support_edge_ratio: float = 0.0
    unit_count_ratio: float = 0.0
    path_channel_count_ratio: float = 0.0
    feature_cache_size_ratio: float = 0.0
    total_storage_bytes: int = 0
    total_storage_ratio_vs_full_stc: float = 0.0
    total_storage_ratio_vs_full_graph: float = 0.0


def compute_feature_cache_bytes(array_or_shape: Any, dtype: Any = np.float32) -> int:
    if isinstance(array_or_shape, np.ndarray):
        arr = np.asarray(array_or_shape)
        return int(arr.size * np.dtype(dtype if dtype is not None else arr.dtype).itemsize)
    shape = tuple(int(value) for value in array_or_shape)
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def count_model_parameters_bytes(model: Any) -> int:
    if model is None:
        return 0
    total = 0
    try:
        iterator = model.parameters()
    except AttributeError:
        return 0
    for param in iterator:
        numel = int(param.numel())
        try:
            itemsize = int(param.element_size())
        except AttributeError:
            itemsize = int(np.asarray(param.detach().cpu().numpy()).dtype.itemsize)
        total += numel * itemsize
    return int(total)


def _ratio(value: int | float, full: int | float) -> float:
    full_value = float(full)
    if full_value <= 0.0:
        return 0.0
    return float(value) / full_value


def compute_total_storage_ratio(cost: CompressionCost) -> CompressionCost:
    total = int(cost.feature_cache_bytes + cost.logit_cache_bytes + cost.model_param_bytes + cost.support_node_count * 8 + cost.support_edge_count * 16 + cost.unit_count * 8)
    full_stc = int(cost.full_feature_cache_bytes + cost.full_logit_cache_bytes + cost.full_model_param_bytes)
    full_graph = int(full_stc + cost.full_support_node_count * 8 + cost.full_support_edge_count * 16 + cost.full_unit_count * 8)
    if full_stc <= 0:
        full_stc = int(cost.full_feature_cache_bytes or cost.feature_cache_bytes or 1)
    if full_graph <= 0:
        full_graph = int(full_stc)
    return replace(
        cost,
        support_node_ratio=_ratio(cost.support_node_count, cost.full_support_node_count),
        support_edge_ratio=_ratio(cost.support_edge_count, cost.full_support_edge_count),
        unit_count_ratio=_ratio(cost.unit_count, cost.full_unit_count),
        path_channel_count_ratio=_ratio(cost.path_channel_count, cost.full_path_channel_count),
        feature_cache_size_ratio=_ratio(cost.feature_cache_bytes, cost.full_feature_cache_bytes),
        total_storage_bytes=int(total),
        total_storage_ratio_vs_full_stc=float(total / max(full_stc, 1)),
        total_storage_ratio_vs_full_graph=float(total / max(full_graph, 1)),
    )


def assert_cost_finite(cost: CompressionCost) -> None:
    values = asdict(cost)
    for key, value in values.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"non-finite cost field {key}: {value}")
        if isinstance(value, int) and value < 0:
            raise ValueError(f"negative cost field {key}: {value}")
    if str(cost.method).startswith("STC") and int(cost.feature_cache_bytes) <= 0:
        raise ValueError("STC methods must report nonzero feature_cache_bytes")


def cost_to_row(cost: CompressionCost, *, cost_axis_used: str = "total_storage_ratio_vs_full_stc") -> dict[str, Any]:
    computed = compute_total_storage_ratio(cost)
    assert_cost_finite(computed)
    return {
        "dataset": computed.dataset,
        "seed": int(computed.seed),
        "method": computed.method,
        "requested_budget": float(computed.requested_budget),
        "support_node_ratio": float(computed.support_node_ratio),
        "support_edge_ratio": float(computed.support_edge_ratio),
        "unit_count_ratio": float(computed.unit_count_ratio),
        "feature_cache_size_ratio": float(computed.feature_cache_size_ratio),
        "path_channel_count_ratio": float(computed.path_channel_count_ratio),
        "feature_cache_bytes": int(computed.feature_cache_bytes),
        "logit_cache_bytes": int(computed.logit_cache_bytes),
        "model_param_bytes": int(computed.model_param_bytes),
        "total_storage_bytes": int(computed.total_storage_bytes),
        "total_storage_ratio_vs_full_stc": float(computed.total_storage_ratio_vs_full_stc),
        "total_storage_ratio_vs_full_graph": float(computed.total_storage_ratio_vs_full_graph),
        "cost_axis_used": str(cost_axis_used),
    }
