from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SYSTEM_WORKLOAD_FIELDS = (
    "dataset",
    "artifact_method",
    "protocol",
    "artifact_bytes",
    "ratio_vs_original_native_full_hgb_text",
    "ratio_vs_export_full_hgb_text",
    "ratio_vs_current_control_artifact",
    "compression_time_seconds",
    "export_time_seconds",
    "load_time_seconds",
    "decompress_time_seconds",
    "adapter_load_time_seconds",
    "official_sehgnn_preprocess_time_seconds",
    "training_time_seconds",
    "eval_time_seconds",
    "total_workload_time_seconds",
    "peak_cpu_rss_mb",
    "peak_gpu_memory_mb",
    "preprocessed_cache_bytes",
    "cache_file_count",
    "training_executed",
    "task_micro_f1",
    "task_macro_f1",
    "loader_supported",
    "official_sehgnn_unmodified",
    "uses_adapter",
    "failure_type",
    "failure_message",
)


def system_workload_cost_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready_rows = [row for row in rows if _bool(row.get("training_executed"))]
    return bool(ready_rows) and all(_row_ready(row) for row in ready_rows)


def normalize_system_workload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    total = sum(_float(out.get(field)) or 0.0 for field in ("load_time_seconds", "decompress_time_seconds", "adapter_load_time_seconds", "official_sehgnn_preprocess_time_seconds", "training_time_seconds", "eval_time_seconds"))
    out.setdefault("total_workload_time_seconds", total if total else "")
    for field in SYSTEM_WORKLOAD_FIELDS:
        out.setdefault(field, "")
    return {field: out.get(field, "") for field in SYSTEM_WORKLOAD_FIELDS}


def _row_ready(row: Mapping[str, Any]) -> bool:
    return all(_positive(row.get(field)) for field in ("artifact_bytes", "load_time_seconds", "official_sehgnn_preprocess_time_seconds", "training_time_seconds", "peak_cpu_rss_mb", "preprocessed_cache_bytes")) and _finite(row.get("task_micro_f1")) and _finite(row.get("task_macro_f1"))


def _positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
