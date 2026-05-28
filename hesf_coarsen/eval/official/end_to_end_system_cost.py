from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence


REQUIRED_SYSTEM_COST_FIELDS = (
    "official_sehgnn_preprocess_time_seconds",
    "training_time_seconds",
    "peak_cpu_rss_mb",
    "preprocessed_cache_bytes",
    "test_micro_f1",
    "test_macro_f1",
)


def gate21_11_system_cost_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    ready = [row for row in rows if _bool(row.get("training_executed"))]
    return bool(ready) and all(all(_positive(row.get(field)) for field in REQUIRED_SYSTEM_COST_FIELDS[:4]) and _finite(row.get("test_micro_f1")) and _finite(row.get("test_macro_f1")) for row in ready)


def summarize_gate21_11_system_cost(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(grouped.items()):
        ready = [row for row in group if _bool(row.get("training_executed")) and all(_finite(row.get(field)) for field in ("test_micro_f1", "test_macro_f1"))]
        out.append(
            {
                "method": method,
                "row_count": len(group),
                "success_count": len(ready),
                "official_sehgnn_preprocess_time_seconds_mean": _mean(ready, "official_sehgnn_preprocess_time_seconds"),
                "training_time_seconds_mean": _mean(ready, "training_time_seconds"),
                "peak_cpu_rss_mb_mean": _mean(ready, "peak_cpu_rss_mb"),
                "preprocessed_cache_bytes_mean": _mean(ready, "preprocessed_cache_bytes"),
                "test_micro_f1_mean": _mean(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean(ready, "test_macro_f1"),
                "system_cost_end_to_end_ready": gate21_11_system_cost_ready(group),
            }
        )
    return out


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    vals = [_float(row.get(field)) for row in rows]
    finite = [val for val in vals if val is not None]
    return "NaN" if not finite else mean(finite)


def _positive(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _float(value: Any) -> float | None:
    if value in {"", None, "NaN", "nan"}:
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

