from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_gate21_13_cross_dataset(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)

    out: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(grouped.items()):
        ready = [row for row in group if _bool(row.get("training_executed")) and _finite(row.get("test_micro_f1"))]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "row_count": len(group),
                "success_count": len(ready),
                "cross_dataset_ready": bool(ready),
                "test_micro_f1_mean": _mean(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean(ready, "test_macro_f1"),
            }
        )
    return out


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "NaN" if not finite else sum(finite) / len(finite)


def _finite(value: object) -> bool:
    return _float(value) is not None


def _float(value: object) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
