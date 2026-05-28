from __future__ import annotations


FREEHGC_STANDARD_RATIOS = (0.012, 0.024, 0.048, 0.096, 0.120)


def freehgc_standard_ratios() -> tuple[float, ...]:
    return FREEHGC_STANDARD_RATIOS


def summarize_gate21_11_freehgc_standard(rows: list[dict[str, object]], *, expected_seed_count: int = 5) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("ratio", row.get("reduction_rate", ""))), []).append(row)
    out: list[dict[str, object]] = []
    for ratio, group in sorted(grouped.items()):
        ready = [
            row
            for row in group
            if _bool(row.get("success"))
            and _bool(row.get("training_executed"))
            and _finite(row.get("test_micro_f1"))
            and _finite(row.get("test_macro_f1"))
        ]
        imported_unverified = any(_bool(row.get("imported_unverified_metric")) for row in group)
        success_count = len(ready)
        out.append(
            {
                "method": f"FreeHGC-standard-ratio{ratio}",
                "ratio": ratio,
                "row_count": len(group),
                "success_count": success_count,
                "expected_seed_count": int(expected_seed_count),
                "seed_count": len({str(row.get("seed", row.get("training_seed", ""))) for row in ready if str(row.get("seed", row.get("training_seed", "")))}),
                "test_micro_f1_mean": _mean_or_nan(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean_or_nan(ready, "test_macro_f1"),
                "imported_unverified_metric": imported_unverified,
                "eligible_for_decision": success_count >= int(expected_seed_count),
                "eligible_for_standard_condensation_table": True,
                "eligible_for_tp_workload_table": False,
            }
        )
    return out


def _mean_or_nan(rows: list[dict[str, object]], field: str) -> float | str:
    vals = [_float(row.get(field)) for row in rows]
    finite = [val for val in vals if val is not None]
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
