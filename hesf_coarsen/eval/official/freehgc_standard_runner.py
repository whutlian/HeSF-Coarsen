from __future__ import annotations

from typing import Any, Iterable, Mapping


FREEHGC_STANDARD_RATIOS = (0.012, 0.024, 0.048, 0.096, 0.120)


def freehgc_standard_ratios() -> tuple[float, ...]:
    return FREEHGC_STANDARD_RATIOS


def build_gate21_15_freehgc_standard_rows(
    *,
    datasets: Iterable[str],
    repo_audit_rows: Iterable[Mapping[str, Any]],
    ratios: Iterable[float] = FREEHGC_STANDARD_RATIOS,
) -> list[dict[str, Any]]:
    repo = next((dict(row) for row in repo_audit_rows if str(row.get("baseline_name")) == "FreeHGC"), {})
    repo_url = repo.get("repo_url", "https://github.com/GooLiang/FreeHGC")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for ratio in ratios:
            rows.append(
                {
                    "dataset": str(dataset).upper(),
                    "method": "FreeHGC-standard",
                    "ratio": float(ratio),
                    "method_family": "standard_condensation",
                    "protocol": "freehgc_standard_condensation",
                    "repo_url": repo_url,
                    "clone_success": repo.get("clone_success", False),
                    "required_files_present": repo.get("required_files_present", False),
                    "success": False,
                    "seed_count": 0,
                    "success_count": 0,
                    "expected_seed_count": 5,
                    "training_executed": False,
                    "eligible_for_main_table": False,
                    "eligible_for_standard_condensation_table": True,
                    "eligible_for_tp_workload_table": False,
                    "mean_micro": "NaN",
                    "mean_macro": "NaN",
                    "test_micro_f1_mean": "NaN",
                    "test_macro_f1_mean": "NaN",
                    "failure_type": "freehgc_standard_not_runnable",
                    "failure_reason": "FreeHGC-standard is a separate condensation protocol and no local runnable standard HGB task result is available under the current pytorch environment.",
                }
            )
    return rows


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


def summarize_gate21_12_freehgc_standard(rows: list[dict[str, object]], *, expected_seed_count: int = 5) -> list[dict[str, object]]:
    out = summarize_gate21_11_freehgc_standard(rows, expected_seed_count=expected_seed_count)
    for row in out:
        row["protocol"] = "freehgc_standard_condensation"
        row["FREEHGC_STANDARD_TASK_RESULTS_READY"] = bool(
            _bool(row.get("eligible_for_decision"))
            and _finite(row.get("test_micro_f1_mean"))
            and _finite(row.get("test_macro_f1_mean"))
        )
    return out


def summarize_gate21_13_freehgc_standard(rows: list[dict[str, object]], *, expected_seed_count: int = 5) -> list[dict[str, object]]:
    out = summarize_gate21_12_freehgc_standard(rows, expected_seed_count=expected_seed_count)
    for row in out:
        row["expected_seed_count"] = int(expected_seed_count)
        row["ready"] = bool(_float(row.get("success_count")) is not None and (_float(row.get("success_count")) or 0.0) >= expected_seed_count and _finite(row.get("test_micro_f1_mean")))
        row.setdefault("failure_reason", "" if row["ready"] else "FreeHGC standard 5-seed evidence is missing or invalid.")
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
