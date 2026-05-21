from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        item = {name: value for name, value in zip(group_keys, key)}
        item["runs"] = int(len(group))
        for metric in metrics:
            values = []
            for row in group:
                try:
                    values.append(float(row.get(metric)))
                except (TypeError, ValueError):
                    pass
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def ratio_matched_gaps(
    method_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_names: set[str],
    tolerance: float = 0.035,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in baseline_rows:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("method"))), []).append(row)
    for row in method_rows:
        ratio = _float(row.get("realized_support_ratio"))
        macro = _float(row.get("macro_f1"))
        acc = _float(row.get("accuracy"))
        for baseline in sorted(baseline_names):
            candidates = by_key.get((str(row.get("dataset")), str(row.get("seed")), baseline), [])
            if ratio is None or not candidates:
                out.append({"dataset": row.get("dataset"), "seed": row.get("seed"), "method": row.get("method"), "baseline": baseline, "comparison_status": "missing_baseline"})
                continue
            scored = []
            for candidate in candidates:
                b_ratio = _float(candidate.get("realized_support_ratio"))
                if b_ratio is not None:
                    scored.append((abs(float(ratio) - float(b_ratio)), candidate))
            if not scored:
                continue
            gap, best = min(scored, key=lambda item: item[0])
            b_macro = _float(best.get("macro_f1"))
            b_acc = _float(best.get("accuracy"))
            status = "matched" if gap <= tolerance else "nearest_flagged"
            out.append(
                {
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "method": row.get("method"),
                    "baseline": baseline,
                    "requested_support_ratio": row.get("requested_support_ratio"),
                    "realized_support_ratio": ratio,
                    "baseline_realized_support_ratio": b_ratio,
                    "ratio_gap": float(gap),
                    "comparison_status": status,
                    "delta_macro_f1": "" if macro is None or b_macro is None else float(macro - b_macro),
                    "delta_accuracy": "" if acc is None or b_acc is None else float(acc - b_acc),
                }
            )
    return out


def _float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
