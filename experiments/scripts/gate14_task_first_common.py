from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.scripts.gate13_task_first_common import (  # noqa: F401
    DEFAULT_METRICS,
    DEFAULT_SEEDS,
    DATASETS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    build_gate13_candidates,
    build_random_support_candidates,
    build_sketch_candidates,
    evaluate_graph,
    load_hgb_graph,
    method_token,
    ratio_token,
    run_full_graph_ceiling_row,
    run_multilevel_task_first,
    run_parallel,
    run_support_baseline,
    task_first_config,
    write_summary_md,
)


GATE14_RATIOS = (0.048, 0.096, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PRIMARY_GATE14_RATIOS = (0.12, 0.15, 0.20, 0.25, 0.30, 0.40)
BASELINE_METHODS = (
    "flatten-sum-support-only",
    "H6-no-spec-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    return str(value).strip()


def _norm_seed(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def build_ratio_matched_rows(
    hesf_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.025,
    non_comparable_gap: float = 0.05,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    baselines_by_key: dict[tuple[Any, Any, str], list[Mapping[str, Any]]] = {}
    for row in baseline_rows:
        key = (_norm_text(row.get("dataset")), _norm_seed(row.get("seed")), str(row.get("method")))
        baselines_by_key.setdefault(key, []).append(row)
    baseline_methods = sorted({str(row.get("method")) for row in baseline_rows})
    for hesf in hesf_rows:
        h_ratio = _float_or_none(hesf.get("realized_support_ratio"))
        h_macro = _float_or_none(hesf.get("macro_f1", hesf.get("task.macro_f1")))
        h_acc = _float_or_none(hesf.get("accuracy", hesf.get("task.accuracy")))
        for baseline in baseline_methods:
            candidates = baselines_by_key.get((_norm_text(hesf.get("dataset")), _norm_seed(hesf.get("seed")), baseline), [])
            scored: list[tuple[float, Mapping[str, Any]]] = []
            for row in candidates:
                b_ratio = _float_or_none(row.get("realized_support_ratio"))
                if h_ratio is None or b_ratio is None:
                    continue
                scored.append((abs(float(b_ratio) - float(h_ratio)), row))
            if not scored:
                out.append(
                    {
                        "dataset": hesf.get("dataset"),
                        "seed": hesf.get("seed"),
                        "method": hesf.get("method"),
                        "baseline": baseline,
                        "comparison_status": "missing_baseline",
                        "ratio_gap": "",
                        "delta_macro_f1": "",
                        "delta_accuracy": "",
                    }
                )
                continue
            gap, matched = min(scored, key=lambda item: (item[0], _float_or_none(item[1].get("requested_support_ratio")) or 0.0))
            gap = float(round(gap, 12))
            b_macro = _float_or_none(matched.get("macro_f1", matched.get("task.macro_f1")))
            b_acc = _float_or_none(matched.get("accuracy", matched.get("task.accuracy")))
            if gap > float(non_comparable_gap):
                status = "non_comparable"
                delta_macro: float | str = ""
                delta_acc: float | str = ""
            elif gap <= float(tolerance):
                status = "matched"
                delta_macro = float(h_macro - b_macro) if h_macro is not None and b_macro is not None else ""
                delta_acc = float(h_acc - b_acc) if h_acc is not None and b_acc is not None else ""
            else:
                status = "nearest_flagged"
                delta_macro = float(h_macro - b_macro) if h_macro is not None and b_macro is not None else ""
                delta_acc = float(h_acc - b_acc) if h_acc is not None and b_acc is not None else ""
            out.append(
                {
                    "dataset": hesf.get("dataset"),
                    "seed": hesf.get("seed"),
                    "method": hesf.get("method"),
                    "baseline": baseline,
                    "requested_support_ratio": hesf.get("requested_support_ratio", hesf.get("ratio")),
                    "realized_support_ratio": h_ratio,
                    "baseline_requested_support_ratio": matched.get("requested_support_ratio", matched.get("ratio")),
                    "baseline_realized_support_ratio": _float_or_none(matched.get("realized_support_ratio")),
                    "ratio_gap": gap,
                    "comparison_status": status,
                    "delta_macro_f1": delta_macro,
                    "delta_accuracy": delta_acc,
                }
            )
    return out


def compute_recovery_vs_ceiling(
    compressed_rows: Sequence[Mapping[str, Any]],
    ceiling_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ceiling_by_key = {
        (_norm_text(row.get("dataset")), _norm_seed(row.get("seed"))): row
        for row in ceiling_rows
        if row.get("dataset") not in {"", None} and row.get("seed") not in {"", None}
    }
    out: list[dict[str, Any]] = []
    for row in compressed_rows:
        ceiling = ceiling_by_key.get((_norm_text(row.get("dataset")), _norm_seed(row.get("seed"))))
        macro = _float_or_none(row.get("macro_f1", row.get("task.macro_f1")))
        acc = _float_or_none(row.get("accuracy", row.get("task.accuracy")))
        if ceiling is None:
            out.append(
                {
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "method": row.get("method"),
                    "recovery_status": "missing_full_graph_lite_ceiling",
                    "recovery_vs_full_graph_lite_macro": "",
                    "recovery_vs_full_graph_lite_accuracy": "",
                }
            )
            continue
        ceiling_macro = _float_or_none(ceiling.get("macro_f1", ceiling.get("task.macro_f1")))
        ceiling_acc = _float_or_none(ceiling.get("accuracy", ceiling.get("task.accuracy")))
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "ratio": row.get("ratio", row.get("requested_support_ratio")),
                "recovery_status": "ok",
                "recovery_vs_full_graph_lite_macro": round(float(macro / ceiling_macro), 12) if macro is not None and ceiling_macro else "",
                "recovery_vs_full_graph_lite_accuracy": round(float(acc / ceiling_acc), 12) if acc is not None and ceiling_acc else "",
            }
        )
    return out


def select_validation_best_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status", "success") == "success":
            groups.setdefault((row.get("dataset"), row.get("seed")), []).append(row)
    selected: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        best = max(
            group,
            key=lambda row: (
                _float_or_none(row.get("validation_macro_f1")) if _float_or_none(row.get("validation_macro_f1")) is not None else -1.0,
                _float_or_none(row.get("validation_accuracy")) if _float_or_none(row.get("validation_accuracy")) is not None else -1.0,
            ),
        )
        out = dict(best)
        out["selected_by_validation"] = True
        selected.append(out)
    return selected


def evaluator_status_rows() -> dict[str, str]:
    return {
        "official_sehgnn_status": "not_integrated",
        "official_hettree_status": "not_integrated",
        "freehgc_status": "not_integrated",
        "diagnostic_scope": "diagnostic_lite_only",
    }


def write_placeholder_png(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.set_title(title)
        ax.set_xlabel("realized support ratio")
        ax.set_ylabel("metric")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        path.write_bytes(b"")
