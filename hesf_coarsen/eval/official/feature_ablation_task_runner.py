from __future__ import annotations


GATE21_9_KEY_FEATURE_TRANSFORMS = (
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-random-projection64",
)

GATE21_10_METHODS = ("full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16")
GATE21_10_FEATURE_TRANSFORMS = (
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-venue-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-only-preserve-original-dims",
    "term-only-preserve-original-dims",
    "paper-random-projection64",
    "paper-pca64",
)
GATE21_10_LABEL_GRAPH_SETTINGS = (
    "default",
    "no_label_feats",
    "num_feature_hops_0",
    "num_label_hops_0",
    "feature_only_mlp_adapter",
)

GATE21_12_REQUIRED_FEATURE_TRANSFORMS = (
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-only-preserve-original-dims",
    "term-only-preserve-original-dims",
    "paper-random-projection64",
    "paper-pca64",
)


def key_feature_transforms() -> tuple[str, ...]:
    return GATE21_9_KEY_FEATURE_TRANSFORMS


def feature_ablation_ready(rows: list[dict[str, object]]) -> bool:
    seen = {
        (str(row.get("method", "")), str(row.get("feature_transform", "")), str(row.get("label_graph_setting", "")))
        for row in rows
        if _bool(row.get("training_executed")) and _finite(row.get("test_micro_f1")) and _finite(row.get("test_macro_f1"))
    }
    return all(
        (method, transform, setting) in seen
        for method in GATE21_10_METHODS
        for transform in GATE21_10_FEATURE_TRANSFORMS
        for setting in GATE21_10_LABEL_GRAPH_SETTINGS
    )


def summarize_gate21_12_feature_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    out: list[dict[str, object]] = []
    for method, group in sorted(grouped.items()):
        ready = [
            row
            for row in group
            if _bool(row.get("training_executed"))
            and _finite(row.get("test_micro_f1"))
            and _finite(row.get("test_macro_f1"))
        ]
        out.append(
            {
                "method": method,
                "row_count": len(group),
                "success_count": len(ready),
                "test_micro_f1_mean": _mean_or_nan(ready, "test_micro_f1"),
                "test_macro_f1_mean": _mean_or_nan(ready, "test_macro_f1"),
                "answers_ready": bool(ready),
                "pt_tp_removal_answer": "not_ready_task_metrics_missing" if not ready else "see_transform_deltas",
                "pa_vp50_feedback_answer": "not_ready_task_metrics_missing" if not ready else "see_apv12_vs_apv16",
                "support_raw_feature_answer": "not_ready_task_metrics_missing" if not ready else "see_zero_support_rows",
                "paper_only_survival_answer": "not_ready_task_metrics_missing" if not ready else "see_paper_only_rows",
            }
        )
    return out


def summarize_gate21_13_feature_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return summarize_gate21_12_feature_ablation(rows)


def _mean_or_nan(rows: list[dict[str, object]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return "NaN" if not finite else sum(finite) / len(finite)


def _float(value: object) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _finite(value: object) -> bool:
    if value in {"", None}:
        return False
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return parsed == parsed and parsed not in {float("inf"), float("-inf")}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
