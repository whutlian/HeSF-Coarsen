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
