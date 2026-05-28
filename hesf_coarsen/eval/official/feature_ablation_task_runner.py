from __future__ import annotations


GATE21_9_KEY_FEATURE_TRANSFORMS = (
    "raw",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-all-support-preserve-dim",
    "paper-random-projection64",
)


def key_feature_transforms() -> tuple[str, ...]:
    return GATE21_9_KEY_FEATURE_TRANSFORMS
