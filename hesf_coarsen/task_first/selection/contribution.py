from __future__ import annotations

from typing import Any

import numpy as np


def _scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1.0e-12:
        return np.ones_like(values, dtype=np.float32) if hi > 0.0 else np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def compute_teacher_support_importance(
    support_features: dict[str, Any],
    teacher_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = support_features["component_matrices"]
    class_mass = np.sum(components.get("class_footprint", np.empty((0, 0))), axis=1)
    anchor_mass = np.sum(components.get("anchor_distribution", np.empty((len(class_mass), 0))), axis=1)
    relation_mass = np.sum(components.get("relation_profile", np.empty((len(class_mass), 0))), axis=1)
    confidence = np.zeros(len(class_mass), dtype=np.float32)
    if teacher_outputs and "logits" in teacher_outputs:
        support_nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
        logits = np.asarray(teacher_outputs["logits"], dtype=np.float32)
        if logits.ndim == 2 and len(support_nodes):
            confidence = np.max(_softmax(logits[support_nodes]), axis=1)
    importance = (
        0.35 * _scale(class_mass)
        + 0.25 * _scale(anchor_mass)
        + 0.20 * _scale(relation_mass)
        + 0.20 * _scale(confidence)
    )
    return {
        "importance": importance.astype(np.float32),
        "components": {
            "class_mass": class_mass.astype(np.float32),
            "anchor_mass": anchor_mass.astype(np.float32),
            "relation_mass": relation_mass.astype(np.float32),
            "teacher_confidence": confidence.astype(np.float32),
        },
    }


def compute_response_support_importance(support_features: dict[str, Any]) -> dict[str, Any]:
    response = support_features["component_matrices"].get("target_response_signature", np.empty((0, 0)))
    relation = support_features["component_matrices"].get("relation_response_signature", np.empty((len(response), 0)))
    response_norm = np.linalg.norm(response, axis=1) if response.size else np.zeros(len(relation), dtype=np.float32)
    relation_norm = np.linalg.norm(relation, axis=1) if relation.size else np.zeros(len(response_norm), dtype=np.float32)
    importance = 0.65 * _scale(response_norm) + 0.35 * _scale(relation_norm)
    return {
        "importance": importance.astype(np.float32),
        "components": {
            "response_norm": response_norm.astype(np.float32),
            "relation_response_norm": relation_norm.astype(np.float32),
        },
    }


def compute_validation_occlusion_importance(
    support_features: dict[str, Any],
    teacher_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    teacher = compute_teacher_support_importance(support_features, teacher_outputs)
    diversity = np.linalg.norm(
        support_features["component_matrices"].get("anchor_distribution", np.empty((len(teacher["importance"]), 0))),
        axis=1,
    )
    importance = 0.75 * teacher["importance"] + 0.25 * _scale(diversity)
    return {
        "importance": importance.astype(np.float32),
        "components": {**teacher["components"], "validation_proxy_diversity": diversity.astype(np.float32)},
    }


def compute_sensitivity_block_importance(
    support_features: dict[str, Any],
    teacher_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    teacher = compute_teacher_support_importance(support_features, teacher_outputs)
    relation = support_features["component_matrices"].get(
        "relation_profile",
        np.empty((len(teacher["importance"]), 0)),
    )
    anchor = support_features["component_matrices"].get(
        "anchor_distribution",
        np.empty((len(teacher["importance"]), 0)),
    )
    class_fp = support_features["component_matrices"].get(
        "class_footprint",
        np.empty((len(teacher["importance"]), 0)),
    )
    relation_mass = np.sum(relation, axis=1) if relation.size else np.zeros(len(teacher["importance"]))
    anchor_mass = np.sum(anchor, axis=1) if anchor.size else np.zeros(len(teacher["importance"]))
    class_margin = np.zeros(len(teacher["importance"]), dtype=np.float32)
    if class_fp.size:
        sorted_fp = np.sort(class_fp, axis=1)
        if sorted_fp.shape[1] >= 2:
            class_margin = (sorted_fp[:, -1] - sorted_fp[:, -2]).astype(np.float32)
        elif sorted_fp.shape[1] == 1:
            class_margin = sorted_fp[:, -1].astype(np.float32)
    diversity_bonus = np.linalg.norm(anchor, axis=1) if anchor.size else np.zeros(len(teacher["importance"]))
    importance = (
        1.0 * teacher["importance"]
        + 0.2 * _scale(relation_mass)
        + 0.2 * _scale(class_margin)
        + 0.1 * _scale(diversity_bonus + anchor_mass)
    )
    return {
        "importance": _scale(importance).astype(np.float32),
        "components": {
            **teacher["components"],
            "sensitivity_relation_mass": np.asarray(relation_mass, dtype=np.float32),
            "sensitivity_class_margin": np.asarray(class_margin, dtype=np.float32),
            "sensitivity_diversity_bonus": np.asarray(diversity_bonus, dtype=np.float32),
        },
    }


def compute_support_importance(
    support_features: dict[str, Any],
    teacher_outputs: dict[str, Any] | None = None,
    *,
    mode: str = "teacher_topk",
    lambda_response: float = 0.05,
) -> dict[str, Any]:
    mode = str(mode)
    if mode in {"teacher_topk", "teacher_diverse_topk", "mlp_importance"}:
        return compute_teacher_support_importance(support_features, teacher_outputs)
    if mode in {"validation_greedy", "validation_proxy_diverse"}:
        return compute_validation_occlusion_importance(support_features, teacher_outputs)
    if mode in {"sensitivity_block_selector", "true_validation_block_greedy"}:
        return compute_sensitivity_block_importance(support_features, teacher_outputs)
    if mode == "hybrid_teacher_response":
        teacher = compute_teacher_support_importance(support_features, teacher_outputs)
        response = compute_response_support_importance(support_features)
        lam = float(lambda_response)
        importance = (1.0 - lam) * teacher["importance"] + lam * response["importance"]
        return {
            "importance": importance.astype(np.float32),
            "components": {**teacher["components"], **response["components"]},
        }
    if mode == "response":
        return compute_response_support_importance(support_features)
    raise ValueError(f"unsupported support importance mode: {mode}")
