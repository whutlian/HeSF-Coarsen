from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Union

import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


ValidationEvaluator = Callable[[np.ndarray], Union[float, Mapping[str, Any]]]


def _argmax_or_unknown(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0 or values.ndim != 2 or values.shape[1] == 0:
        return np.full(values.shape[0] if values.ndim else 0, -1, dtype=np.int64)
    ids = np.argmax(values, axis=1).astype(np.int64)
    ids[np.sum(values, axis=1) <= 1.0e-12] = -1
    return ids


def _degree_buckets(degree_profile: np.ndarray, n: int) -> np.ndarray:
    if degree_profile.size == 0:
        return np.zeros(int(n), dtype=np.int64)
    degree = np.sum(np.asarray(degree_profile, dtype=np.float64), axis=1)
    if degree.size == 0:
        return np.zeros(int(n), dtype=np.int64)
    nonzero = degree[degree > 0.0]
    if nonzero.size == 0:
        return np.zeros_like(degree, dtype=np.int64)
    p25, p75, p95 = np.percentile(nonzero, [25, 75, 95])
    buckets = np.zeros_like(degree, dtype=np.int64)
    buckets[(degree > 0.0) & (degree <= p25)] = 1
    buckets[(degree > p25) & (degree <= p75)] = 2
    buckets[(degree > p75) & (degree <= p95)] = 3
    buckets[degree > p95] = 4
    return buckets.astype(np.int64, copy=False)


def _bridge_flags(relation: np.ndarray, anchor: np.ndarray, class_fp: np.ndarray, degree_bucket: np.ndarray) -> np.ndarray:
    n = len(degree_bucket)
    relation_nnz = np.count_nonzero(np.asarray(relation) > 1.0e-12, axis=1) if relation.size else np.zeros(n)
    anchor_nnz = np.count_nonzero(np.asarray(anchor) > 1.0e-12, axis=1) if anchor.size else np.zeros(n)
    class_nnz = np.count_nonzero(np.asarray(class_fp) > 1.0e-12, axis=1) if class_fp.size else np.zeros(n)
    return (
        (relation_nnz > 1)
        | (anchor_nnz > 1)
        | (class_nnz > 1)
        | (np.asarray(degree_bucket) >= 4)
    ).astype(np.int64)


def build_support_block_keys(support_features: dict[str, Any], mode: str = "class_anchor_relation") -> list[tuple[int, ...]]:
    nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    types = np.asarray(support_features["support_node_types"], dtype=np.int64)
    components = support_features.get("component_matrices", {})
    relation = np.asarray(components.get("relation_profile", np.empty((len(nodes), 0))), dtype=np.float32)
    anchor = np.asarray(components.get("anchor_distribution", np.empty((len(nodes), 0))), dtype=np.float32)
    class_fp = np.asarray(components.get("class_footprint", np.empty((len(nodes), 0))), dtype=np.float32)
    relation_bucket = _argmax_or_unknown(relation)
    anchor_bucket = _argmax_or_unknown(anchor)
    class_bucket = _argmax_or_unknown(class_fp)
    if str(mode) in {"default", "class_anchor_relation"}:
        return [
            (int(types[idx]), int(relation_bucket[idx]), int(anchor_bucket[idx]), int(class_bucket[idx]))
            for idx in range(len(nodes))
        ]
    if str(mode) != "dblp_aware":
        raise ValueError(f"unsupported support block key mode: {mode}")
    degree = np.asarray(components.get("degree_profile", np.empty((len(nodes), 0))), dtype=np.float32)
    degree_bucket = _degree_buckets(degree, len(nodes))
    bridge_flag = _bridge_flags(relation, anchor, class_fp, degree_bucket)
    return [
        (
            int(types[idx]),
            int(relation_bucket[idx]),
            int(anchor_bucket[idx]),
            int(class_bucket[idx]),
            int(degree_bucket[idx]),
            int(bridge_flag[idx]),
        )
        for idx in range(len(nodes))
    ]


def group_support_by_block(block_keys: list[tuple[int, ...]]) -> dict[tuple[int, ...], np.ndarray]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, key in enumerate(block_keys):
        groups.setdefault(tuple(int(value) for value in key), []).append(int(idx))
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def _proxy_block_scores(
    block_groups: Mapping[tuple[int, ...], np.ndarray],
    importance: np.ndarray,
) -> dict[tuple[int, ...], float]:
    values = np.asarray(importance, dtype=np.float64).reshape(-1)
    return {key: float(np.sum(values[indices])) for key, indices in block_groups.items()}


def _node_order(indices: np.ndarray, nodes: np.ndarray, importance: np.ndarray) -> list[int]:
    return sorted(
        [int(idx) for idx in np.asarray(indices, dtype=np.int64)],
        key=lambda idx: (-float(importance[idx]), int(nodes[idx])),
    )


def _eval_float(result: float | Mapping[str, Any], primary_key: str = "validation_macro_f1") -> float:
    if isinstance(result, Mapping):
        try:
            return float(result.get(primary_key, result.get("macro_f1", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(result)
    except (TypeError, ValueError):
        return 0.0


def _metric_or_nan(result: float | Mapping[str, Any], key: str) -> float:
    if not isinstance(result, Mapping):
        return float(result) if key == "validation_macro_f1" else float("nan")
    if key not in result or result.get(key) in {"", None}:
        return float("nan")
    try:
        return float(result.get(key))
    except (TypeError, ValueError):
        return float("nan")


def _safe_delta(left: float, right: float, *, mode: str) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    if mode == "drop":
        return float(left - right)
    return float(right - left)


def _nanmean(values: list[float]) -> float:
    finite = [float(value) for value in values if not np.isnan(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _nanmax(values: list[float]) -> float:
    finite = [float(value) for value in values if not np.isnan(float(value))]
    return float(np.max(finite)) if finite else float("nan")


def select_blocks_by_validation_feedback(
    support_features: dict[str, Any],
    importance: np.ndarray,
    budget: int,
    cfg: SupportSelectorConfig,
    validation_evaluator: ValidationEvaluator,
) -> dict[str, Any]:
    nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    values = np.asarray(importance, dtype=np.float32).reshape(-1)
    block_keys = build_support_block_keys(support_features, mode=str(cfg.block_key_mode))
    block_groups = group_support_by_block(block_keys)
    block_scores = _proxy_block_scores(block_groups, values)
    remaining = sorted(block_groups, key=lambda key: (-block_scores[key], key))
    selected: list[int] = []
    selected_set: set[int] = set()
    accepted_blocks: list[tuple[int, ...]] = []
    rejected_blocks: list[tuple[int, ...]] = []
    trial_rows: list[dict[str, Any]] = []
    gain_history: list[float] = []
    block_trim_count = 0
    oversized_block_count = 0
    proxy_fallback_fill_count = 0
    previous_score = _eval_float(validation_evaluator(np.empty(0, dtype=np.int64)))
    score_history: list[float] = [float(previous_score)]
    max_steps = max(1, int(cfg.max_validation_greedy_steps))
    candidate_pool_size = max(1, int(cfg.candidate_pool_size))
    for step in range(max_steps):
        if len(selected) >= int(budget) or not remaining:
            break
        pool = remaining[:candidate_pool_size]
        best_key: tuple[int, ...] | None = None
        best_members: list[int] = []
        best_score = -float("inf")
        for candidate_rank, key in enumerate(pool, start=1):
            members = [idx for idx in _node_order(block_groups[key], nodes, values) if idx not in selected_set]
            remaining_budget = int(budget) - len(selected)
            trimmed_to_budget = False
            if len(members) > remaining_budget:
                oversized_block_count += 1
                block_trim_count += 1
                members = members[:remaining_budget]
                trimmed_to_budget = True
            if not members:
                continue
            trial_indices = np.asarray([*selected, *members], dtype=np.int64)
            trial_nodes = nodes[trial_indices]
            score = _eval_float(validation_evaluator(trial_nodes))
            score_history.append(float(score))
            gain = float(score - previous_score)
            trial_rows.append(
                {
                    "step": int(step + 1),
                    "candidate_rank": int(candidate_rank),
                    "block_key": repr(key),
                    "block_size": int(len(block_groups[key])),
                    "trial_selected_support_count": int(len(trial_nodes)),
                    "validation_score": float(score),
                    "validation_gain": float(gain),
                    "accepted": False,
                    "trimmed_to_budget": bool(trimmed_to_budget),
                }
            )
            if score > best_score or (score == best_score and key < (best_key or key)):
                best_key = key
                best_members = members
                best_score = float(score)
        if best_key is None:
            break
        best_gain = float(best_score - previous_score)
        required_gain = max(float(cfg.min_gain), 1.0e-4 * abs(float(previous_score)))
        if best_gain <= required_gain:
            rejected_blocks.extend(pool)
            break
        selected.extend(best_members)
        selected_set.update(best_members)
        accepted_blocks.append(best_key)
        gain_history.append(best_gain)
        previous_score = float(best_score)
        for row in trial_rows:
            if row["step"] == step + 1 and row["block_key"] == repr(best_key):
                row["accepted"] = True
        remaining = [key for key in remaining if key != best_key]
    if bool(getattr(cfg, "allow_proxy_fill", True)) and len(selected) < int(budget):
        for key in remaining:
            for idx in _node_order(block_groups[key], nodes, values):
                if idx in selected_set:
                    continue
                selected.append(int(idx))
                selected_set.add(int(idx))
                proxy_fallback_fill_count += 1
                if len(selected) >= int(budget):
                    break
            if len(selected) >= int(budget):
                break
    best_gain_max = float(np.max(gain_history)) if gain_history else 0.0
    score_keys = {f"{float(value):.12g}" for value in score_history}
    real_validation_degenerate = bool(
        len(accepted_blocks) == 0
        or best_gain_max <= max(float(cfg.min_gain), 1.0e-4 * abs(float(previous_score)))
        or len(score_keys) <= 1
    )
    return {
        "selected_local_indices": np.asarray(selected[: int(budget)], dtype=np.int64),
        "validation_greedy_trials": trial_rows,
        "diagnostics": {
            "selector_uses_true_validation_feedback": True,
            "selector_feedback_source": "real_validation_degenerate" if real_validation_degenerate else "real_validation_block_greedy",
            "selection_split_source": "train_val_only",
            "selector_family": "real_validation_selector",
            "validation_greedy_steps": int(len(accepted_blocks)),
            "validation_trial_count": int(len(trial_rows)),
            "validation_candidate_pool_size": int(candidate_pool_size),
            "validation_short_eval_epochs": int(cfg.short_eval_epochs),
            "validation_objective": "projected_val_macro_f1",
            "validation_greedy_best_gain_mean": float(np.mean(gain_history)) if gain_history else 0.0,
            "validation_greedy_best_gain_max": best_gain_max,
            "validation_greedy_gain_history": [float(value) for value in gain_history],
            "validation_gain_history": [float(value) for value in gain_history],
            "validation_scores_unique_count": int(len(score_keys)),
            "accepted_block_count": int(len(accepted_blocks)),
            "rejected_block_count": int(len(rejected_blocks)),
            "selected_block_keys": [repr(key) for key in accepted_blocks],
            "block_trim_count": int(block_trim_count),
            "oversized_block_count": int(oversized_block_count),
            "proxy_fallback_fill_count": int(proxy_fallback_fill_count),
            "real_validation_degenerate": real_validation_degenerate,
            "validation_signal_pass": bool(not real_validation_degenerate and proxy_fallback_fill_count == 0),
        },
    }


def _metric_map(result: float | Mapping[str, Any]) -> dict[str, float]:
    return {
        "validation_macro_f1": _eval_float(result),
        "validation_cross_entropy": _metric_or_nan(result, "validation_cross_entropy"),
        "margin": _metric_or_nan(result, "margin"),
        "teacher_kl": _metric_or_nan(result, "teacher_kl"),
        "class_recall": _metric_or_nan(result, "class_recall"),
        "tree_tensor_l2_delta_when_occluded": _metric_or_nan(result, "tree_tensor_l2_delta_when_occluded"),
    }


def select_blocks_by_occlusion_feedback(
    support_features: dict[str, Any],
    importance: np.ndarray,
    budget: int,
    cfg: SupportSelectorConfig,
    occlusion_evaluator: ValidationEvaluator,
) -> dict[str, Any]:
    nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    values = np.asarray(importance, dtype=np.float32).reshape(-1)
    block_keys = build_support_block_keys(support_features, mode=str(cfg.block_key_mode))
    block_groups = group_support_by_block(block_keys)
    block_scores = _proxy_block_scores(block_groups, values)
    ordered = sorted(block_groups, key=lambda key: (-block_scores[key], key))
    pool = ordered[: max(1, int(cfg.occlusion_candidate_pool_size))]
    baseline = _metric_map(occlusion_evaluator(np.empty(0, dtype=np.int64)))
    required_metric_keys = [
        "validation_cross_entropy",
        "margin",
        "teacher_kl",
        "class_recall",
        "tree_tensor_l2_delta_when_occluded",
    ]
    score_rows: list[dict[str, Any]] = []
    for key in pool:
        member_indices = _node_order(block_groups[key], nodes, values)
        member_nodes = nodes[np.asarray(member_indices, dtype=np.int64)]
        masked = _metric_map(occlusion_evaluator(member_nodes))
        delta_macro = _safe_delta(baseline["validation_macro_f1"], masked["validation_macro_f1"], mode="drop")
        delta_ce = _safe_delta(baseline["validation_cross_entropy"], masked["validation_cross_entropy"], mode="rise")
        delta_margin = _safe_delta(baseline["margin"], masked["margin"], mode="drop")
        delta_kl = _safe_delta(baseline["teacher_kl"], masked["teacher_kl"], mode="rise")
        delta_recall = _safe_delta(baseline["class_recall"], masked["class_recall"], mode="drop")
        final = (
            (0.0 if np.isnan(delta_ce) else delta_ce)
            + float(cfg.alpha_teacher_kl) * (0.0 if np.isnan(delta_kl) else delta_kl)
            + float(cfg.beta_margin) * (0.0 if np.isnan(delta_margin) else delta_margin)
            + float(cfg.gamma_class_recall) * (0.0 if np.isnan(delta_recall) else delta_recall)
            + max(0.0 if np.isnan(delta_macro) else delta_macro, 0.0)
        )
        score_rows.append(
            {
                "block_key": repr(key),
                "block_size": int(len(block_groups[key])),
                "proxy_importance": float(block_scores[key]),
                "base_validation_macro_f1": float(baseline["validation_macro_f1"]),
                "occluded_validation_macro_f1": float(masked["validation_macro_f1"]),
                "base_val_ce": float(baseline["validation_cross_entropy"]),
                "occluded_val_ce": float(masked["validation_cross_entropy"]),
                "delta_val_ce": float(delta_ce),
                "delta_val_macro_f1": float(delta_macro),
                "delta_margin": float(delta_margin),
                "delta_teacher_kl": float(delta_kl),
                "delta_class_recall": float(delta_recall),
                "tree_tensor_l2_delta_when_occluded": float(masked["tree_tensor_l2_delta_when_occluded"]),
                "occlusion_metric_complete": bool(
                    all(not np.isnan(float(baseline[key])) for key in required_metric_keys)
                    and all(not np.isnan(float(masked[key])) for key in required_metric_keys)
                ),
                "final_block_importance": float(final),
                "selected": False,
                "rank": 0,
                "local_indices": member_indices,
            }
        )
    score_rows.sort(key=lambda row: (-float(row["final_block_importance"]), -float(row["proxy_importance"]), str(row["block_key"])))
    selected: list[int] = []
    selected_set: set[int] = set()
    for rank, row in enumerate(score_rows, start=1):
        row["rank"] = int(rank)
        row["selected"] = len(selected) < int(budget)
        for idx in row.pop("local_indices"):
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= int(budget):
                break
    proxy_fallback_fill_count = 0
    if bool(getattr(cfg, "allow_proxy_fill", True)) and len(selected) < int(budget):
        for key in ordered:
            for idx in _node_order(block_groups[key], nodes, values):
                if idx in selected_set:
                    continue
                selected.append(int(idx))
                selected_set.add(int(idx))
                proxy_fallback_fill_count += 1
                if len(selected) >= int(budget):
                    break
            if len(selected) >= int(budget):
                break
    ce_values = [float(row["delta_val_ce"]) for row in score_rows]
    macro_values = [float(row["delta_val_macro_f1"]) for row in score_rows]
    margin_values = [float(row["delta_margin"]) for row in score_rows]
    kl_values = [float(row["delta_teacher_kl"]) for row in score_rows]
    tree_delta_values = [float(row["tree_tensor_l2_delta_when_occluded"]) for row in score_rows]
    signal_values = [
        value
        for value in [*ce_values, *macro_values, *tree_delta_values]
        if not np.isnan(float(value))
    ]
    metric_complete = bool(score_rows) and all(bool(row.get("occlusion_metric_complete")) for row in score_rows)
    nonzero_delta_count = sum(1 for value in [*ce_values, *macro_values, *margin_values, *kl_values] if not np.isnan(float(value)) and abs(float(value)) > 1.0e-12)
    nonzero_delta_den = sum(1 for value in [*ce_values, *macro_values, *margin_values, *kl_values] if not np.isnan(float(value)))
    tree_nonzero_count = sum(1 for value in tree_delta_values if not np.isnan(float(value)) and abs(float(value)) > 1.0e-12)
    tree_nonzero_den = sum(1 for value in tree_delta_values if not np.isnan(float(value)))
    occlusion_degenerate = not any(abs(float(value)) > 1.0e-12 for value in signal_values)
    proxy_fallback_used = bool(proxy_fallback_fill_count > 0 or occlusion_degenerate or not metric_complete)
    if not bool(getattr(cfg, "allow_proxy_fill", True)):
        proxy_fallback_used = bool(occlusion_degenerate or not metric_complete)
    return {
        "selected_local_indices": np.asarray(selected[: int(budget)], dtype=np.int64),
        "occlusion_block_scores": score_rows,
        "diagnostics": {
            "selector_family": "real_occlusion_selector",
            "selector_feedback_source": "occlusion_proxy_fallback" if occlusion_degenerate else "real_validation_occlusion",
            "selection_split_source": "train_val_only",
            "occlusion_trial_count": int(len(score_rows)),
            "occlusion_candidate_pool_size": int(max(1, int(cfg.occlusion_candidate_pool_size))),
            "occlusion_objective": str(cfg.primary_occlusion_term),
            "occlusion_delta_ce_mean": _nanmean(ce_values),
            "occlusion_delta_ce_max": _nanmax(ce_values),
            "occlusion_delta_macro_f1_mean": _nanmean(macro_values),
            "occlusion_delta_macro_f1_max": _nanmax(macro_values),
            "occlusion_delta_margin_mean": _nanmean(margin_values),
            "occlusion_delta_margin_max": _nanmax(margin_values),
            "occlusion_delta_teacher_kl_mean": _nanmean(kl_values),
            "occlusion_delta_teacher_kl_max": _nanmax(kl_values),
            "occlusion_tree_tensor_l2_delta_mean": _nanmean(tree_delta_values),
            "occlusion_tree_tensor_l2_delta_max": _nanmax(tree_delta_values),
            "occlusion_metric_complete": metric_complete,
            "occlusion_nonzero_delta_rate": float(nonzero_delta_count / max(nonzero_delta_den, 1)),
            "occlusion_tree_delta_nonzero_rate": float(tree_nonzero_count / max(tree_nonzero_den, 1)),
            "occlusion_cache_hit_count": 0,
            "occlusion_cache_miss_count": int(len(score_rows)),
            "occlusion_degenerate": occlusion_degenerate,
            "occlusion_proxy_fallback_used": proxy_fallback_used,
            "occlusion_proxy_fallback_fill_count": int(proxy_fallback_fill_count),
            "occlusion_signal_pass": bool(metric_complete and not occlusion_degenerate and not proxy_fallback_used),
            "selector_uses_true_validation_feedback": False,
        },
    }
