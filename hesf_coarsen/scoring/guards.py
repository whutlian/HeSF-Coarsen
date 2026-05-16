from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Optional

import numpy as np


SourceLookup = Callable[[int, int], Optional[str]]


def _guard_config(config: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _spec_terms(terms: dict[str, np.ndarray], expected: int) -> np.ndarray | None:
    spec = np.asarray(terms.get("spec", np.empty(0)), dtype=np.float32)
    if spec.shape != (expected,):
        return None
    return spec


def _sources(scored: np.ndarray, source_lookup: SourceLookup | None) -> np.ndarray:
    if source_lookup is None:
        return np.asarray(["unknown"] * int(scored.shape[0]), dtype=object)
    return np.asarray(
        [str(source_lookup(int(left), int(right)) or "unknown") for left, right in scored[:, :2]],
        dtype=object,
    )


def _source_share(source_values: np.ndarray) -> dict[str, float]:
    total = max(int(len(source_values)), 1)
    counts = Counter(str(value) for value in source_values)
    return {source: float(count / total) for source, count in sorted(counts.items())}


def _source_avg(spec: np.ndarray, source_values: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for source in sorted(set(str(value) for value in source_values)):
        mask = source_values == source
        values[source] = float(np.mean(spec[mask])) if np.any(mask) else 0.0
    return values


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    if values.size < 8 and float(quantile) >= 0.9:
        return float(np.max(values))
    return float(np.quantile(values, float(quantile)))


def _filter_terms(terms: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[keep] for name, values in terms.items()}


def _candidate_counts(scored: np.ndarray, keep: np.ndarray, num_nodes: int) -> np.ndarray:
    counts = np.zeros(max(int(num_nodes), 0), dtype=np.int32)
    for left, right in scored[keep, :2].astype(np.int64, copy=False):
        if 0 <= left < len(counts):
            counts[left] += 1
        if 0 <= right < len(counts):
            counts[right] += 1
    return counts


def apply_spectral_guard(
    scored: np.ndarray,
    terms: dict[str, np.ndarray],
    *,
    node_type: np.ndarray | None = None,
    source_lookup: SourceLookup | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    guard = _guard_config(config, "spectral_guard")
    base_diag: dict[str, Any] = {
        "guard_enabled": bool(guard.get("enabled", False)),
        "guard_triggered": False,
        "trigger_reason": "",
        "pairs_before": int(scored.shape[0]),
        "pairs_after": int(scored.shape[0]),
        "rejected_by_spec_count": 0,
        "rejected_by_spec_share": 0.0,
        "fallback_used_count": 0,
        "target_pressure_accept_count": 0,
    }
    if not base_diag["guard_enabled"] or scored.size == 0:
        return scored, terms, base_diag
    spec = _spec_terms(terms, int(scored.shape[0]))
    if spec is None:
        base_diag["trigger_reason"] = "missing_spec_terms"
        return scored, terms, base_diag

    quantile = float(guard.get("quantile", 0.95))
    warmup_from_bucket = bool(guard.get("warmup_from_bucket", True))
    reject = bool(guard.get("reject_high_delta_spec", True))
    downrank = bool(guard.get("downrank_instead_of_reject", False))
    min_after = max(0, int(guard.get("min_candidates_per_node_after_guard", 2)))
    sources = _sources(scored, source_lookup)
    node_type_values = (
        np.asarray(node_type, dtype=np.int32)
        if node_type is not None
        else np.zeros(int(np.max(scored[:, :2])) + 1 if scored.size else 0, dtype=np.int32)
    )
    left = scored[:, 0].astype(np.int64, copy=False)
    type_key = np.asarray(
        [int(node_type_values[node]) if 0 <= node < len(node_type_values) else -1 for node in left],
        dtype=np.int32,
    )
    thresholds: dict[int, float] = {}
    finite = spec[np.isfinite(spec)]
    global_threshold = _safe_quantile(finite, quantile) if finite.size else float("inf")
    for type_id in sorted(set(int(value) for value in type_key)):
        type_mask = type_key == type_id
        bucket_mask = type_mask & (sources == "bucket")
        values = spec[bucket_mask] if warmup_from_bucket and np.any(bucket_mask) else spec[type_mask]
        values = values[np.isfinite(values)]
        thresholds[type_id] = _safe_quantile(values, quantile) if values.size else global_threshold
    high = np.asarray(
        [float(value) > thresholds.get(int(t), global_threshold) for value, t in zip(spec, type_key)],
        dtype=bool,
    )
    keep = np.ones(int(scored.shape[0]), dtype=bool)
    pressure_accept = 0
    if reject and not downrank:
        proposed_keep = ~high
        if min_after > 0:
            num_nodes = int(len(node_type_values))
            counts = _candidate_counts(scored, proposed_keep, num_nodes)
            for idx in np.flatnonzero(high):
                lnode = int(scored[idx, 0])
                rnode = int(scored[idx, 1])
                left_low = 0 <= lnode < len(counts) and counts[lnode] < min_after
                right_low = 0 <= rnode < len(counts) and counts[rnode] < min_after
                if left_low or right_low:
                    proposed_keep[idx] = True
                    if left_low:
                        counts[lnode] += 1
                    if right_low and rnode != lnode:
                        counts[rnode] += 1
                    pressure_accept += 1
        keep = proposed_keep
    elif downrank:
        scored = scored.copy()
        penalty = np.maximum(spec - np.asarray([thresholds.get(int(t), global_threshold) for t in type_key]), 0.0)
        scored[:, 2] += penalty.astype(np.float64, copy=False)

    rejected = int(np.count_nonzero(~keep))
    base_diag.update(
        {
            "guard_triggered": bool(np.any(high)),
            "trigger_reason": "high_delta_spec" if np.any(high) else "",
            "pairs_after": int(np.count_nonzero(keep)),
            "rejected_by_spec_count": rejected,
            "rejected_by_spec_share": float(rejected / max(int(scored.shape[0]), 1)),
            "target_pressure_accept_count": int(pressure_accept),
            "thresholds_by_type": {str(key): float(value) for key, value in sorted(thresholds.items())},
        }
    )
    return scored[keep], _filter_terms(terms, keep), base_diag


def apply_source_aware_auto_guard(
    scored: np.ndarray,
    terms: dict[str, np.ndarray],
    *,
    source_lookup: SourceLookup | None,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    guard = _guard_config(config, "source_aware_guard")
    enabled = bool(guard.get("enabled", False))
    spec = _spec_terms(terms, int(scored.shape[0]))
    sources = _sources(scored, source_lookup)
    diag: dict[str, Any] = {
        "guard_enabled": enabled,
        "guard_triggered": False,
        "trigger_reason": "",
        "pairs_before": int(scored.shape[0]),
        "pairs_after": int(scored.shape[0]),
        "source_selected_share_before": _source_share(sources),
        "source_selected_share_after": _source_share(sources),
        "source_avg_delta_spec_before": _source_avg(spec, sources) if spec is not None else {},
        "source_avg_delta_spec_after": _source_avg(spec, sources) if spec is not None else {},
        "rejected_by_spec_count": 0,
        "rejected_by_spec_share": 0.0,
        "fallback_used_count": 0,
        "target_pressure_accept_count": 0,
        "cluster_size_hist": {},
    }
    if not enabled or scored.size == 0 or spec is None:
        if enabled and spec is None:
            diag["trigger_reason"] = "missing_spec_terms"
        return scored, terms, diag

    trigger = guard.get("trigger", {}) if isinstance(guard.get("trigger", {}), dict) else {}
    action = guard.get("action", {}) if isinstance(guard.get("action", {}), dict) else {}
    onehop_mask = sources == "onehop"
    bucket_mask = sources == "bucket"
    onehop_share = float(np.count_nonzero(onehop_mask) / max(len(sources), 1))
    onehop_avg = float(np.mean(spec[onehop_mask])) if np.any(onehop_mask) else 0.0
    bucket_avg = float(np.mean(spec[bucket_mask])) if np.any(bucket_mask) else 0.0
    ratio = float(onehop_avg / max(bucket_avg, 1.0e-12)) if np.any(bucket_mask) else float("inf")
    tail_q = float(trigger.get("onehop_delta_spec_tail_quantile", 0.95))
    onehop_tail = _safe_quantile(spec[onehop_mask], tail_q) if np.any(onehop_mask) else 0.0
    bucket_q95 = _safe_quantile(spec[bucket_mask], 0.95) if np.any(bucket_mask) else float("inf")
    share_threshold = float(trigger.get("onehop_selected_share_above", 0.30))
    ratio_threshold = float(trigger.get("onehop_avg_delta_spec_ratio_to_bucket_above", 4.0))
    triggered = bool(
        onehop_share > share_threshold
        and ratio > ratio_threshold
        and onehop_tail > bucket_q95
    )
    diag.update(
        {
            "onehop_selected_share": onehop_share,
            "onehop_avg_delta_spec_ratio_to_bucket": ratio,
            "onehop_delta_spec_tail": onehop_tail,
            "bucket_delta_spec_q95": bucket_q95,
            "guard_triggered": triggered,
        }
    )
    if not triggered:
        return scored, terms, diag

    keep = np.ones(int(scored.shape[0]), dtype=bool)
    if bool(action.get("reject_if_delta_spec_above_bucket_q95", True)):
        keep &= ~(onehop_mask & (spec > bucket_q95))
    topk = action.get("onehop_topk_per_node")
    if topk is not None:
        topk = max(0, int(topk))
        by_node: dict[int, list[int]] = defaultdict(list)
        for idx in np.flatnonzero(onehop_mask & keep):
            left = int(scored[idx, 0])
            right = int(scored[idx, 1])
            by_node[left].append(int(idx))
            by_node[right].append(int(idx))
        drop: set[int] = set()
        for indices in by_node.values():
            if len(indices) <= topk:
                continue
            ordered = sorted(indices, key=lambda idx: (float(spec[idx]), float(scored[idx, 2]), idx))
            drop.update(ordered[topk:])
        if drop:
            keep[np.fromiter(drop, dtype=np.int64)] = False

    rejected = int(np.count_nonzero(~keep))
    after_sources = sources[keep]
    after_spec = spec[keep]
    diag.update(
        {
            "trigger_reason": "onehop spectral pollution",
            "pairs_after": int(np.count_nonzero(keep)),
            "source_selected_share_after": _source_share(after_sources),
            "source_avg_delta_spec_after": _source_avg(after_spec, after_sources),
            "rejected_by_spec_count": rejected,
            "rejected_by_spec_share": float(rejected / max(int(scored.shape[0]), 1)),
        }
    )
    return scored[keep], _filter_terms(terms, keep), diag


def apply_candidate_guards(
    scored: np.ndarray,
    terms: dict[str, np.ndarray],
    *,
    node_type: np.ndarray | None = None,
    source_lookup: SourceLookup | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    scored, terms, spectral_diag = apply_spectral_guard(
        scored,
        terms,
        node_type=node_type,
        source_lookup=source_lookup,
        config=config,
    )
    scored, terms, source_diag = apply_source_aware_auto_guard(
        scored,
        terms,
        source_lookup=source_lookup,
        config=config,
    )
    return scored, terms, {
        "spectral_guard": spectral_diag,
        "source_aware_guard": source_diag,
    }
