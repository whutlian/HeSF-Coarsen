from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph


def _relation_name(graph: HeteroGraph, relation_id: int) -> str:
    spec = graph.relation_specs.get(int(relation_id))
    return spec.name if spec is not None else f"relation_{int(relation_id)}"


def _edge_count_distribution(graph: HeteroGraph) -> dict[int, float]:
    counts = {int(rid): float(rel.num_edges) for rid, rel in graph.relations.items()}
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {rid: 0.0 for rid in counts}
    return {rid: value / total for rid, value in counts.items()}


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1.0e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def relation_distribution_drift(original: HeteroGraph, coarse: HeteroGraph) -> dict[str, Any]:
    """Compare relation edge-count mass distributions before and after coarsening."""

    relation_ids = sorted(set(original.relations) | set(coarse.relations))
    original_dist = _edge_count_distribution(original)
    coarse_dist = _edge_count_distribution(coarse)
    p = np.asarray([original_dist.get(rid, 0.0) for rid in relation_ids], dtype=np.float64)
    q = np.asarray([coarse_dist.get(rid, 0.0) for rid in relation_ids], dtype=np.float64)
    return {
        "relation_edge_mass_original": {
            str(rid): float(original_dist.get(rid, 0.0)) for rid in relation_ids
        },
        "relation_edge_mass_coarse": {
            str(rid): float(coarse_dist.get(rid, 0.0)) for rid in relation_ids
        },
        "relation_edge_count_original": {
            str(rid): int(original.relations[rid].num_edges) if rid in original.relations else 0
            for rid in relation_ids
        },
        "relation_edge_count_coarse": {
            str(rid): int(coarse.relations[rid].num_edges) if rid in coarse.relations else 0
            for rid in relation_ids
        },
        "relation_mass_l1_drift": float(np.sum(np.abs(p - q))),
        "relation_mass_js_drift": float(_js_divergence(p, q)),
    }


def coarse_edge_collapse_by_relation(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
) -> list[dict[str, Any]]:
    """Report relation-wise mapped-edge collapse before and after dedup/reduction."""

    rows: list[dict[str, Any]] = []
    for relation_id in sorted(set(original.relations) | set(coarse.relations)):
        original_rel = original.relations.get(relation_id)
        coarse_rel = coarse.relations.get(relation_id)
        if original_rel is None:
            before = 0
            self_loop_count = 0
            original_weight = 0.0
        else:
            mapped_src = assignment.assignment[original_rel.src]
            mapped_dst = assignment.assignment[original_rel.dst]
            before = int(original_rel.num_edges)
            self_loop_count = int(np.count_nonzero(mapped_src == mapped_dst))
            original_weight = float(np.sum(original_rel.weight.astype(np.float64, copy=False)))
        after = int(coarse_rel.num_edges) if coarse_rel is not None else 0
        coarse_weight = (
            float(np.sum(coarse_rel.weight.astype(np.float64, copy=False)))
            if coarse_rel is not None
            else 0.0
        )
        uniqueness = float(after / before) if before else 0.0
        self_loop_share = float(self_loop_count / before) if before else 0.0
        rows.append(
            {
                "relation_id": int(relation_id),
                "relation_name": _relation_name(original, int(relation_id)),
                "original_edges": before,
                "coarse_edges_before_dedup": before,
                "coarse_edges_after_dedup": after,
                "coarse_edge_uniqueness_ratio": uniqueness,
                "self_loop_share": self_loop_share,
                "duplicate_collapse_ratio": float(max(0.0, 1.0 - uniqueness)),
                "edge_weight_original_sum": original_weight,
                "edge_weight_coarse_sum": coarse_weight,
                "edge_weight_abs_error": float(abs(original_weight - coarse_weight)),
            }
        )
    return rows


def _cluster_mean_signals(signals: np.ndarray, assignment: Assignment) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 1:
        signals = signals[:, None]
    out = np.zeros((assignment.num_supernodes, signals.shape[1]), dtype=np.float64)
    counts = np.bincount(assignment.assignment, minlength=assignment.num_supernodes).astype(np.float64)
    for col in range(signals.shape[1]):
        out[:, col] = np.bincount(
            assignment.assignment,
            weights=signals[:, col].astype(np.float64, copy=False),
            minlength=assignment.num_supernodes,
        )
    out /= np.maximum(counts[:, None], 1.0)
    return out.astype(np.float32)


def _relation_energy(rel, signals: np.ndarray) -> float:
    if rel is None or rel.num_edges == 0:
        return 0.0
    diff = signals[rel.src] - signals[rel.dst]
    sq = np.sum(diff.astype(np.float64, copy=False) ** 2, axis=1)
    return float(np.sum(rel.weight.astype(np.float64, copy=False) * sq))


def relation_energy_error_by_relation(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    signals: np.ndarray,
) -> dict[str, Any]:
    """Estimate relation-wise energy preservation with supplied low-pass/sketch signals."""

    coarse_signals = _cluster_mean_signals(signals, assignment)
    errors: dict[str, float] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    for relation_id in sorted(set(original.relations) | set(coarse.relations)):
        b = _relation_energy(original.relations.get(relation_id), np.asarray(signals, dtype=np.float32))
        a = _relation_energy(coarse.relations.get(relation_id), coarse_signals)
        before[str(relation_id)] = b
        after[str(relation_id)] = a
        errors[str(relation_id)] = float(abs(a - b) / max(abs(b), 1.0e-12))
    values = np.asarray(list(errors.values()), dtype=np.float64)
    return {
        "relation_energy_before": before,
        "relation_energy_after": after,
        "relation_energy_error": errors,
        "relation_energy_error_mean": float(values.mean()) if values.size else 0.0,
        "relation_energy_error_max": float(values.max(initial=0.0)) if values.size else 0.0,
        "relation_energy_error_p95": float(np.percentile(values, 95)) if values.size else 0.0,
    }


def sampled_metapath_connectivity(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    *,
    max_pairs: int = 512,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Bounded two-relation metapath connectivity sanity without materializing A^2."""

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for first_id, first_rel in sorted(original.relations.items()):
        for second_id, second_rel in sorted(original.relations.items()):
            if first_rel.dst_type != second_rel.src_type:
                continue
            if first_rel.num_edges == 0:
                continue
            sample_size = min(int(max_pairs), int(first_rel.num_edges))
            if sample_size <= 0:
                continue
            if sample_size == first_rel.num_edges:
                sample_indices = np.arange(first_rel.num_edges, dtype=np.int64)
            else:
                sample_indices = rng.choice(first_rel.num_edges, size=sample_size, replace=False)
            sampled_middle = first_rel.dst[sample_indices]
            sampled_starts = first_rel.src[sample_indices]
            middle_set = set(int(value) for value in sampled_middle)
            original_adj: dict[int, set[int]] = defaultdict(set)
            mask = np.fromiter((int(src) in middle_set for src in second_rel.src), dtype=bool, count=second_rel.num_edges)
            for src, dst in zip(second_rel.src[mask], second_rel.dst[mask]):
                original_adj[int(src)].add(int(dst))
            original_counts = [len(original_adj.get(int(mid), set())) for mid in sampled_middle]

            coarse_first = coarse.relations.get(first_id)
            coarse_second = coarse.relations.get(second_id)
            coarse_adj: dict[int, set[int]] = defaultdict(set)
            if coarse_second is not None:
                for src, dst in zip(coarse_second.src, coarse_second.dst):
                    coarse_adj[int(src)].add(int(dst))
            coarse_counts = []
            if coarse_first is not None:
                for start, middle in zip(sampled_starts, sampled_middle):
                    coarse_mid = int(assignment.assignment[int(middle)])
                    coarse_counts.append(len(coarse_adj.get(coarse_mid, set())))
            else:
                coarse_counts = [0 for _ in sampled_middle]
            original_score = float(np.mean(original_counts)) if original_counts else 0.0
            coarse_score = float(np.mean(coarse_counts)) if coarse_counts else 0.0
            rows.append(
                {
                    "metapath_name": f"{_relation_name(original, first_id)}>{_relation_name(original, second_id)}",
                    "relation_path": f"{first_id}>{second_id}",
                    "sampled_pair_count": int(sample_size),
                    "original_connectivity_score": original_score,
                    "coarse_projected_connectivity_score": coarse_score,
                    "relative_error": float(abs(coarse_score - original_score) / max(abs(original_score), 1.0e-12)),
                }
            )
    return rows


def relation_diagnostics_summary(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    *,
    signals: np.ndarray | None = None,
    metapath_max_pairs: int = 512,
    seed: int = 12345,
) -> dict[str, Any]:
    out = {
        "relation_distribution_drift": relation_distribution_drift(original, coarse),
        "coarse_edge_collapse_by_relation": coarse_edge_collapse_by_relation(
            original,
            coarse,
            assignment,
        ),
        "metapath_connectivity_sampled": sampled_metapath_connectivity(
            original,
            coarse,
            assignment,
            max_pairs=metapath_max_pairs,
            seed=seed,
        ),
    }
    if signals is not None:
        out["relation_energy"] = relation_energy_error_by_relation(
            original,
            coarse,
            assignment,
            signals,
        )
    return out
