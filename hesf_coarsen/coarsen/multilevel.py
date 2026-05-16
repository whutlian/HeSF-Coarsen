from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.candidates.bucket import generate_bucket_candidates, generate_bucket_candidates_chunked
from hesf_coarsen.candidates.capped_twohop import (
    generate_capped_twohop_candidates,
    generate_capped_twohop_candidates_chunked,
)
from hesf_coarsen.candidates.onehop import generate_onehop_candidates, generate_onehop_candidates_chunked
from hesf_coarsen.candidates.partition_ann import generate_partition_ann_candidates
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph, coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.diagnostics import compute_diagnostics, save_diagnostics
from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics
from hesf_coarsen.io.edge_list import load_graph, save_graph
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.matching.greedy import (
    finalize_mutual_best,
    initialize_mutual_best_state,
    mutual_best_update_block,
    run_matching,
    selected_pair_sources,
)
from hesf_coarsen.ops.fusion_weights import compute_relation_fusion_weights
from hesf_coarsen.partition.type_partition import default_partition
from hesf_coarsen.progress import progress_iter, progress_message
from hesf_coarsen.scoring.conv_response import compute_conv_response_sketch
from hesf_coarsen.scoring.merge_cost import (
    ScoreTermAccumulator,
    prepare_pair_scoring_context,
    score_term_contributions,
    score_pair_block_with_terms,
)
from hesf_coarsen.scoring.relation_profile import compute_relation_profiles
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch
from hesf_coarsen.sketch.simhash import compute_simhash_buckets


@dataclass
class LevelResult:
    level: int
    graph: HeteroGraph
    assignment: Assignment
    diagnostics: dict


@dataclass(frozen=True)
class CompletedLevel:
    level: int
    directory: Path
    num_nodes: int
    legacy: bool


def _parse_level_dir(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("level_"):
        return None
    try:
        level = int(path.name.removeprefix("level_"))
    except ValueError:
        return None
    return level if level > 0 else None


def _has_level_dirs(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(_parse_level_dir(path) is not None for path in output_dir.iterdir())


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _save_assignment(assignment: Assignment, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            assignment=assignment.assignment,
            supernode_type=assignment.supernode_type,
        )
    tmp_path.replace(path)


def _reset_cuda_peak_memory_stats() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _completed_level(
    level_dir: Path,
    level: int,
    allow_legacy_checkpoints: bool,
) -> CompletedLevel | None:
    diagnostics_path = level_dir / "diagnostics.json"
    checkpoint_path = level_dir / "checkpoint.json"
    if not diagnostics_path.exists():
        return None
    try:
        graph = load_graph(level_dir)
    except Exception:
        return None

    if checkpoint_path.exists():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        if not bool(checkpoint.get("complete", False)):
            return None
        if int(checkpoint.get("level", -1)) != level:
            return None
        if not (level_dir / "assignment.npz").exists():
            return None
        return CompletedLevel(level=level, directory=level_dir, num_nodes=graph.num_nodes, legacy=False)

    if allow_legacy_checkpoints:
        return CompletedLevel(level=level, directory=level_dir, num_nodes=graph.num_nodes, legacy=True)
    return None


def discover_completed_levels(
    output_dir: str | Path,
    allow_legacy_checkpoints: bool = False,
) -> list[CompletedLevel]:
    root = Path(output_dir)
    completed: list[CompletedLevel] = []
    level = 1
    while True:
        level_dir = root / f"level_{level}"
        if not level_dir.exists():
            break
        found = _completed_level(level_dir, level, allow_legacy_checkpoints)
        if found is None:
            break
        completed.append(found)
        level += 1
    return completed


def _save_checkpoint(
    level_dir: Path,
    level: int,
    input_nodes: int,
    coarse_nodes: int,
    target_nodes: int,
    legacy_resume: bool,
) -> None:
    checkpoint = {
        "version": 1,
        "complete": True,
        "level": int(level),
        "input_nodes": int(input_nodes),
        "coarse_nodes": int(coarse_nodes),
        "target_nodes": int(target_nodes),
        "legacy_resume": bool(legacy_resume),
    }
    _write_json_atomic(level_dir / "checkpoint.json", checkpoint)


def _add_fallback_candidates(
    graph: HeteroGraph,
    partition_id: np.ndarray,
    store: BoundedCandidateStore | ArrayCandidateStore,
    config: dict,
) -> dict[str, int]:
    candidate_cfg = config.get("candidates", {})
    same_partition = bool(config.get("coarsening", {}).get("same_partition_only", True))
    penalty = float(candidate_cfg.get("fallback_penalty", 1.0e6))
    max_fraction = float(candidate_cfg.get("fallback_max_fraction", 1.0))
    max_pairs = max(0, int(ceil(graph.num_nodes * max_fraction)))
    added = 0
    if max_pairs == 0:
        return {"pairs_considered": 0}
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = nodes_of_type(graph, int(type_id))
        if same_partition:
            keys = sorted(np.unique(partition_id[nodes]))
            groups = [nodes[partition_id[nodes] == key] for key in keys]
        else:
            groups = [nodes]
        for group in groups:
            for left, right in zip(group[::2], group[1::2]):
                store.add(int(left), int(right), penalty, "fallback")
                added += 1
                if added >= max_pairs:
                    return {"pairs_considered": added}
    return {"pairs_considered": added}


def _config_for_level(config: dict, num_nodes: int, target_nodes: int) -> dict:
    level_config = deepcopy(config)
    per_level_ratio = float(config.get("coarsening", {}).get("per_level_ratio", 0.55))
    remaining_ratio = float(target_nodes / max(num_nodes, 1))
    level_ratio = max(per_level_ratio, remaining_ratio)
    level_ratio = min(max(level_ratio, 0.0), 1.0)
    desired_coarse_nodes = int(ceil(num_nodes * level_ratio - 1.0e-12))
    desired_coarse_nodes = max(int(target_nodes), min(int(num_nodes), desired_coarse_nodes))
    max_pairs = max(0, int(num_nodes) - desired_coarse_nodes)
    level_config.setdefault("coarsening", {})["remaining_ratio"] = remaining_ratio
    level_config.setdefault("coarsening", {})["level_ratio"] = level_ratio
    level_config.setdefault("coarsening", {})["desired_coarse_nodes"] = desired_coarse_nodes
    level_config.setdefault("coarsening", {})["max_matched_pairs"] = max_pairs
    return level_config


def _target_control_diagnostics(
    config: dict,
    *,
    original_nodes: int,
    input_nodes: int,
    target_nodes: int,
    level_config: dict,
) -> dict:
    coarsening = level_config.get("coarsening", {})
    return {
        "target_ratio": float(config.get("coarsening", {}).get("target_ratio", 0.0)),
        "per_level_ratio": float(config.get("coarsening", {}).get("per_level_ratio", 0.0)),
        "original_nodes": int(original_nodes),
        "input_nodes": int(input_nodes),
        "target_nodes": int(target_nodes),
        "remaining_ratio": float(coarsening.get("remaining_ratio", 0.0)),
        "level_ratio": float(coarsening.get("level_ratio", 0.0)),
        "desired_coarse_nodes": int(coarsening.get("desired_coarse_nodes", input_nodes)),
        "max_matched_pairs": int(coarsening.get("max_matched_pairs", 0)),
    }


def _resolved_config_diagnostics(config: dict) -> dict:
    coarsening = config.get("coarsening", {})
    sketch = config.get("sketch", {})
    fusion = config.get("fusion", {})
    metapath = config.get("metapath_sketch", {})
    scoring = config.get("scoring", {})
    candidates = config.get("candidates", {})
    relation_weighting = fusion.get("relation_weighting", {})
    if not isinstance(relation_weighting, dict):
        relation_weighting = {"method": relation_weighting}
    return {
        "coarsening": {
            "target_ratio": coarsening.get("target_ratio"),
            "per_level_ratio": coarsening.get("per_level_ratio"),
            "max_levels": coarsening.get("max_levels"),
            "matching_method": coarsening.get("matching_method"),
            "max_cluster_size": coarsening.get("max_cluster_size"),
            "same_type_only": coarsening.get("same_type_only"),
            "same_partition_only": coarsening.get("same_partition_only"),
            "terminal_guard": coarsening.get("terminal_guard"),
            "cumulative_guard": coarsening.get("cumulative_guard"),
        },
        "sketch": {
            "method": sketch.get("method"),
            "dim": sketch.get("dim"),
            "order": sketch.get("order"),
        },
        "fusion": {
            "relation_weighting": {
                "method": relation_weighting.get("method"),
            },
            "relation_operator_mode": fusion.get("relation_operator_mode", "relationwise"),
        },
        "metapath_sketch": {
            "enabled": metapath.get("enabled"),
            "operator_weight_total": metapath.get("operator_weight_total"),
        },
        "scoring": {
            key: scoring.get(key)
            for key in (
                "lambda_spec",
                "lambda_rel",
                "lambda_feat",
                "lambda_conv",
                "lambda_boundary",
                "normalization",
                "normalization_scope",
                "relation_profile_mode",
            )
        },
        "candidates": {
            key: candidates.get(key)
            for key in (
                "total_budget_K",
                "twohop_budget_K2",
                "twohop_mode",
                "twohop_budget_per_node",
                "twohop_max_time_budget_sec",
                "ann_budget_K",
                "enable_onehop",
                "enable_capped_twohop",
                "enable_bucket",
                "enable_partition_ann",
                "enable_fallback",
                "simhash_bits",
                "bucket_pair_cap",
                "hash_tables",
                "multi_probe",
                "hamming_radius",
                "adaptive_hamming_radius",
                "quotas",
            )
        },
    }


def _bucket_hash_bits(candidate_cfg: dict) -> list[int]:
    raw = candidate_cfg.get("hash_tables", None)
    default_bits = int(candidate_cfg.get("simhash_bits", 16))
    if raw in (None, "", False):
        return [default_bits]
    if isinstance(raw, int):
        return [default_bits for _ in range(max(raw, 1))]
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw] or [default_bits]
    return [int(raw)]


def _score_contribution_share(summary: dict[str, dict]) -> dict[str, float]:
    means = {
        name: max(float(stats.get("mean", 0.0) or 0.0), 0.0)
        for name, stats in summary.items()
    }
    total = float(sum(means.values()))
    if total <= 0.0:
        return {name: 0.0 for name in means}
    return {name: float(value / total) for name, value in means.items()}


def _repair_objective_name(guard: dict) -> str:
    raw = guard.get("repair_objective", guard.get("objective", "current"))
    name = str(raw or "current").lower().replace("-", "_")
    if name in {"spectral", "fixed", "cumulative"}:
        return "energy"
    if name in {"energy", "relation", "task"}:
        return name
    return "current"


def _repair_bad_clusters(
    graph: HeteroGraph,
    assignment: Assignment,
    Z: np.ndarray,
    config: dict,
) -> tuple[Assignment, dict]:
    guard = config.get("coarsening", {}).get("cumulative_guard", {})
    enabled = bool(guard.get("enabled", False))
    repair_enabled = bool(guard.get("repair_bad_clusters", False))
    strategy = str(guard.get("repair_strategy", "current")).lower().replace("-", "_")
    objective_name = _repair_objective_name(guard)
    if not enabled or not repair_enabled or strategy == "off":
        return assignment, {
            "enabled": enabled,
            "repair_bad_clusters": repair_enabled,
            "repair_strategy": strategy,
            "repair_objective_name": objective_name,
            "repair_accepted": False,
        }

    labels = None if graph.labels is None else np.asarray(graph.labels)
    relation_profiles = compute_relation_profiles(graph) if graph.relations else None

    def cluster_metrics(
        candidate: Assignment,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        sizes = candidate.cluster_sizes()
        spreads = np.zeros(candidate.num_supernodes, dtype=np.float64)
        label_entropy = np.zeros(candidate.num_supernodes, dtype=np.float64)
        relation_variance = np.zeros(candidate.num_supernodes, dtype=np.float64)
        for supernode in range(candidate.num_supernodes):
            members = np.flatnonzero(candidate.assignment == supernode)
            if len(members) <= 1:
                continue
            block = Z[members].astype(np.float64, copy=False)
            center = block.mean(axis=0, keepdims=True)
            spreads[supernode] = float(np.mean(np.sum((block - center) ** 2, axis=1)))
            if labels is not None:
                cluster_labels = labels[members]
                cluster_labels = cluster_labels[cluster_labels >= 0]
                if len(cluster_labels):
                    _values, counts = np.unique(cluster_labels, return_counts=True)
                    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
                    label_entropy[supernode] = float(
                        -np.sum(probs * np.log(np.maximum(probs, 1.0e-12)))
                    )
            if relation_profiles is not None:
                profile_block = relation_profiles[members].astype(np.float64, copy=False)
                if profile_block.size:
                    relation_variance[supernode] = float(np.mean(np.var(profile_block, axis=0)))
        objective = {
            "cluster_sketch_spread": float(np.mean(spreads[sizes > 1])) if np.any(sizes > 1) else 0.0,
            "relation_profile_variance": float(np.mean(relation_variance[sizes > 1]))
            if np.any(sizes > 1)
            else 0.0,
            "train_label_entropy": float(np.mean(label_entropy[sizes > 1]))
            if np.any(sizes > 1)
            else 0.0,
        }
        return sizes, spreads, label_entropy, relation_variance, objective

    sizes, spreads, label_entropy, relation_variance, before_objective = cluster_metrics(assignment)
    before_dee_proxy = float(
        before_objective["cluster_sketch_spread"] + before_objective["relation_profile_variance"]
    )
    before_sipe_proxy = float(before_objective["cluster_sketch_spread"])
    large = sizes > 2
    spread_cutoff = float(np.percentile(spreads[large], 75)) if np.any(large) else np.inf
    entropy_cutoff = float(guard.get("label_entropy_cutoff", 0.0))

    def objective_mask(values: np.ndarray) -> np.ndarray:
        if not np.any(large):
            return np.zeros_like(large, dtype=bool)
        active = values[large]
        cutoff = float(np.percentile(active, 75))
        return large & (values >= cutoff) & (values > 0.0)

    if objective_name == "energy":
        bad = objective_mask(spreads)
    elif objective_name == "relation":
        bad = objective_mask(relation_variance)
    elif objective_name == "task":
        bad = objective_mask(label_entropy)
    else:
        bad = large & ((spreads >= spread_cutoff) | (label_entropy > entropy_cutoff))
    selected_clusters = np.flatnonzero(bad).astype(np.int64).tolist()
    if not np.any(bad):
        return assignment, {
            "enabled": True,
            "repair_bad_clusters": True,
            "repair_strategy": strategy,
            "repair_objective_name": objective_name,
            "repair_accepted": False,
            "repaired_cluster_count": 0,
            "repair_selected_clusters": [],
            "repair_trace_signature": f"{objective_name}:none",
            "repair_objective": {
                **before_objective,
                "cumulative_energy_delta": 0.0,
            },
            "estimated_cumulative_dee_before": before_dee_proxy,
            "estimated_cumulative_dee_after": before_dee_proxy,
            "estimated_cumulative_sipe_before": before_sipe_proxy,
            "estimated_cumulative_sipe_after": before_sipe_proxy,
            "node_reduction_before": int(np.sum(np.maximum(sizes - 1, 0))),
            "node_reduction_after": int(np.sum(np.maximum(sizes - 1, 0))),
        }

    new_assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    new_types: list[int] = []

    def emit(nodes: np.ndarray) -> None:
        super_id = len(new_types)
        new_assignment[nodes] = super_id
        new_types.append(int(graph.node_type[int(nodes[0])]))

    for supernode in range(assignment.num_supernodes):
        members = np.flatnonzero(assignment.assignment == supernode).astype(np.int64)
        if len(members) == 0:
            continue
        if not bad[supernode]:
            emit(members)
            continue
        if strategy == "split_local_swap_accept":
            if objective_name == "relation" and relation_profiles is not None:
                profile_block = relation_profiles[members].astype(np.float64, copy=False)
                center = profile_block.mean(axis=0)
                first_axis = profile_block[:, 0] if profile_block.shape[1] else members.astype(np.float64)
                distance = np.sum((profile_block - center) ** 2, axis=1)
                order = np.lexsort((members, distance, first_axis))
            elif objective_name == "task" and labels is not None:
                member_labels = labels[members].astype(np.int64, copy=False)
                label_key = np.where(member_labels >= 0, member_labels, np.iinfo(np.int64).max)
                first_axis = Z[members, 0].astype(np.float64)
                order = np.lexsort((members, first_axis, label_key))
            else:
                center = Z[members].astype(np.float64, copy=False).mean(axis=0)
                first_axis = Z[members, 0].astype(np.float64)
                distance = np.sum((Z[members].astype(np.float64, copy=False) - center) ** 2, axis=1)
                order = np.lexsort((members, distance, first_axis))
        else:
            order = np.lexsort((members, Z[members, 0].astype(np.float64)))
        ordered = members[order]
        for start in range(0, len(ordered), 2):
            emit(ordered[start : start + 2])
    repaired = Assignment(new_assignment, np.asarray(new_types, dtype=np.int32))
    repaired_sizes = repaired.cluster_sizes()
    _after_sizes, _after_spreads, _after_entropy, _after_relation_variance, after_objective = cluster_metrics(
        repaired
    )
    after_dee_proxy = float(
        after_objective["cluster_sketch_spread"] + after_objective["relation_profile_variance"]
    )
    after_sipe_proxy = float(after_objective["cluster_sketch_spread"])
    cumulative_energy_delta = float(after_dee_proxy - before_dee_proxy)
    accept_only_if_improves = bool(guard.get("accept_only_if_cumulative_improves", False))
    accept_metric = str(guard.get("accept_metric", "proxy")).lower().replace("-", "_")
    objective_metric = {
        "energy": "cluster_sketch_spread",
        "relation": "relation_profile_variance",
        "task": "train_label_entropy",
    }.get(objective_name)
    before_score = (
        float(before_objective.get(objective_metric, 0.0)) if objective_metric else before_dee_proxy
    )
    after_score = float(after_objective.get(objective_metric, 0.0)) if objective_metric else after_dee_proxy
    accepted = after_score < before_score
    if objective_name == "current":
        accepted = (after_dee_proxy < before_dee_proxy) or (after_sipe_proxy < before_sipe_proxy)
    if accept_only_if_improves and accept_metric != "true_cumulative" and not accepted:
        repaired = assignment
        repaired_sizes = sizes
        after_dee_proxy = before_dee_proxy
        after_sipe_proxy = before_sipe_proxy
        cumulative_energy_delta = 0.0
        after_score = before_score
    else:
        accepted = True
    trace_signature = (
        f"{objective_name}:"
        f"{','.join(str(value) for value in selected_clusters)}:"
        f"{int(np.sum(np.maximum(repaired_sizes - 1, 0)))}:"
        f"{int(bool(accepted))}"
    )
    return repaired, {
        "enabled": True,
        "repair_bad_clusters": True,
        "repair_strategy": strategy,
        "repair_objective_name": objective_name,
        "repair_accepted": bool(accepted),
        "repaired_cluster_count": int(np.sum(bad)),
        "repair_selected_clusters": selected_clusters,
        "repair_trace_signature": trace_signature,
        "spread_cutoff": float(spread_cutoff),
        "repair_objective_score_before": before_score,
        "repair_objective_score_after": after_score,
        "repair_objective": {
            **before_objective,
            "cumulative_energy_delta": cumulative_energy_delta,
        },
        "repair_objective_after": after_objective,
        "estimated_cumulative_dee_before": before_dee_proxy,
        "estimated_cumulative_dee_after": after_dee_proxy,
        "estimated_cumulative_sipe_before": before_sipe_proxy,
        "estimated_cumulative_sipe_after": after_sipe_proxy,
        "node_reduction_before": int(np.sum(np.maximum(sizes - 1, 0))),
        "node_reduction_after": int(np.sum(np.maximum(repaired_sizes - 1, 0))),
    }


def _maybe_apply_true_cumulative_repair_gate(
    *,
    original: HeteroGraph,
    current: HeteroGraph,
    before_assignment: Assignment,
    repaired_assignment: Assignment,
    cumulative_assignment: np.ndarray | None,
    root_spectral_input: np.ndarray | None,
    current_spectral_input: np.ndarray,
    root_relation_weights: dict[int, float] | None,
    current_relation_weights: dict[int, float],
    config: dict,
    diagnostics: dict,
    seed: int,
) -> tuple[Assignment, dict]:
    guard = config.get("coarsening", {}).get("cumulative_guard", {})
    accept_metric = str(guard.get("accept_metric", "proxy")).lower().replace("-", "_")
    objective_name = _repair_objective_name(guard)
    if (
        accept_metric != "true_cumulative"
        or not bool(guard.get("accept_only_if_cumulative_improves", False))
        or not bool(diagnostics.get("repair_accepted", False))
        or cumulative_assignment is None
    ):
        return repaired_assignment, diagnostics

    spectral_input = root_spectral_input
    relation_weights = root_relation_weights
    if spectral_input is None and current is original:
        spectral_input = current_spectral_input
        relation_weights = current_relation_weights
    if spectral_input is None or relation_weights is None:
        diagnostics["true_cumulative_accept"] = None
        diagnostics["repair_rejected_by_true_cumulative"] = False
        diagnostics["true_cumulative_skipped_reason"] = "missing_root_spectral_state"
        return repaired_assignment, diagnostics

    before_cumulative = before_assignment.assignment[cumulative_assignment]
    after_cumulative = repaired_assignment.assignment[cumulative_assignment]
    feature_aggregation, feature_weights, _feature_diag = _feature_aggregation_options(config)
    before_coarse = coarsen_graph(
        current,
        before_assignment,
        feature_aggregation=feature_aggregation,
        feature_weights=feature_weights,
        pagerank_iterations=int(
            config.get("coarsening", {}).get("feature_aggregation_pagerank_iterations", 20)
        ),
        pagerank_damping=float(
            config.get("coarsening", {}).get("feature_aggregation_pagerank_damping", 0.85)
        ),
    )
    after_coarse = coarsen_graph(
        current,
        repaired_assignment,
        feature_aggregation=feature_aggregation,
        feature_weights=feature_weights,
        pagerank_iterations=int(
            config.get("coarsening", {}).get("feature_aggregation_pagerank_iterations", 20)
        ),
        pagerank_damping=float(
            config.get("coarsening", {}).get("feature_aggregation_pagerank_damping", 0.85)
        ),
    )
    diagnostics_cfg = config.get("diagnostics", {})
    smoothing_steps = int(diagnostics_cfg.get("spectral_smoothing_steps", 1))
    eigen_max_nodes = diagnostics_cfg.get("cumulative_spectral_exact_eigenvalue_max_nodes", 0)
    before_metrics = compute_spectral_diagnostics(
        original=original,
        coarse=before_coarse,
        assignment=Assignment(
            assignment=before_cumulative.astype(np.int64, copy=False),
            supernode_type=before_coarse.node_type.astype(np.int32, copy=False),
        ),
        seed=int(seed),
        num_signals=int(spectral_input.shape[1]),
        smoothing_steps=smoothing_steps,
        relation_weights=relation_weights,
        Z=spectral_input,
        exact_eigenvalue_max_nodes=eigen_max_nodes,
        baseline_methods=None,
    )
    after_metrics = compute_spectral_diagnostics(
        original=original,
        coarse=after_coarse,
        assignment=Assignment(
            assignment=after_cumulative.astype(np.int64, copy=False),
            supernode_type=after_coarse.node_type.astype(np.int32, copy=False),
        ),
        seed=int(seed) + 1,
        num_signals=int(spectral_input.shape[1]),
        smoothing_steps=smoothing_steps,
        relation_weights=relation_weights,
        Z=spectral_input,
        exact_eigenvalue_max_nodes=eigen_max_nodes,
        baseline_methods=None,
    )
    before_dee = float(before_metrics.get("dirichlet_energy_relative_error", 0.0))
    after_dee = float(after_metrics.get("dirichlet_energy_relative_error", 0.0))
    before_sipe = float(before_metrics.get("sketch_inner_product_relative_error", 0.0))
    after_sipe = float(after_metrics.get("sketch_inner_product_relative_error", 0.0))
    before_ree = float(before_metrics.get("relation_energy_relative_error_max", 0.0))
    after_ree = float(after_metrics.get("relation_energy_relative_error_max", 0.0))
    before_task_macro = None
    after_task_macro = None
    if objective_name == "task":
        before_task_macro = float(_task_diagnostics(original, before_coarse, before_cumulative).get("macro_f1", 0.0))
        after_task_macro = float(_task_diagnostics(original, after_coarse, after_cumulative).get("macro_f1", 0.0))
        accepted = after_task_macro >= before_task_macro
    elif objective_name == "relation":
        accepted = after_ree < before_ree
    else:
        accepted = (after_dee < before_dee) or (after_sipe < before_sipe)
    diagnostics.update(
        {
            "repair_objective_name": objective_name,
            "true_cumulative_accept": bool(accepted),
            "repair_rejected_by_true_cumulative": bool(not accepted),
            "true_cumulative_dee_before": before_dee,
            "true_cumulative_dee_after": after_dee,
            "true_cumulative_sipe_before": before_sipe,
            "true_cumulative_sipe_after": after_sipe,
            "true_cumulative_ree_max_before": before_ree,
            "true_cumulative_ree_max_after": after_ree,
            "true_cumulative_task_macro_f1_before": before_task_macro,
            "true_cumulative_task_macro_f1_after": after_task_macro,
        }
    )
    if accepted:
        return repaired_assignment, diagnostics
    diagnostics["repair_accepted"] = False
    diagnostics["node_reduction_after"] = diagnostics.get("node_reduction_before")
    diagnostics["estimated_cumulative_dee_after"] = diagnostics.get("estimated_cumulative_dee_before")
    diagnostics["estimated_cumulative_sipe_after"] = diagnostics.get("estimated_cumulative_sipe_before")
    return before_assignment, diagnostics


def _classification_f1_from_labels(truth: np.ndarray, predicted: np.ndarray) -> dict:
    truth = np.asarray(truth).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError("truth and predicted labels must have the same shape")
    valid = (truth >= 0) & (predicted >= 0)
    base = {
        "model": "majority_label_projection",
        "train_on": "coarse_graph_majority_labels",
        "eval_on": "original_labels_projected_from_coarse",
        "labeled_nodes": int(np.sum(valid)),
    }
    if not np.any(valid):
        return {**base, "micro_f1": 0.0, "macro_f1": 0.0}
    y_true = truth[valid].astype(np.int64, copy=False)
    y_pred = predicted[valid].astype(np.int64, copy=False)
    f1_values: list[float] = []
    for label in np.union1d(y_true, y_pred):
        true_pos = int(np.sum((y_true == label) & (y_pred == label)))
        false_pos = int(np.sum((y_true != label) & (y_pred == label)))
        false_neg = int(np.sum((y_true == label) & (y_pred != label)))
        denom = 2 * true_pos + false_pos + false_neg
        f1_values.append(0.0 if denom == 0 else float(2 * true_pos / denom))
    return {
        **base,
        "micro_f1": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_values) if f1_values else 0.0),
    }


def _task_diagnostics(
    original: HeteroGraph,
    coarse: HeteroGraph,
    cumulative_assignment: np.ndarray | None,
) -> dict:
    if original.labels is None or coarse.labels is None or cumulative_assignment is None:
        return {
            "model": "majority_label_projection",
            "train_on": "coarse_graph_majority_labels",
            "eval_on": "original_labels_projected_from_coarse",
            "labeled_nodes": 0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "skipped": True,
        }
    projected = np.asarray(coarse.labels).reshape(-1)[cumulative_assignment]
    result = _classification_f1_from_labels(np.asarray(original.labels).reshape(-1), projected)
    result["skipped"] = False
    return result


def _config_with_level_feature_store(config: dict, level: int) -> dict:
    feature_cfg = config.get("features", {})
    mmap_dir = feature_cfg.get("projection_mmap_dir")
    if mmap_dir in {None, ""}:
        return config
    level_config = deepcopy(config)
    level_config.setdefault("features", {})["projection_mmap_dir"] = str(Path(mmap_dir) / f"level_{level}")
    return level_config


def _feature_aggregation_options(config: dict) -> tuple[str, np.ndarray | dict[int, np.ndarray] | None, dict]:
    coarsening_cfg = config.get("coarsening", {})
    method = str(coarsening_cfg.get("feature_aggregation", "mean")).lower()
    weights = coarsening_cfg.get("feature_aggregation_weights")
    weight_path = coarsening_cfg.get("feature_aggregation_weight_path")
    if weight_path not in {None, ""}:
        weights = np.load(Path(weight_path))
    elif isinstance(weights, dict):
        weights = {
            int(type_id): np.asarray(values, dtype=np.float32)
            for type_id, values in weights.items()
        }
    elif weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
    pagerank_iterations = int(coarsening_cfg.get("feature_aggregation_pagerank_iterations", 20))
    pagerank_damping = float(coarsening_cfg.get("feature_aggregation_pagerank_damping", 0.85))
    diagnostics = {
        "method": method,
        "uses_weights": bool(method != "mean"),
        "weight_source": {
            "mean": "cluster_count",
            "degree_weighted": "incident_edge_weight_mass",
            "pagerank_weighted": "pagerank",
            "custom_weight": "custom",
        }.get(method, method),
        "custom_weight_path": None if weight_path in {None, ""} else str(weight_path),
        "pagerank_iterations": pagerank_iterations if method == "pagerank_weighted" else None,
        "pagerank_damping": pagerank_damping if method == "pagerank_weighted" else None,
    }
    return method, weights, diagnostics


def _make_candidate_store(
    graph: HeteroGraph,
    config: dict,
    level: int,
) -> BoundedCandidateStore | ArrayCandidateStore:
    candidate_cfg = config.get("candidates", {})
    same_type_only = bool(config.get("coarsening", {}).get("same_type_only", True))
    K = int(candidate_cfg["total_budget_K"])
    backend = str(candidate_cfg.get("store_backend", "heap")).lower()
    if backend in {"heap", "bounded_heap"}:
        return BoundedCandidateStore(graph.node_type, K=K, same_type_only=same_type_only)
    if backend in {"array", "mmap", "memmap"}:
        mmap_dir = candidate_cfg.get("mmap_dir")
        level_mmap_dir = None if mmap_dir is None else Path(mmap_dir) / f"level_{level}"
        return ArrayCandidateStore(
            graph.node_type,
            K=K,
            same_type_only=same_type_only,
            mmap_dir=level_mmap_dir,
        )
    raise ValueError(f"unsupported candidate store_backend: {backend}")


def _flush_candidate_store(store: BoundedCandidateStore | ArrayCandidateStore) -> None:
    flush = getattr(store, "flush", None)
    if callable(flush):
        flush()


def _store_source_node_coverage(store: BoundedCandidateStore | ArrayCandidateStore) -> dict[str, float]:
    coverage = getattr(store, "source_node_coverage", None)
    if callable(coverage):
        return {str(key): float(value) for key, value in coverage().items()}
    return {}


def _store_buffer_nbytes(store: BoundedCandidateStore | ArrayCandidateStore) -> dict[str, int]:
    buffer_nbytes = getattr(store, "buffer_nbytes", None)
    if callable(buffer_nbytes):
        return {str(key): int(value) for key, value in buffer_nbytes().items()}
    return {}


def _selected_source_score_breakdown(
    *,
    scoring_context,
    selected_pairs: np.ndarray,
    source_lookup,
    assignment: Assignment,
    max_pairs: int,
) -> dict:
    selected_pairs = np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)
    total_pairs = int(selected_pairs.shape[0])
    if total_pairs == 0 or source_lookup is None:
        return {
            "selected_source_avg_score": {},
            "selected_source_avg_delta_spec": {},
            "selected_source_avg_delta_conv": {},
            "selected_source_cluster_size_hist": {},
            "selected_merge_stats_by_source": {},
            "selected_source_analysis_total_pairs": total_pairs,
            "selected_source_analysis_sampled_pairs": 0,
        }
    limit = max(1, int(max_pairs))
    if total_pairs > limit:
        indices = np.linspace(0, total_pairs - 1, num=limit, dtype=np.int64)
        selected_pairs = selected_pairs[indices]
    scored, terms = score_pair_block_with_terms(scoring_context, selected_pairs)
    if scored.size == 0:
        sampled = 0
        score = np.empty(0, dtype=np.float64)
        spec = np.empty(0, dtype=np.float32)
        conv = np.empty(0, dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        sampled = int(scored.shape[0])
        pairs = scored[:, :2].astype(np.int64, copy=False)
        score = scored[:, 2].astype(np.float64, copy=False)
        spec = np.asarray(terms.get("spec", np.empty(0)), dtype=np.float32)
        conv = np.asarray(terms.get("conv", np.empty(0)), dtype=np.float32)
    cluster_sizes = assignment.cluster_sizes()
    values: dict[str, dict[str, list[float] | dict[str, int]]] = {}
    for index, (raw_i, raw_j) in enumerate(pairs):
        i = int(raw_i)
        j = int(raw_j)
        source = str(source_lookup(i, j) or "unknown")
        item = values.setdefault(
            source,
            {"score": [], "spec": [], "conv": [], "cluster_size_hist": {}},
        )
        item["score"].append(float(score[index]))
        item["spec"].append(float(spec[index]) if index < len(spec) else 0.0)
        item["conv"].append(float(conv[index]) if index < len(conv) else 0.0)
        supernode = int(assignment.assignment[i])
        size = int(cluster_sizes[supernode]) if 0 <= supernode < len(cluster_sizes) else 0
        hist = item["cluster_size_hist"]
        assert isinstance(hist, dict)
        hist[str(size)] = int(hist.get(str(size), 0)) + 1

    avg_score: dict[str, float] = {}
    avg_spec: dict[str, float] = {}
    avg_conv: dict[str, float] = {}
    hist_by_source: dict[str, dict[str, int]] = {}
    nested: dict[str, dict] = {}
    for source, item in sorted(values.items()):
        source_scores = item["score"]
        source_specs = item["spec"]
        source_convs = item["conv"]
        assert isinstance(source_scores, list)
        assert isinstance(source_specs, list)
        assert isinstance(source_convs, list)
        hist = item["cluster_size_hist"]
        assert isinstance(hist, dict)
        avg_score[source] = float(np.mean(source_scores)) if source_scores else 0.0
        avg_spec[source] = float(np.mean(source_specs)) if source_specs else 0.0
        avg_conv[source] = float(np.mean(source_convs)) if source_convs else 0.0
        hist_by_source[source] = {str(key): int(value) for key, value in sorted(hist.items())}
        nested[source] = {
            "avg_score": avg_score[source],
            "avg_delta_spec": avg_spec[source],
            "avg_delta_conv": avg_conv[source],
            "cluster_size_hist": hist_by_source[source],
        }
    return {
        "selected_source_avg_score": avg_score,
        "selected_source_avg_delta_spec": avg_spec,
        "selected_source_avg_delta_conv": avg_conv,
        "selected_source_cluster_size_hist": hist_by_source,
        "selected_merge_stats_by_source": nested,
        "selected_source_analysis_total_pairs": total_pairs,
        "selected_source_analysis_sampled_pairs": sampled,
    }


def run_multilevel_coarsening(graph: HeteroGraph, config: dict) -> list[LevelResult]:
    original_nodes = graph.num_nodes
    target_nodes = max(1, int(np.ceil(original_nodes * float(config["coarsening"]["target_ratio"]))))
    max_levels = int(config["coarsening"]["max_levels"])
    output_dir = Path(config.get("output", {}).get("dir", "outputs/default_run"))
    resume_cfg = config.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", False))
    allow_legacy_checkpoints = bool(resume_cfg.get("allow_legacy_checkpoints", False))
    if _has_level_dirs(output_dir) and not resume_enabled:
        raise FileExistsError(
            f"{output_dir} already contains level outputs; rerun with --resume or use a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_levels = discover_completed_levels(
        output_dir,
        allow_legacy_checkpoints=allow_legacy_checkpoints,
    )
    legacy_resume = any(level.legacy for level in completed_levels)
    if completed_levels:
        last_completed = completed_levels[-1]
        current = load_graph(last_completed.directory)
        start_level = last_completed.level + 1
        progress_message(
            config,
            f"resume: using level {last_completed.level} from {last_completed.directory}",
        )
    else:
        current = graph
        start_level = 1
    cumulative_assignment: np.ndarray | None = (
        np.arange(original_nodes, dtype=np.int64)
        if not completed_levels
        else None
    )
    root_spectral_input: np.ndarray | None = None
    root_relation_weights: dict[int, float] | None = None
    results: list[LevelResult] = []

    for level in range(start_level, max_levels + 1):
        if current.num_nodes <= target_nodes:
            break
        _reset_cuda_peak_memory_stats()
        level_dir = output_dir / f"level_{level}"
        runtime: dict[str, float] = {}
        progress_message(
            config,
            f"level {level}: start ({current.num_nodes} nodes, target {target_nodes})",
        )

        progress_message(config, f"level {level}: sketch start")
        start = perf_counter()
        sketch_diagnostics: dict = {}
        Z = compute_lowpass_sketch(current, config, diagnostics=sketch_diagnostics)
        runtime["sketch"] = perf_counter() - start
        progress_message(config, f"level {level}: sketch done in {runtime['sketch']:.2f}s")

        progress_message(config, f"level {level}: candidates start")
        start = perf_counter()
        partition_id = default_partition(current)
        candidate_cfg = config.get("candidates", {})
        store = _make_candidate_store(current, config, level)
        use_chunked = bool(candidate_cfg.get("use_chunked_generation", False))
        candidate_substage_times: dict[str, float] = {
            "onehop": 0.0,
            "incident_index_build": 0.0,
            "twohop_expansion": 0.0,
            "simhash": 0.0,
            "bucket_emit": 0.0,
            "partition_ann": 0.0,
            "fallback": 0.0,
            "store_finalize": 0.0,
        }
        candidate_source_generation: dict[str, dict[str, float | int]] = {}

        def _add_source_stats(source: str, stats: dict | None) -> None:
            if not stats:
                return
            target = candidate_source_generation.setdefault(source, {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    target[key] = float(target.get(key, 0.0)) + float(value)
                else:
                    target[key] = value

        if config["candidates"].get("enable_onehop", True):
            source_start = perf_counter()
            onehop_stats = None
            if use_chunked:
                onehop_stats = generate_onehop_candidates_chunked(
                    current,
                    Z,
                    partition_id,
                    config,
                    store,
                    edge_chunk_size=int(candidate_cfg.get("edge_chunk_size", 1_000_000)),
                )
            else:
                onehop_stats = generate_onehop_candidates(current, Z, partition_id, config, store)
            candidate_substage_times["onehop"] += float(perf_counter() - source_start)
            _add_source_stats("onehop", onehop_stats)
        if config["candidates"].get("enable_capped_twohop", True):
            source_start = perf_counter()
            twohop_stats = None
            if use_chunked:
                twohop_config = config
                incident_index_mmap_dir = candidate_cfg.get("incident_index_mmap_dir")
                if incident_index_mmap_dir is not None:
                    twohop_config = deepcopy(config)
                    twohop_config.setdefault("candidates", {})["incident_index_mmap_dir"] = str(
                        Path(incident_index_mmap_dir) / f"level_{level}"
                    )
                twohop_stats = generate_capped_twohop_candidates_chunked(
                    current,
                    Z,
                    partition_id,
                    twohop_config,
                    store,
                    middle_chunk_size=int(candidate_cfg.get("middle_chunk_size", 100_000)),
                    edge_chunk_size=int(candidate_cfg.get("edge_chunk_size", 1_000_000)),
                )
            else:
                twohop_stats = generate_capped_twohop_candidates(current, Z, partition_id, config, store)
            twohop_elapsed = float(perf_counter() - source_start)
            if twohop_stats:
                candidate_substage_times["incident_index_build"] += float(
                    twohop_stats.get("incident_index_build_time", 0.0) or 0.0
                )
                candidate_substage_times["twohop_expansion"] += float(
                    twohop_stats.get(
                        "twohop_expansion_time",
                        max(0.0, twohop_elapsed - candidate_substage_times["incident_index_build"]),
                    )
                    or 0.0
                )
            else:
                candidate_substage_times["twohop_expansion"] += twohop_elapsed
            _add_source_stats("capped_twohop", twohop_stats)
        if config["candidates"].get("enable_bucket", True):
            for table_id, bits in enumerate(_bucket_hash_bits(candidate_cfg)):
                simhash_start = perf_counter()
                buckets = compute_simhash_buckets(
                    Z,
                    current.node_type,
                    partition_id,
                    bits=int(bits),
                    seed=int(config.get("seed", 12345)) + level + 1009 * table_id,
                )
                candidate_substage_times["simhash"] += float(perf_counter() - simhash_start)
                bucket_config = deepcopy(config)
                bucket_config.setdefault("candidates", {})["active_hash_bits"] = int(bits)
                bucket_config["candidates"]["active_hash_table"] = int(table_id)
                emit_start = perf_counter()
                bucket_stats = None
                if use_chunked:
                    bucket_stats = generate_bucket_candidates_chunked(
                        buckets,
                        current.node_type,
                        partition_id,
                        bucket_config,
                        store,
                        node_chunk_size=int(candidate_cfg.get("node_chunk_size", 1_000_000)),
                    )
                else:
                    bucket_stats = generate_bucket_candidates(
                        buckets,
                        current.node_type,
                        partition_id,
                        bucket_config,
                        store,
                    )
                candidate_substage_times["bucket_emit"] += float(perf_counter() - emit_start)
                _add_source_stats("bucket", bucket_stats)
        if config["candidates"].get("enable_partition_ann", False):
            source_start = perf_counter()
            ann_stats = generate_partition_ann_candidates(current, Z, partition_id, config, store)
            candidate_substage_times["partition_ann"] += float(perf_counter() - source_start)
            _add_source_stats("partition_ann", ann_stats)
        if bool(candidate_cfg.get("enable_fallback", True)):
            source_start = perf_counter()
            fallback_stats = _add_fallback_candidates(current, partition_id, store, config)
            candidate_substage_times["fallback"] += float(perf_counter() - source_start)
            _add_source_stats("fallback", fallback_stats)
        finalize_start = perf_counter()
        _flush_candidate_store(store)
        pair_count_fn = getattr(store, "pair_count", None)
        pair_count = int(pair_count_fn()) if callable(pair_count_fn) else int(store.to_pairs().shape[0])
        candidate_counts = store.counts()
        source_counts = store.source_counts()
        candidate_substage_times["store_finalize"] += float(perf_counter() - finalize_start)
        runtime["candidates"] = perf_counter() - start
        candidate_generation = {
            "total_time": float(runtime["candidates"]),
            "retained_pair_count": int(pair_count),
            "candidate_pairs_per_sec": float(pair_count / runtime["candidates"])
            if runtime["candidates"] > 0
            else 0.0,
            "substage_times": candidate_substage_times,
            "source_generation": candidate_source_generation,
            "source_node_coverage": _store_source_node_coverage(store),
            "memory_by_candidate_buffers": _store_buffer_nbytes(store),
        }
        progress_message(
            config,
            f"level {level}: candidates done in {runtime['candidates']:.2f}s "
            f"({pair_count} pairs)",
        )

        progress_message(config, f"level {level}: scoring start")
        start = perf_counter()
        progress_message(config, f"level {level}: scoring relation profiles start")
        relation_profile_mode = str(
            config.get("scoring", {}).get("relation_profile_mode", "relationwise")
        )
        relation_profiles = compute_relation_profiles(current, mode=relation_profile_mode)
        progress_message(config, f"level {level}: scoring relation profiles done")
        progress_message(config, f"level {level}: scoring fusion weights start")
        relation_weights = compute_relation_fusion_weights(current, Z.astype(np.float32), config)
        progress_message(config, f"level {level}: scoring fusion weights done")
        progress_message(config, f"level {level}: scoring conv response start")
        conv = compute_conv_response_sketch(
            current,
            Z.astype(np.float32, copy=False),
            relation_weights,
            operator=str(config.get("scoring", {}).get("conv_response_operator", "fused_operator")),
            relation_operator_mode=str(config.get("fusion", {}).get("relation_operator_mode", "relationwise")),
        )
        progress_message(config, f"level {level}: scoring conv response done")
        progress_message(config, f"level {level}: scoring candidate pairs start")
        scoring_config = _config_with_level_feature_store(config, level)
        matching_method = str(config.get("coarsening", {}).get("matching_method", "mutual_best"))
        matching_method_normalized = matching_method.lower().replace("-", "_")
        level_matching_config = _config_for_level(config, current.num_nodes, target_nodes)
        streaming_mutual_best = matching_method_normalized == "mutual_best"
        score_term_accumulator = ScoreTermAccumulator.from_config(scoring_config)
        score_contribution_accumulator = ScoreTermAccumulator.from_config(scoring_config)
        scored = None
        scored_pair_count = 0
        streaming_state = None
        if streaming_mutual_best:
            streaming_state = initialize_mutual_best_state(current)
            scoring_context = prepare_pair_scoring_context(
                current,
                Z,
                relation_profiles,
                conv,
                current.features,
                scoring_config,
                partition_id=partition_id,
            )
            pair_block_size = max(
                int(
                    candidate_cfg.get(
                        "pair_block_size",
                        scoring_config.get("acceleration", {}).get("scoring_batch_size", 65_536),
                    )
                ),
                1,
            )
            block_total = ceil(pair_count / pair_block_size) if pair_count else 0
            pair_blocks = store.iter_pair_blocks(block_size=pair_block_size)
            for pair_block in progress_iter(
                pair_blocks,
                total=block_total,
                desc="score/match pair blocks",
                config=config,
                unit="block",
            ):
                scored_block, term_values = score_pair_block_with_terms(scoring_context, pair_block)
                score_term_accumulator.update(term_values)
                score_contribution_accumulator.update(
                    score_term_contributions(scoring_context, term_values)
                )
                scored_pair_count += int(scored_block.shape[0])
                mutual_best_update_block(
                    current,
                    streaming_state,
                    scored_block,
                    level_matching_config,
                    partition_id=partition_id,
                    source_lookup=getattr(store, "source_for_pair", None),
                )
        else:
            pairs = store.to_pairs()
            scoring_context = prepare_pair_scoring_context(
                current,
                Z,
                relation_profiles,
                conv,
                current.features,
                scoring_config,
                partition_id=partition_id,
            )
            scored, term_values = score_pair_block_with_terms(scoring_context, pairs)
            score_term_accumulator.update(term_values)
            score_contribution_accumulator.update(
                score_term_contributions(scoring_context, term_values)
            )
            scored_pair_count = int(scored.shape[0])
        score_term_summary = score_term_accumulator.summary()
        score_contribution_summary = score_contribution_accumulator.summary()
        progress_message(config, f"level {level}: scoring candidate pairs done")
        runtime["scoring"] = perf_counter() - start
        progress_message(
            config,
            f"level {level}: scoring done in {runtime['scoring']:.2f}s "
            f"({scored_pair_count} pairs)",
        )

        progress_message(config, f"level {level}: matching and aggregation start")
        start = perf_counter()
        matching_start = perf_counter()
        progress_message(config, f"level {level}: matching start (method={matching_method})")
        source_lookup = getattr(store, "source_for_pair", None)
        if streaming_mutual_best:
            assert streaming_state is not None
            assignment = finalize_mutual_best(
                current,
                streaming_state,
                level_matching_config,
                source_lookup=source_lookup,
            )
        else:
            assert scored is not None
            assignment = run_matching(
                current,
                scored,
                level_matching_config,
                partition_id=partition_id,
                source_lookup=source_lookup,
            )
        progress_message(config, f"level {level}: matching done")
        matching_diag = dict(getattr(assignment, "diagnostics", {}) or {})
        pre_repair_assignment = assignment
        assignment, cumulative_guard_diag = _repair_bad_clusters(
            current,
            assignment,
            Z.astype(np.float32, copy=False),
            level_matching_config,
        )
        repair_spectral_num_signals = int(
            config.get("diagnostics", {}).get("spectral_num_signals", min(Z.shape[1], 4))
        )
        repair_spectral_input = Z[:, : max(1, min(repair_spectral_num_signals, Z.shape[1]))].astype(
            np.float32,
            copy=False,
        )
        assignment, cumulative_guard_diag = _maybe_apply_true_cumulative_repair_gate(
            original=graph,
            current=current,
            before_assignment=pre_repair_assignment,
            repaired_assignment=assignment,
            cumulative_assignment=cumulative_assignment,
            root_spectral_input=root_spectral_input,
            current_spectral_input=repair_spectral_input,
            root_relation_weights=root_relation_weights,
            current_relation_weights=relation_weights,
            config=level_matching_config,
            diagnostics=cumulative_guard_diag,
            seed=int(config.get("seed", 12345)) + level + 20_000,
        )
        runtime["matching"] = perf_counter() - matching_start
        matched_pairs_by_source = selected_pair_sources(
            assignment,
            source_lookup or (lambda _i, _j: None),
        )
        selected_merges_by_source = dict(
            matching_diag.get("selected_merges_by_source") or matched_pairs_by_source
        )
        selected_merge_pairs = np.asarray(
            matching_diag.get("_selected_merge_pairs", np.empty((0, 2), dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1, 2)
        next_cumulative_assignment = (
            assignment.assignment[cumulative_assignment]
            if cumulative_assignment is not None
            else None
        )
        aggregation_chunk_size = int(config.get("coarsening", {}).get("aggregation_chunk_size", 1_000_000))
        aggregation_reducer = str(config.get("coarsening", {}).get("aggregation_reducer", "sort"))
        feature_aggregation, feature_weights, feature_aggregation_diag = _feature_aggregation_options(config)
        progress_message(
            config,
            f"level {level}: chunked aggregation start "
            f"(chunk_size={aggregation_chunk_size}, reducer={aggregation_reducer}, "
            f"feature_aggregation={feature_aggregation})",
        )
        aggregation_start = perf_counter()
        coarse = coarsen_graph_chunked(
            current,
            assignment,
            chunk_size=aggregation_chunk_size,
            output_dir=level_dir,
            reducer=aggregation_reducer,
            feature_aggregation=feature_aggregation,
            feature_weights=feature_weights,
            pagerank_iterations=int(
                config.get("coarsening", {}).get("feature_aggregation_pagerank_iterations", 20)
            ),
            pagerank_damping=float(
                config.get("coarsening", {}).get("feature_aggregation_pagerank_damping", 0.85)
            ),
        )
        progress_message(config, f"level {level}: chunked aggregation done")
        runtime["aggregation"] = perf_counter() - aggregation_start
        runtime["matching_and_aggregation"] = perf_counter() - start
        progress_message(
            config,
            f"level {level}: matching and aggregation done in "
            f"{runtime['matching_and_aggregation']:.2f}s ({coarse.num_nodes} nodes)",
        )

        progress_message(config, f"level {level}: diagnostics and save start")
        diagnostics = compute_diagnostics(
            current,
            coarse,
            assignment,
            candidate_counts,
            source_counts,
            runtime_by_stage=runtime,
            config=config,
            artifact_dirs={
                name: path
                for name, path in {
                    "candidate_mmap": (
                        Path(candidate_cfg["mmap_dir"]) / f"level_{level}"
                        if candidate_cfg.get("mmap_dir") is not None
                        else None
                    ),
                    "incident_index_mmap": (
                        Path(candidate_cfg["incident_index_mmap_dir"]) / f"level_{level}"
                        if candidate_cfg.get("incident_index_mmap_dir") is not None
                        else None
                    ),
                    "projected_features": (
                        Path(config["features"]["projection_mmap_dir"]) / f"level_{level}"
                        if config.get("features", {}).get("projection_mmap_dir") is not None
                        else None
                    ),
                    "aggregation_shards": (
                        level_dir / "_aggregation_shards"
                        if aggregation_reducer == "sort"
                        else None
                    ),
                }.items()
                if path is not None
            },
            candidate_generation=candidate_generation,
            Z=Z.astype(np.float32, copy=False),
            relation_profiles=relation_profiles,
            conv_response=conv,
        )
        diagnostics["sketch"] = {
            key: value
            for key, value in sketch_diagnostics.items()
            if key not in {"fusion", "metapath_sketch"}
        }
        diagnostics["fusion"] = sketch_diagnostics.get("fusion", {})
        diagnostics["metapath_sketch"] = sketch_diagnostics.get(
            "metapath_sketch",
            {"enabled": False, "num_paths": 0, "paths": []},
        )
        diagnostics["feature_aggregation"] = feature_aggregation_diag
        diagnostics["score_terms"] = score_term_summary
        diagnostics["score_contributions"] = score_contribution_summary
        diagnostics["score_contribution_share"] = _score_contribution_share(score_contribution_summary)
        diagnostics["cumulative_guard"] = cumulative_guard_diag
        diagnostics["terminal_guard"] = dict(
            getattr(assignment, "diagnostics", {}).get(
                "terminal_guard",
                {
                    "enabled": False,
                    "protected_node_count": 0,
                    "protected_node_fraction": 0.0,
                    "protected_by_reason": {
                        "hub": 0,
                        "rare_relation": 0,
                        "boundary": 0,
                        "label_entropy": 0,
                    },
                    "merge_blocked_count": 0,
                    "merge_blocked_fraction": 0.0,
                    "cluster_size_reduction_due_to_guard": 0,
                },
            )
        )
        diagnostics["target_control"] = _target_control_diagnostics(
            config,
            original_nodes=original_nodes,
            input_nodes=current.num_nodes,
            target_nodes=target_nodes,
            level_config=level_matching_config,
        )
        diagnostics_cfg = config.get("diagnostics", {})
        diagnostics["config"] = _resolved_config_diagnostics(scoring_config)
        diagnostics["matched_pairs_by_source"] = matched_pairs_by_source
        diagnostics["selected_merges_by_source"] = selected_merges_by_source
        diagnostics.update(
            _selected_source_score_breakdown(
                scoring_context=scoring_context,
                selected_pairs=selected_merge_pairs,
                source_lookup=source_lookup,
                assignment=assignment,
                max_pairs=int(
                    diagnostics_cfg.get(
                        "selected_source_analysis_max_pairs",
                        diagnostics_cfg.get("selected_merge_analysis_max_pairs", 20_000),
                    )
                ),
            )
        )
        if streaming_state is not None and streaming_state.selected_quota_diagnostics is not None:
            quota_diag = streaming_state.selected_quota_diagnostics
            diagnostics["selected_match_quota"] = quota_diag
            diagnostics["selected_match_source_distribution_before_quota"] = quota_diag.get(
                "selected_match_source_distribution_before_quota",
                {},
            )
            diagnostics["selected_match_source_distribution_after_quota"] = quota_diag.get(
                "selected_match_source_distribution_after_quota",
                {},
            )
            diagnostics["selected_source_fraction_before_quota"] = quota_diag.get(
                "selected_source_fraction_before_quota",
                {},
            )
            diagnostics["selected_source_fraction_after_quota"] = quota_diag.get(
                "selected_source_fraction_after_quota",
                {},
            )
            diagnostics["quota"] = quota_diag.get("quota", {})
            diagnostics["quota_violation"] = quota_diag.get("quota_violation", {})
        diagnostics["fallback_selected_fraction"] = float(
            matched_pairs_by_source.get("fallback", 0)
            / max(int(diagnostics.get("matched_pairs", 0)), 1)
        )
        diagnostics["task"] = _task_diagnostics(
            original=graph,
            coarse=coarse,
            cumulative_assignment=next_cumulative_assignment,
        )
        if bool(diagnostics_cfg.get("enable_spectral", True)):
            progress_message(config, f"level {level}: spectral diagnostics start")
            start_spectral = perf_counter()
            spectral_num_signals = int(
                diagnostics_cfg.get("spectral_num_signals", min(Z.shape[1], 4))
            )
            spectral_input = Z[:, : max(1, min(spectral_num_signals, Z.shape[1]))].astype(
                np.float32,
                copy=False,
            )
            if root_spectral_input is None and cumulative_assignment is not None:
                root_spectral_input = spectral_input
                root_relation_weights = dict(relation_weights)
            diagnostics["spectral"] = compute_spectral_diagnostics(
                original=current,
                coarse=coarse,
                assignment=assignment,
                seed=int(config.get("seed", 12345)) + level,
                num_signals=int(spectral_input.shape[1]),
                smoothing_steps=int(diagnostics_cfg.get("spectral_smoothing_steps", 1)),
                relation_weights=relation_weights,
                Z=spectral_input,
                exact_eigenvalue_max_nodes=diagnostics_cfg.get(
                    "spectral_exact_eigenvalue_max_nodes",
                    256,
                ),
                baseline_methods=diagnostics_cfg.get(
                    "spectral_baselines",
                    ["random", "heavy_edge", "graphzoom_style", "convmatch_style"],
                ),
                baseline_max_nodes=diagnostics_cfg.get("spectral_baseline_max_nodes", 5000),
                relation_operator_mode=str(
                    config.get("fusion", {}).get("relation_operator_mode", "relationwise")
                ),
                relation_detail=bool(diagnostics_cfg.get("spectral_relation_detail", True)),
            )
            if (
                next_cumulative_assignment is not None
                and root_spectral_input is not None
                and root_relation_weights is not None
            ):
                cumulative_spectral_baselines = diagnostics_cfg.get("cumulative_spectral_baselines", [])
                diagnostics["cumulative_spectral"] = compute_spectral_diagnostics(
                    original=graph,
                    coarse=coarse,
                    assignment=Assignment(
                        assignment=next_cumulative_assignment.astype(np.int64, copy=False),
                        supernode_type=coarse.node_type.astype(np.int32, copy=False),
                    ),
                    seed=int(config.get("seed", 12345)) + level + 10_000,
                    num_signals=int(root_spectral_input.shape[1]),
                    smoothing_steps=int(diagnostics_cfg.get("spectral_smoothing_steps", 1)),
                    relation_weights=root_relation_weights,
                    Z=root_spectral_input,
                    exact_eigenvalue_max_nodes=diagnostics_cfg.get(
                        "cumulative_spectral_exact_eigenvalue_max_nodes",
                        0,
                    ),
                    baseline_methods=cumulative_spectral_baselines,
                    baseline_max_nodes=diagnostics_cfg.get("spectral_baseline_max_nodes", 5000),
                    baseline_target_ratio=float(config.get("coarsening", {}).get("target_ratio", 0.0)),
                    baseline_target_tolerance=float(
                        diagnostics_cfg.get(
                            "baseline_target_tolerance",
                            diagnostics_cfg.get("target_tolerance", 0.02),
                        )
                    ),
                    baseline_max_levels=int(
                        diagnostics_cfg.get(
                            "baseline_max_levels",
                            config.get("coarsening", {}).get("max_levels", 4),
                        )
                    ),
                    baseline_task_eval=bool(
                        diagnostics_cfg.get("cumulative_spectral_baseline_task_eval", False)
                    ),
                    baseline_task_eval_params=dict(
                        diagnostics_cfg.get("cumulative_spectral_baseline_task_eval_params", {})
                    ),
                    relation_operator_mode=str(
                        config.get("fusion", {}).get("relation_operator_mode", "relationwise")
                    ),
                    relation_detail=bool(diagnostics_cfg.get("spectral_relation_detail", True)),
                )
            runtime["spectral_diagnostics"] = perf_counter() - start_spectral
            diagnostics["runtime_by_stage"] = dict(runtime)
            progress_message(
                config,
                f"level {level}: spectral diagnostics done in "
                f"{runtime['spectral_diagnostics']:.2f}s",
            )
        save_graph(coarse, level_dir)
        _save_assignment(assignment, level_dir / "assignment.npz")
        if next_cumulative_assignment is not None:
            _save_assignment(
                Assignment(
                    assignment=next_cumulative_assignment.astype(np.int64, copy=False),
                    supernode_type=coarse.node_type.astype(np.int32, copy=False),
                ),
                level_dir / "cumulative_assignment.npz",
            )
        save_diagnostics(diagnostics, level_dir / "diagnostics.json")
        _save_checkpoint(
            level_dir,
            level=level,
            input_nodes=current.num_nodes,
            coarse_nodes=coarse.num_nodes,
            target_nodes=target_nodes,
            legacy_resume=legacy_resume,
        )
        results.append(LevelResult(level, coarse, assignment, diagnostics))
        progress_message(config, f"level {level}: saved {level_dir}")

        if coarse.num_nodes >= current.num_nodes:
            progress_message(
                config,
                f"level {level}: stop because node count did not decrease",
            )
            break
        current = coarse
        cumulative_assignment = next_cumulative_assignment

    return results
