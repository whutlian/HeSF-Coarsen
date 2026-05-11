from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.diagnostics import compute_diagnostics, save_diagnostics
from hesf_coarsen.io.edge_list import save_graph
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.matching.greedy import run_greedy_matching
from hesf_coarsen.partition.type_partition import default_partition
from hesf_coarsen.progress import progress_message
from hesf_coarsen.scoring.conv_response import compute_conv_response_sketch
from hesf_coarsen.scoring.merge_cost import score_candidate_pairs
from hesf_coarsen.scoring.relation_profile import compute_relation_profiles
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch
from hesf_coarsen.sketch.simhash import compute_simhash_buckets


@dataclass
class LevelResult:
    level: int
    graph: HeteroGraph
    assignment: Assignment
    diagnostics: dict


def _global_feature_matrix(graph: HeteroGraph) -> np.ndarray | None:
    if graph.features is None:
        return None
    width = max(feature.shape[1] for feature in graph.features.values())
    X = np.zeros((graph.num_nodes, width), dtype=np.float32)
    for type_id, feature in graph.features.items():
        X[nodes_of_type(graph, type_id), : feature.shape[1]] = feature
    return X


def _add_fallback_candidates(
    graph: HeteroGraph,
    partition_id: np.ndarray,
    store: BoundedCandidateStore | ArrayCandidateStore,
    config: dict,
) -> None:
    same_partition = bool(config.get("coarsening", {}).get("same_partition_only", True))
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = nodes_of_type(graph, int(type_id))
        if same_partition:
            keys = sorted(np.unique(partition_id[nodes]))
            groups = [nodes[partition_id[nodes] == key] for key in keys]
        else:
            groups = [nodes]
        for group in groups:
            for left, right in zip(group[::2], group[1::2]):
                store.add(int(left), int(right), 1e6, "fallback")


def _config_for_level(config: dict, num_nodes: int) -> dict:
    level_config = deepcopy(config)
    per_level_ratio = float(config.get("coarsening", {}).get("per_level_ratio", 0.55))
    if per_level_ratio > 0.0:
        max_pairs = max(1, int(num_nodes * max(0.0, 1.0 - per_level_ratio)))
        level_config.setdefault("coarsening", {})["max_matched_pairs"] = max_pairs
    return level_config


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


def run_multilevel_coarsening(graph: HeteroGraph, config: dict) -> list[LevelResult]:
    current = graph
    original_nodes = graph.num_nodes
    target_nodes = max(1, int(np.ceil(original_nodes * float(config["coarsening"]["target_ratio"]))))
    max_levels = int(config["coarsening"]["max_levels"])
    output_dir = Path(config.get("output", {}).get("dir", "outputs/default_run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[LevelResult] = []

    for level in range(1, max_levels + 1):
        if current.num_nodes <= target_nodes:
            break
        runtime: dict[str, float] = {}
        progress_message(
            config,
            f"level {level}: start ({current.num_nodes} nodes, target {target_nodes})",
        )

        progress_message(config, f"level {level}: sketch start")
        start = perf_counter()
        Z = compute_lowpass_sketch(current, config)
        runtime["sketch"] = perf_counter() - start
        progress_message(config, f"level {level}: sketch done in {runtime['sketch']:.2f}s")

        progress_message(config, f"level {level}: candidates start")
        start = perf_counter()
        partition_id = default_partition(current)
        candidate_cfg = config.get("candidates", {})
        store = _make_candidate_store(current, config, level)
        use_chunked = bool(candidate_cfg.get("use_chunked_generation", False))
        if config["candidates"].get("enable_onehop", True):
            if use_chunked:
                generate_onehop_candidates_chunked(
                    current,
                    Z,
                    partition_id,
                    config,
                    store,
                    edge_chunk_size=int(candidate_cfg.get("edge_chunk_size", 1_000_000)),
                )
            else:
                generate_onehop_candidates(current, Z, partition_id, config, store)
        if config["candidates"].get("enable_capped_twohop", True):
            if use_chunked:
                twohop_config = config
                incident_index_mmap_dir = candidate_cfg.get("incident_index_mmap_dir")
                if incident_index_mmap_dir is not None:
                    twohop_config = deepcopy(config)
                    twohop_config.setdefault("candidates", {})["incident_index_mmap_dir"] = str(
                        Path(incident_index_mmap_dir) / f"level_{level}"
                    )
                generate_capped_twohop_candidates_chunked(
                    current,
                    Z,
                    partition_id,
                    twohop_config,
                    store,
                    middle_chunk_size=int(candidate_cfg.get("middle_chunk_size", 100_000)),
                    edge_chunk_size=int(candidate_cfg.get("edge_chunk_size", 1_000_000)),
                )
            else:
                generate_capped_twohop_candidates(current, Z, partition_id, config, store)
        if config["candidates"].get("enable_bucket", True):
            buckets = compute_simhash_buckets(
                Z,
                current.node_type,
                partition_id,
                bits=int(config["candidates"].get("simhash_bits", 16)),
                seed=int(config.get("seed", 12345)) + level,
            )
            if use_chunked:
                generate_bucket_candidates_chunked(
                    buckets,
                    current.node_type,
                    partition_id,
                    config,
                    store,
                    node_chunk_size=int(candidate_cfg.get("node_chunk_size", 1_000_000)),
                )
            else:
                generate_bucket_candidates(buckets, current.node_type, partition_id, config, store)
        if config["candidates"].get("enable_partition_ann", False):
            generate_partition_ann_candidates(current, Z, partition_id, config, store)
        _add_fallback_candidates(current, partition_id, store, config)
        _flush_candidate_store(store)
        pairs = store.to_pairs()
        candidate_counts = store.counts()
        source_counts = store.source_counts()
        runtime["candidates"] = perf_counter() - start
        progress_message(
            config,
            f"level {level}: candidates done in {runtime['candidates']:.2f}s "
            f"({pairs.shape[0]} pairs)",
        )

        progress_message(config, f"level {level}: scoring start")
        start = perf_counter()
        relation_profiles = compute_relation_profiles(current)
        X = _global_feature_matrix(current)
        H = Z.astype(np.float32) if X is None else np.concatenate([Z.astype(np.float32), X], axis=1)
        conv = compute_conv_response_sketch(current, H, None)
        scored = score_candidate_pairs(
            current,
            pairs,
            Z,
            relation_profiles,
            conv,
            current.features,
            config,
            partition_id=partition_id,
        )
        runtime["scoring"] = perf_counter() - start
        progress_message(
            config,
            f"level {level}: scoring done in {runtime['scoring']:.2f}s ({scored.shape[0]} pairs)",
        )

        progress_message(config, f"level {level}: matching and aggregation start")
        start = perf_counter()
        assignment = run_greedy_matching(
            current,
            scored,
            _config_for_level(config, current.num_nodes),
            partition_id=partition_id,
        )
        coarse = coarsen_graph(current, assignment)
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
                }.items()
                if path is not None
            },
        )
        level_dir = output_dir / f"level_{level}"
        save_graph(coarse, level_dir)
        save_diagnostics(diagnostics, level_dir / "diagnostics.json")
        results.append(LevelResult(level, coarse, assignment, diagnostics))
        progress_message(config, f"level {level}: saved {level_dir}")

        if coarse.num_nodes >= current.num_nodes:
            progress_message(
                config,
                f"level {level}: stop because node count did not decrease",
            )
            break
        current = coarse

    return results
