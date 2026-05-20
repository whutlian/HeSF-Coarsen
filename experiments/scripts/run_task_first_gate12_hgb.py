from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, write_csv, write_json
from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.candidates.bucket import generate_bucket_candidates_chunked
from hesf_coarsen.candidates.capped_twohop import generate_capped_twohop_candidates_chunked
from hesf_coarsen.candidates.onehop import generate_onehop_candidates_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.edge_list import load_graph, save_graph
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.partition.type_partition import default_partition
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch
from hesf_coarsen.sketch.simhash import compute_simhash_buckets
from hesf_coarsen.task_first.config import TaskFirstConfig, TaskFirstScoringConfig
from hesf_coarsen.task_first.pipeline import (
    build_support_only_task_first_coarsening,
    task_first_support_merge_budget,
)


DATASETS = {
    "ACM": "acm_hesf",
    "DBLP": "dblp_hesf",
    "IMDB": "imdb_hesf",
}

DEFAULT_METHODS = ("HeSF-TC-P", "HeSF-TC-S", "HeSF-TC-no-rel")
DEFAULT_RATIOS = (0.012, 0.024, 0.048, 0.096)
DEFAULT_SEEDS = (12345, 23456, 34567, 45678, 56789)


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def _method_token(method: str) -> str:
    return method.lower().replace("-", "_").replace(" ", "_")


def _ratio_token(ratio: float) -> str:
    return f"{float(ratio):.4f}".replace(".", "p").rstrip("0").rstrip("p")


def _candidate_config(seed: int, candidate_k: int, twohop_budget_per_node: int, twohop_time_budget_sec: float) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["seed"] = int(seed)
    cfg.setdefault("progress", {})["enabled"] = False
    cfg["sketch"] = dict(
        cfg["sketch"],
        dim=16,
        order=5,
        dtype="float16",
        row_normalize=True,
    )
    cfg["candidates"] = dict(
        cfg["candidates"],
        total_budget_K=int(candidate_k),
        twohop_budget_K2=max(1, int(candidate_k) // 2),
        bucket_pair_cap=max(4, int(candidate_k)),
        per_middle_pair_cap=max(4, int(candidate_k)),
        twohop_mode="capped_sampled",
        twohop_budget_per_node=max(1, int(twohop_budget_per_node)),
        twohop_max_time_budget_sec=float(twohop_time_budget_sec),
        middle_degree_cap_policy="p95",
        enable_onehop=True,
        enable_capped_twohop=True,
        enable_bucket=True,
        enable_fallback=False,
    )
    return cfg


def _build_candidates(
    graph: HeteroGraph,
    *,
    seed: int,
    candidate_k: int,
    candidate_source: str,
    twohop_budget_per_node: int,
    twohop_time_budget_sec: float,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    if str(candidate_source) == "random_support":
        return _build_random_support_candidates(graph, seed=seed, candidate_k=candidate_k)
    config = _candidate_config(seed, candidate_k, twohop_budget_per_node, twohop_time_budget_sec)
    partition = default_partition(graph)
    start = perf_counter()
    sketch = compute_lowpass_sketch(graph, config)
    sketch_sec = float(perf_counter() - start)
    store = ArrayCandidateStore(
        graph.node_type,
        K=int(candidate_k),
        same_type_only=True,
    )
    candidate_diag: dict[str, Any] = {"sketch_sec": sketch_sec}
    start = perf_counter()
    candidate_diag["onehop"] = generate_onehop_candidates_chunked(
        graph,
        sketch,
        partition,
        config,
        store,
        edge_chunk_size=int(config["candidates"]["edge_chunk_size"]),
    )
    candidate_diag["onehop_sec"] = float(perf_counter() - start)
    start = perf_counter()
    candidate_diag["twohop"] = generate_capped_twohop_candidates_chunked(
        graph,
        sketch,
        partition,
        config,
        store,
        middle_chunk_size=int(config["candidates"]["middle_chunk_size"]),
        edge_chunk_size=int(config["candidates"]["edge_chunk_size"]),
    )
    candidate_diag["twohop_sec"] = float(perf_counter() - start)
    start = perf_counter()
    buckets = compute_simhash_buckets(
        sketch,
        graph.node_type,
        partition,
        bits=int(config["candidates"]["simhash_bits"]),
        seed=int(seed),
    )
    candidate_diag["bucket"] = generate_bucket_candidates_chunked(
        buckets,
        graph.node_type,
        partition,
        config,
        store,
        node_chunk_size=int(config["candidates"]["node_chunk_size"]),
    )
    candidate_diag["bucket_sec"] = float(perf_counter() - start)
    candidate_diag["pair_count"] = int(store.pair_count())
    candidate_diag["source_counts"] = store.source_counts()
    return store, candidate_diag


def _build_random_support_candidates(
    graph: HeteroGraph,
    *,
    seed: int,
    candidate_k: int,
) -> tuple[ArrayCandidateStore, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    store = ArrayCandidateStore(
        graph.node_type,
        K=int(candidate_k),
        same_type_only=True,
    )
    start = perf_counter()
    emitted = 0
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if len(nodes) < 2:
            continue
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        span = min(max(1, int(candidate_k)), max(1, len(shuffled) - 1))
        for offset in range(1, span + 1):
            left = shuffled[:-offset]
            right = shuffled[offset:]
            scores = rng.random(len(left), dtype=np.float32)
            store.add_many(left, right, scores, "random_support")
            emitted += int(len(left))
    return store, {
        "candidate_source": "random_support",
        "random_support_sec": float(perf_counter() - start),
        "random_support_pairs_considered": int(emitted),
        "pair_count": int(store.pair_count()),
        "source_counts": store.source_counts(),
    }


def _task_first_config(method: str, *, target_type: int, ratio: float, ratio_mode: str, pair_delta_mode: str) -> TaskFirstConfig:
    scoring = TaskFirstScoringConfig(pair_delta_mode=pair_delta_mode)
    if method == "HeSF-TC-S":
        scoring = replace(
            scoring,
            lambda_target_spec=2.0,
            lambda_rel_response=1.0,
            lambda_support_coverage=1.0,
            lambda_support_purity=1.0,
            lambda_feat=0.1,
        )
    elif method == "HeSF-TC-no-rel":
        scoring = replace(scoring, lambda_rel_response=0.0)
    elif method != "HeSF-TC-P":
        raise ValueError(f"unsupported task-first method: {method}")
    kwargs: dict[str, Any] = {"target_ratio": float(ratio)} if ratio_mode == "full" else {"support_ratio": float(ratio)}
    return TaskFirstConfig(
        target_node_type=int(target_type),
        scoring=scoring,
        **kwargs,
    )


def _compose(cumulative: np.ndarray, assignment: Assignment) -> np.ndarray:
    return assignment.assignment[np.asarray(cumulative, dtype=np.int64)]


def _train_mask_for_current(
    current: HeteroGraph,
    original_train_nodes: np.ndarray,
    cumulative: np.ndarray,
) -> np.ndarray:
    train_mask = np.zeros(current.num_nodes, dtype=bool)
    mapped = np.asarray(cumulative, dtype=np.int64)[np.asarray(original_train_nodes, dtype=np.int64)]
    mapped = mapped[(mapped >= 0) & (mapped < current.num_nodes)]
    train_mask[np.unique(mapped)] = True
    return train_mask


def _run_multilevel_task_first(
    original: HeteroGraph,
    *,
    method: str,
    ratio: float,
    ratio_mode: str,
    seed: int,
    max_levels: int,
    per_level_ratio: float,
    candidate_k: int,
    candidate_source: str,
    twohop_budget_per_node: int,
    twohop_time_budget_sec: float,
    pair_delta_mode: str,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    target_type = infer_target_node_type(original)
    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
        original,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
    )
    original_support = int(np.sum(original.node_type != int(target_type)))
    if ratio_mode == "support":
        desired_final_support = max(0, int(np.ceil(original_support * float(ratio) - 1.0e-12)))
    else:
        requested_total = int(np.ceil(original.num_nodes * float(ratio) - 1.0e-12))
        target_count = int(np.sum(original.node_type == int(target_type)))
        desired_final_support = max(0, requested_total - target_count)
    current = original
    cumulative = np.arange(original.num_nodes, dtype=np.int64)
    levels: list[dict[str, Any]] = []
    peak_rss = _rss_mb()
    start_total = perf_counter()
    for level in range(1, int(max_levels) + 1):
        current_support = int(np.sum(current.node_type != int(target_type)))
        if current_support <= desired_final_support:
            break
        next_support = max(
            desired_final_support,
            int(np.ceil(current_support * float(per_level_ratio) - 1.0e-12)),
        )
        max_support_merges = max(0, current_support - next_support)
        if max_support_merges <= 0:
            break
        candidate_start = perf_counter()
        store, candidate_diag = _build_candidates(
            current,
            seed=int(seed) + level * 1009,
            candidate_k=int(candidate_k),
            candidate_source=str(candidate_source),
            twohop_budget_per_node=int(twohop_budget_per_node),
            twohop_time_budget_sec=float(twohop_time_budget_sec),
        )
        train_mask = _train_mask_for_current(current, train_nodes, cumulative)
        cfg = _task_first_config(
            method,
            target_type=int(target_type),
            ratio=float(ratio),
            ratio_mode=ratio_mode,
            pair_delta_mode=pair_delta_mode,
        )
        cfg = replace(cfg, max_support_merges=int(max_support_merges), target_ratio=None, support_ratio=None)
        coarsen_start = perf_counter()
        result = build_support_only_task_first_coarsening(
            current,
            store,
            np.asarray(current.labels if current.labels is not None else np.full(current.num_nodes, -1)),
            train_mask,
            cfg,
        )
        selected = int(result.diagnostics.get("selected_support_merges", 0))
        levels.append(
            {
                "level": int(level),
                "input_nodes": int(current.num_nodes),
                "input_support_nodes": int(current_support),
                "desired_next_support_nodes": int(next_support),
                "candidate_total_sec": float(perf_counter() - candidate_start),
                "coarsen_sec": float(perf_counter() - coarsen_start),
                "candidate_pair_count": int(candidate_diag.get("pair_count", 0)),
                **{f"candidate_{key}": value for key, value in candidate_diag.items() if not isinstance(value, dict)},
                **result.diagnostics,
                "output_nodes": int(result.graph.num_nodes),
                "output_support_nodes": int(np.sum(result.graph.node_type != int(target_type))),
            }
        )
        current = result.graph
        cumulative = _compose(cumulative, result.assignment)
        peak_rss = max(peak_rss, _rss_mb())
        if selected <= 0:
            break
    total_sec = float(perf_counter() - start_total)
    target_nodes = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    final_target_supernodes = cumulative[target_nodes]
    target_hit = bool(len(np.unique(final_target_supernodes)) == len(target_nodes))
    diagnostics = {
        "target_node_type": int(target_type),
        "train_nodes": int(len(train_nodes)),
        "val_nodes": int(len(val_nodes)),
        "test_nodes": int(len(test_nodes)),
        **split_protocol,
        "requested_ratio": float(ratio),
        "ratio_mode": ratio_mode,
        "original_nodes": int(original.num_nodes),
        "original_support_nodes": int(original_support),
        "desired_final_support_nodes": int(desired_final_support),
        "final_nodes": int(current.num_nodes),
        "final_support_nodes": int(np.sum(current.node_type != int(target_type))),
        "realized_full_ratio": float(current.num_nodes / max(original.num_nodes, 1)),
        "realized_support_ratio": float(np.sum(current.node_type != int(target_type)) / max(original_support, 1)),
        "target_hit": target_hit,
        "levels": levels,
        "num_levels": int(len(levels)),
        "total_coarsen_sec": total_sec,
        "peak_rss_mb": peak_rss,
    }
    if levels:
        for key in (
            "target_spec_error",
            "relation_response_error",
            "support_coverage_error",
            "support_purity_error",
            "num_support_candidates_scored",
            "num_support_candidates_rejected_by_purity",
            "num_support_candidates_rejected_by_constraints",
            "selected_support_merges",
        ):
            values = [float(level.get(key, 0.0) or 0.0) for level in levels]
            diagnostics[f"{key}_last"] = values[-1]
            diagnostics[f"{key}_mean"] = float(np.mean(values))
    return current, cumulative, diagnostics


def _flatten(prefix: str, payload: Mapping[str, Any], row: dict[str, Any]) -> None:
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            _flatten(name + ".", value, row)
        elif isinstance(value, (list, tuple)):
            row[name] = json.dumps(value, sort_keys=True)
        else:
            row[name] = value


def run_one_combo(args: argparse.Namespace, dataset: str, method: str, ratio: float, seed: int) -> dict[str, Any]:
    graph_dir = Path(args.data_root) / DATASETS[str(dataset).upper()]
    original = load_graph(graph_dir)
    run_name = f"task_first_gate12_{dataset.lower()}_{_method_token(method)}_{args.ratio_mode}_r{_ratio_token(ratio)}_seed{int(seed)}"
    run_dir = Path(args.output) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dataset": str(dataset).upper(),
        "method": str(method),
        "ratio": float(ratio),
        "ratio_mode": str(args.ratio_mode),
        "seed": int(seed),
        "git_commit": git_commit_hash(),
        "status": "running",
    }
    try:
        coarse, assignment, diag = _run_multilevel_task_first(
            original,
            method=str(method),
            ratio=float(ratio),
            ratio_mode=str(args.ratio_mode),
            seed=int(seed),
            max_levels=int(args.max_levels),
            per_level_ratio=float(args.per_level_ratio),
            candidate_k=int(args.candidate_k),
            candidate_source=str(args.candidate_source),
            twohop_budget_per_node=int(args.twohop_budget_per_node),
            twohop_time_budget_sec=float(args.twohop_time_budget_sec),
            pair_delta_mode=str(args.pair_delta_mode),
        )
        save_graph(coarse, run_dir / "final_graph")
        np.savez_compressed(run_dir / "cumulative_assignment.npz", assignment=assignment)
        write_json(run_dir / "coarsening_diagnostics.json", diag)
        spectral = compute_spectral_diagnostics(
            original,
            coarse,
            Assignment(assignment, coarse.node_type.astype(np.int32, copy=False)),
            seed=int(seed),
            num_signals=int(args.spectral_signals),
            smoothing_steps=1,
            relation_detail=False,
        )
        write_json(run_dir / "spectral_diagnostics.json", spectral)
        task = evaluate_hettree_task(
            original,
            coarse,
            assignment,
            seed=int(seed),
            epochs=int(args.task_epochs),
            hidden_dim=int(args.task_hidden_dim),
            device=str(args.device),
        ).metrics
        task["eval_protocol_warning"] = "diagnostic_hettree_lite_not_real_full_target_claim"
        write_json(run_dir / "task_eval_hettree_lite.json", task)
        row["status"] = "success"
        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
        _flatten("spectral.", spectral, row)
        _flatten("task.", task, row)
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
        write_json(run_dir / "error.json", row)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        write_json(run_dir / "error.json", row)
    write_json(run_dir / "summary.json", row)
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("method")),
            str(row.get("ratio_mode")),
            float(row.get("ratio", 0.0)),
        )
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    metrics = (
        "realized_full_ratio",
        "realized_support_ratio",
        "target_spec_error_last",
        "relation_response_error_last",
        "support_coverage_error_last",
        "support_purity_error_last",
        "spectral.dirichlet_energy_relative_error",
        "spectral.fused_sketch_energy_relative_error",
        "spectral.sketch_inner_product_relative_error",
        "task.macro_f1",
        "task.micro_f1",
        "task.accuracy",
        "total_coarsen_sec",
        "peak_rss_mb",
    )
    for (dataset, method, ratio_mode, ratio), group in sorted(groups.items()):
        out: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "ratio_mode": ratio_mode,
            "ratio": ratio,
            "seeds": len(group),
        }
        for metric in metrics:
            values = []
            for row in group:
                value = row.get(metric)
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HeSF-TC Gate1/Gate2 HGB validation.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--ratio-mode", choices=["support", "full"], default="support")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_task_first_gate12_hgb_20260520"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--per-level-ratio", type=float, default=0.55)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--candidate-source", choices=["random_support", "sketch"], default="random_support")
    parser.add_argument("--twohop-budget-per-node", type=int, default=2)
    parser.add_argument("--twohop-time-budget-sec", type=float, default=2.0)
    parser.add_argument("--pair-delta-mode", choices=["exact", "local_surrogate"], default="local_surrogate")
    parser.add_argument("--task-epochs", type=int, default=20)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spectral-signals", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [
        (dataset, method, ratio, seed)
        for dataset in args.datasets
        for method in args.methods
        for ratio in args.ratios
        for seed in args.seeds
    ]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    write_json(
        args.output / "manifest.json",
        {
            "datasets": list(args.datasets),
            "methods": list(args.methods),
            "ratios": [float(value) for value in args.ratios],
            "ratio_mode": args.ratio_mode,
            "seeds": [int(value) for value in args.seeds],
            "jobs": int(args.jobs),
            "pair_delta_mode": args.pair_delta_mode,
            "git_commit": git_commit_hash(),
            "combos": len(combos),
        },
    )
    rows: list[dict[str, Any]] = []
    if int(args.jobs) <= 1:
        for combo in combos:
            rows.append(run_one_combo(args, *combo))
            write_csv(args.output / "gate12_runs.csv", rows)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {pool.submit(run_one_combo, args, *combo): combo for combo in combos}
            for future in as_completed(futures):
                rows.append(future.result())
                write_csv(args.output / "gate12_runs.csv", rows)
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0.0)), int(row.get("seed", 0))))
    write_csv(args.output / "gate12_runs.csv", rows)
    summary = _aggregate(rows)
    write_csv(args.output / "gate12_summary.csv", summary)
    failures = [row for row in rows if row.get("status") != "success"]
    write_csv(args.output / "gate12_failures.csv", failures)
    write_json(
        args.output / "result.json",
        {
            "rows": len(rows),
            "success": int(sum(1 for row in rows if row.get("status") == "success")),
            "failed": len(failures),
            "output": str(args.output),
            "summary_rows": len(summary),
        },
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
