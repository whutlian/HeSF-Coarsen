from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.gate17_2_effective_budget import compute_effective_budget_fields
from experiments.scripts.run_gate17_1_support_sensitivity import (
    _full_graph_row,
    _semantic_row_for_graph,
    _target_only_empty_support_graph,
    _target_only_row,
    _teacher_consistency_rows,
)
from experiments.scripts.run_gate17_support_selection import (
    _flat_payload,
    _mask,
    _row_from_task,
    _run_fast_selection_method,
    _split_values,
)
from experiments.scripts.summarize_gate17_2 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline


GATE17_2_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
EXPLICIT_ALL_SEEDS = (12345, 23456, 34567, 45678, 56789)
DEFAULT_METHODS = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    "H6-no-spec-support-only",
    "random-support-only",
    "flatten-sum-support-only",
    "HeSF-SS-sensitivity-plus-prototype-no-free-raw",
    "HeSF-SS-dblp-aware-prototype-no-free-raw",
    "HeSF-SS-real-validation-no-fallback",
    "HeSF-SS-real-occlusion-no-fallback",
    "HeSF-SS-real-occlusion-plus-dblp-prototype-budgeted",
)
BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only", "random-support-only"}


def parse_dataset_seed_map(value: str | None) -> dict[str, int]:
    if value is None or str(value).strip() == "":
        return {}
    out: dict[str, int] = {}
    for token in str(value).replace(";", ",").split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"dataset-seed-map token must be DATASET:SEED, got {item!r}")
        dataset, seed = item.split(":", 1)
        dataset = dataset.strip()
        if dataset not in GATE17_2_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unknown Gate17.2 dataset in seed map: {dataset}")
        out[dataset] = int(seed.strip())
    return out


def resolve_dataset_seed_pairs(
    datasets: list[str],
    seed_policy: str,
    dataset_seed_map: str | None,
    explicit_seeds: list[int] | None = None,
) -> list[tuple[str, int]]:
    mapping = parse_dataset_seed_map(dataset_seed_map)
    pairs: list[tuple[str, int]] = []
    policy = str(seed_policy)
    for dataset in datasets:
        if dataset not in GATE17_2_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate17.2 dataset: {dataset}")
        if mapping:
            if dataset not in mapping:
                raise ValueError(f"missing seed for dataset {dataset} in dataset-seed-map")
            pairs.append((dataset, int(mapping[dataset])))
        elif policy == "best_single":
            pairs.append((dataset, int(GATE17_2_SINGLE_SEED_BY_DATASET[dataset])))
        elif policy in {"explicit", "explicit_all"}:
            seeds = explicit_seeds or list(EXPLICIT_ALL_SEEDS)
            pairs.extend((dataset, int(seed)) for seed in seeds)
        else:
            raise ValueError(f"unsupported seed policy: {policy}")
    return pairs


def _selector_for_gate17_2_method(method: str, args: argparse.Namespace) -> SupportSelectorConfig:
    common = {
        "candidate_pool_size": int(args.candidate_pool_size),
        "short_eval_epochs": int(args.short_eval_epochs),
        "max_validation_greedy_steps": int(args.max_validation_greedy_steps),
        "occlusion_candidate_pool_size": int(args.occlusion_candidate_pool_size),
        "occlusion_short_eval_epochs": int(args.occlusion_short_eval_epochs),
        "occlusion_short_patience": int(args.occlusion_short_patience),
        "max_members_per_prototype": int(args.max_members_per_prototype),
        "force_raw_bridge_nodes": False,
        "force_raw_keep_high_degree_bridges": False,
        "raw_bridge_mode": "off",
    }
    if method == "HeSF-SS-sensitivity-plus-prototype-no-free-raw":
        return SupportSelectorConfig(
            selector="sensitivity_block_selector",
            background_strategy="class_anchor_relation_prototype",
            **common,
        )
    if method == "HeSF-SS-dblp-aware-prototype-no-free-raw":
        return SupportSelectorConfig(
            selector="sensitivity_block_selector",
            background_strategy="dblp_aware_prototype",
            block_key_mode="dblp_aware",
            **common,
        )
    if method == "HeSF-SS-real-validation-no-fallback":
        return SupportSelectorConfig(
            selector="real_validation_block_greedy",
            background_strategy="class_anchor_relation_prototype",
            min_gain=1.0e-4,
            allow_proxy_fill=False,
            **common,
        )
    if method == "HeSF-SS-real-occlusion-no-fallback":
        return SupportSelectorConfig(
            selector="real_occlusion_block_selector",
            background_strategy="class_anchor_relation_prototype",
            allow_proxy_fill=False,
            **common,
        )
    if method == "HeSF-SS-real-occlusion-plus-dblp-prototype-budgeted":
        return SupportSelectorConfig(
            selector="occlusion_plus_dblp_prototype",
            background_strategy="dblp_aware_prototype",
            block_key_mode="dblp_aware",
            allow_proxy_fill=False,
            **common,
        )
    raise ValueError(f"unsupported Gate17.2 method: {method}")


def _effective_fields_for_row(
    row: dict[str, Any],
    graph_support_count: int,
    graph_diag: dict[str, Any] | None,
    semantic_row: dict[str, Any] | None,
) -> dict[str, Any]:
    method = str(row.get("method", ""))
    candidate_allclose = bool((semantic_row or {}).get("allclose_to_full", False)) if method.startswith("HeSF-SS") else False
    return compute_effective_budget_fields(
        original_support_nodes=int(graph_support_count),
        requested_support_ratio=float(row.get("requested_support_ratio", 0.0) or 0.0),
        selected_support_count=int(row.get("selected_support_count", row.get("realized_support_count", 0)) or 0),
        graph_diagnostics=graph_diag or row,
        candidate_allclose_to_full=candidate_allclose,
    )


def _merge_semantic_fields(row: dict[str, Any], semantic_row: dict[str, Any] | None) -> None:
    if not semantic_row:
        return
    for key in [
        "tree_tensor_l2_delta_vs_full",
        "tree_tensor_l1_delta_vs_full",
        "tree_tensor_cosine_delta_vs_full",
        "tree_tensor_linf_delta_vs_full",
        "target_path_feature_changed_fraction",
        "support_dependent_path_count",
        "nonself_path_count",
        "allclose_to_full",
        "allclose_to_target_only",
        "coarse_tree_hash",
        "full_tree_hash",
        "target_only_tree_hash",
    ]:
        row[key] = semantic_row.get(key, "")
    if str(row.get("method", "")).startswith("HeSF-SS"):
        row["candidate_allclose_to_full"] = bool(semantic_row.get("allclose_to_full", False))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    diag_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    effective_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    candidate_semantic_rows: list[dict[str, Any]] = []
    teacher_rows_all: list[dict[str, Any]] = []
    teacher_consistency: list[dict[str, Any]] = []

    args.run_mode = "gate17_2_single_seed_effective_budget"
    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
        target_type = infer_target_node_type(graph)
        support_count = int(np.sum(graph.node_type != int(target_type)))
        train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
            graph,
            labels,
            seed=int(seed),
            target_node_type=int(target_type),
        )
        split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
        train_mask = _mask(train_nodes, graph.num_nodes)
        val_mask = _mask(val_nodes, graph.num_nodes)
        test_mask = _mask(test_nodes, graph.num_nodes)
        target_only_graph, target_only_assignment = _target_only_empty_support_graph(graph, int(target_type))
        teacher = {
            "metrics": {
                "teacher_uses_test_labels_for_training": False,
                "teacher_reliable_for_importance": False,
                "logits_source": "disabled_gate17_2_runner",
            },
            "teacher_uses_test_labels_for_training": False,
        }
        teacher_consistency.extend(_teacher_consistency_rows(str(dataset), int(seed), []))
        teacher_rows_all.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "teacher_eval_source": "disabled_gate17_2_runner",
                "teacher_reliable_for_importance": False,
                "teacher_uses_test_labels_for_training": False,
            }
        )
        for ratio in args.ratios:
            for method in args.methods:
                start = perf_counter()
                row: dict[str, Any] = {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "method": str(method),
                    "requested_support_ratio": float(ratio),
                    "seed_policy": str(args.seed_policy),
                    **split_protocol,
                }
                coarse_for_delta = None
                assignment_for_delta: np.ndarray | None = None
                result: dict[str, Any] | None = None
                graph_diag: dict[str, Any] | None = None
                semantic_row: dict[str, Any] | None = None
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, str(dataset), int(seed), float(ratio), args, split))
                        coarse_for_delta = graph
                        assignment_for_delta = np.arange(graph.num_nodes, dtype=np.int64)
                        graph_diag = {"prototype_background_count": 0, "forced_raw_bridge_count": 0, "prototype_member_count_sum": 0}
                    elif method == "target-only-empty-support":
                        target_row, coarse_for_delta, assignment_for_delta = _target_only_row(
                            graph, str(dataset), int(seed), float(ratio), args, split
                        )
                        row.update(target_row)
                        graph_diag = {"prototype_background_count": 0, "forced_raw_bridge_count": 0, "prototype_member_count_sum": 0}
                    elif method in BASELINES:
                        coarse, assignment, diag = run_support_baseline(
                            graph,
                            baseline=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            candidate_k=int(args.candidate_k),
                        )
                        coarse_for_delta = coarse
                        assignment_for_delta = np.asarray(assignment, dtype=np.int64)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
                        final_support = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
                        row["selected_support_count"] = int(final_support)
                        row.update(
                            budget_diagnostics(
                                num_support=support_count,
                                support_ratio=float(ratio),
                                realized_support_count=final_support,
                            )
                        )
                        task = evaluate_hettree_task(
                            graph,
                            coarse,
                            assignment_for_delta,
                            seed=int(seed),
                            epochs=int(args.task_epochs),
                            hidden_dim=int(args.task_hidden_dim),
                            device=str(args.device),
                            target_node_type=int(target_type),
                            official_split_nodes=split,
                            primary_eval_mode=str(args.primary_eval_mode),
                            early_stopping=True,
                            monitor=str(args.monitor),
                            max_paths=int(args.max_paths),
                        ).metrics
                        _row_from_task(row, task)
                        row.setdefault("selector_uses_test_labels", False)
                        row.setdefault("teacher_uses_test_labels_for_training", False)
                        row.setdefault("validation_trial_count", 0)
                        row.setdefault("occlusion_trial_count", 0)
                        graph_diag = {"prototype_background_count": 0, "forced_raw_bridge_count": 0, "prototype_member_count_sum": 0}
                    else:
                        cfg = Gate15Config(
                            target_node_type=int(target_type),
                            selector=replace(_selector_for_gate17_2_method(str(method), args), support_ratios=(float(ratio),)),
                        )
                        if str(args.feature_mode) == "full":
                            result = run_supervised_support_selection_pipeline(
                                graph,
                                labels,
                                train_mask,
                                val_mask,
                                test_mask,
                                cfg,
                                support_ratio=float(ratio),
                                teacher_outputs=teacher,
                                method_name=str(method),
                                seed=int(seed),
                                task_epochs=int(args.task_epochs),
                                task_hidden_dim=int(args.task_hidden_dim),
                                task_max_paths=int(args.max_paths),
                                device=str(args.device),
                            )
                        else:
                            result = _run_fast_selection_method(
                                graph,
                                labels,
                                train_mask,
                                val_mask,
                                test_mask,
                                cfg,
                                str(method),
                                float(ratio),
                                int(seed),
                                args,
                            )
                        row.update(_flat_payload(result))
                        coarse_for_delta = result["coarse_graph"]
                        assignment_for_delta = np.asarray(result["assignment"].assignment, dtype=np.int64)
                        graph_diag = dict(result["graph_diagnostics"])
                        selection_rows.append(
                            {
                                "dataset": str(dataset),
                                "seed": int(seed),
                                "method": str(method),
                                "requested_support_ratio": float(ratio),
                                **result["selection"]["diagnostics"],
                            }
                        )
                        graph_rows.append(
                            {
                                "dataset": str(dataset),
                                "seed": int(seed),
                                "method": str(method),
                                "requested_support_ratio": float(ratio),
                                **{key: value for key, value in graph_diag.items() if not isinstance(value, dict)},
                            }
                        )
                        prototype_rows.append(
                            {
                                "dataset": str(dataset),
                                "seed": int(seed),
                                "method": str(method),
                                "requested_support_ratio": float(ratio),
                                **graph_diag,
                                "max_members_per_prototype": int(args.max_members_per_prototype),
                            }
                        )
                        for item in result["selection"].get("occlusion_block_scores", []):
                            occlusion_rows.append(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "method": str(method),
                                    "requested_support_ratio": float(ratio),
                                    **item,
                                }
                            )
                        for item in result["selection"].get("validation_greedy_trials", []):
                            validation_rows.append(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "method": str(method),
                                    "requested_support_ratio": float(ratio),
                                    **item,
                                }
                            )
                    if coarse_for_delta is not None and assignment_for_delta is not None:
                        semantic_row = _semantic_row_for_graph(
                            graph=graph,
                            coarse=coarse_for_delta,
                            assignment=assignment_for_delta,
                            target_only=target_only_graph,
                            target_only_assignment=target_only_assignment,
                            target_type=int(target_type),
                            dataset=str(dataset),
                            seed=int(seed),
                            method=str(method),
                            ratio=float(ratio),
                            args=args,
                        )
                        semantic_rows.append(semantic_row)
                        if str(method).startswith("HeSF-SS"):
                            candidate_semantic_rows.append(
                                {
                                    **semantic_row,
                                    "candidate_allclose_to_full": bool(semantic_row.get("allclose_to_full", False)),
                                }
                            )
                        _merge_semantic_fields(row, semantic_row)
                    row.update(_effective_fields_for_row(row, support_count, graph_diag, semantic_row))
                    row.setdefault("effective_budget_exact_match", row.get("support_budget_exact_match", False))
                    row.setdefault("max_members_per_prototype", int(args.max_members_per_prototype))
                    row.setdefault("status", "success")
                except RuntimeError as exc:
                    text = str(exc)
                    row["status"] = "oom_or_runtime_error" if "out of memory" in text.lower() else "failed"
                    row["error"] = text
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                row["wall_clock_sec"] = float(perf_counter() - start)
                row["run_mode"] = args.run_mode
                rows.append(row)
                if row.get("status") == "success":
                    effective_rows.append(
                        {
                            "dataset": row.get("dataset"),
                            "seed": row.get("seed"),
                            "method": row.get("method"),
                            "requested_support_ratio": row.get("requested_support_ratio"),
                            **{key: row.get(key) for key in row if key.startswith("effective_") or key.endswith("_leak_ratio")},
                            "original_support_nodes": row.get("original_support_nodes"),
                            "requested_support_count": row.get("requested_support_count"),
                            "selected_budget_support_count": row.get("selected_budget_support_count"),
                            "forced_raw_support_count": row.get("forced_raw_support_count"),
                            "prototype_background_count": row.get("prototype_background_count"),
                            "prototype_member_count_sum": row.get("prototype_member_count_sum"),
                            "represented_support_context_count": row.get("represented_support_context_count"),
                            "represented_support_context_ratio": row.get("represented_support_context_ratio"),
                            "candidate_allclose_to_full": row.get("candidate_allclose_to_full", False),
                        }
                    )
                write_csv(output_dir / "gate17_2_raw_rows.csv", rows)

    write_csv(diag_dir / "effective_budget.csv", effective_rows)
    write_csv(diag_dir / "candidate_semantic_delta.csv", candidate_semantic_rows)
    write_csv(diag_dir / "semantic_tree_delta.csv", semantic_rows)
    write_csv(diag_dir / "validation_feedback_trials.csv", validation_rows)
    write_csv(diag_dir / "occlusion_feedback_scores.csv", occlusion_rows)
    write_csv(diag_dir / "prototype_budget_saturation.csv", prototype_rows)
    write_csv(diag_dir / "compressed_graph_summary.csv", graph_rows)
    write_csv(diag_dir / "support_selection_diagnostics.csv", selection_rows)
    write_csv(diag_dir / "full_graph_teacher_by_dataset_seed.csv", teacher_rows_all)
    write_csv(diag_dir / "teacher_metric_consistency.csv", teacher_consistency)
    result = summarize(output_dir, output_dir, diag_dir)
    failed = [row for row in rows if str(row.get("status", "success")) != "success"]
    if any(row.get("status") == "oom_or_runtime_error" for row in failed):
        result["local_oom_or_runtime_error"] = True
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate17.2 single-seed effective-budget feedback gate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate17_2_single_seed"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seed-policy", choices=["best_single", "explicit", "explicit_all"], default="best_single")
    parser.add_argument("--dataset-seed-map", default="")
    parser.add_argument("--seeds", nargs="*", default=[])
    parser.add_argument("--support-ratios", "--ratios", nargs="*", default=[0.30, 0.70])
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-eval-epochs", type=int, default=3)
    parser.add_argument("--occlusion-short-patience", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--occlusion-candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-validation-greedy-steps", type=int, default=3)
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--feature-mode", choices=["fast", "full", "support_sensitive_fast"], default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-members-per-prototype", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.datasets = _split_values(args.datasets, str) or ["ACM", "DBLP", "IMDB"]
    args.seeds = _split_values(args.seeds, int)
    args.ratios = _split_values(args.support_ratios, float) or [0.30, 0.70]
    args.methods = _split_values(args.methods, str) or list(DEFAULT_METHODS)
    if str(args.feature_mode) == "support_sensitive_fast":
        args.feature_mode = "fast"
    args.dataset_seed_pairs = resolve_dataset_seed_pairs(
        args.datasets,
        str(args.seed_policy),
        str(args.dataset_seed_map),
        [int(seed) for seed in args.seeds],
    )
    result = run(args)
    return 3 if bool(result.get("local_oom_or_runtime_error", False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
