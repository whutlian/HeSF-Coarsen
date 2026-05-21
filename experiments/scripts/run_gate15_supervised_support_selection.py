from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.audit_gate15_code import write_audit
from experiments.scripts.gate13_task_first_common import (
    add_task_and_optional_spectral,
    load_hgb_graph,
    run_multilevel_task_first,
    run_support_baseline,
)
from experiments.scripts.run_task_first_gate14_final import _run_a0_reference
from experiments.scripts.summarize_gate15_supervised_support_selection import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher


DATASETS = ("ACM", "DBLP", "IMDB")
SEEDS = (12345, 23456, 34567, 45678, 56789)
RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
NEW_METHODS = (
    "HeSF-SS-teacher-topk",
    "HeSF-SS-teacher-diverse-topk",
    "HeSF-SS-hybrid-teacher-response",
    "HeSF-SS-validation-greedy",
    "HeSF-SS-selection-background-condense",
)
GATE14_REFERENCES = (
    "HeSF-TC-best-Gate14-validation-selected",
    "HeSF-TC-coverage-v2",
    "HeSF-TC-stateful-v1",
)

_GATE14_REFERENCE_CACHE: dict[Path, list[dict[str, Any]]] = {}
BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
    "A0-current-all-type-coarse-transfer-reference",
)


def _selector_for_method(method: str) -> SupportSelectorConfig:
    if method == "HeSF-SS-teacher-topk":
        return SupportSelectorConfig(selector="teacher_topk", background_strategy="typed_background")
    if method == "HeSF-SS-teacher-diverse-topk":
        return SupportSelectorConfig(selector="teacher_diverse_topk", background_strategy="typed_background")
    if method == "HeSF-SS-hybrid-teacher-response":
        return SupportSelectorConfig(selector="hybrid_teacher_response", background_strategy="typed_background")
    if method == "HeSF-SS-validation-greedy":
        return SupportSelectorConfig(selector="validation_greedy", background_strategy="typed_background")
    if method == "HeSF-SS-selection-background-condense":
        return SupportSelectorConfig(selector="teacher_diverse_topk", background_strategy="typed_background")
    raise ValueError(f"unsupported Gate15 method: {method}")


def _gate14_reference_settings(method: str) -> tuple[str, str, str, str]:
    if method == "HeSF-TC-stateful-v1":
        return "HeSF-TC-stateful-v1", "stateful_signature", "combined", "unknown_blocks_known"
    if method == "HeSF-TC-coverage-v2":
        return "HeSF-TC-coverage-v2", "response_signature", "coverage_v2", "unknown_blocks_known"
    return "HeSF-TC-coverage-v2", "response_signature", "coverage_v2", "unknown_blocks_known"


def _read_gate14_reference_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path not in _GATE14_REFERENCE_CACHE:
        if not path.exists():
            _GATE14_REFERENCE_CACHE[path] = []
        else:
            with path.open("r", encoding="utf-8", newline="") as handle:
                _GATE14_REFERENCE_CACHE[path] = list(csv.DictReader(handle))
    return _GATE14_REFERENCE_CACHE[path]


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _external_gate14_reference_row(
    path: Path,
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
) -> dict[str, Any] | None:
    actual_method, _pair_delta, _coverage, _purity = _gate14_reference_settings(method)
    candidates = []
    for row in _read_gate14_reference_rows(path):
        if str(row.get("dataset")).upper() != str(dataset).upper():
            continue
        try:
            row_seed = int(float(row.get("seed", -1)))
        except (TypeError, ValueError):
            continue
        if row_seed != int(seed) or row.get("method") != actual_method or row.get("status", "success") != "success":
            continue
        row_ratio = _float_or_none(row.get("requested_support_ratio", row.get("ratio")))
        if row_ratio is None:
            continue
        candidates.append((abs(float(row_ratio) - float(ratio)), row_ratio, row))
    if not candidates:
        return None
    _gap, source_ratio, source = min(candidates, key=lambda item: (item[0], item[1]))
    out = {
        "dataset": str(dataset).upper(),
        "seed": int(seed),
        "method": method,
        "requested_support_ratio": float(ratio),
        "source_method": actual_method,
        "source_requested_support_ratio": float(source_ratio),
        "external_reference_source": str(path),
        "external_reference_policy": "nearest_gate14_output",
        "status": "success",
        "evaluator_status": "diagnostic_lite_only",
    }
    for key in (
        "realized_support_ratio",
        "realized_full_ratio",
        "selected_support_count",
        "selected_support_merges",
        "background_node_count",
        "dropped_support_count",
        "target_hit",
        "macro_f1",
        "micro_f1",
        "accuracy",
        "validation_macro_f1",
        "validation_accuracy",
        "macro_recovery_vs_full_graph",
        "accuracy_recovery_vs_full_graph",
        "total_coarsen_sec",
        "peak_rss_mb",
    ):
        value = source.get(key)
        if value in {"", None} and key in {"macro_f1", "micro_f1", "accuracy"}:
            value = source.get(f"task.{key}")
        if value in {"", None} and key == "validation_macro_f1":
            value = source.get("task.validation_macro_f1")
        if value in {"", None} and key == "validation_accuracy":
            value = source.get("task.validation_accuracy")
        if value not in {"", None}:
            out[key] = value
    return out


def _mask(nodes: np.ndarray, total: int) -> np.ndarray:
    out = np.zeros(int(total), dtype=bool)
    out[np.asarray(nodes, dtype=np.int64)] = True
    return out


def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
    skip_keys = {
        "coarse_graph",
        "assignment",
        "support_features",
        "importance",
        "selection",
        "graph_diagnostics",
        "task_metrics",
        "teacher_outputs",
    }
    return {key: value for key, value in row.items() if key not in skip_keys and not isinstance(value, (dict, list, np.ndarray))}


def _add_task_metrics(row: dict[str, Any], task: dict[str, Any]) -> None:
    row["macro_f1"] = task.get("macro_f1", 0.0)
    row["micro_f1"] = task.get("micro_f1", 0.0)
    row["accuracy"] = task.get("accuracy", 0.0)
    row["validation_macro_f1"] = task.get("validation_macro_f1", 0.0)
    row["validation_accuracy"] = task.get("validation_accuracy", 0.0)


def _full_graph_row(graph, dataset: str, seed: int, args: argparse.Namespace, train_nodes, val_nodes, test_nodes) -> dict[str, Any]:
    task = evaluate_hettree_task(
        graph,
        graph,
        np.arange(graph.num_nodes, dtype=np.int64),
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=infer_target_node_type(graph),
        official_split_nodes={"train": train_nodes, "val": val_nodes, "test": test_nodes},
    ).metrics
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "full-graph-hettree-lite-tuned",
        "requested_support_ratio": 1.0,
        "realized_support_ratio": 1.0,
        "realized_full_ratio": 1.0,
        "selected_support_count": int(np.sum(graph.node_type != infer_target_node_type(graph))),
        "background_node_count": 0,
        "dropped_support_count": 0,
        "target_hit": True,
        "status": "success" if not task.get("skipped", False) else "skipped",
        "evaluator_status": "diagnostic_lite_only",
    }
    _add_task_metrics(row, task)
    row["macro_recovery_vs_full_graph"] = 1.0
    row["accuracy_recovery_vs_full_graph"] = 1.0
    return row


def _run_group(args: argparse.Namespace, dataset: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    graph = load_hgb_graph(Path(args.data_root), dataset)
    labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
    target_type = infer_target_node_type(graph)
    train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(
        graph,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
    )
    train_mask = _mask(train_nodes, graph.num_nodes)
    val_mask = _mask(val_nodes, graph.num_nodes)
    test_mask = _mask(test_nodes, graph.num_nodes)
    teacher_dir = Path(args.output) / "teacher" / f"{dataset}_seed{seed}"
    teacher = train_full_graph_lite_teacher(
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        Gate15Config(target_node_type=int(target_type)).teacher,
        output_dir=teacher_dir,
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
    )
    rows: list[dict[str, Any]] = [_full_graph_row(graph, dataset, int(seed), args, train_nodes, val_nodes, test_nodes)]
    teacher_rows = [{"dataset": dataset, "seed": int(seed), **teacher["metrics"]}]
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for ratio in args.ratios:
        for method in args.methods:
            start = perf_counter()
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "requested_support_ratio": float(ratio),
                "status": "running",
                **split_protocol,
            }
            try:
                if method in NEW_METHODS:
                    cfg = Gate15Config(
                        target_node_type=int(target_type),
                        selector=replace(_selector_for_method(method), support_ratios=(float(ratio),)),
                    )
                    result = run_supervised_support_selection_pipeline(
                        graph,
                        labels,
                        train_mask,
                        val_mask,
                        test_mask,
                        cfg,
                        support_ratio=float(ratio),
                        teacher_outputs=teacher,
                        method_name=method,
                        seed=int(seed),
                        task_epochs=int(args.task_epochs),
                        task_hidden_dim=int(args.task_hidden_dim),
                        device=str(args.device),
                    )
                    row.update(_flat_row(result))
                    imp = result["importance"]
                    row["status"] = "success"
                    importance_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            **{key: float(np.mean(value)) for key, value in imp.get("components", {}).items()},
                            "importance_mean": row.get("support_importance_mean"),
                            "selected_importance_mean": row.get("selected_importance_mean"),
                        }
                    )
                    selection_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            **result["selection"]["diagnostics"],
                        }
                    )
                    graph_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            **{key: value for key, value in result["graph_diagnostics"].items() if not isinstance(value, dict)},
                        }
                    )
                elif method in BASELINES:
                    if method == "A0-current-all-type-coarse-transfer-reference":
                        coarse, assignment, diag = _run_a0_reference(graph, float(ratio), int(seed))
                    else:
                        coarse, assignment, diag = run_support_baseline(
                            graph,
                            baseline=method,
                            ratio=float(ratio),
                            seed=int(seed),
                            candidate_k=int(args.candidate_k),
                        )
                    row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
                    task = evaluate_hettree_task(
                        graph,
                        coarse,
                        np.asarray(assignment, dtype=np.int64),
                        seed=int(seed),
                        epochs=int(args.task_epochs),
                        hidden_dim=int(args.task_hidden_dim),
                        device=str(args.device),
                        target_node_type=int(target_type),
                        official_split_nodes={"train": train_nodes, "val": val_nodes, "test": test_nodes},
                    ).metrics
                    _add_task_metrics(row, task)
                    row["status"] = "success" if not task.get("skipped", False) else "skipped"
                    row["evaluator_status"] = "diagnostic_lite_only"
                elif method in GATE14_REFERENCES:
                    external = (
                        None
                        if bool(args.rerun_gate14_references)
                        else _external_gate14_reference_row(
                            Path(args.gate14_reference_runs),
                            dataset=dataset,
                            seed=int(seed),
                            method=method,
                            ratio=float(ratio),
                        )
                    )
                    if external is not None:
                        row.update(external)
                    else:
                        actual_method, pair_delta, coverage, purity = _gate14_reference_settings(method)
                        coarse, assignment, diag = run_multilevel_task_first(
                            graph,
                            method=actual_method,
                            ratio=float(ratio),
                            ratio_mode="support",
                            seed=int(seed),
                            max_levels=int(args.max_levels),
                            per_level_ratio=float(args.per_level_ratio),
                            candidate_k=int(args.candidate_k),
                            candidate_source="hybrid_task_aware",
                            pair_delta_mode=pair_delta,
                            coverage_mode=coverage,
                            purity_policy=purity,
                            candidate_pair_cap=int(args.candidate_pair_cap),
                        )
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
                        task = evaluate_hettree_task(
                            graph,
                            coarse,
                            np.asarray(assignment, dtype=np.int64),
                            seed=int(seed),
                            epochs=int(args.task_epochs),
                            hidden_dim=int(args.task_hidden_dim),
                            device=str(args.device),
                            target_node_type=int(target_type),
                            official_split_nodes={"train": train_nodes, "val": val_nodes, "test": test_nodes},
                        ).metrics
                        _add_task_metrics(row, task)
                        row["status"] = "success" if not task.get("skipped", False) else "skipped"
                        row["evaluator_status"] = "diagnostic_lite_only"
                else:
                    raise ValueError(f"unsupported method: {method}")
                ceiling = rows[0]
                row["macro_recovery_vs_full_graph"] = float(row.get("macro_f1", 0.0) or 0.0) / max(float(ceiling.get("macro_f1", 0.0) or 0.0), 1.0e-12)
                row["accuracy_recovery_vs_full_graph"] = float(row.get("accuracy", 0.0) or 0.0) / max(float(ceiling.get("accuracy", 0.0) or 0.0), 1.0e-12)
            except RuntimeError as exc:
                row["status"] = "oom_or_runtime_error" if "out of memory" in str(exc).lower() else "failed"
                row["error"] = str(exc)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
            row["wall_clock_sec"] = float(perf_counter() - start)
            rows.append(row)
    return {
        "rows": rows,
        "teacher_rows": teacher_rows,
        "importance_rows": importance_rows,
        "selection_rows": selection_rows,
        "graph_rows": graph_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate15 supervised support selection experiments.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gate15_supervised_support_selection_20260521"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(NEW_METHODS) + list(GATE14_REFERENCES) + list(BASELINES))
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--per-level-ratio", type=float, default=0.55)
    parser.add_argument("--candidate-pair-cap", type=int, default=20000)
    parser.add_argument("--gate14-reference-runs", type=Path, default=Path("outputs/exp_task_first_gate14_hgb_20260521/gate14_all_runs.csv"))
    parser.add_argument("--rerun-gate14-references", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit-groups", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    write_audit(args.output / "code_audit")
    groups = [(dataset, seed) for dataset in args.datasets for seed in args.seeds]
    if args.limit_groups is not None:
        groups = groups[: max(0, int(args.limit_groups))]
    all_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    def consume(payload: dict[str, list[dict[str, Any]]]) -> None:
        nonlocal all_rows, teacher_rows, importance_rows, selection_rows, graph_rows
        all_rows.extend(payload["rows"])
        teacher_rows.extend(payload["teacher_rows"])
        importance_rows.extend(payload["importance_rows"])
        selection_rows.extend(payload["selection_rows"])
        graph_rows.extend(payload["graph_rows"])
        write_csv(args.output / "runs" / "gate15_all_runs.csv", all_rows)
        write_csv(args.output / "teacher" / "full_graph_teacher_by_dataset_seed.csv", teacher_rows)
        write_csv(args.output / "selection" / "support_importance.csv", importance_rows)
        write_csv(args.output / "selection" / "support_selection_diagnostics.csv", selection_rows)
        write_csv(args.output / "graphs" / "compressed_graph_summary.csv", graph_rows)

    if int(args.jobs) <= 1:
        for dataset, seed in groups:
            consume(_run_group(args, str(dataset), int(seed)))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {pool.submit(_run_group, args, str(dataset), int(seed)): (dataset, seed) for dataset, seed in groups}
            for future in as_completed(futures):
                consume(future.result())
    write_csv(args.output / "teacher" / "full_graph_teacher_metrics.csv", teacher_rows)
    result = summarize(args.output)
    failures = [row for row in all_rows if row.get("status") not in {"success"}]
    write_json(
        args.output / "result.json",
        {"rows": len(all_rows), "success": len(all_rows) - len(failures), "failed": len(failures), **result},
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
