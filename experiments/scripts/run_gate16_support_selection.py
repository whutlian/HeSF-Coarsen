from __future__ import annotations

import argparse
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
from experiments.scripts.audit_gate16_code import write_audit
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.summarize_gate16 import summarize
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.budget import budget_diagnostics
from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher


DATASETS = ("ACM", "DBLP", "IMDB")
SEEDS = (12345, 23456, 34567, 45678, 56789)
RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
METHODS = (
    "full-graph-hettree-lite-tuned",
    *BASELINES,
    "HeSF-SS-teacher-topk",
    "HeSF-SS-teacher-diverse-topk",
    "HeSF-SS-validation-proxy-diverse",
    "HeSF-SS-true-validation-block-greedy",
    "HeSF-SS-sensitivity-block-selector",
    "HeSF-SS-prototype-residual-condense",
    "HeSF-SS-sensitivity-plus-prototype",
    "HeSF-SS-hybrid-teacher-response",
)


def _mask(nodes: np.ndarray, total: int) -> np.ndarray:
    out = np.zeros(int(total), dtype=bool)
    out[np.asarray(nodes, dtype=np.int64)] = True
    return out


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row_from_task(row: dict[str, Any], task: dict[str, Any]) -> None:
    row.update(
        {
            "primary_eval_mode": task.get("primary_eval_mode", "compressed_projected"),
            "primary_task_metric_name": task.get("primary_task_metric_name", "projected_original_macro_f1"),
            "macro_f1": _metric(task, "macro_f1"),
            "micro_f1": _metric(task, "micro_f1"),
            "accuracy": _metric(task, "accuracy"),
            "validation_macro_f1": _metric(task, "validation_macro_f1"),
            "validation_accuracy": _metric(task, "validation_accuracy"),
            "projected_macro_f1": _metric(task, "projected_original_macro_f1"),
            "transfer_macro_f1": _metric(task, "transfer_original_macro_f1"),
            "projected_accuracy": _metric(task, "projected_original_accuracy"),
            "transfer_accuracy": _metric(task, "transfer_original_accuracy"),
            "hybrid_target_macro_f1": _metric(task, "hybrid_target_original_macro_f1"),
            "hybrid_target_accuracy": _metric(task, "hybrid_target_original_accuracy"),
            "projected_vs_transfer_macro_gap": _metric(task, "projected_vs_transfer_macro_gap"),
            "projected_vs_transfer_accuracy_gap": _metric(task, "projected_vs_transfer_accuracy_gap"),
            "best_epoch": int(task.get("best_epoch", -1) or -1),
            "early_stopped": bool(task.get("early_stopped", False)),
            "status": "success" if not task.get("skipped", False) else "skipped",
            "skip_reason": task.get("skip_reason", ""),
            "evaluator_status": "diagnostic_lite_only",
        }
    )


def _selector_for_method(method: str) -> SupportSelectorConfig:
    if method == "HeSF-SS-teacher-topk":
        return SupportSelectorConfig(selector="teacher_topk", background_strategy="typed_background")
    if method == "HeSF-SS-teacher-diverse-topk":
        return SupportSelectorConfig(selector="teacher_diverse_topk", background_strategy="typed_background")
    if method == "HeSF-SS-validation-proxy-diverse":
        return SupportSelectorConfig(selector="validation_proxy_diverse", background_strategy="typed_background")
    if method == "HeSF-SS-true-validation-block-greedy":
        return SupportSelectorConfig(selector="true_validation_block_greedy", background_strategy="typed_background")
    if method == "HeSF-SS-sensitivity-block-selector":
        return SupportSelectorConfig(selector="sensitivity_block_selector", background_strategy="typed_background")
    if method == "HeSF-SS-prototype-residual-condense":
        return SupportSelectorConfig(selector="teacher_diverse_topk", background_strategy="class_anchor_relation_prototype")
    if method == "HeSF-SS-sensitivity-plus-prototype":
        return SupportSelectorConfig(selector="sensitivity_block_selector", background_strategy="class_anchor_relation_prototype")
    if method == "HeSF-SS-hybrid-teacher-response":
        return SupportSelectorConfig(selector="hybrid_teacher_response", background_strategy="class_anchor_relation_prototype")
    raise ValueError(f"unsupported Gate16 method: {method}")


def _flat_payload(result: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "coarse_graph",
        "assignment",
        "support_features",
        "importance",
        "selection",
        "graph_diagnostics",
        "task_metrics",
        "teacher_outputs",
    }
    return {key: value for key, value in result.items() if key not in skip}


def _full_graph_row(graph, dataset: str, seed: int, args: argparse.Namespace, split: dict[str, np.ndarray]) -> dict[str, Any]:
    target_type = infer_target_node_type(graph)
    support_count = int(np.sum(graph.node_type != int(target_type)))
    task = evaluate_hettree_task(
        graph,
        graph,
        np.arange(graph.num_nodes, dtype=np.int64),
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(target_type),
        official_split_nodes=split,
        primary_eval_mode="compressed_projected",
        early_stopping=True,
        monitor="projected_val_macro_f1",
    ).metrics
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "full-graph-hettree-lite-tuned",
        "requested_support_ratio": 1.0,
        "requested_support_count": int(support_count),
        "realized_support_count": int(support_count),
        "realized_support_ratio": 1.0,
        "support_budget_error": 0,
        "support_budget_exact_match": True,
        "realized_full_ratio": 1.0,
        "selected_support_count": int(support_count),
        "background_node_count": 0,
        "prototype_background_count": 0,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "test_label_usage": "metrics_only",
        "target_hit": True,
    }
    _row_from_task(row, task)
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
    split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
    train_mask = _mask(train_nodes, graph.num_nodes)
    val_mask = _mask(val_nodes, graph.num_nodes)
    test_mask = _mask(test_nodes, graph.num_nodes)
    teacher = train_full_graph_lite_teacher(
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        Gate15Config(target_node_type=int(target_type)).teacher,
        output_dir=Path(args.output_root) / "gate16_teacher" / f"{dataset}_seed{seed}",
        seed=int(seed),
        epochs=int(args.teacher_epochs),
        hidden_dim=int(args.teacher_hidden_dim),
        device=str(args.device),
        restarts=int(args.teacher_restarts),
    )
    rows: list[dict[str, Any]] = []
    teacher_rows = [{"dataset": dataset, "seed": int(seed), **teacher["metrics"]}]
    teacher_grid_rows = [{"dataset": dataset, "seed": int(seed), **item} for item in teacher.get("grid_results", [])]
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    if "full-graph-hettree-lite-tuned" in args.methods:
        rows.append(_full_graph_row(graph, dataset, int(seed), args, split))
    ceiling = rows[0] if rows else None
    support_count = int(np.sum(graph.node_type != int(target_type)))
    for ratio in args.ratios:
        for method in args.methods:
            if method == "full-graph-hettree-lite-tuned":
                continue
            start = perf_counter()
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "requested_support_ratio": float(ratio),
                **split_protocol,
            }
            try:
                if method in BASELINES:
                    coarse, assignment, diag = run_support_baseline(
                        graph,
                        baseline=method,
                        ratio=float(ratio),
                        seed=int(seed),
                        candidate_k=int(args.candidate_k),
                    )
                    row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list))})
                    final_support = int(diag.get("final_support_nodes", np.sum(coarse.node_type != int(target_type))))
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
                        np.asarray(assignment, dtype=np.int64),
                        seed=int(seed),
                        epochs=int(args.task_epochs),
                        hidden_dim=int(args.task_hidden_dim),
                        device=str(args.device),
                        target_node_type=int(target_type),
                        official_split_nodes=split,
                        primary_eval_mode="compressed_projected",
                        early_stopping=True,
                        monitor="projected_val_macro_f1",
                    ).metrics
                    _row_from_task(row, task)
                    row.setdefault("selector_uses_test_labels", False)
                    row.setdefault("teacher_uses_test_labels_for_training", False)
                    row.setdefault("test_label_usage", "metrics_only")
                else:
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
                    row.update(_flat_payload(result))
                    row["status"] = "success" if row.get("status") == "success" else row.get("status", "success")
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
                    components = result["importance"].get("components", {})
                    importance_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            "requested_support_ratio": float(ratio),
                            "importance_mean": row.get("support_importance_mean", 0.0),
                            "selected_importance_mean": row.get("selected_importance_mean", 0.0),
                            **{key: float(np.mean(value)) for key, value in components.items()},
                        }
                    )
                if ceiling is not None:
                    row["macro_recovery_vs_full_graph"] = _metric(row, "macro_f1") / max(_metric(ceiling, "macro_f1"), 1.0e-12)
                    row["accuracy_recovery_vs_full_graph"] = _metric(row, "accuracy") / max(_metric(ceiling, "accuracy"), 1.0e-12)
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
        "teacher_grid_rows": teacher_grid_rows,
        "importance_rows": importance_rows,
        "selection_rows": selection_rows,
        "graph_rows": graph_rows,
    }


def _write_smoke_report(path: Path, rows: list[dict[str, Any]]) -> None:
    skipped = [row for row in rows if row.get("status") != "success"]
    missing_primary = [
        row for row in rows
        if row.get("primary_eval_mode") == "compressed_projected"
        and abs(_metric(row, "macro_f1") - _metric(row, "projected_macro_f1")) > 1.0e-12
    ]
    lines = [
        "# Gate16 Smoke Report",
        "",
        f"- rows: `{len(rows)}`",
        f"- skipped_or_failed: `{len(skipped)}`",
        f"- primary_metric_mismatch: `{len(missing_primary)}`",
        f"- selector_uses_test_labels: `{any(str(row.get('selector_uses_test_labels', 'False')).lower() not in {'false', '0', ''} for row in rows)}`",
        f"- teacher_uses_test_labels_for_training: `{any(str(row.get('teacher_uses_test_labels_for_training', 'False')).lower() not in {'false', '0', ''} for row in rows)}`",
        f"- budget_fields_present: `{all('support_budget_exact_match' in row for row in rows if row.get('method') != 'full-graph-hettree-lite-tuned')}`",
        f"- projected_vs_transfer_gap_reported: `{all('projected_vs_transfer_macro_gap' in row for row in rows if row.get('status') == 'success')}`",
    ]
    if skipped:
        lines += ["", "## Skipped/Failed", ""]
        for row in skipped:
            lines.append(f"- {row.get('dataset')} {row.get('seed')} {row.get('method')}: {row.get('status')} {row.get('skip_reason', row.get('error', ''))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate16 support selection experiments.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--teacher-hidden-dim", type=int, default=32)
    parser.add_argument("--teacher-restarts", type=int, default=1)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit-groups", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.datasets = ["ACM"]
        args.seeds = [12345]
        args.ratios = [0.20]
        args.methods = [
            "full-graph-hettree-lite-tuned",
            "H6-no-spec-support-only",
            "TypedHash-ChebHeat-support-only",
            "HeSF-SS-teacher-topk",
            "HeSF-SS-sensitivity-block-selector",
            "HeSF-SS-prototype-residual-condense",
        ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_audit(args.output_root / "gate16_code_audit")
    groups = [(dataset, seed) for dataset in args.datasets for seed in args.seeds]
    if args.limit_groups is not None:
        groups = groups[: max(0, int(args.limit_groups))]
    all_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    teacher_grid_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    def consume(payload: dict[str, list[dict[str, Any]]]) -> None:
        nonlocal all_rows, teacher_rows, teacher_grid_rows, importance_rows, selection_rows, graph_rows
        all_rows.extend(payload["rows"])
        teacher_rows.extend(payload["teacher_rows"])
        teacher_grid_rows.extend(payload["teacher_grid_rows"])
        importance_rows.extend(payload["importance_rows"])
        selection_rows.extend(payload["selection_rows"])
        graph_rows.extend(payload["graph_rows"])
        if args.smoke:
            write_csv(args.output_root / "gate16_smoke" / "gate16_smoke_all_runs.csv", all_rows)
        else:
            write_csv(args.output_root / "gate16_tables" / "gate16_all_runs.csv", all_rows)
            write_csv(args.output_root / "gate16_teacher" / "full_graph_teacher_by_dataset_seed.csv", teacher_rows)
            write_csv(args.output_root / "gate16_teacher" / "full_graph_teacher_grid_results.csv", teacher_grid_rows)
            write_csv(args.output_root / "gate16_diag" / "support_importance.csv", importance_rows)
            write_csv(args.output_root / "gate16_diag" / "support_selection_diagnostics.csv", selection_rows)
            write_csv(args.output_root / "gate16_diag" / "compressed_graph_summary.csv", graph_rows)
            write_csv(args.output_root / "gate16_diag" / "full_graph_teacher_metrics.csv", teacher_rows)

    if int(args.jobs) <= 1:
        for dataset, seed in groups:
            consume(_run_group(args, str(dataset), int(seed)))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {pool.submit(_run_group, args, str(dataset), int(seed)): (dataset, seed) for dataset, seed in groups}
            for future in as_completed(futures):
                consume(future.result())
    if args.smoke:
        _write_smoke_report(args.output_root / "gate16_smoke" / "gate16_smoke_report.md", all_rows)
        return 0 if all(row.get("status") == "success" for row in all_rows) else 2
    write_csv(args.output_root / "gate16_tables" / "gate16_all_runs.csv", all_rows)
    write_csv(args.output_root / "gate16_teacher" / "full_graph_teacher_by_dataset_seed.csv", teacher_rows)
    write_csv(args.output_root / "gate16_teacher" / "full_graph_teacher_grid_results.csv", teacher_grid_rows)
    reliability_lines = ["# Gate16 Teacher Stability Report", ""]
    for dataset in sorted({str(row.get("dataset")) for row in teacher_rows}):
        values = [float(row.get("full_graph_teacher_macro_f1", 0.0) or 0.0) for row in teacher_rows if str(row.get("dataset")) == dataset]
        acc = [float(row.get("full_graph_teacher_accuracy", 0.0) or 0.0) for row in teacher_rows if str(row.get("dataset")) == dataset]
        reliability_lines.append(f"- {dataset}: macro mean `{float(np.mean(values)) if values else 0.0}`, std `{float(np.std(values)) if values else 0.0}`, accuracy mean `{float(np.mean(acc)) if acc else 0.0}`")
    (args.output_root / "gate16_teacher").mkdir(parents=True, exist_ok=True)
    (args.output_root / "gate16_teacher" / "teacher_stability_report.md").write_text("\n".join(reliability_lines) + "\n", encoding="utf-8")
    write_csv(args.output_root / "gate16_diag" / "support_importance.csv", importance_rows)
    write_csv(args.output_root / "gate16_diag" / "support_selection_diagnostics.csv", selection_rows)
    write_csv(args.output_root / "gate16_diag" / "compressed_graph_summary.csv", graph_rows)
    write_csv(args.output_root / "gate16_diag" / "full_graph_teacher_metrics.csv", teacher_rows)
    result = summarize(args.output_root)
    write_json(args.output_root / "gate16_tables" / "result.json", result)
    failed = [row for row in all_rows if row.get("status") != "success"]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
