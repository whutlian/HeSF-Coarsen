from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split


def _row(dataset: str, seed: int, method: str, ratio: float, args: argparse.Namespace) -> dict[str, Any]:
    graph = load_hgb_graph(Path(args.data_root), dataset)
    labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
    target_type = infer_target_node_type(graph)
    train_nodes, val_nodes, test_nodes, _split = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
    if method == "full-graph":
        coarse = graph
        assignment = np.arange(graph.num_nodes, dtype=np.int64)
    else:
        coarse, assignment, _diag = run_support_baseline(
            graph,
            baseline=method,
            ratio=float(ratio),
            seed=int(seed),
            candidate_k=int(args.candidate_k),
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
        official_split_nodes={"train": train_nodes, "val": val_nodes, "test": test_nodes},
        primary_eval_mode="compressed_projected",
        early_stopping=True,
        monitor="projected_val_macro_f1",
    ).metrics
    return {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "support_ratio": float(ratio),
        "primary_eval_mode": task.get("primary_eval_mode", ""),
        "projected_macro_f1": task.get("projected_original_macro_f1", 0.0),
        "transfer_macro_f1": task.get("transfer_original_macro_f1", 0.0),
        "projected_vs_transfer_macro_gap": task.get("projected_vs_transfer_macro_gap", 0.0),
        "projected_accuracy": task.get("projected_original_accuracy", 0.0),
        "transfer_accuracy": task.get("transfer_original_accuracy", 0.0),
        "projected_vs_transfer_accuracy_gap": task.get("projected_vs_transfer_accuracy_gap", 0.0),
        "validation_projected_macro_f1": task.get("projected_original_val_macro_f1", 0.0),
        "validation_transfer_macro_f1": task.get("transfer_original_val_macro_f1", 0.0),
        "skipped": bool(task.get("skipped", False)),
        "skip_reason": task.get("skip_reason", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate16 evaluator projected-vs-transfer comparison.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gate16_evaluator"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--methods", nargs="+", default=["full-graph", "TypedHash-ChebHeat-support-only"])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.20])
    parser.add_argument("--task-epochs", type=int, default=5)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for method in args.methods:
                for ratio in ([1.0] if method == "full-graph" else args.ratios):
                    rows.append(_row(str(dataset), int(seed), str(method), float(ratio), args))
    write_csv(args.output / "gate16_evaluator_comparison.csv", rows)
    mean_gap = float(np.mean([float(row["projected_vs_transfer_macro_gap"]) for row in rows])) if rows else 0.0
    report = "# Gate16 Evaluator Report\n\n"
    report += f"- rows: `{len(rows)}`\n"
    report += f"- mean_projected_vs_transfer_macro_gap: `{mean_gap}`\n"
    report += "- primary_eval_mode: `compressed_projected`\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate16_evaluator_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
