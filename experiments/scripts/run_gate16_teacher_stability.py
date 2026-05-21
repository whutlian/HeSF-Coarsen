from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.selection.config import Gate15Config
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher


def _mask(nodes: np.ndarray, total: int) -> np.ndarray:
    out = np.zeros(int(total), dtype=bool)
    out[np.asarray(nodes, dtype=np.int64)] = True
    return out


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gate16 full-graph teacher stability.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gate16_teacher"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        graph = load_hgb_graph(Path(args.data_root), dataset)
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
        target_type = infer_target_node_type(graph)
        for seed in args.seeds:
            train_nodes, val_nodes, test_nodes, _split = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
            teacher = train_full_graph_lite_teacher(
                graph,
                labels,
                _mask(train_nodes, graph.num_nodes),
                _mask(val_nodes, graph.num_nodes),
                _mask(test_nodes, graph.num_nodes),
                Gate15Config(target_node_type=int(target_type)).teacher,
                output_dir=args.output / f"{dataset}_seed{seed}",
                seed=int(seed),
                epochs=int(args.epochs),
                hidden_dim=int(args.hidden_dim),
                device=str(args.device),
                restarts=int(args.restarts),
            )
            rows.append({"dataset": dataset, "seed": int(seed), **teacher["metrics"]})
            for item in teacher.get("grid_results", []):
                grid_rows.append({"dataset": dataset, "seed": int(seed), **item})
    write_csv(args.output / "full_graph_teacher_by_dataset_seed.csv", rows)
    write_csv(args.output / "full_graph_teacher_grid_results.csv", grid_rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    acc_grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(_float(row.get("full_graph_teacher_macro_f1")))
        acc_grouped[str(row["dataset"])].append(_float(row.get("full_graph_teacher_accuracy")))
    lines = ["# Gate16 Teacher Stability Report", ""]
    for dataset in sorted(grouped):
        values = grouped[dataset]
        acc = acc_grouped[dataset]
        worst = min((row for row in rows if row["dataset"] == dataset), key=lambda row: _float(row.get("full_graph_teacher_macro_f1")))
        lines.append(
            f"- {dataset}: macro mean `{float(np.mean(values))}`, std `{float(np.std(values))}`, "
            f"accuracy mean `{float(np.mean(acc))}`, worst seed `{worst['seed']}`"
        )
    reliable = all(values and float(np.mean(values)) >= 0.35 and (float(np.std(values)) <= 0.20 or len(values) < 2) for values in grouped.values())
    lines += ["", f"teacher_not_reliable_for_importance = `{not reliable}`"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "teacher_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
