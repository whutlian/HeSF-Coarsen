from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
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


def _split_values(values: list[Any] | None, cast=str) -> list[Any]:
    if not values:
        return []
    out: list[Any] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                out.append(cast(item))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gate17 full-graph teacher stability.")
    parser.add_argument("--output", "--output-dir", type=Path, default=Path("outputs/gate17_diagnostics"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", nargs="*", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--epochs", nargs="*", default=[100])
    parser.add_argument("--hidden-dim", "--hidden-dims", nargs="*", default=[64])
    parser.add_argument("--lr", nargs="*", default=[0.003])
    parser.add_argument("--dropout", nargs="*", default=[0.25, 0.5])
    parser.add_argument("--weight-decay", nargs="*", default=[1.0e-5])
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    datasets = _split_values(args.datasets, str) or ["ACM", "DBLP", "IMDB"]
    seeds = _split_values(args.seeds, int) or [12345, 23456, 34567, 45678, 56789]
    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        graph = load_hgb_graph(Path(args.data_root), dataset)
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1))
        target_type = infer_target_node_type(graph)
        teacher_cfg = replace(
            Gate15Config(target_node_type=int(target_type)).teacher,
            epochs_grid=tuple(_split_values(args.epochs, int) or [100]),
            hidden_dim_grid=tuple(_split_values(args.hidden_dim, int) or [64]),
            lr_grid=tuple(_split_values(args.lr, float) or [0.003]),
            dropout_grid=tuple(_split_values(args.dropout, float) or [0.25, 0.5]),
            weight_decay_grid=tuple(_split_values(args.weight_decay, float) or [1.0e-5]),
            restarts=int(args.restarts),
            patience=30,
            monitor="projected_val_macro_f1",
        )
        for seed in seeds:
            train_nodes, val_nodes, test_nodes, _split = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
            teacher = train_full_graph_lite_teacher(
                graph,
                labels,
                _mask(train_nodes, graph.num_nodes),
                _mask(val_nodes, graph.num_nodes),
                _mask(test_nodes, graph.num_nodes),
                teacher_cfg,
                output_dir=args.output / f"{dataset}_seed{seed}",
                seed=int(seed),
                epochs=int(teacher_cfg.epochs_grid[0]),
                hidden_dim=int(teacher_cfg.hidden_dim_grid[0]),
                device=str(args.device),
                use_config_grid=True,
                restarts=int(args.restarts),
            )
            rows.append({"dataset": dataset, "seed": int(seed), **teacher["metrics"]})
            for item in teacher.get("grid_results", []):
                grid_rows.append({"dataset": dataset, "seed": int(seed), **item})
    write_csv(args.output / "full_graph_teacher_by_dataset_seed.csv", rows)
    write_csv(args.output / "teacher_config_sweep.csv", grid_rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    acc_grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(_float(row.get("full_graph_teacher_macro_f1")))
        acc_grouped[str(row["dataset"])].append(_float(row.get("full_graph_teacher_accuracy")))
    lines = ["# Gate17 Teacher Stability Report", ""]
    for dataset in sorted(grouped):
        values = grouped[dataset]
        acc = acc_grouped[dataset]
        worst = min((row for row in rows if row["dataset"] == dataset), key=lambda row: _float(row.get("full_graph_teacher_macro_f1")))
        lines.append(
            f"- {dataset}: macro mean `{float(np.mean(values))}`, std `{float(np.std(values))}`, "
            f"accuracy mean `{float(np.mean(acc))}`, worst seed `{worst['seed']}`"
        )
    acm = grouped.get("ACM", [])
    dblp = grouped.get("DBLP", [])
    imdb = grouped.get("IMDB", [])
    reliable = (
        bool(acm) and float(np.mean(acm)) > 0.80 and float(np.std(acm)) < 0.10
        and bool(dblp) and float(np.mean(dblp)) > 0.70
        and bool(imdb) and float(np.mean(imdb)) > 0.35
    )
    lines += [
        "",
        f"teacher_reliable_for_primary_selection = `{reliable}`",
        "teacher usage rule: auxiliary diagnostics/tie-breaker only unless reliability thresholds are met.",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "teacher_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
