from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json
from experiments.scripts.gate14_task_first_common import aggregate_rows, evaluate_graph, load_hgb_graph


FOCUSED_GRID = (
    (64, 0.0, 0.001, 50, 2, 32),
    (128, 0.3, 0.005, 100, 2, 64),
    (256, 0.5, 0.01, 200, 3, 32),
    (128, 0.0, 0.005, 50, 3, 64),
    (64, 0.5, 0.01, 100, 2, 32),
    (256, 0.3, 0.001, 200, 3, 64),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate14 full-graph hettree_lite tuning.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def _grid(full: bool) -> list[tuple[int, float, float, int, int, int]]:
    if not full:
        return list(FOCUSED_GRID)
    return list(
        itertools.product(
            [64, 128, 256],
            [0.0, 0.3, 0.5],
            [0.001, 0.005, 0.01],
            [50, 100, 200],
            [2, 3],
            [32, 64],
        )
    )


def _worker(args: argparse.Namespace, dataset: str, seed: int, hp: tuple[int, float, float, int, int, int]) -> dict[str, Any]:
    hidden_dim, dropout, lr, epochs, max_hops, max_paths = hp
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "model": "hettree_lite",
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lr": float(lr),
        "epochs": int(epochs),
        "max_hops": int(max_hops),
        "max_paths": int(max_paths),
        "hyperparameters": json.dumps(
            {
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "lr": float(lr),
                "epochs": int(epochs),
                "max_hops": int(max_hops),
                "max_paths": int(max_paths),
            },
            sort_keys=True,
        ),
        "status": "running",
    }
    try:
        graph = load_hgb_graph(Path(args.data_root), dataset)
        assignment = np.arange(graph.num_nodes, dtype=np.int64)
        metrics = evaluate_graph(
            graph,
            graph,
            assignment,
            seed=int(seed),
            task_epochs=int(epochs),
            task_hidden_dim=int(hidden_dim),
            lr=float(lr),
            dropout=float(dropout),
            max_hops=int(max_hops),
            max_paths=int(max_paths),
            device=str(args.device),
        )
        row.update(metrics)
        row["split_policy"] = metrics.get("split_policy", metrics.get("task_protocol", "deterministic_random"))
        row["status"] = "success"
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def _run(args: argparse.Namespace, combos: list[tuple[str, int, tuple[int, float, float, int, int, int]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_csv = args.output / "full_graph_lite_runs.csv"
    if int(args.jobs) <= 1:
        for combo in combos:
            rows.append(_worker(args, *combo))
            write_csv(out_csv, rows)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = {pool.submit(_worker, args, *combo): combo for combo in combos}
            for future in as_completed(futures):
                rows.append(future.result())
                write_csv(out_csv, rows)
    return rows


def _select_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "success":
            groups.setdefault((str(row.get("dataset")), int(row.get("seed"))), []).append(row)
    selected = []
    for _key, group in sorted(groups.items()):
        best = max(group, key=lambda row: (float(row.get("validation_macro_f1") or -1), float(row.get("validation_accuracy") or -1)))
        out = dict(best)
        out["selected_by_validation"] = True
        selected.append(out)
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [(dataset, seed, hp) for dataset in args.datasets for seed in args.seeds for hp in _grid(bool(args.full_grid))]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = _run(args, combos)
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), int(row.get("seed", 0)), str(row.get("hyperparameters"))))
    selected = _select_best(rows)
    for row in rows:
        row["selected_by_validation"] = any(
            row.get("dataset") == best.get("dataset")
            and row.get("seed") == best.get("seed")
            and row.get("hyperparameters") == best.get("hyperparameters")
            for best in selected
        )
    write_csv(args.output / "full_graph_lite_runs.csv", rows)
    write_csv(args.output / "full_graph_lite_best_by_validation.csv", selected)
    by_dataset = aggregate_rows(selected, ["dataset", "model"], ("macro_f1", "micro_f1", "accuracy", "validation_macro_f1", "validation_accuracy"))
    write_csv(args.output / "full_graph_lite_by_dataset.csv", by_dataset)
    summary = "# Full Graph Lite Ceiling Summary\n\n"
    summary += "Validation-selected full-graph `hettree_lite` ceiling. Official evaluator status remains diagnostic lite only.\n\n"
    summary += markdown_table(by_dataset, ["dataset", "model", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean"])
    (args.output / "full_graph_lite_ceiling_summary.md").write_text(summary + "\n", encoding="utf-8")
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures), "focused_grid": not bool(args.full_grid)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
