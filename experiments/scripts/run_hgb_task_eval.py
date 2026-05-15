from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import discover_run_dirs, read_json, write_csv, write_json
from hesf_coarsen.eval.task_gnn import compose_assignments, evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph


def _level_number(path: Path) -> int:
    return int(path.name.removeprefix("level_"))


def _final_level_dir(run_dir: Path) -> Path | None:
    levels = [
        path
        for path in run_dir.glob("level_*")
        if path.is_dir() and (path / "schema.json").exists() and (path / "diagnostics.json").exists()
    ]
    if not levels:
        return None
    return max(levels, key=_level_number)


def _assignment_paths(run_dir: Path) -> list[str]:
    levels = sorted(
        [path for path in run_dir.glob("level_*") if (path / "assignment.npz").exists()],
        key=_level_number,
    )
    return [str(level / "assignment.npz") for level in levels]


def _cumulative_assignment(run_dir: Path, original_nodes: int, final_level: Path) -> Any:
    cumulative = final_level / "cumulative_assignment.npz"
    if cumulative.exists():
        import numpy as np

        return np.load(cumulative)["assignment"].astype(np.int64, copy=False)
    return compose_assignments(original_nodes, _assignment_paths(run_dir))


def evaluate_run(
    run_dir: Path,
    *,
    graph_root: Path,
    seed: int,
    epochs: int,
    refine_epochs: int,
    refine_epochs_list: list[int] | None,
    hidden_dim: int,
    device: str,
    full_graph_rgcn_lite: bool,
) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    dataset = str(metadata.get("dataset", ""))
    if not dataset:
        raise ValueError(f"{run_dir} metadata does not contain dataset")
    original_dir = graph_root / f"{dataset.lower()}_hesf"
    final_level = _final_level_dir(run_dir)
    if final_level is None:
        raise ValueError(f"{run_dir} has no completed level graph")
    original = load_graph(original_dir)
    coarse = load_graph(final_level)
    mapping = _cumulative_assignment(run_dir, original.num_nodes, final_level)
    result = evaluate_rgcn_task(
        original,
        coarse,
        mapping,
        seed=int(metadata.get("seed", seed) or seed),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        refine_epochs=int(refine_epochs),
        refine_epochs_list=refine_epochs_list,
        device=device,
        full_graph_rgcn_lite=bool(full_graph_rgcn_lite),
    ).metrics
    result.update(
        {
            "run_name": metadata.get("run_name", run_dir.name),
            "dataset": dataset,
            "variant": metadata.get("variant", ""),
            "experiment_block": metadata.get("experiment_block", ""),
            "unique_run_key": metadata.get("unique_run_key", metadata.get("run_name", run_dir.name)),
            "final_level": _level_number(final_level),
            "coarse_nodes": coarse.num_nodes,
            "original_nodes": original.num_nodes,
        }
    )
    write_json(run_dir / "task_eval.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real HGB task evaluation on existing coarse graphs.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--refine-epochs", type=int, default=10)
    parser.add_argument("--refine-epochs-list", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--full-graph-rgcn-lite",
        "--include-full-graph-baseline",
        action="store_true",
        dest="full_graph_rgcn_lite",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = discover_run_dirs([args.runs_root])
    rows: list[dict[str, Any]] = []
    wanted_datasets = {name.upper() for name in args.datasets}
    wanted_variants = None if args.variants is None else {str(value) for value in args.variants}
    selected: list[Path] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        if str(metadata.get("status", "success")) != "success":
            continue
        if str(metadata.get("dataset", "")).upper() not in wanted_datasets:
            continue
        if wanted_variants is not None and str(metadata.get("variant", "")) not in wanted_variants:
            continue
        selected.append(run_dir)
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    for index, run_dir in enumerate(selected, start=1):
        if args.progress:
            print(f"[task-eval] {index}/{len(selected)} {run_dir.name}", flush=True)
        try:
            row = evaluate_run(
                run_dir,
                graph_root=args.graph_root,
                seed=args.seed,
                epochs=args.epochs,
                refine_epochs=args.refine_epochs,
                refine_epochs_list=args.refine_epochs_list,
                hidden_dim=args.hidden_dim,
                device=args.device,
                full_graph_rgcn_lite=args.full_graph_rgcn_lite,
            )
        except Exception as exc:
            row = {
                "run_name": run_dir.name,
                "status": "failed",
                "failure_reason": str(exc),
            }
        rows.append(row)
    output = args.output or (args.runs_root / "task_eval_summary.csv")
    write_csv(output, rows)
    return 0 if all(str(row.get("status", "success")) != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
