from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, write_csv, write_json
from experiments.scripts.summarize_next16_sehgnn_compression import summarize_next16_sehgnn_compression
from hesf_coarsen.eval.sehgnn_task import evaluate_sehgnn_task
from hesf_coarsen.eval.task_gnn import compose_assignments
from hesf_coarsen.io.edge_list import load_graph


DATASETS = {
    "ACM": Path("data/acm_hesf"),
    "DBLP": Path("data/dblp_hesf"),
    "IMDB": Path("data/imdb_hesf"),
}
DEFAULT_METHODS = ["HeSF-LVC-P", "HeSF-LVC-S"]
DEFAULT_SEEDS = [12345, 23456, 34567, 45678, 56789]
DEFAULT_RATIOS = [0.012, 0.024, 0.048, 0.096]


def _method_token(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def _ratio_token(ratio: float) -> str:
    return f"{float(ratio):.4f}".replace(".", "p").rstrip("0").rstrip("p")


def _source_run_name(dataset: str, method: str, ratio: float, seed: int) -> str:
    return f"next15_hettree_{dataset.lower()}_{_method_token(method)}_r{_ratio_token(ratio)}_seed{int(seed)}"


def _level_number(path: Path) -> int:
    return int(path.name.removeprefix("level_"))


def _final_level_dir(run_dir: Path) -> Path | None:
    levels = [
        path
        for path in run_dir.glob("level_*")
        if path.is_dir() and (path / "schema.json").exists()
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


def _cumulative_assignment(run_dir: Path, original_nodes: int, final_level: Path) -> np.ndarray:
    cumulative = final_level / "cumulative_assignment.npz"
    if cumulative.exists():
        return np.load(cumulative)["assignment"].astype(np.int64, copy=False)
    return compose_assignments(original_nodes, _assignment_paths(run_dir))


def _edge_count(graph: Any) -> int:
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_oom(text: str) -> bool:
    lowered = str(text).lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "cuda error: out of memory" in lowered


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "experiments.scripts.run_next16_sehgnn_compression",
        "--source-runs-root",
        str(args.source_runs_root),
        "--datasets",
        *[str(item) for item in args.datasets],
        "--methods",
        *[str(item) for item in args.methods],
        "--compression-ratios",
        *[str(item) for item in args.compression_ratios],
        "--seeds",
        *[str(item) for item in args.seeds],
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--device",
        "cuda",
        "--output",
        str(args.output),
        "--skip-existing",
    ]


def _row_from_failure(
    *,
    dataset: str,
    method: str,
    ratio: float,
    seed: int,
    source_run_dir: Path,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "target_ratio": float(ratio),
        "seed": int(seed),
        "run_status": status,
        "skipped": True,
        "skip_reason": reason,
        "source_run_dir": str(source_run_dir),
    }


def _metadata(path: Path) -> Mapping[str, Any]:
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _evaluate_source_run(
    *,
    root: Path,
    output: Path,
    source_run_dir: Path,
    dataset: str,
    method: str,
    ratio: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    graph_dir = root / DATASETS[dataset]
    original = load_graph(graph_dir)
    final_level = _final_level_dir(source_run_dir)
    if final_level is None:
        raise ValueError(f"{source_run_dir} has no completed coarse graph")
    coarse = load_graph(final_level)
    mapping = _cumulative_assignment(source_run_dir, original.num_nodes, final_level)
    eval_metrics = evaluate_sehgnn_task(
        original,
        coarse,
        mapping,
        seed=int(seed),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        input_dropout=float(args.input_dropout),
        attention_dropout=float(args.attention_dropout),
        num_heads=int(args.num_heads),
        num_feature_projection_layers=int(args.num_feature_projection_layers),
        num_task_layers=int(args.num_task_layers),
        max_hops=int(args.max_hops),
        max_paths=int(args.max_paths) if args.max_paths is not None else None,
        device=str(args.device),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
    ).metrics
    actual_ratio = float(coarse.num_nodes / max(original.num_nodes, 1))
    eval_dir = output / "runs" / source_run_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "target_ratio": float(ratio),
        "seed": int(seed),
        "source_run_name": source_run_dir.name,
        "source_status": _metadata(source_run_dir).get("status", ""),
        "run_status": "success" if not eval_metrics.get("skipped") else "skipped",
        "skipped": bool(eval_metrics.get("skipped", False)),
        "skip_reason": eval_metrics.get("skip_reason", ""),
        "original_nodes": int(original.num_nodes),
        "coarse_nodes": int(coarse.num_nodes),
        "actual_ratio": actual_ratio,
        "target_hit": bool(actual_ratio <= float(ratio) * 1.05),
        "original_edges": _edge_count(original),
        "coarse_edges": _edge_count(coarse),
        "edge_ratio": float(_edge_count(coarse) / max(_edge_count(original), 1)),
        "final_level": _level_number(final_level),
        "source_run_dir": str(source_run_dir),
        "final_level_dir": str(final_level),
        "eval_dir": str(eval_dir),
    }
    for key, value in eval_metrics.items():
        if key not in row:
            row[key] = value
    write_json(eval_dir / "sehgnn_eval.json", row)
    return row


def run_next16_sehgnn_compression(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "sehgnn_runs.csv"
    rows: list[dict[str, Any]] = []
    if raw_path.exists() and args.skip_existing:
        rows = list(_read_existing(raw_path))
    existing = {
        (str(row.get("dataset")), str(row.get("method")), str(row.get("target_ratio")), str(row.get("seed")))
        for row in rows
    }
    server_commands: list[list[str]] = []

    for dataset in args.datasets:
        for method in args.methods:
            for ratio in args.compression_ratios:
                for seed in args.seeds:
                    key = (str(dataset), str(method), str(float(ratio)), str(int(seed)))
                    if key in existing:
                        continue
                    source_run_dir = args.source_runs_root / _source_run_name(str(dataset), str(method), float(ratio), int(seed))
                    if not source_run_dir.exists():
                        rows.append(
                            _row_from_failure(
                                dataset=str(dataset),
                                method=str(method),
                                ratio=float(ratio),
                                seed=int(seed),
                                source_run_dir=source_run_dir,
                                status="missing_source",
                                reason="source coarse run not found",
                            )
                        )
                        write_csv(raw_path, rows)
                        continue
                    try:
                        rows.append(
                            _evaluate_source_run(
                                root=root,
                                output=args.output,
                                source_run_dir=source_run_dir,
                                dataset=str(dataset),
                                method=str(method),
                                ratio=float(ratio),
                                seed=int(seed),
                                args=args,
                            )
                        )
                    except RuntimeError as exc:
                        reason = str(exc)
                        if _is_oom(reason):
                            server_commands.append(_server_command(args))
                            status = "oom"
                        else:
                            status = "error"
                        rows.append(
                            _row_from_failure(
                                dataset=str(dataset),
                                method=str(method),
                                ratio=float(ratio),
                                seed=int(seed),
                                source_run_dir=source_run_dir,
                                status=status,
                                reason=reason,
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            _row_from_failure(
                                dataset=str(dataset),
                                method=str(method),
                                ratio=float(ratio),
                                seed=int(seed),
                                source_run_dir=source_run_dir,
                                status="error",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        )
                    write_csv(raw_path, rows)

    summary_result = summarize_next16_sehgnn_compression(input=args.output, output=args.output)
    if server_commands:
        write_json(args.output / "server_commands.json", {"commands": server_commands})
    return {"rows": len(rows), "summary": summary_result, "server_commands": len(server_commands)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate existing Next15 coarse graphs with a local SeHGNN-style model.")
    parser.add_argument("--source-runs-root", type=Path, default=Path("outputs/exp_next15_hettree_compression_20260518/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_next16_sehgnn_compression_20260518"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--compression-ratios", "--ratios", dest="compression_ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--input-dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-feature-projection-layers", type=int, default=2)
    parser.add_argument("--num-task-layers", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_next16_sehgnn_compression(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
