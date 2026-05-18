from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import (
    repo_root,
    run_subprocess_with_log,
    write_command_metadata,
    write_config_snapshot,
    write_csv,
    write_json,
)
from experiments.scripts.summarize_next15_hettree_compression import summarize_next15_hettree_compression
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task
from hesf_coarsen.eval.task_gnn import compose_assignments
from hesf_coarsen.io.edge_list import load_graph


DATASETS = {
    "ACM": Path("data/acm_hesf"),
    "DBLP": Path("data/dblp_hesf"),
    "IMDB": Path("data/imdb_hesf"),
}
METHOD_CONFIGS = {
    "HeSF-LVC-P": Path("configs/paper/hgb_hesf_lvc_p.yaml"),
    "HeSF-LVC-S": Path("configs/paper/hgb_hesf_lvc_s.yaml"),
}
DEFAULT_SEEDS = [12345, 23456, 34567, 45678, 56789]
DEFAULT_RATIOS = [0.012, 0.024, 0.048, 0.096]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _method_token(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def _ratio_token(ratio: float) -> str:
    return f"{float(ratio):.4f}".replace(".", "p").rstrip("0").rstrip("p")


def _run_name(dataset: str, method: str, ratio: float, seed: int) -> str:
    return f"next15_hettree_{dataset.lower()}_{_method_token(method)}_r{_ratio_token(ratio)}_seed{int(seed)}"


def _prepare_config(root: Path, run_dir: Path, *, method: str, ratio: float, seed: int, device: str) -> dict[str, Any]:
    config = deepcopy(_load_yaml(root / METHOD_CONFIGS[method]))
    config["seed"] = int(seed)
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["experiment_block"] = "next15_hettree_compression"
    config.setdefault("paper", {})["downstream_model"] = "hettree_lite"
    config.setdefault("coarsening", {})["target_ratio"] = float(ratio)
    config.setdefault("coarsening", {})["max_levels"] = max(
        int(config.get("coarsening", {}).get("max_levels", 4)),
        10,
    )
    config.setdefault("output", {})["dir"] = str(run_dir)
    config.setdefault("candidates", {})["mmap_dir"] = str(run_dir / "_candidate_mmap")
    config.setdefault("candidates", {})["incident_index_mmap_dir"] = str(run_dir / "_incident_index_mmap")
    config.setdefault("acceleration", {})["device"] = str(device)
    return config


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


def _edge_count(graph_path: Path) -> int:
    graph = load_graph(graph_path)
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _is_oom(text: str) -> bool:
    lowered = str(text).lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "cuda error: out of memory" in lowered


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "experiments.scripts.run_next15_hettree_compression",
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
        "--progress",
        "--skip-existing",
    ]


def _row_from_failure(
    *,
    dataset: str,
    method: str,
    ratio: float,
    seed: int,
    run_dir: Path,
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
        "run_dir": str(run_dir),
    }


def _evaluate_run(
    *,
    graph_dir: Path,
    run_dir: Path,
    dataset: str,
    method: str,
    ratio: float,
    seed: int,
    args: argparse.Namespace,
    coarsen_sec: float,
) -> dict[str, Any]:
    original = load_graph(graph_dir)
    final_level = _final_level_dir(run_dir)
    if final_level is None:
        raise ValueError(f"{run_dir} has no completed coarse graph")
    coarse = load_graph(final_level)
    mapping = _cumulative_assignment(run_dir, original.num_nodes, final_level)
    eval_metrics = evaluate_hettree_task(
        original,
        coarse,
        mapping,
        seed=int(seed),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        max_hops=int(args.max_hops),
        max_paths=int(args.max_paths) if args.max_paths is not None else None,
        device=str(args.device),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
    ).metrics
    original_edges = sum(rel.num_edges for rel in original.relations.values())
    coarse_edges = sum(rel.num_edges for rel in coarse.relations.values())
    actual_ratio = float(coarse.num_nodes / max(original.num_nodes, 1))
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "target_ratio": float(ratio),
        "seed": int(seed),
        "run_name": run_dir.name,
        "run_status": "success" if not eval_metrics.get("skipped") else "skipped",
        "skipped": bool(eval_metrics.get("skipped", False)),
        "skip_reason": eval_metrics.get("skip_reason", ""),
        "original_nodes": int(original.num_nodes),
        "coarse_nodes": int(coarse.num_nodes),
        "actual_ratio": actual_ratio,
        "target_hit": bool(actual_ratio <= float(ratio) * 1.05),
        "original_edges": int(original_edges),
        "coarse_edges": int(coarse_edges),
        "edge_ratio": float(coarse_edges / max(original_edges, 1)),
        "final_level": _level_number(final_level),
        "coarsen_wall_clock_sec": float(coarsen_sec),
        "run_dir": str(run_dir),
        "final_level_dir": str(final_level),
    }
    for key, value in eval_metrics.items():
        if key not in row:
            row[key] = value
    write_json(run_dir / "hettree_eval.json", row)
    return row


def run_next15_hettree_compression(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_path = args.output / "hettree_runs.csv"
    if raw_path.exists() and args.skip_existing:
        import csv

        with raw_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    seen = {
        (str(row.get("dataset")), str(row.get("method")), str(row.get("target_ratio")), str(row.get("seed")))
        for row in rows
    }
    server_command_written = False
    for dataset in args.datasets:
        dataset_name = str(dataset).upper()
        graph_dir = root / DATASETS[dataset_name]
        for method in args.methods:
            for ratio in args.compression_ratios:
                for seed in args.seeds:
                    key = (dataset_name, str(method), str(float(ratio)), str(int(seed)))
                    if args.skip_existing and key in seen:
                        continue
                    run_name = _run_name(dataset_name, str(method), float(ratio), int(seed))
                    run_dir = args.output / "runs" / run_name
                    if args.skip_existing and (run_dir / "hettree_eval.json").exists():
                        try:
                            rows.append(json.loads((run_dir / "hettree_eval.json").read_text(encoding="utf-8")))
                            write_csv(raw_path, rows)
                            continue
                        except json.JSONDecodeError:
                            pass
                    config = _prepare_config(
                        root,
                        run_dir,
                        method=str(method),
                        ratio=float(ratio),
                        seed=int(seed),
                        device=str(args.device),
                    )
                    write_config_snapshot(run_dir / "config.yaml", config)
                    command = [
                        str(args.python),
                        "-m",
                        "hesf_coarsen.cli.main",
                        "coarsen",
                        "--config",
                        str(run_dir / "config.yaml"),
                        "--input",
                        str(graph_dir),
                        "--output",
                        str(run_dir),
                    ]
                    if args.progress:
                        command.append("--progress")
                    write_command_metadata(
                        run_dir,
                        run_name=run_name,
                        command=command,
                        status="running",
                        dataset=dataset_name,
                        method=str(method),
                        target_ratio=float(ratio),
                        seed=int(seed),
                        experiment_block="next15_hettree_compression",
                    )
                    coarsen_start = perf_counter()
                    completed = run_subprocess_with_log(
                        command,
                        cwd=root,
                        log_path=run_dir / "coarsen.log",
                        stream_output=bool(args.progress),
                    )
                    coarsen_sec = float(perf_counter() - coarsen_start)
                    output_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
                    if completed.returncode != 0:
                        status = "oom" if _is_oom(output_text) else "failed"
                        reason = f"coarsen_returncode={completed.returncode}"
                        write_command_metadata(
                            run_dir,
                            run_name=run_name,
                            command=command,
                            status="failed",
                            returncode=completed.returncode,
                            dataset=dataset_name,
                            method=str(method),
                            target_ratio=float(ratio),
                            seed=int(seed),
                            experiment_block="next15_hettree_compression",
                        )
                        row = _row_from_failure(
                            dataset=dataset_name,
                            method=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            run_dir=run_dir,
                            status=status,
                            reason=reason,
                        )
                        rows.append(row)
                        write_csv(raw_path, rows)
                        if status == "oom" and args.stop_on_oom:
                            write_json(args.output / "server_command.json", {"command": _server_command(args)})
                            server_command_written = True
                            break
                        continue
                    write_command_metadata(
                        run_dir,
                        run_name=run_name,
                        command=command,
                        status="success",
                        returncode=completed.returncode,
                        dataset=dataset_name,
                        method=str(method),
                        target_ratio=float(ratio),
                        seed=int(seed),
                        experiment_block="next15_hettree_compression",
                    )
                    try:
                        row = _evaluate_run(
                            graph_dir=graph_dir,
                            run_dir=run_dir,
                            dataset=dataset_name,
                            method=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            args=args,
                            coarsen_sec=coarsen_sec,
                        )
                    except RuntimeError as exc:
                        status = "oom" if _is_oom(str(exc)) else "failed"
                        row = _row_from_failure(
                            dataset=dataset_name,
                            method=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            run_dir=run_dir,
                            status=status,
                            reason=str(exc),
                        )
                        if status == "oom" and args.stop_on_oom:
                            write_json(args.output / "server_command.json", {"command": _server_command(args)})
                            server_command_written = True
                    except Exception as exc:
                        row = _row_from_failure(
                            dataset=dataset_name,
                            method=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            run_dir=run_dir,
                            status="failed",
                            reason=str(exc),
                        )
                    rows.append(row)
                    write_csv(raw_path, rows)
                    if server_command_written:
                        break
                if server_command_written:
                    break
            if server_command_written:
                break
        if server_command_written:
            break
    summarize_next15_hettree_compression(input=args.output, output=args.output)
    return {"rows": len(rows), "server_command_written": server_command_written}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_CONFIGS))
    parser.add_argument("--compression-ratios", "--ratios", dest="compression_ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-oom", action="store_true", default=True)
    parser.add_argument("--continue-after-oom", action="store_false", dest="stop_on_oom")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        if str(dataset).upper() not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
    for method in args.methods:
        if str(method) not in METHOD_CONFIGS:
            raise ValueError(f"unknown method: {method}")
    run_next15_hettree_compression(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
