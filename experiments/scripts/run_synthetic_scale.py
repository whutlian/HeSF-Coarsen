from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, read_json, repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot, write_csv
from experiments.scripts.make_synthetic_scale import estimate_scale_bytes
from hesf_coarsen.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or prepare synthetic scale experiments.")
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--input-root", type=Path, help="Root containing synthetic scale graph directories or estimate manifests.")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/synthetic_scale"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--edges", type=int, default=10_000_000)
    parser.add_argument("--max-levels", type=int, default=1)
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input_root:
        return _run_input_root(args)
    if args.graph_dir is None:
        parser.error("--graph-dir or --input-root is required")
    estimate = estimate_scale_bytes(args.nodes, args.edges)
    config_path = _write_run_config(args.config, args.output, args.max_levels, args.target_ratio)
    command = [
        args.python,
        "-m",
        "hesf_coarsen.cli.main",
        "coarsen",
        "--config",
        str(config_path),
        "--input",
        str(args.graph_dir),
        "--output",
        str(args.output),
    ]
    write_command_metadata(args.output, run_name=args.output.name, command=command, dataset="synthetic_scale", status="estimate" if args.estimate_only else "running", **estimate)
    if args.estimate_only:
        return 0
    completed = run_subprocess_with_log(
        command,
        cwd=repo_root(),
        log_path=args.output / "run.log",
    )
    write_command_metadata(args.output, run_name=args.output.name, dataset="synthetic_scale", status="success" if completed.returncode == 0 else "failed", returncode=completed.returncode, **estimate)
    return completed.returncode


def _run_input_root(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    failed = False
    for graph_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        run_dir = args.output / graph_dir.name
        estimate_path = graph_dir / "estimate.json"
        estimate: dict[str, object] = read_json(estimate_path) if estimate_path.exists() else estimate_scale_bytes(args.nodes, args.edges)
        command = [
            args.python,
            "-m",
            "hesf_coarsen.cli.main",
            "coarsen",
            "--config",
            str(_write_run_config(args.config, run_dir, args.max_levels, args.target_ratio)),
            "--input",
            str(graph_dir),
            "--output",
            str(run_dir),
        ]
        row: dict[str, object] = {
            "run_name": graph_dir.name,
            "graph_dir": str(graph_dir),
            "run_dir": str(run_dir),
            "nodes": estimate.get("nodes", ""),
            "edges": estimate.get("edges", ""),
        }
        if args.estimate_only or not (graph_dir / "schema.json").exists():
            status = "estimate" if args.estimate_only else "skipped_missing_graph"
            write_command_metadata(run_dir, run_name=graph_dir.name, command=command, dataset="synthetic_scale", status=status, **estimate)
            rows.append({**row, "status": status})
            continue
        write_command_metadata(run_dir, run_name=graph_dir.name, command=command, dataset="synthetic_scale", status="running", **estimate)
        completed = run_subprocess_with_log(command, cwd=repo_root(), log_path=run_dir / "run.log")
        status = "success" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            failed = True
        write_command_metadata(run_dir, run_name=graph_dir.name, command=command, dataset="synthetic_scale", status=status, returncode=completed.returncode, **estimate)
        rows.append({**row, "status": status, "returncode": completed.returncode})
    write_csv(args.output / "summary.csv", rows)
    (args.output / "report.md").write_text(
        "# Synthetic Scale Report\n\n"
        + markdown_table(rows, ["run_name", "status", "nodes", "edges", "graph_dir", "run_dir"])
        + "\n",
        encoding="utf-8",
    )
    return 1 if failed else 0


def _write_run_config(config_path: Path, run_dir: Path, max_levels: int, target_ratio: float) -> Path:
    config = load_config(config_path)
    config["coarsening"] = dict(
        config.get("coarsening", {}),
        max_levels=int(max_levels),
        target_ratio=float(target_ratio),
    )
    config.setdefault("progress", {})["backend"] = "plain"
    config.setdefault("resume", {})["enabled"] = True
    config.setdefault("output", {})["dir"] = str(run_dir)
    return write_config_snapshot(run_dir / "config.yaml", config)


if __name__ == "__main__":
    raise SystemExit(main())
