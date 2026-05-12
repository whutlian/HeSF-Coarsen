from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, repo_root, run_subprocess_with_log, write_command_metadata
from hesf_coarsen.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OGBN-MAG envelope configs A/B/C.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/ogbn_mag_envelope"))
    parser.add_argument("--configs", nargs="+", type=Path, default=[
        Path("configs/ogbn_mag_A_cpu_chunked.yaml"),
        Path("configs/ogbn_mag_B_cpu_ann.yaml"),
        Path("configs/ogbn_mag_C_torch_ann.yaml"),
    ])
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument("--max-levels", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    rows: list[dict[str, object]] = []
    for config_path in args.configs:
        run_name = config_path.stem
        run_dir = args.output / run_name
        try:
            config = deepcopy(load_config(config_path))
        except Exception as exc:
            write_command_metadata(run_dir, run_name=run_name, dataset="ogbn-mag", status="failed", failure_reason=str(exc))
            rows.append({"run_name": run_name, "status": "failed", "failure_reason": str(exc)})
            continue
        config.setdefault("coarsening", {})["target_ratio"] = args.target_ratio
        config.setdefault("coarsening", {})["max_levels"] = args.max_levels
        config.setdefault("progress", {})["enabled"] = True
        config.setdefault("progress", {})["backend"] = "plain"
        config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
        config.setdefault("resume", {})["enabled"] = bool(args.resume)
        config["output"] = {"dir": str(run_dir)}
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
        command = [
            args.python,
            "-m",
            "hesf_coarsen.cli.main",
            "coarsen",
            "--config",
            str(run_dir / "config.yaml"),
            "--input",
            str(args.input),
            "--output",
            str(run_dir),
            "--progress",
            "--progress-backend",
            "plain",
        ]
        if args.resume:
            command.append("--resume")
        write_command_metadata(run_dir, run_name=run_name, command=command, dataset="ogbn-mag", status="created")
        if args.dry_run:
            rows.append({"run_name": run_name, "status": "created"})
            continue
        completed = run_subprocess_with_log(command, cwd=root, log_path=run_dir / "run.log")
        status = "success" if completed.returncode == 0 else "failed"
        write_command_metadata(run_dir, run_name=run_name, command=command, dataset="ogbn-mag", status=status, returncode=completed.returncode)
        rows.append({"run_name": run_name, "status": status, "returncode": completed.returncode})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "report.md").write_text("# OGBN-MAG Envelope\n\n" + markdown_table(rows, ["run_name", "status", "returncode"]) + "\n", encoding="utf-8")
    return 0 if all(row.get("status") != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
