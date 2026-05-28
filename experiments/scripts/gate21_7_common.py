from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("outputs/gate21_7_icde_ready")
GATE21_6_SOURCE = Path("results/gate21_6_icde_ready")


def add_gate21_7_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--run-task-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-epochs", type=int, default=200)
    return parser


def training_seeds(args: argparse.Namespace) -> list[int]:
    seeds = args.training_seeds if args.training_seeds is not None else args.seeds
    return [int(seed) for seed in seeds]


def datasets(args: argparse.Namespace) -> list[str]:
    values = args.datasets if args.datasets is not None else [args.dataset]
    return [str(value).upper() for value in values]


def ensure_layout(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    subdirs = {
        "root": root,
        "main_official": root / "main_official",
        "apv16_stability": root / "apv16_stability",
        "external_tp": root / "external_tp",
        "standard_condensation": root / "standard_condensation",
        "semantic_audit": root / "semantic_audit",
        "feature_ablation_repaired": root / "feature_ablation_repaired",
        "adapter_package_repaired": root / "adapter_package_repaired",
        "storage_system_costs": root / "storage_system_costs",
        "cross_dataset": root / "cross_dataset",
        "summaries": root / "summaries",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_csv(src: str | Path, dst: str | Path, *, fallback_fieldnames: Sequence[str] | None = None) -> list[dict[str, str]]:
    rows = read_csv(src)
    write_csv(Path(dst), rows, fieldnames=fallback_fieldnames)
    return rows


def write_plan(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)
