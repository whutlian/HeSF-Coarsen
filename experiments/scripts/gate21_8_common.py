from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("outputs/gate21_8_icde_evidence")
DEFAULT_GATE21_7_ROOT = Path("outputs/gate21_7_icde_ready")
DEFAULT_GATE21_6_DIR = Path("results/gate21_6_icde_ready")

GATE21_8_SUBDIRS = (
    "apv16_5x5",
    "external_tp_5x5",
    "freehgc_protocols",
    "metapath_cache_dump",
    "feature_ablation_tasks",
    "adapter_package_v3",
    "storage_system_costs",
    "cross_dataset_auto_channel",
    "audits",
    "logs",
)


def add_gate21_8_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--components", default="apv16_5x5,external_tp,freehgc,metapath_cache,feature_ablation,adapter,storage_system,cross_dataset,audits")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--gate21-7-root", type=Path, default=DEFAULT_GATE21_7_ROOT)
    parser.add_argument("--gate21-6-dir", type=Path, default=DEFAULT_GATE21_6_DIR)
    return parser


def components(args: argparse.Namespace) -> set[str]:
    raw = args.components
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = [str(part).strip() for part in raw]
    return {value for value in values if value}


def datasets(args: argparse.Namespace) -> list[str]:
    values = args.datasets if args.datasets is not None else [args.dataset]
    return [str(value).upper() for value in values]


def ensure_layout(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    paths = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for name in GATE21_8_SUBDIRS:
        paths[name] = root / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def write_component_readmes(paths: Mapping[str, Path]) -> None:
    descriptions = {
        "apv16_5x5": "APV12/APV16 seed-stability evidence. Rows must distinguish real 5x5 evidence from deterministic or not-validated status.",
        "external_tp_5x5": "Schema-preserving target-preserving external baseline evidence with budget alignment and hard failure rows.",
        "freehgc_protocols": "FreeHGC standard condensation and FreeHGC-TP are separated here. Standard condensation is not the TP protocol.",
        "metapath_cache_dump": "Real SeHGNN tensor/cache introspection evidence. Fallback audits do not pass Gate21.8.",
        "feature_ablation_tasks": "Feature/label ablation task rows plus shape safety. Unsupported settings must be explicit failures.",
        "adapter_package_v3": "Adapter task and package accounting with static snapshot vs reproducible transform package fields.",
        "storage_system_costs": "Storage-only and system-cost rows with explicit ratio denominators.",
        "cross_dataset_auto_channel": "DBLP/ACM/IMDB auto-channel plans and task-result evidence boundaries.",
        "audits": "Coverage v3, denominator, leakage, and protocol-separation audits.",
        "logs": "Runner manifests and execution summaries.",
    }
    for name, description in descriptions.items():
        path = paths[name] / "README.md"
        if not path.exists():
            path.write_text(f"# Gate21.8 {name}\n\n{description}\n", encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_csv(src: str | Path, dst: str | Path, *, fieldnames: Sequence[str] | None = None) -> list[dict[str, str]]:
    rows = read_csv(src)
    write_csv(Path(dst), rows, fieldnames=fieldnames)
    return rows


def write_plan(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)

