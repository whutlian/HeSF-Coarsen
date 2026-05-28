from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("outputs/gate21_9_icde_evidence")
DEFAULT_GATE21_8_ROOT = Path("outputs/gate21_8_icde_evidence")
DEFAULT_GATE21_7_ROOT = Path("outputs/gate21_7_icde_ready")

GATE21_9_SUBDIRS = (
    "auto_selector_alignment",
    "external_tp_5x5",
    "freehgc_protocols",
    "metapath_cache_dump",
    "feature_ablation_tasks",
    "adapter_package_v4",
    "storage_system_costs",
    "cross_dataset_auto_channel",
    "audits",
    "logs",
)


def add_gate21_9_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--components", default="auto_selector,external_tp,freehgc,metapath,feature_ablation,adapter,storage,cross_dataset")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--gate21-8-root", type=Path, default=DEFAULT_GATE21_8_ROOT)
    parser.add_argument("--gate21-7-root", type=Path, default=DEFAULT_GATE21_7_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def components(args: argparse.Namespace) -> set[str]:
    raw = str(args.components)
    if raw.strip().lower() == "all":
        return {
            "auto_selector",
            "external_tp",
            "freehgc",
            "metapath",
            "feature_ablation",
            "adapter",
            "storage",
            "cross_dataset",
        }
    return {part.strip() for part in raw.split(",") if part.strip()}


def ensure_layout(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    paths = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for name in GATE21_9_SUBDIRS:
        paths[name] = root / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def write_component_readmes(paths: Mapping[str, Path]) -> None:
    descriptions = {
        "auto_selector_alignment": "Validation-only channel utility, removal probes, and DBLP APV auto-selector alignment.",
        "external_tp_5x5": "Schema-preserving TP external baseline task grid. Missing cells remain explicit failures.",
        "freehgc_protocols": "FreeHGC standard condensation and TP adapter/hard-gap evidence are separated.",
        "metapath_cache_dump": "Official SeHGNN metapath/cache tensor dump and cache hash assertions.",
        "feature_ablation_tasks": "Feature ablation task metrics or explicit unsupported task rows.",
        "adapter_package_v4": "Feature adapter task rows with static, transform recipe, and reconstructable package ratios.",
        "storage_system_costs": "Storage bytes, explicit denominator audit, loader support, and workload cost trace.",
        "cross_dataset_auto_channel": "DBLP/ACM/IMDB auto-channel task evidence and failure rows.",
        "audits": "Coverage v4 and semantic sanity assertions.",
        "logs": "Run manifests and component summaries.",
    }
    for name, text in descriptions.items():
        path = paths[name] / "README.md"
        if not path.exists():
            path.write_text(f"# Gate21.9 {name}\n\n{text}\n", encoding="utf-8")


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
