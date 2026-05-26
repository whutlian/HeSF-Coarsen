from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.coverage_diagnostics import compute_apv_coverage_diagnostics
from hesf_coarsen.eval.official.metapath_cache_introspection import fallback_metapath_cache_row, cache_hash_comparison_row
from hesf_coarsen.eval.official.storage_only_baselines import build_storage_only_row
from hesf_coarsen.eval.official.system_resource_logger import conservative_resource_row


QUICK_METHODS = ["export-full-SeHGNN", "H6-node30", "HeSF-RCS-APV12", "Random-HG-TP"]
ARTIFACTS = [
    "gate21_6_decision.json",
    "gate21_6_decision.md",
    "gate21_6_main_table_official.csv",
    "gate21_6_adapter_table.csv",
    "gate21_6_external_tp_table.csv",
    "gate21_6_storage_system_table.csv",
    "gate21_6_ablation_table.csv",
]


def planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = [str(item).upper() for item in args.datasets]
    graph_seeds = [int(args.graph_seeds[0])] if args.quick else [int(seed) for seed in args.graph_seeds]
    training_seeds = [int(args.training_seeds[0])] if args.quick else [int(seed) for seed in args.training_seeds]
    methods = list(QUICK_METHODS) if args.quick else [
        "full-native-SeHGNN",
        "export-full-SeHGNN",
        "H6-node30",
        "H6-APV-skeleton",
        "HeSF-RCS-APV12",
        "HeSF-RCS-APV16",
        "Random-HG-TP",
        "Herding-HG-TP",
        "KCenter-HG-TP",
        "Coarsening-HG-TP",
        "GraphSparsify-TP",
        "FreeHGC-TP",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            stochastic = method in {"Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "Coarsening-HG-TP", "GraphSparsify-TP", "FreeHGC-TP"}
            for graph_seed in (graph_seeds if stochastic else [graph_seeds[0]]):
                for training_seed in training_seeds:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "graph_seed": graph_seed,
                            "training_seed": training_seed,
                            "protocol": "schema_preserving_tp_workload" if method.endswith("-TP") or method.startswith("HeSF") or method.startswith("H6") else "official_reference",
                            "planned_only": bool(args.dry_run),
                        }
                    )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = planned_rows(args)
    write_csv(out / "planned_runs.csv", rows)
    write_json(out / "planned_methods.json", {"methods": sorted({row["method"] for row in rows}), "quick": bool(args.quick), "dry_run": bool(args.dry_run)})
    write_json(out / "planned_artifacts.json", {"artifacts": ARTIFACTS, "output_dir": str(out)})
    if args.dry_run:
        return {"planned_runs": len(rows), "dry_run": True, "output_dir": str(out)}

    from experiments.scripts.run_gate21_6_directed_skeleton_stability import run as run_directed
    from experiments.scripts.run_gate21_6_feature_ablation_safe import run as run_ablation
    from experiments.scripts.run_gate21_6_feature_adapter_package import run as run_adapter
    from experiments.scripts.run_gate21_6_external_baselines_tp import run as run_external
    from experiments.scripts.run_gate21_6_standard_condensation_baselines import run as run_standard
    from experiments.scripts.run_gate21_6_cross_dataset_auto_channel import run as run_cross

    base_kwargs = {
        "datasets": args.datasets,
        "graph_seeds": args.graph_seeds,
        "training_seeds": args.training_seeds,
        "quick": args.quick,
        "dry_run": False,
        "output_dir": out,
        "force_reprocess": args.force_reprocess,
        "skip_existing": args.skip_existing,
        "freehgc_root": args.freehgc_root,
        "official_sehgnn_root": args.official_sehgnn_root,
        "max_runs": args.max_runs,
        "fail_fast": args.fail_fast,
        "device": args.device,
    }
    directed = run_directed(argparse.Namespace(**base_kwargs, gate21_5_dir=Path("results/gate21_5_directed_apv_feature_adapter")))
    ablation = run_ablation(argparse.Namespace(**base_kwargs, methods=["full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"], gate21_5_dir=Path("results/gate21_5_directed_apv_feature_adapter")))
    adapter = run_adapter(argparse.Namespace(**base_kwargs, base_methods=["H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"], adapters=["raw", "fp16", "int8", "pca128", "pca64", "rp128", "rp64"], gate21_5_dir=Path("results/gate21_5_directed_apv_feature_adapter")))
    external = run_external(argparse.Namespace(**base_kwargs, methods=["Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "Coarsening-HG-TP", "GraphSparsify-TP", "FreeHGC-TP"], budgets=[0.50, 0.30, 0.20, 0.10]))
    standard = run_standard(argparse.Namespace(**base_kwargs, methods=["FreeHGC", "HGCond", "GCond-HG", "Random-HG", "Herding-HG", "KCenter-HG", "Coarsening-HG"]))
    cross = run_cross(argparse.Namespace(**base_kwargs, selector=["coverage_greedy", "validation_probe_greedy"]))
    _write_storage_system_metapath_coverage(out)
    return {
        "planned_runs": len(rows),
        "dry_run": False,
        "output_dir": str(out),
        "directed": directed,
        "feature_ablation": ablation,
        "adapter": adapter,
        "external": external,
        "standard": standard,
        "cross_dataset": cross,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _gzip_dir_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += len(gzip.compress(item.read_bytes()))
    return int(total)


def _node_counts_and_features(export_dir: Path) -> tuple[dict[int, int], int]:
    counts: dict[int, int] = {}
    feature_values = 0
    node_path = export_dir / "node.dat"
    if not node_path.exists():
        return counts, 0
    with node_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            typ = int(parts[2])
            counts[typ] = counts.get(typ, 0) + 1
            if len(parts) >= 4 and parts[3]:
                feature_values += len([item for item in parts[3].split(",") if item != ""])
    return counts, feature_values


def _relations(export_dir: Path) -> dict[str, list[tuple[int, int]]]:
    names = {0: "AP", 1: "PA", 2: "PT", 3: "PV", 4: "TP", 5: "VP"}
    out: dict[str, list[tuple[int, int]]] = {name: [] for name in names.values()}
    link_path = export_dir / "link.dat"
    if not link_path.exists():
        return out
    with link_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            out.setdefault(names.get(int(parts[2]), str(parts[2])), []).append((int(parts[0]), int(parts[1])))
    return out


def _write_storage_system_metapath_coverage(out: Path) -> None:
    gate5 = Path("results/gate21_5_directed_apv_feature_adapter")
    cache_rows = _read_csv(gate5 / "gate21_5_cache_audit.csv")
    first_export = next((Path(row["export_dir"]) for row in cache_rows if row.get("method") == "export-full-SeHGNN" and Path(row.get("export_dir", "")).exists()), None)
    if first_export is None:
        first_export = next((Path(row["export_dir"]) for row in cache_rows if Path(row.get("export_dir", "")).exists()), Path("."))
    native_bytes = max(1, _dir_bytes(first_export))
    raw_bytes = _dir_bytes(first_export)
    gzip_bytes = _gzip_dir_bytes(first_export) if first_export.exists() else 0
    counts, feature_values = _node_counts_and_features(first_export)
    relation_edges = sum(len(values) for values in _relations(first_export).values())
    storage_rows = [
        build_storage_only_row(dataset="DBLP", artifact_name="raw_hgb_text", native_full_text_bytes=native_bytes, total_artifact_bytes=raw_bytes, changes_training_semantics=False, requires_loader_adapter=False, raw_hgb_text_bytes=raw_bytes, loader_supported=True),
        build_storage_only_row(dataset="DBLP", artifact_name="gzip_hgb_text", native_full_text_bytes=native_bytes, total_artifact_bytes=gzip_bytes, changes_training_semantics=False, requires_loader_adapter=True, raw_hgb_text_bytes=raw_bytes, gzip_bytes=gzip_bytes, loader_supported=False),
        build_storage_only_row(dataset="DBLP", artifact_name="binary_csr_relation_tables", native_full_text_bytes=native_bytes, total_artifact_bytes=relation_edges * 12, changes_training_semantics=False, requires_loader_adapter=True, binary_relation_bytes=relation_edges * 12, loader_supported=False),
        build_storage_only_row(dataset="DBLP", artifact_name="binary_csr_plus_fp16_features", native_full_text_bytes=native_bytes, total_artifact_bytes=relation_edges * 12 + feature_values * 2, changes_training_semantics=False, requires_loader_adapter=True, binary_relation_bytes=relation_edges * 12, binary_feature_bytes=feature_values * 2, loader_supported=False),
        build_storage_only_row(dataset="DBLP", artifact_name="binary_csr_plus_int8_features", native_full_text_bytes=native_bytes, total_artifact_bytes=relation_edges * 12 + feature_values, changes_training_semantics=False, requires_loader_adapter=True, binary_relation_bytes=relation_edges * 12, binary_feature_bytes=feature_values, loader_supported=False),
        build_storage_only_row(dataset="DBLP", artifact_name="zstd_hgb_text", native_full_text_bytes=native_bytes, total_artifact_bytes=raw_bytes, changes_training_semantics=False, requires_loader_adapter=True, raw_hgb_text_bytes=raw_bytes, zstd_bytes="", loader_supported=False, notes="zstd executable/library not required for Gate21.6 local run; row kept as unsupported storage-only baseline."),
    ]
    write_csv(out / "gate21_6_storage_only_baselines.csv", storage_rows)
    resource_rows = [
        conservative_resource_row(stage_name="compression_construction", input_paths=[gate5 / "gate21_5_directed_by_method.csv"], output_paths=[out / "gate21_6_directed_skeleton_by_method.csv"], num_edge_passes=1),
        conservative_resource_row(stage_name="feature_adapter_construction", input_paths=[gate5 / "gate21_5_feature_adapter_raw_rows.csv"], output_paths=[out / "gate21_6_feature_adapter_by_run.csv"], num_feature_passes=1),
        conservative_resource_row(stage_name="summarization", input_paths=[out], output_paths=[out]),
    ]
    write_csv(out / "gate21_6_system_resource_by_stage.csv", resource_rows)
    metapath_rows = [fallback_metapath_cache_row(row) for row in cache_rows if row.get("method") in {"H6-node30", "H6-APV-skeleton", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00", "H6-dirskel-AP100-PA00-PV75-VP00-PTTP00", "export-full-SeHGNN"}]
    write_csv(out / "gate21_6_metapath_cache_audit.csv", metapath_rows)
    pttp00 = next((row for row in cache_rows if row.get("method") == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"), {})
    pttp10 = next((row for row in cache_rows if row.get("method") == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP10"), {})
    write_csv(out / "gate21_6_cache_hash_assertions.csv", [cache_hash_comparison_row(pttp00, pttp10)] if pttp00 and pttp10 else [])
    coverage_rows = []
    summary_rows = []
    for method in ["H6-APV-skeleton", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00", "H6-dirskel-AP100-PA00-PV75-VP00-PTTP00"]:
        row = next((item for item in cache_rows if item.get("method") == method and item.get("training_seed") == "1"), None)
        if row is None:
            continue
        export_dir = Path(row.get("export_dir", ""))
        counts, _feature_values = _node_counts_and_features(export_dir)
        rels = _relations(export_dir)
        cov = compute_apv_coverage_diagnostics(
            method=method,
            graph_seed=int(row.get("graph_seed") or 1),
            relation_keep_plan={},
            num_authors=counts.get(0, 0),
            num_papers=counts.get(1, 0),
            num_terms=counts.get(2, 0),
            num_venues=counts.get(3, 0),
            relations=rels,
        )
        coverage_rows.append(cov)
        summary_rows.append({key: cov.get(key, "") for key in ["method", "graph_seed", "fraction_target_authors_with_AP_edge", "fraction_target_authors_reaching_venue", "venue_coverage_fraction", "paper_coverage_fraction", "coverage_warning_flags"]})
    write_csv(out / "gate21_6_coverage_diagnostics.csv", coverage_rows)
    write_csv(out / "gate21_6_coverage_summary_by_method.csv", summary_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate21_6_icde_ready"))
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--freehgc-root", type=Path, default=None)
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
