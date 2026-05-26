from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.run_gate21_4_apv_skeleton_validation import build_parser as build_gate21_4_parser
from experiments.scripts.run_gate21_4_apv_skeleton_validation import run_gate21_4
from experiments.scripts.summarize_gate21_5_directed_apv import summarize_gate21_5
from hesf_coarsen.eval.official.directed_relation_skeleton import canonicalize_directed_method, expand_directed_methods, is_directed_skeleton_method
from hesf_coarsen.eval.official.metapath_channel_audit import loaded_relation_audit_rows, metapath_placeholder_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


MANIFEST_FIELDS = [
    "run_id",
    "dataset",
    "method",
    "canonical_method",
    "run_group",
    "graph_seed",
    "training_seed",
    "relation_channel_spec",
    "deterministic_graph_method",
    "graph_seed_independence_required",
    "graph_seed_independence_status",
    "sehgnn_command_json",
    "export_dir",
    "cache_dir",
    "output_dir",
    "status",
]

DIRECTED_BY_METHOD_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "method_family",
    "budget_strategy",
    "edge_score_strategy",
    "relation_channel_spec",
    "runs",
    "success_count",
    "failed_count",
    "graph_seed_count",
    "training_seed_count",
    "num_effective_graph_variants",
    "num_effective_training_seeds",
    "deterministic_graph_method",
    "graph_seed_independence_required",
    "graph_seed_independence_status",
    "mean_semantic_structural_storage_ratio",
    "std_semantic_structural_storage_ratio",
    "mean_hgb_raw_file_byte_ratio",
    "mean_official_text_hgb_byte_ratio",
    "mean_preprocessed_cache_byte_ratio",
    "mean_effective_total_byte_ratio",
    "mean_support_node_ratio",
    "mean_support_edge_ratio",
    "mean_total_node_ratio",
    "mean_total_edge_ratio",
    "mean_test_micro_f1",
    "std_test_micro_f1",
    "mean_test_macro_f1",
    "std_test_macro_f1",
    "mean_validation_micro_f1",
    "mean_validation_macro_f1",
    "mean_recovery_vs_native_full_micro",
    "mean_recovery_vs_native_full_macro",
    "mean_val_test_micro_gap",
    "schema_complete_all",
    "relation_mapping_audit_pass_all",
    "relation_retention_audit_pass_all",
    "cache_hygiene_pass_all",
    "no_test_label_export_leakage_all",
    "no_test_label_scoring_leakage_all",
    "official_sehgnn_unmodified_all",
    "eligible_for_main_decision",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _method_safe(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(method))


def _is_deterministic(method: str) -> bool:
    return str(method).startswith("H6-dirskel-") or str(method) == "H6-APV-skeleton"


def _method_meta(method: str) -> dict[str, Any]:
    if str(method).startswith("H6-dirskel-"):
        return canonicalize_directed_method(method)
    if str(method) == "H6-APV-skeleton":
        return {
            "method": method,
            "canonical_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "relation_channel_spec": "APPA100-PVVP100-PTTP00",
            "deterministic_graph_method": True,
            "graph_seed_independence_required": False,
            "graph_seed_independence_status": "not_applicable_deterministic",
        }
    return {
        "method": method,
        "canonical_method": method,
        "relation_channel_spec": "",
        "deterministic_graph_method": False,
        "graph_seed_independence_required": True,
        "graph_seed_independence_status": "required",
    }


def _run_pairs(method: str, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[tuple[int | str, int]]:
    if method in {"full-native-SeHGNN", "export-full-SeHGNN"}:
        return [("", int(seed)) for seed in training_seeds]
    seeds = [int(graph_seeds[0])] if _is_deterministic(method) else [int(seed) for seed in graph_seeds]
    return [(int(graph_seed), int(training_seed)) for graph_seed in seeds for training_seed in training_seeds]


def _manifest_rows(args: argparse.Namespace, methods: Sequence[str]) -> list[dict[str, Any]]:
    dataset = str(args.dataset).upper()
    out = Path(args.out_dir)
    rows = []
    for method in methods:
        meta = _method_meta(method)
        for graph_seed, training_seed in _run_pairs(method, args.graph_seeds, args.training_seeds):
            graph_part = "graph_seed_none" if graph_seed == "" else f"graph_seed_{graph_seed}"
            run_id = f"{dataset}_{_method_safe(method)}_{graph_part}_train_seed_{training_seed}"
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": meta["method"],
                    "canonical_method": meta["canonical_method"],
                    "run_group": "directed_apv_skeleton",
                    "graph_seed": graph_seed,
                    "training_seed": int(training_seed),
                    "relation_channel_spec": meta["relation_channel_spec"],
                    "deterministic_graph_method": bool(meta["deterministic_graph_method"]),
                    "graph_seed_independence_required": bool(meta["graph_seed_independence_required"]),
                    "graph_seed_independence_status": meta["graph_seed_independence_status"],
                    "sehgnn_command_json": "",
                    "export_dir": str(out / "exports" / dataset / graph_part / _method_safe(method)),
                    "cache_dir": str(out / "cache" / dataset / _method_safe(method) / graph_part / f"training_seed_{training_seed}"),
                    "output_dir": str(out / "logs" / dataset / graph_part / f"training_seed_{training_seed}" / _method_safe(method)),
                    "status": "planned" if _bool_arg(args.dry_run) else "pending",
                }
            )
    return rows


def _empty_outputs(out: Path) -> None:
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    write_csv(out / "gate21_5_directed_by_method.csv", [], fieldnames=DIRECTED_BY_METHOD_FIELDS)
    write_csv(out / "gate21_5_directed_raw_rows.csv", [])
    write_csv(out / "gate21_5_directed_storage_frontier.csv", [])
    write_csv(out / "gate21_5_relation_edge_retention.csv", [])
    write_csv(out / "gate21_5_relation_mapping_audit.csv", [])
    write_csv(out / "gate21_5_loaded_relation_audit.csv", [])
    write_csv(out / "gate21_5_metapath_channel_audit.csv", [])
    write_csv(out / "gate21_5_cache_audit.csv", [])


def _write_dry_run(args: argparse.Namespace, methods: Sequence[str]) -> dict[str, Any]:
    out = Path(args.out_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    manifest = _manifest_rows(args, methods)
    write_csv(out / "manifests" / "gate21_5_directed_run_manifest.csv", manifest, fieldnames=MANIFEST_FIELDS)
    write_json(out / "gate21_5_plan.json", {"dataset": str(args.dataset).upper(), "methods": list(methods), "dry_run": True})
    summarize_gate21_5(out, out, native_full_micro=0.9533802, native_full_macro=0.9498198, write_md=True, write_json_flag=True)
    return {"dry_run": True, "planned_runs": len(manifest)}


def _copy_gate21_4_to_gate21_5(out: Path, internal: Path) -> None:
    _copy_gate21_4_many_to_gate21_5(out, [internal])


def _copy_gate21_4_many_to_gate21_5(out: Path, internals: Sequence[Path]) -> None:
    by_method = []
    raw_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, str]] = []
    retention_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []
    cache_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for internal in internals:
        for row in _read_csv(internal / "gate21_4_by_method.csv"):
            deterministic = _is_deterministic(str(row.get("method", "")))
            row = {
                **row,
                "num_effective_graph_variants": 1 if deterministic else row.get("graph_seed_count", ""),
                "num_effective_training_seeds": row.get("training_seed_count", ""),
                "deterministic_graph_method": deterministic,
                "graph_seed_independence_required": not deterministic,
                "graph_seed_independence_status": "not_applicable_deterministic" if deterministic else "required",
                "mean_official_text_hgb_byte_ratio": row.get("mean_hgb_raw_file_byte_ratio", ""),
            }
            by_method.append({field: row.get(field, "") for field in DIRECTED_BY_METHOD_FIELDS})
        for row in _read_csv(internal / "gate21_4_raw_rows.csv"):
            deterministic = _is_deterministic(row.get("method", ""))
            row["deterministic_graph_method"] = deterministic
            row["graph_seed_independence_required"] = not deterministic
            row["graph_seed_independence_status"] = "not_applicable_deterministic" if deterministic else "required"
            row["official_text_hgb_byte_ratio"] = row.get("hgb_raw_file_byte_ratio", "")
            raw_rows.append(row)
        storage_rows.extend(_read_csv(internal / "gate21_4_storage_frontier.csv"))
        retention_rows.extend(_read_csv(internal / "gate21_4_relation_edge_retention.csv"))
        mapping_rows.extend(_read_csv(internal / "gate21_4_relation_mapping_audit.csv"))
        cache_rows.extend(_read_csv(internal / "gate21_4_cache_audit.csv"))
        manifest_rows.extend(_read_csv(internal / "gate21_4_run_manifest.csv"))
    write_csv(out / "gate21_5_directed_by_method.csv", by_method, fieldnames=DIRECTED_BY_METHOD_FIELDS)
    write_csv(out / "gate21_5_directed_raw_rows.csv", raw_rows)
    write_csv(out / "gate21_5_directed_storage_frontier.csv", storage_rows)
    write_csv(out / "gate21_5_relation_edge_retention.csv", retention_rows)
    write_csv(out / "gate21_5_relation_mapping_audit.csv", mapping_rows)
    write_csv(out / "gate21_5_cache_audit.csv", cache_rows)
    write_csv(out / "manifests" / "gate21_5_directed_run_manifest.csv", manifest_rows)
    _write_loaded_relation_sidecars_many(out, internals)


def _write_loaded_relation_sidecars(out: Path, internal: Path) -> None:
    _write_loaded_relation_sidecars_many(out, [internal])


def _write_loaded_relation_sidecars_many(out: Path, internals: Sequence[Path]) -> None:
    cache_rows: list[dict[str, str]] = []
    retention_rows: list[dict[str, str]] = []
    for internal in internals:
        cache_rows.extend(_read_csv(internal / "gate21_4_cache_audit.csv"))
        retention_rows.extend(_read_csv(internal / "gate21_4_relation_edge_retention.csv"))
    expected: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in retention_rows:
        key = (row.get("method", ""), row.get("graph_seed", ""), row.get("training_seed", ""))
        expected.setdefault(key, {})[str(row.get("official_relation_id", ""))] = int(float(row.get("actual_relation_budget") or 0))
    loaded_rows: list[dict[str, Any]] = []
    metapath_rows: list[dict[str, Any]] = []
    for row in cache_rows:
        key = (row.get("method", ""), row.get("graph_seed", ""), row.get("training_seed", ""))
        export_dir = Path(row.get("export_dir", ""))
        if not export_dir.exists():
            continue
        loaded_rows.extend(
            loaded_relation_audit_rows(
                dataset=row.get("dataset", "DBLP"),
                method=row.get("method", ""),
                canonical_method=row.get("canonical_method", row.get("method", "")),
                graph_seed=int(float(row.get("graph_seed") or 0)),
                training_seed=int(float(row.get("training_seed") or 0)),
                export_dir=export_dir,
                expected_relation_counts=expected.get(key, {}),
            )
        )
        metapath_rows.extend(
            metapath_placeholder_rows(
                dataset=row.get("dataset", "DBLP"),
                method=row.get("method", ""),
                canonical_method=row.get("canonical_method", row.get("method", "")),
                graph_seed=int(float(row.get("graph_seed") or 0)),
                training_seed=int(float(row.get("training_seed") or 0)),
                export_dir=export_dir,
                relation_channel_spec="",
            )
        )
    write_csv(out / "gate21_5_loaded_relation_audit.csv", loaded_rows)
    write_csv(out / "gate21_5_metapath_channel_audit.csv", metapath_rows)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def run(args: argparse.Namespace) -> dict[str, Any]:
    methods = expand_directed_methods(args.methods, args.custom_methods)
    if _bool_arg(args.quick):
        args.training_seeds = list(args.training_seeds[:3])
        if args.methods == "full":
            methods = expand_directed_methods("core")
    if _bool_arg(args.dry_run):
        return _write_dry_run(args, methods)
    out = Path(args.out_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    deterministic_methods = [method for method in methods if _is_deterministic(method)]
    random_or_reference_methods = [method for method in methods if method not in deterministic_methods]
    internals: list[Path] = []
    results: list[dict[str, Any]] = []
    groups = []
    if deterministic_methods:
        groups.append(("deterministic", deterministic_methods, [int(args.graph_seeds[0])]))
    if random_or_reference_methods:
        groups.append(("controls", random_or_reference_methods, [int(seed) for seed in args.graph_seeds]))
    for name, group_methods, group_graph_seeds in groups:
        internal = out / "diagnostics" / f"gate21_4_directed_internal_{name}"
        gate_args = build_gate21_4_parser().parse_args(
            [
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(internal),
                "--graph-seeds",
                *[str(seed) for seed in group_graph_seeds],
                "--training-seeds",
                *[str(seed) for seed in args.training_seeds],
                "--methods",
                *group_methods,
                "--force",
                "--force-reprocess",
                "--sehgnn-root",
                str(args.sehgnn_root),
                "--hgb-data-root",
                str(args.hgb_data_root),
                "--device",
                str(args.device),
            ]
        )
        results.append(run_gate21_4(gate_args))
        internals.append(internal)
    _copy_gate21_4_many_to_gate21_5(out, internals)
    summarize_gate21_5(out, out, native_full_micro=0.9533802, native_full_macro=0.9498198, write_md=True, write_json_flag=True)
    return {"methods": len(methods), "gate21_4_results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", choices=["core", "full", "custom"], required=True)
    parser.add_argument("--custom-methods", default="")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-native-full", default="")
    parser.add_argument("--reuse-export-full", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
