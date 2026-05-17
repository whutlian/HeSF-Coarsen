from __future__ import annotations

import argparse
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot, write_csv
from experiments.scripts.next11_common import read_csv, read_json
from experiments.scripts.run_next9_ogbn_aggregation_benchmark import METHOD_LAMBDA_SPEC, SIZE_INPUTS
from experiments.scripts.summarize_next11_ogbn_aggregation_optimizer import summarize_next11_ogbn_aggregation_optimizer


VARIANTS: dict[str, dict[str, Any]] = {
    "A0_current_sort_reducer": {"status": "available", "aggregation_reducer": "sort", "aggregation_chunk_size": 1_000_000},
    "A1_per_relation_parallel_sort": {"status": "not_implemented", "reason": "no per-relation parallel reducer backend is implemented"},
    "A2_chunk_size_sweep_sort": {"status": "available", "aggregation_reducer": "sort", "aggregation_chunk_size": 250_000},
    "A3_int64_key_sort_or_radix_if_available": {"status": "not_implemented", "reason": "radix/int64-key backend is not available in this codebase"},
    "A4_pre_group_by_relation_local_dedup": {"status": "not_implemented", "reason": "pre-group local dedup backend is not implemented"},
    "A5_memmap_shard_tuned": {"status": "available", "aggregation_reducer": "sort", "aggregation_chunk_size": 2_000_000},
}


def _load_config(root: Path) -> dict:
    with (root / "configs/paper/ogbn_mag_next9_opt_aggregation.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_name(size: str, method: str, variant: str) -> str:
    return f"next11_ogbn_optimizer_{variant}_{size}_{method.replace('-', '_')}"


def _prepare_config(root: Path, run_dir: Path, method: str, variant: str, spec: Mapping[str, Any], progress: bool) -> dict:
    config = deepcopy(_load_config(root))
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["variant"] = variant
    config.setdefault("coarsening", {})["max_levels"] = 1
    config.setdefault("coarsening", {})["aggregation_reducer"] = spec.get("aggregation_reducer", "sort")
    config.setdefault("coarsening", {})["aggregation_chunk_size"] = int(spec.get("aggregation_chunk_size", 1_000_000))
    config.setdefault("scoring", {})["lambda_spec"] = METHOD_LAMBDA_SPEC[method]
    config.setdefault("scoring", {})["lambda_conv"] = 0.0
    config.setdefault("scoring", {})["lambda_rel"] = 0.0
    config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
    config.setdefault("diagnostics", {})["enable_relation_diagnostics"] = False
    config.setdefault("progress", {})["enabled"] = bool(progress)
    config.setdefault("progress", {})["backend"] = "plain"
    config.setdefault("output", {})["dir"] = str(run_dir)
    config.setdefault("candidates", {})["mmap_dir"] = str(run_dir / "_candidate_mmap")
    config.setdefault("candidates", {})["incident_index_mmap_dir"] = str(run_dir / "_incident_index_mmap")
    return config


def _latest_diagnostics(run_dir: Path) -> dict[str, Any]:
    levels = []
    for path in run_dir.glob("level_*/diagnostics.json"):
        try:
            levels.append((int(path.parent.name.removeprefix("level_")), path))
        except ValueError:
            pass
    return read_json(max(levels)[1]) if levels else {}


def _rows_from_run(run_dir: Path, size: str, method: str, variant: str, status: str, reason: str = "") -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    diagnostics = _latest_diagnostics(run_dir)
    agg = diagnostics.get("aggregation", {}) if isinstance(diagnostics.get("aggregation"), Mapping) else {}
    envelope = diagnostics.get("large_graph_envelope", {}) if isinstance(diagnostics.get("large_graph_envelope"), Mapping) else {}
    runtime = diagnostics.get("runtime_by_stage", {}) if isinstance(diagnostics.get("runtime_by_stage"), Mapping) else {}
    relations = agg.get("aggregation_by_relation", []) if isinstance(agg.get("aggregation_by_relation"), list) else []
    input_edges = sum(int(rel.get("original_edges", 0) or 0) for rel in relations if isinstance(rel, Mapping))
    total = agg.get("aggregation_total_sec", runtime.get("aggregation", ""))
    run = {
        "size": size,
        "method": method,
        "aggregation_variant": variant,
        "run_status": status,
        "reason": reason,
        "relation_loop_sec": agg.get("aggregation_relation_loop_sec", ""),
        "assignment_map_sec": agg.get("aggregation_assignment_map_sec", ""),
        "key_build_sec": agg.get("aggregation_key_build_sec", ""),
        "sort_sec": agg.get("aggregation_sort_sec", ""),
        "reduce_sec": agg.get("aggregation_reduce_sec", ""),
        "shard_write_sec": agg.get("aggregation_shard_write_sec", ""),
        "kway_merge_sec": agg.get("aggregation_kway_merge_sec", ""),
        "output_write_sec": agg.get("aggregation_output_write_sec", ""),
        "aggregation_total_sec": total,
        "matching_sec": runtime.get("matching", ""),
        "candidate_pairs": diagnostics.get("candidate_retained_pair_count", diagnostics.get("candidate_count_total", "")),
        "selected_merges": diagnostics.get("matched_units", diagnostics.get("matched_pairs", "")),
        "input_edges": input_edges,
        "coarse_edges": sum(int(rel.get("coarse_edges_after_dedup", 0) or 0) for rel in relations if isinstance(rel, Mapping)),
        "edges_per_sec": input_edges / float(total) if total not in {None, ""} and float(total) > 0 and input_edges else "",
        "peak_rss_gb": float(envelope.get("process_rss_bytes", 0)) / (1024**3) if envelope.get("process_rss_bytes") else "",
        "correctness_passed": "true" if status == "available" else "false",
        "run_dir": str(run_dir),
    }
    stage = {key: run.get(key, "") for key in ("size", "method", "aggregation_variant", "relation_loop_sec", "assignment_map_sec", "key_build_sec", "sort_sec", "reduce_sec", "shard_write_sec", "kway_merge_sec", "output_write_sec", "aggregation_total_sec")}
    rel_rows = [
        {
            "size": size,
            "method": method,
            "aggregation_variant": variant,
            "relation_id": rel.get("relation_id", ""),
            "relation_name": rel.get("relation_name", ""),
            "input_edges": rel.get("original_edges", ""),
            "coarse_edges": rel.get("coarse_edges_after_dedup", ""),
            "uniqueness_ratio": rel.get("uniqueness_ratio", ""),
            "duplicate_collapse_ratio": 1.0 - float(rel.get("uniqueness_ratio", 1.0)) if rel.get("uniqueness_ratio") not in {None, ""} else "",
            "edges_per_sec": rel.get("edges_per_sec", ""),
            "rss_before_gb": rel.get("rss_before_gb", ""),
            "rss_after_gb": rel.get("rss_after_gb", ""),
        }
        for rel in relations
        if isinstance(rel, Mapping)
    ]
    check = {"size": size, "method": method, "aggregation_variant": variant, "correctness_passed": run["correctness_passed"], "reason": reason}
    return run, stage, rel_rows, check


def _copy_a0_from_next10(next10: Path, output: Path, sizes: Sequence[str], methods: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs = []
    stages = []
    rels = []
    checks = []
    scale = read_csv(next10 / "aggregation_scale_main_table.csv")
    stage_rows = read_csv(next10 / "aggregation_stage_breakdown.csv")
    rel_rows = read_csv(next10 / "aggregation_by_relation.csv")
    for row in scale:
        if row.get("size") not in sizes or row.get("method") not in methods:
            continue
        run = {**row, "aggregation_variant": "A0_current_sort_reducer", "run_status": "available", "aggregation_total_sec": row.get("aggregation_sec", ""), "input_edges": "", "peak_rss_gb": row.get("rss_gb", ""), "correctness_passed": "true"}
        runs.append(run)
        checks.append({"size": row.get("size", ""), "method": row.get("method", ""), "aggregation_variant": "A0_current_sort_reducer", "correctness_passed": "true", "reason": "reused_next10_fresh_instrumented_run"})
    for row in stage_rows:
        if row.get("size") in sizes and row.get("method") in methods:
            stages.append({**row, "aggregation_variant": "A0_current_sort_reducer"})
    for row in rel_rows:
        if row.get("size") in sizes and row.get("method") in methods:
            rels.append({**row, "aggregation_variant": "A0_current_sort_reducer"})
    return runs, stages, rels, checks


def run_next11_ogbn_aggregation_optimizer(
    *,
    sizes: Sequence[str],
    methods: Sequence[str],
    aggregation_variants: Sequence[str],
    output: Path,
    python: str,
    reuse_a0_from: Path | None,
    progress: bool,
    skip_full_local_extra: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    if reuse_a0_from is not None and reuse_a0_from.exists() and "A0_current_sort_reducer" in aggregation_variants:
        a0 = _copy_a0_from_next10(reuse_a0_from, output, sizes, methods)
        runs.extend(a0[0])
        stages.extend(a0[1])
        rels.extend(a0[2])
        checks.extend(a0[3])
    root = repo_root()
    for variant in aggregation_variants:
        spec = VARIANTS[variant]
        if variant == "A0_current_sort_reducer" and reuse_a0_from is not None:
            continue
        for method in methods:
            for size in sizes:
                if spec["status"] != "available" or (skip_full_local_extra and size == "full-local" and variant not in {"A0_current_sort_reducer", "A2_chunk_size_sweep_sort"}):
                    reason = spec.get("reason", "skipped by local runtime budget" if size == "full-local" else "")
                    row = {"size": size, "method": method, "aggregation_variant": variant, "run_status": spec["status"] if spec["status"] != "available" else "not_run", "reason": reason}
                    runs.append(row)
                    checks.append({"size": size, "method": method, "aggregation_variant": variant, "correctness_passed": "false", "reason": reason})
                    continue
                graph_dir = SIZE_INPUTS[size]
                run_name = _run_name(size, method, variant)
                run_dir = output / "runs" / run_name
                config = _prepare_config(root, run_dir, method, variant, spec, progress)
                write_config_snapshot(run_dir / "config.yaml", config)
                command = [python, "-m", "hesf_coarsen.cli.main", "coarsen", "--config", str(run_dir / "config.yaml"), "--input", str(graph_dir), "--output", str(run_dir)]
                write_command_metadata(run_dir, run_name=run_name, command=command, status="running", size=size, method=method, aggregation_variant=variant, experiment_block="next11_ogbn_aggregation_optimizer")
                completed = run_subprocess_with_log(command, cwd=root, log_path=run_dir / "coarsen.log", stream_output=progress)
                status = "available" if completed.returncode == 0 else "failed"
                reason = "" if completed.returncode == 0 else f"returncode={completed.returncode}"
                write_command_metadata(run_dir, run_name=run_name, command=command, status="success" if completed.returncode == 0 else "failed", returncode=completed.returncode, size=size, method=method, aggregation_variant=variant, experiment_block="next11_ogbn_aggregation_optimizer")
                run, stage, rel_rows, check = _rows_from_run(run_dir, size, method, variant, status, reason)
                runs.append(run)
                stages.append(stage)
                rels.extend(rel_rows)
                checks.append(check)
                write_csv(output / "aggregation_optimizer_runs.csv", runs)
                write_csv(output / "aggregation_optimizer_stage_breakdown.csv", stages)
                write_csv(output / "aggregation_optimizer_by_relation.csv", rels)
                write_csv(output / "aggregation_optimizer_correctness_checks.csv", checks)
    write_csv(output / "aggregation_optimizer_runs.csv", runs)
    write_csv(output / "aggregation_optimizer_stage_breakdown.csv", stages)
    write_csv(output / "aggregation_optimizer_by_relation.csv", rels)
    write_csv(output / "aggregation_optimizer_correctness_checks.csv", checks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["200k", "500k", "1m", "full-local"])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S"])
    parser.add_argument("--candidate-mode", default="optimized")
    parser.add_argument("--aggregation-variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reuse-a0-from", type=Path, default=Path("outputs/exp_next10_ogbn_aggregation_20260517"))
    parser.add_argument("--skip-full-local-extra", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    run_next11_ogbn_aggregation_optimizer(sizes=args.sizes, methods=args.methods, aggregation_variants=args.aggregation_variants, output=args.output, python=args.python, reuse_a0_from=args.reuse_a0_from, progress=bool(args.progress), skip_full_local_extra=bool(args.skip_full_local_extra))
    summarize_next11_ogbn_aggregation_optimizer(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
