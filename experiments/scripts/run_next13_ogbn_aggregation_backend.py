from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot, write_csv
from experiments.scripts.next11_common import read_json
from experiments.scripts.run_next9_ogbn_aggregation_benchmark import METHOD_LAMBDA_SPEC, SIZE_INPUTS
from experiments.scripts.summarize_next13_ogbn_aggregation_backend import summarize_next13_ogbn_aggregation_backend


BACKENDS: dict[str, dict[str, Any]] = {
    "A0_current_sort_reducer": {"aggregation_reducer": "sort", "aggregation_chunk_size": 1_000_000},
    "A4_local_prededup_sort_reducer": {"aggregation_reducer": "local_prededup_sort", "aggregation_chunk_size": 1_000_000},
}


def _load_config(root: Path) -> dict[str, Any]:
    with (root / "configs/paper/ogbn_mag_next9_opt_aggregation.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_name(size: str, method: str, backend: str) -> str:
    return f"next13_ogbn_backend_{backend}_{size}_{method.replace('-', '_')}"


def _prepare_config(root: Path, run_dir: Path, method: str, backend: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(_load_config(root))
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["aggregation_backend"] = backend
    config.setdefault("coarsening", {})["max_levels"] = 1
    config.setdefault("coarsening", {})["aggregation_reducer"] = spec["aggregation_reducer"]
    config.setdefault("coarsening", {})["aggregation_chunk_size"] = int(spec["aggregation_chunk_size"])
    config.setdefault("scoring", {})["lambda_spec"] = METHOD_LAMBDA_SPEC[method]
    config.setdefault("scoring", {})["lambda_conv"] = 0.0
    config.setdefault("scoring", {})["lambda_rel"] = 0.0
    config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
    config.setdefault("diagnostics", {})["enable_relation_diagnostics"] = False
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


def _rows_from_run(run_dir: Path, size: str, method: str, backend: str, status: str, reason: str = ""):
    diagnostics = _latest_diagnostics(run_dir)
    agg = diagnostics.get("aggregation", {}) if isinstance(diagnostics.get("aggregation"), Mapping) else {}
    envelope = diagnostics.get("large_graph_envelope", {}) if isinstance(diagnostics.get("large_graph_envelope"), Mapping) else {}
    rels = agg.get("aggregation_by_relation", []) if isinstance(agg.get("aggregation_by_relation"), list) else []
    total = agg.get("aggregation_total_sec", "")
    input_edges = sum(int(rel.get("original_edges", 0) or 0) for rel in rels if isinstance(rel, Mapping))
    ok = status == "available" and all(float(rel.get("edge_weight_abs_error", 0) or 0) <= 1.0e-5 for rel in rels if isinstance(rel, Mapping))
    run = {
        "size": size,
        "method": method,
        "backend": backend,
        "run_status": status,
        "reason": reason,
        "aggregation_total_sec": total,
        "input_edges": input_edges,
        "coarse_edges": sum(int(rel.get("coarse_edges_after_dedup", 0) or 0) for rel in rels if isinstance(rel, Mapping)),
        "edges_per_sec": input_edges / float(total) if total not in {None, ""} and float(total) > 0 and input_edges else "",
        "peak_rss_gb": float(envelope.get("process_rss_bytes", 0)) / (1024**3) if envelope.get("process_rss_bytes") else "",
        "correctness_passed": "true" if ok else "false",
        "edge_weight_preservation_checks": "passed" if ok else "failed",
        "run_dir": str(run_dir),
    }
    timing_keys = [
        "exclusive_relation_loop_compute_sec",
        "exclusive_assignment_map_sec",
        "exclusive_key_build_sec",
        "exclusive_sort_sec",
        "exclusive_reduce_sec",
        "exclusive_shard_write_sec",
        "exclusive_kway_merge_sec",
        "exclusive_output_write_sec",
        "total_aggregation_sec",
        "timing_inclusive_fields_present",
        "exclusive_timing_sum_sec",
        "exclusive_timing_residual_sec",
    ]
    timing = {"size": size, "method": method, "backend": backend, **{key: agg.get(key, "") for key in timing_keys}}
    by_relation = [{**rel, "size": size, "method": method, "backend": backend} for rel in rels if isinstance(rel, Mapping)]
    check = {"size": size, "method": method, "backend": backend, "correctness_passed": run["correctness_passed"], "edge_weight_preservation_checks": run["edge_weight_preservation_checks"], "reason": reason}
    return run, timing, by_relation, check


def run_next13_ogbn_aggregation_backend(*, sizes: Sequence[str], methods: Sequence[str], backends: Sequence[str], output: Path, python: str, progress: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    root = repo_root()
    for backend in backends:
        spec = BACKENDS[backend]
        for method in methods:
            for size in sizes:
                run_name = _run_name(size, method, backend)
                run_dir = output / "runs" / run_name
                config = _prepare_config(root, run_dir, method, backend, spec)
                write_config_snapshot(run_dir / "config.yaml", config)
                command = [python, "-m", "hesf_coarsen.cli.main", "coarsen", "--config", str(run_dir / "config.yaml"), "--input", str(SIZE_INPUTS[size]), "--output", str(run_dir)]
                write_command_metadata(run_dir, run_name=run_name, command=command, status="running", size=size, method=method, backend=backend, experiment_block="next13_ogbn_aggregation_backend")
                completed = run_subprocess_with_log(command, cwd=root, log_path=run_dir / "coarsen.log", stream_output=progress)
                status = "available" if completed.returncode == 0 else "failed"
                reason = "" if completed.returncode == 0 else f"returncode={completed.returncode}"
                write_command_metadata(run_dir, run_name=run_name, command=command, status="success" if completed.returncode == 0 else "failed", returncode=completed.returncode, size=size, method=method, backend=backend, experiment_block="next13_ogbn_aggregation_backend")
                run, timing, rel_rows, check = _rows_from_run(run_dir, size, method, backend, status, reason)
                runs.append(run)
                timings.append(timing)
                rels.extend(rel_rows)
                checks.append(check)
                write_csv(output / "aggregation_backend_runs.csv", runs)
                write_csv(output / "aggregation_backend_exclusive_timing.csv", timings)
                write_csv(output / "aggregation_backend_by_relation.csv", rels)
                write_csv(output / "aggregation_backend_correctness_checks.csv", checks)
    write_csv(output / "aggregation_backend_runs.csv", runs)
    write_csv(output / "aggregation_backend_exclusive_timing.csv", timings)
    write_csv(output / "aggregation_backend_by_relation.csv", rels)
    write_csv(output / "aggregation_backend_correctness_checks.csv", checks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["200k", "500k", "1m", "full-local"])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S"])
    parser.add_argument("--backends", nargs="+", default=["A0_current_sort_reducer", "A4_local_prededup_sort_reducer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    run_next13_ogbn_aggregation_backend(sizes=args.sizes, methods=args.methods, backends=args.backends, output=args.output, python=args.python, progress=bool(args.progress))
    summarize_next13_ogbn_aggregation_backend(input=args.output, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
