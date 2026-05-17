from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.run_next12_metapath_retention import _coarse_for_method
from experiments.scripts.summarize_next13_metapath_mass import summarize_next13_metapath_mass
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.metapath_mass import evaluate_metapath_transition_mass, infer_schema_paths
from hesf_coarsen.io.edge_list import load_graph


def _coarse_for_next13(method: str, dataset: str, seed: int, original, resource_runs: Path, guard_runs: Path):
    if method in {"AH-UGC-style-tuned", "AH-UGC-style tuned-global"}:
        coarse, assignment, diag = coarsen_type_isolated_lsh(
            original,
            target_ratio=0.5,
            seed=int(seed),
            hash_bits=20,
            bucket_topk=4,
            assignment_source="chebheat_sketch",
        )
        return coarse, assignment, {"coarse_source": "ahugc_style_tuned_global", **diag}
    coarse, assignment, diag = _coarse_for_method(method, dataset, int(seed), original, resource_runs, guard_runs)
    return coarse, assignment, diag


def run_next13_metapath_mass(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    schema_path_lengths: Sequence[int],
    num_probes: int,
    max_schema_paths: int,
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        schema_paths = infer_schema_paths(original, lengths=schema_path_lengths, max_paths=int(max_schema_paths))
        for seed in seeds:
            for method in methods:
                try:
                    coarse, assignment, source = _coarse_for_next13(method, dataset, int(seed), original, resource_runs, guard_runs)
                    method_rows = evaluate_metapath_transition_mass(original, coarse, assignment, schema_paths, num_probes=int(num_probes), sample_seed=int(seed), include_untyped_control=True)
                    for row in method_rows:
                        row.update({"dataset": dataset, "seed": int(seed), "method": method, **{k: v for k, v in source.items() if isinstance(v, (str, int, float, bool))}})
                    rows.extend(method_rows)
                    status.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "available", "schema_paths": len(schema_paths), **{k: v for k, v in source.items() if isinstance(v, (str, int, float, bool))}})
                except Exception as exc:
                    status.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "failed", "reason": str(exc), "schema_paths": len(schema_paths)})
                write_csv(output / "metapath_mass_by_run.csv", rows)
                write_csv(output / "run_status.csv", status)
    write_csv(output / "metapath_mass_by_run.csv", rows)
    write_csv(output / "run_status.csv", status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "AH-UGC-style-tuned", "GraphZoom-style", "ConvMatch-style", "random"])
    parser.add_argument("--schema-path-lengths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--max-schema-paths", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next13_metapath_mass(datasets=args.datasets, seeds=args.seeds, methods=args.methods, schema_path_lengths=args.schema_path_lengths, num_probes=args.num_probes, max_schema_paths=args.max_schema_paths, graph_root=args.graph_root, resource_runs=args.resource_runs, guard_runs=args.guard_runs, output=args.output)
    summarize_next13_metapath_mass(input=args.output, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
