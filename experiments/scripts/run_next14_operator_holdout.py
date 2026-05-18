from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.run_next13_metapath_mass import _coarse_for_next13
from experiments.scripts.summarize_next14_operator_holdout import summarize_next14_operator_holdout
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.holdout_operator_probes import evaluate_holdout_operator_probe
from hesf_coarsen.io.edge_list import load_graph


def _method_for_coarsening(method: str) -> str:
    if method in {"TypedHash-ChebHeat", "TypedHash-ChebHeat tuned-global"}:
        return "AH-UGC-style-tuned"
    return method


def run_next14_operator_holdout(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    probe_dim: int,
    cheb_order: int,
    heat_time: float,
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        for seed in seeds:
            for method in methods:
                try:
                    coarse, assignment, source = _coarse_for_next13(_method_for_coarsening(method), dataset, int(seed), original, resource_runs, guard_runs)
                    if not isinstance(assignment, Assignment):
                        assignment = Assignment(assignment, coarse.node_type)
                    metrics = evaluate_holdout_operator_probe(
                        original,
                        coarse,
                        assignment,
                        dataset=dataset,
                        seed=int(seed),
                        probe_dim=int(probe_dim),
                        cheb_order=int(cheb_order),
                        heat_time=float(heat_time),
                        probe_namespace="holdout_operator_next14",
                    )
                    row = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "method": method,
                        "run_status": "available",
                        **metrics,
                        **{k: v for k, v in source.items() if isinstance(v, (str, int, float, bool))},
                    }
                except Exception as exc:
                    row = {"dataset": dataset, "seed": int(seed), "method": method, "run_status": "failed", "reason": str(exc)}
                rows.append(row)
                write_csv(output / "holdout_operator_runs.csv", rows)
    write_csv(output / "holdout_operator_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "TypedHash-ChebHeat", "GraphZoom-style", "ConvMatch-style", "random"])
    parser.add_argument("--probe-dim", type=int, default=32)
    parser.add_argument("--cheb-order", type=int, default=5)
    parser.add_argument("--heat-time", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next14_operator_holdout(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        probe_dim=args.probe_dim,
        cheb_order=args.cheb_order,
        heat_time=args.heat_time,
        graph_root=args.graph_root,
        resource_runs=args.resource_runs,
        guard_runs=args.guard_runs,
        output=args.output,
    )
    summarize_next14_operator_holdout(input=args.output, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
