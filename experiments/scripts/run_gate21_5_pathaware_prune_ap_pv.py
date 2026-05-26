from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.summarize_gate21_5_directed_apv import summarize_gate21_5
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


SPECS = [
    "AP50-PA00-PV100-VP00-PTTP00",
    "AP30-PA00-PV100-VP00-PTTP00",
    "AP10-PA00-PV100-VP00-PTTP00",
    "AP100-PA00-PV50-VP00-PTTP00",
    "AP100-PA00-PV30-VP00-PTTP00",
    "AP100-PA00-PV10-VP00-PTTP00",
    "AP50-PA00-PV50-VP00-PTTP00",
    "AP30-PA00-PV30-VP00-PTTP00",
]

STRATEGIES = [
    "random_edge_within_relation",
    "degree_within_relation",
    "pathaware_v2_topk",
    "pathaware_v2_stratified",
    "coverage_sampler",
]

RAW_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "relation_channel_spec",
    "edge_score_strategy",
    "graph_seed",
    "training_seed",
    "semantic_structural_storage_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "success",
    "status",
    "failed_reason",
]

COVERAGE_FIELDS = [
    "dataset",
    "method",
    "relation_channel_spec",
    "edge_score_strategy",
    "graph_seed",
    "num_isolated_target_authors_after_pruning",
    "fraction_target_authors_with_AP_edge",
    "fraction_papers_with_PV_edge",
    "fraction_retained_papers_connected_to_venue",
    "coverage_score_mean",
    "coverage_score_std",
    "hub_penalty_mean",
    "edge_jaccard_across_graph_seeds",
]

SCORE_FIELDS = [
    "dataset",
    "method",
    "relation_channel_spec",
    "edge_score_strategy",
    "graph_seed",
    "relation_name",
    "edge_score_quantiles_by_relation",
]


def _method(spec: str, strategy: str) -> str:
    suffix = {
        "random_edge_within_relation": "random",
        "degree_within_relation": "degree",
        "pathaware_v2_topk": "pathaware-v2-topk",
        "pathaware_v2_stratified": "pathaware-v2-stratified",
        "coverage_sampler": "coverage",
    }[strategy]
    return f"H6-dirskel-{spec}-{suffix}"


def _raw_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        for strategy in STRATEGIES:
            method = _method(spec, strategy)
            for graph_seed in args.graph_seeds:
                for training_seed in args.training_seeds:
                    rows.append(
                        {
                            "dataset": str(args.dataset).upper(),
                            "method": method,
                            "canonical_method": method,
                            "relation_channel_spec": spec,
                            "edge_score_strategy": strategy,
                            "graph_seed": int(graph_seed),
                            "training_seed": int(training_seed),
                            "success": False,
                            "status": "planned" if args.dry_run else "pending_pathaware_not_executed",
                            "failed_reason": "",
                        }
                    )
    return rows


def _by_method(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(groups.items()):
        micros = [float(row["test_micro_f1"]) for row in group if str(row.get("test_micro_f1", "")) not in {"", "nan"}]
        first = group[0]
        out.append(
            {
                "dataset": first.get("dataset", "DBLP"),
                "method": method,
                "canonical_method": first.get("canonical_method", method),
                "relation_channel_spec": first.get("relation_channel_spec", ""),
                "edge_score_strategy": first.get("edge_score_strategy", ""),
                "runs": len(group),
                "success_count": sum(1 for row in group if str(row.get("status")) == "success"),
                "mean_test_micro_f1": float(mean(micros)) if micros else "",
                "std_test_micro_f1": float(pstdev(micros)) if len(micros) > 1 else (0.0 if micros else ""),
                "pathaware_beats_random_by_005": "",
                "diagnostic_only": True,
            }
        )
    return out


def _diagnostics(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for spec in SPECS:
        for strategy in STRATEGIES:
            for graph_seed in args.graph_seeds:
                coverage.append(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": _method(spec, strategy),
                        "relation_channel_spec": spec,
                        "edge_score_strategy": strategy,
                        "graph_seed": int(graph_seed),
                        "num_isolated_target_authors_after_pruning": "",
                        "fraction_target_authors_with_AP_edge": "",
                        "fraction_papers_with_PV_edge": "",
                        "fraction_retained_papers_connected_to_venue": "",
                        "coverage_score_mean": "",
                        "coverage_score_std": "",
                        "hub_penalty_mean": "",
                        "edge_jaccard_across_graph_seeds": "",
                    }
                )
                scores.append(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": _method(spec, strategy),
                        "relation_channel_spec": spec,
                        "edge_score_strategy": strategy,
                        "graph_seed": int(graph_seed),
                        "relation_name": "",
                        "edge_score_quantiles_by_relation": "",
                    }
                )
    return coverage, scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if args.force and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _raw_rows(args)
    coverage, scores = _diagnostics(args)
    write_csv(out / "gate21_5_ap_pv_pruning_raw_rows.csv", rows, fieldnames=RAW_FIELDS)
    write_csv(out / "gate21_5_ap_pv_pruning_by_method.csv", _by_method(rows))
    write_csv(out / "gate21_5_edge_score_diagnostics.csv", scores, fieldnames=SCORE_FIELDS)
    write_csv(out / "gate21_5_coverage_diagnostics.csv", coverage, fieldnames=COVERAGE_FIELDS)
    write_json(out / "gate21_5_pathaware_prune_plan.json", {"dataset": str(args.dataset).upper(), "raw_rows": len(rows), "dry_run": bool(args.dry_run), "note": "Gate21.5 keeps path-aware AP/PV pruning as diagnostic unless a real run populates metrics."})
    summarize_gate21_5(out, out, native_full_micro=0.9533802, native_full_macro=0.9498198, write_md=True, write_json_flag=True)
    return {"dry_run": bool(args.dry_run), "pathaware_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--out-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.graph_seeds = list(args.graph_seeds[:1])
        args.training_seeds = list(args.training_seeds[:1])
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
