from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    add_common_args,
    add_task_and_optional_spectral,
    load_hgb_graph,
    run_multilevel_task_first,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 relation-response activation sanity."))
    parser.add_argument("--method", default="HeSF-TC-P")
    parser.add_argument("--candidate-source", default="target_response_knn")
    parser.add_argument("--pair-delta-mode", default="response_signature")
    parser.add_argument("--lambda-rel-response-values", type=float, nargs="+", default=[0.0, 0.5, 1.0, 5.0, 10.0])
    return parser


def _worker(args: argparse.Namespace, dataset: str, ratio: float, lambda_rel: float, seed: int) -> dict:
    row = {"dataset": dataset, "method": args.method, "ratio": float(ratio), "lambda_rel_response": float(lambda_rel), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_multilevel_task_first(
            original,
            method=str(args.method),
            ratio=float(ratio),
            ratio_mode=str(args.ratio_mode),
            seed=int(seed),
            max_levels=int(args.max_levels),
            per_level_ratio=float(args.per_level_ratio),
            candidate_k=int(args.candidate_k),
            candidate_source=str(args.candidate_source),
            pair_delta_mode=str(args.pair_delta_mode),
            coverage_mode="combined",
            purity_policy="unknown_blocks_known",
            lambda_rel_response=float(lambda_rel),
        )
        row["selected_pair_keys"] = json.dumps(diag.get("levels", [{}])[-1].get("selected_pair_keys", []) if diag.get("levels") else [])
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
        add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
        row["status"] = "success"
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def _selected(row: dict) -> set[str]:
    try:
        return set(json.loads(str(row.get("selected_pair_keys", "[]"))))
    except Exception:
        return set()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [
        (dataset, ratio, lambda_rel, seed)
        for dataset in args.datasets
        for ratio in args.ratios
        for lambda_rel in args.lambda_rel_response_values
        for seed in args.seeds
    ]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "relation_response_activation_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), float(row.get("ratio", 0)), int(row.get("seed", 0)), float(row.get("lambda_rel_response", 0))))
    write_csv(args.output / "relation_response_activation_runs.csv", rows)
    grouped: dict[tuple, dict[float, dict]] = {}
    for row in rows:
        if row.get("status") == "success":
            grouped.setdefault((row.get("dataset"), row.get("ratio"), row.get("seed")), {})[float(row.get("lambda_rel_response", 0.0))] = row
    shifts = []
    for key, by_lambda in grouped.items():
        zero = by_lambda.get(0.0)
        if zero is None:
            continue
        zero_set = _selected(zero)
        for value, row in sorted(by_lambda.items()):
            cur = _selected(row)
            union = len(zero_set | cur)
            overlap = float(len(zero_set & cur) / union) if union else 1.0
            shifts.append({"dataset": key[0], "ratio": key[1], "seed": key[2], "lambda_rel_response": value, "selected_pair_overlap_with_lambda0": overlap, "score_rank_spearman_with_lambda0_proxy": overlap})
    write_csv(args.output / "relation_response_rank_shift.csv", shifts)
    write_summary_md(args.output / "relation_response_summary.md", "Gate13 Relation Response Activation", shifts, ["dataset", "ratio", "seed", "lambda_rel_response", "selected_pair_overlap_with_lambda0"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
