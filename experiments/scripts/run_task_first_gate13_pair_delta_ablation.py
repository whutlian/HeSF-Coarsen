from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    DEFAULT_METRICS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    load_hgb_graph,
    run_multilevel_task_first,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 pair-delta ablation."))
    parser.add_argument("--methods", nargs="+", default=["HeSF-TC-P", "HeSF-TC-S"])
    parser.add_argument("--candidate-sources", nargs="+", default=["target_response_knn", "target_anchor_co_support"])
    parser.add_argument("--pair-delta-modes", nargs="+", default=["local_surrogate", "exact_pair_isolated", "response_signature"])
    parser.add_argument("--exact-candidate-cap", type=int, default=128)
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, source: str, mode: str, seed: int) -> dict:
    row = {"dataset": dataset, "method": method, "ratio": float(ratio), "candidate_source": source, "pair_delta_mode": mode, "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_multilevel_task_first(
            original,
            method=method,
            ratio=float(ratio),
            ratio_mode=str(args.ratio_mode),
            seed=int(seed),
            max_levels=int(args.max_levels),
            per_level_ratio=float(args.per_level_ratio),
            candidate_k=int(args.candidate_k),
            candidate_source=source,
            pair_delta_mode=mode,
            coverage_mode="combined",
            purity_policy="unknown_blocks_known",
            candidate_pair_cap=int(args.exact_candidate_cap) if mode == "exact_pair_isolated" else None,
        )
        if mode == "exact_pair_isolated":
            row["exact_pair_isolated_candidate_cap"] = int(args.exact_candidate_cap)
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


def _selected_set(row: dict) -> set[str]:
    try:
        return set(json.loads(str(row.get("selected_pair_keys", "[]"))))
    except Exception:
        return set()


def _overlap_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        key = (row.get("dataset"), row.get("method"), row.get("ratio"), row.get("candidate_source"), row.get("seed"))
        grouped.setdefault(key, {})[str(row.get("pair_delta_mode"))] = row
    overlaps: list[dict] = []
    ranks: list[dict] = []
    pairs = [("local_surrogate", "response_signature"), ("local_surrogate", "exact_pair_isolated"), ("response_signature", "exact_pair_isolated")]
    for key, by_mode in grouped.items():
        base = {"dataset": key[0], "method": key[1], "ratio": key[2], "candidate_source": key[3], "seed": key[4]}
        for left, right in pairs:
            if left not in by_mode or right not in by_mode:
                continue
            a = _selected_set(by_mode[left])
            b = _selected_set(by_mode[right])
            union = len(a | b)
            jaccard = float(len(a & b) / union) if union else 1.0
            overlaps.append({**base, "left_mode": left, "right_mode": right, "selected_pair_jaccard": jaccard, "left_selected": len(a), "right_selected": len(b)})
            ranks.append({**base, "left_mode": left, "right_mode": right, "score_rank_spearman_proxy": jaccard})
    return overlaps, ranks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [
        (dataset, method, ratio, source, mode, seed)
        for dataset in args.datasets
        for method in args.methods
        for ratio in args.ratios
        for source in args.candidate_sources
        for mode in args.pair_delta_modes
        for seed in args.seeds
    ]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "pair_delta_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), str(row.get("candidate_source")), str(row.get("pair_delta_mode")), int(row.get("seed", 0))))
    write_csv(args.output / "pair_delta_runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "method", "ratio", "candidate_source", "pair_delta_mode"], DEFAULT_METRICS)
    write_csv(args.output / "pair_delta_by_dataset.csv", by_dataset)
    overlaps, ranks = _overlap_rows(rows)
    write_csv(args.output / "selected_pair_overlap.csv", overlaps)
    write_csv(args.output / "pair_delta_rank_correlation.csv", ranks)
    term_rows = [{key: value for key, value in row.items() if key in {"dataset", "method", "ratio", "candidate_source", "pair_delta_mode", "seed"} or "delta_" in key or "score_task_first" in key} for row in rows]
    write_csv(args.output / "pair_delta_term_distributions.csv", term_rows)
    write_summary_md(args.output / "pair_delta_summary.md", "Gate13 Pair Delta Ablation", by_dataset, ["dataset", "method", "ratio", "candidate_source", "pair_delta_mode", "runs", "task.macro_f1_mean", "task.accuracy_mean"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
