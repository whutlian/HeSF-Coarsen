from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.run_gate21_21_final_rep_external_baselines import _checklist, _summary
from hesf_coarsen.eval.official.gate21_21_decision import GATE21_21_DECISION_FLAGS, decision_flag_rows, gate21_21_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import normalize_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.21 final representative/external-baseline outputs.")
    parser.add_argument("--results", default=str(ROOT / "results" / "gate21_21_final_rep_external_baselines_quick"))
    parser.add_argument("--out", default="")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [normalize_dataset(item) for item in args.datasets]

    main_rows = _read_csv(results / "gate21_21_main_official_table.csv")
    rep_rows = _read_csv(results / "gate21_21_rep_selection.csv")
    compact_rows = _read_csv(results / "gate21_21_final_compact_table.csv")
    frontier_rows = _read_csv(results / "gate21_21_frontiers.csv")
    external_repo_rows = _read_csv(results / "gate21_21_external_repo_audit.csv")
    freehgc_standard_rows = _read_csv(results / "gate21_21_freehgc_standard.csv")
    freehgc_tp_rows = _read_csv(results / "gate21_21_freehgc_score_tp_local.csv")
    freehgc_selector_rows = _read_csv(results / "gate21_21_freehgc_score_selector.csv")
    acm_overlap_rows = _read_csv(results / "gate21_21_acm_selector_overlap.csv")
    imdb_planner_rows = _read_csv(results / "gate21_21_imdb_channel_planner.csv")
    hgcond_gcond_rows = _read_csv(results / "gate21_21_hgcond_gcond_score_tp.csv")
    failures = _read_csv(results / "gate21_21_training_failures.csv")

    decision = gate21_21_decision(
        main_rows=main_rows,
        rep_rows=rep_rows,
        compact_rows=compact_rows,
        frontier_rows=frontier_rows,
        external_repo_rows=external_repo_rows,
        freehgc_standard_rows=freehgc_standard_rows,
        freehgc_tp_rows=freehgc_tp_rows,
        freehgc_selector_rows=freehgc_selector_rows,
        acm_overlap_rows=acm_overlap_rows,
        imdb_planner_rows=imdb_planner_rows,
        datasets=datasets,
    )

    write_csv(out_dir / "gate21_21_decision_flags.csv", decision_flag_rows(decision))
    write_json(out_dir / "gate21_21_decision.json", decision)
    summary_args = argparse.Namespace(mode="summary", reuse_gate21_20=True, out=str(results))
    (out_dir / "gate21_21_summary.md").write_text(_summary(decision, rep_rows, compact_rows, external_repo_rows, failures, summary_args), encoding="utf-8")
    (out_dir / "gate21_21_requirement_checklist.md").write_text(
        _checklist(decision, results, rep_rows, acm_overlap_rows, imdb_planner_rows, freehgc_selector_rows, hgcond_gcond_rows),
        encoding="utf-8",
    )
    return decision


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.21 FINAL_COMPACT_TABLE_READY={decision['FINAL_COMPACT_TABLE_READY']}")
    print(f"Gate21.21 PAPER_FINAL_TABLE_READY={decision['PAPER_FINAL_TABLE_READY']}")
    for flag in GATE21_21_DECISION_FLAGS:
        print(f"{flag}={decision.get(flag)}")


if __name__ == "__main__":
    main()
