from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.run_gate21_20_final_stage_table import _checklist, _decision_flag_rows, _summary
from hesf_coarsen.eval.official.final_stage_report_tables import (
    BEST_METHOD_COMPARISON_FIELDS,
    FRONTIER_FIELDS,
    build_best_method_comparison,
    build_frontier_rows,
)
from hesf_coarsen.eval.official.critical_robustness_runner import ROBUSTNESS_FIELDS, build_critical_robustness_rows
from hesf_coarsen.eval.official.gate21_20_decision import gate21_20_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import normalize_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.20 final-stage table outputs.")
    parser.add_argument("--in-dir", default=str(ROOT / "outputs" / "gate21_20_final_stage"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick-robust"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [normalize_dataset(item) for item in args.datasets]
    main_rows = _read_csv(in_dir / "gate21_20_main_official_table.csv")
    rep_rows = _read_csv(in_dir / "gate21_20_rep_selection.csv")
    training_runs = _read_csv(in_dir / "gate21_20_training_runs.csv")
    robustness_rows = build_critical_robustness_rows(main_rows, training_runs) if training_runs else _read_csv(in_dir / "gate21_20_robustness_by_method.csv")
    acm_overlap_rows = _read_csv(in_dir / "gate21_20_acm_selector_overlap.csv")
    imdb_upgrade_rows = _read_csv(in_dir / "gate21_20_imdb_planner_upgrade.csv")
    freehgc_selector_rows = _read_csv(in_dir / "gate21_20_freehgc_score_selector.csv")
    failure_rows = _read_csv(in_dir / "gate21_20_training_failures.csv")

    best_rows = build_best_method_comparison(main_rows, rep_rows=rep_rows, datasets=datasets)
    frontier_rows = build_frontier_rows(main_rows, datasets=datasets)
    decision = gate21_20_decision(
        main_rows=main_rows,
        rep_rows=rep_rows,
        robustness_rows=robustness_rows,
        acm_overlap_rows=acm_overlap_rows,
        imdb_upgrade_rows=imdb_upgrade_rows,
        freehgc_selector_rows=freehgc_selector_rows,
        datasets=datasets,
    )

    if out_dir != in_dir:
        write_csv(out_dir / "gate21_20_main_official_table.csv", main_rows)
        write_csv(out_dir / "gate21_20_rep_selection.csv", rep_rows)
        write_csv(out_dir / "gate21_20_acm_selector_overlap.csv", acm_overlap_rows)
        write_csv(out_dir / "gate21_20_imdb_planner_upgrade.csv", imdb_upgrade_rows)
        write_csv(out_dir / "gate21_20_freehgc_score_selector.csv", freehgc_selector_rows)
        write_csv(out_dir / "gate21_20_training_failures.csv", failure_rows)
    write_csv(out_dir / "gate21_20_robustness_by_method.csv", robustness_rows, ROBUSTNESS_FIELDS)
    write_csv(out_dir / "gate21_20_best_method_comparison.csv", best_rows, BEST_METHOD_COMPARISON_FIELDS)
    write_csv(out_dir / "gate21_20_frontiers.csv", frontier_rows, FRONTIER_FIELDS)
    write_csv(out_dir / "gate21_20_decision_flags.csv", _decision_flag_rows(decision))
    write_json(out_dir / "gate21_20_decision.json", decision)
    (out_dir / "gate21_20_summary.md").write_text(_summary(decision, main_rows, rep_rows, robustness_rows, failure_rows), encoding="utf-8")
    (out_dir / "gate21_20_requirement_checklist.md").write_text(_checklist(decision, rep_rows, robustness_rows, failure_rows, str(args.mode)), encoding="utf-8")
    return decision


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.20 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.20 STAGE_REPORT_QUICK_ROBUSTNESS_READY={decision['STAGE_REPORT_QUICK_ROBUSTNESS_READY']}")
    print(f"Gate21.20 STAGE_REPORT_FINAL_TABLE_READY={decision['STAGE_REPORT_FINAL_TABLE_READY']}")


if __name__ == "__main__":
    main()
