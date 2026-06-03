from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hesf_coarsen.eval.official.budget_truth_audit import build_budget_truth_audit
from hesf_coarsen.eval.official.gate21_19_decision import gate21_19_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import normalize_dataset
from experiments.scripts.run_gate21_19_multidataset_frontier import (
    GATE21_19_MAIN_FIELDS,
    _by_method_rows,
    _checklist,
    _decision_flag_rows,
    _dblp_extra_core_tp_rows,
    _frontier_rows,
    _read_csv,
    _summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.19 multi-dataset frontier outputs.")
    parser.add_argument("--output", "--output-dir", dest="output", default=str(ROOT / "outputs" / "gate21_19_smoke"))
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output)
    main_rows = _read_csv(out_dir / "gate21_19_main_official_table.csv")
    failure_rows = _read_csv(out_dir / "gate21_19_training_failures.csv")
    extra_rows = _dblp_extra_core_tp_rows(out_dir)
    existing_methods = {(row.get("dataset"), row.get("method")) for row in main_rows}
    for row in extra_rows:
        key = (row.get("dataset"), row.get("method"))
        if key not in existing_methods:
            main_rows.append(row)
            existing_methods.add(key)
    extra_methods = {row.get("method") for row in extra_rows}
    failure_rows = [row for row in failure_rows if row.get("method") not in extra_methods]
    datasets = [normalize_dataset(item) for item in args.datasets]
    decision = gate21_19_decision(main_rows=main_rows, datasets=datasets, mode=str(args.mode))

    write_csv(out_dir / "gate21_19_main_official_table.csv", main_rows, GATE21_19_MAIN_FIELDS)
    write_csv(out_dir / "gate21_19_dataset_frontier_by_method.csv", _frontier_rows(main_rows))
    write_csv(out_dir / "gate21_19_dblp_frontier.csv", _frontier_rows(main_rows, dataset="DBLP"))
    write_csv(out_dir / "gate21_19_acm_closure_frontier.csv", _frontier_rows(main_rows, dataset="ACM"))
    write_csv(out_dir / "gate21_19_imdb_channel_frontier.csv", _frontier_rows(main_rows, dataset="IMDB"))
    write_csv(out_dir / "gate21_19_external_tp_by_method.csv", _by_method_rows([row for row in main_rows if row.get("method_family") == "external_tp_baseline"]))
    write_csv(out_dir / "gate21_19_budget_truth_audit.csv", build_budget_truth_audit(main_rows))
    write_csv(out_dir / "gate21_19_training_failures.csv", failure_rows)
    write_csv(out_dir / "gate21_19_decision_flags.csv", _decision_flag_rows(decision))
    write_json(out_dir / "gate21_19_decision.json", decision)
    (out_dir / "gate21_19_summary.md").write_text(_summary(decision, main_rows, failure_rows), encoding="utf-8")
    (out_dir / "gate21_19_requirement_checklist.md").write_text(_checklist(decision, failure_rows, str(args.mode)), encoding="utf-8")
    return decision


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.19 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.19 STAGE_REPORT_QUICK_ROBUSTNESS_READY={decision['STAGE_REPORT_QUICK_ROBUSTNESS_READY']}")


if __name__ == "__main__":
    main()
