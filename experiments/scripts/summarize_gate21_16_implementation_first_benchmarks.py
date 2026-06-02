from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hesf_coarsen.eval.official.gate21_16_decision import gate21_16_decision
from hesf_coarsen.eval.official.runner_utils import write_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Gate21.16 decision flags from emitted artifacts.")
    parser.add_argument("--input-dir", default="outputs/gate21_16_quick")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    decision = gate21_16_decision(
        main_rows=_read(in_dir / "gate21_16_main_official_table.csv"),
        acm_consistency_rows=_read(in_dir / "gate21_16_acm_consistency_audit.csv"),
        imdb_consistency_rows=_read(in_dir / "gate21_16_imdb_consistency_audit.csv"),
        rep_rows=_read(in_dir / "gate21_16_hesf_rcs_rep_selection.csv"),
        structural_rows=_read(in_dir / "gate21_16_structural_baseline_results.csv"),
        external_tp_rows=_read(in_dir / "gate21_16_external_tp_results.csv"),
        freehgc_score_rows=_read(in_dir / "gate21_16_freehgc_score_tp_results.csv"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate21_16_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    write_csv(out_dir / "gate21_16_decision_flags.csv", [{"flag": k, "value": json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v} for k, v in decision.items()])
    print(f"Gate21.16 STAGE_REPORT_QUICK_READY={decision['STAGE_REPORT_QUICK_READY']}")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
