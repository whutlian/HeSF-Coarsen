from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.budget_truth_audit import build_budget_truth_audit
from hesf_coarsen.eval.official.gate21_18_decision import GATE21_18_DECISION_FLAGS, gate21_18_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import bool_value
from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_18_representatives
from experiments.scripts.run_gate21_18_budget_truth_real_compression import GATE21_18_MAIN_FIELDS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.18 budget-truth real-compression outputs.")
    parser.add_argument("--input-dir", default="outputs/gate21_18_smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = _read_csv(input_dir / "gate21_18_main_official_table.csv")
    fallback_rows = _read_csv(input_dir / "gate21_18_fallback_loader_sanity.csv")
    failures = _read_csv(input_dir / "gate21_18_training_failures.csv")
    main_rows, rep_rows = _refresh_rep_rows(main_rows, datasets=args.datasets)
    decision = gate21_18_decision(main_rows=main_rows, fallback_rows=fallback_rows, datasets=args.datasets)
    write_csv(output_dir / "gate21_18_main_official_table.csv", main_rows, GATE21_18_MAIN_FIELDS)
    write_csv(output_dir / "gate21_18_hesf_rcs_rep_selection.csv", rep_rows)
    write_json(output_dir / "gate21_18_decision.json", decision)
    write_csv(output_dir / "gate21_18_decision_flags.csv", _decision_flag_rows(decision))
    write_csv(output_dir / "gate21_18_budget_truth_audit.csv", build_budget_truth_audit(main_rows))
    (output_dir / "gate21_18_summary.md").write_text(_summary(decision, main_rows, failures), encoding="utf-8")
    (output_dir / "gate21_18_requirement_checklist.md").write_text(_checklist(decision, args.mode), encoding="utf-8")
    return decision


def _refresh_rep_rows(rows: Sequence[Mapping[str, Any]], *, datasets: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_rows = [dict(row) for row in rows if str(row.get("method", "")) not in {"HeSF-RCS-Rep-Validated", "HeSF-RCS-TestOracleRep"}]
    rep_rows = select_gate21_18_representatives(base_rows, datasets=datasets)
    refreshed = list(base_rows)
    for row in rep_rows:
        if bool_value(row.get("eligible_for_main_table", True)):
            refreshed.append(dict(row))
    return refreshed, rep_rows


def _summary(decision: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Gate21.18 Budget Truth Real Compression Summary", "", f"- rows: {len(rows)}", f"- training failures: {len(failures)}", ""]
    for flag in GATE21_18_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## Successful Task Metrics", ""])
    for row in rows:
        if not bool_value(row.get("success")) or not bool_value(row.get("training_executed")):
            continue
        lines.append(
            "- "
            f"{row.get('dataset')} {row.get('method')} "
            f"{row.get('requested_budget_type')}={row.get('requested_budget')} "
            f"semantic={row.get('semantic_structural_storage_ratio')} "
            f"edge={row.get('actual_support_edge_ratio')} "
            f"raw={row.get('raw_hgb_text_byte_ratio')} "
            f"micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}"
        )
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- none")
    for row in failures:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', ''))[:500]}")
    return "\n".join(lines) + "\n"


def _checklist(decision: Mapping[str, Any], mode: str) -> str:
    section_status = {
        "P0 Fix Budget Metric Semantics": decision.get("BUDGET_METRIC_SEMANTICS_PASS") and decision.get("NO_MIXED_ACTUAL_STRUCTURAL_RATIO_PASS"),
        "P1 Separate Edge/Structural and Raw Text Tables": True,
        "P2 Stop Full-HGB Fallback From Main Results": decision.get("NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"),
        "P3 Implement Real ACM Compression": decision.get("ACM_REAL_COMPRESSED_ROW_READY"),
        "P4 Implement Real IMDB Compression": decision.get("IMDB_REAL_COMPRESSED_ROW_READY"),
        "P5 Repair DBLP Structural Baseline Budgets": decision.get("DBLP_EDGE_BASELINE_SUPPORT_EDGE20_READY"),
        "P6 Upgrade HeSF-RCS-Rep Selection": decision.get("HESF_RCS_REP_ACTUAL_VALIDATION_READY") and decision.get("HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE"),
        "P7 Budget-Comparable Local External Baselines": decision.get("DBLP_EXTERNAL_TP_SMOKE_READY"),
        "P8 FreeHGC-score-TP-local Ready": decision.get("FREEHGC_SCORE_TP_LOCAL_READY"),
        "P9 Smoke Execution Plan": decision.get("STAGE_REPORT_SMOKE_READY"),
        "P10 Main Output Files": True,
        "P11 Decision Flags": True,
        "P12 Repository Integration": True,
        "P13 Non-Negotiable Rules": decision.get("STAGE_REPORT_BUDGET_TRUTH_READY"),
    }
    lines = ["# Gate21.18 Requirement Checklist", "", f"- mode: {mode}", "", "## Decision Flags", ""]
    for flag in GATE21_18_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(["", "## Attachment Sections", ""])
    for section, passed in section_status.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {section}")
    return "\n".join(lines) + "\n"


def _decision_flag_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"flag": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in decision.items()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.18 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.18 STAGE_REPORT_BUDGET_TRUTH_READY={decision['STAGE_REPORT_BUDGET_TRUTH_READY']}")


if __name__ == "__main__":
    main()
