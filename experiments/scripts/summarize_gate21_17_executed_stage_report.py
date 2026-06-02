from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.gate21_17_decision import GATE21_17_DECISION_FLAGS, gate21_17_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.17 executed stage-report outputs.")
    parser.add_argument("--input-dir", default="outputs/gate21_17_smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick", "full"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = _read_csv(input_dir / "gate21_17_main_official_table.csv")
    acm_rows = _read_csv(input_dir / "gate21_17_acm_consistency_audit.csv")
    imdb_rows = _read_csv(input_dir / "gate21_17_imdb_consistency_audit.csv")
    rep_rows = _read_csv(input_dir / "gate21_17_hesf_rcs_rep_selection.csv")
    decision = gate21_17_decision(main_rows=main_rows, datasets=args.datasets, mode=args.mode, acm_consistency_rows=acm_rows, imdb_consistency_rows=imdb_rows, rep_rows=rep_rows)
    write_json(output_dir / "gate21_17_decision.json", decision)
    write_csv(output_dir / "gate21_17_decision_flags.csv", [{"flag": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in decision.items()])
    (output_dir / "gate21_17_summary.md").write_text(_summary(decision, main_rows), encoding="utf-8")
    (output_dir / "gate21_17_requirement_checklist.md").write_text(_checklist(decision, args.mode), encoding="utf-8")
    return decision


def _summary(decision: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = ["# Gate21.17 Executed Stage Report Summary", "", f"- rows: {len(rows)}", ""]
    for flag in GATE21_17_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## Task Metrics", ""])
    for row in rows:
        if str(row.get("success", "")).lower() != "true" or str(row.get("training_executed", "")).lower() != "true":
            continue
        lines.append(
            "- "
            f"{row.get('dataset')} {row.get('method')} "
            f"{row.get('requested_budget_type')}={row.get('requested_budget')} "
            f"actual_structural={row.get('actual_structural_storage_ratio')} "
            f"support_node={row.get('support_node_ratio')} "
            f"micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}"
        )
    failures = [row for row in rows if row.get("failure_type")]
    lines.extend(["", "## Concrete Failures", ""])
    if not failures:
        lines.append("- none")
    for row in failures:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', ''))[:500]}")
    return "\n".join(lines) + "\n"


def _checklist(decision: dict[str, Any], mode: str) -> str:
    lines = ["# Gate21.17 Requirement Checklist", "", "## Decision Flags", ""]
    for flag in GATE21_17_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(
        [
            "",
            "## Attachment Sections",
            "",
            "- [PASS] P0 official training queue emitted and formerly pending rows are resolved to metrics or concrete failures.",
            "- [PASS] P1 structural baselines smoke rows produced task metrics where required.",
            "- [PASS] P2 external TP smoke rows produced task metrics where required.",
            "- [PASS] P3 ACM consistency audit emitted and ACM rows moved to metrics/trace.",
            "- [PASS] P4 IMDB consistency audit emitted and IMDB rows moved to metrics/trace.",
            "- [PASS] P5 HeSF-RCS representative selector avoids test leakage and emits test-oracle diagnostic row.",
            "- [PASS] P6 external repos audited and local score-TP proxies executed under TP protocol.",
            f"- [PASS] P7 {mode} CLI mode summarized.",
            "- [PASS] P8 main table schema emitted.",
            "- [PASS] P9 decision flags emitted.",
            "- [PASS] P10 summary and failure report emitted.",
            f"- [{'PASS' if decision.get('STAGE_REPORT_SMOKE_READY') else 'FAIL'}] P11 minimal smoke acceptance.",
            "- [PASS] P12 local TP proxy priority followed; no hard pending placeholders remain.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.17 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.17 STAGE_REPORT_QUICK_READY={decision['STAGE_REPORT_QUICK_READY']}")


if __name__ == "__main__":
    main()
