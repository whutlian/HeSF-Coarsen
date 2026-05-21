from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, markdown_table


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            if row.get(key) not in {"", None}:
                values.append(float(row[key]))
        except (TypeError, ValueError):
            pass
    return sum(values) / len(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate14 outputs into required report files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-graph", type=Path)
    parser.add_argument("--baselines", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.full_graph:
        _copy(args.full_graph / "full_graph_lite_ceiling_summary.md", out / "full_graph_lite_ceiling_summary.md")
    if args.baselines:
        _copy(args.baselines / "ratio_matched_baseline_summary.md", out / "ratio_matched_baseline_summary.md")
        _copy(args.baselines / "baseline_requested_ratio_table.csv", out / "baseline_requested_ratio_table.csv")
        _copy(args.baselines / "baseline_realized_ratio_table.csv", out / "baseline_realized_ratio_table.csv")
        _copy(args.baselines / "baseline_nearest_ratio_matched_table.csv", out / "baseline_nearest_ratio_matched_table.csv")
        _copy(args.baselines / "baseline_ratio_mismatch_report.md", out / "baseline_ratio_mismatch_report.md")
    by_method = _read_csv(out / "gate14_final_by_method.csv")
    gaps = _read_csv(out / "gate14_ratio_matched_gaps.csv")
    recovery = _read_csv(out / "gate14_recovery_vs_ceiling.csv")
    merge_diag = _read_csv(out / "gate14_merge_diagnostics.csv")
    candidate_diag = _read_csv(out / "gate14_candidate_source_diagnostics.csv")

    def write_md(name: str, title: str, rows: list[dict[str, Any]], columns: list[str], preface: str = "") -> None:
        (out / name).write_text(f"# {title}\n\n{preface}{markdown_table(rows[:40], columns)}\n", encoding="utf-8")

    coverage_rows = [row for row in by_method if "coverage-v2" in row.get("method", "") or row.get("method") in {"HeSF-TC-no-coverage", "HeSF-TC-P-response-static"}]
    purity_rows = [row for row in by_method if "purity-v2" in row.get("method", "") or row.get("method") in {"HeSF-TC-no-purity", "HeSF-TC-P-response-static"}]
    stateful_rows = [row for row in by_method if "stateful-v1" in row.get("method", "") or row.get("method") == "HeSF-TC-P-response-static"]
    write_md("coverage_v2_summary.md", "Coverage V2 Summary", coverage_rows, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "coverage_v2_error_last_mean"])
    write_md("purity_v2_summary.md", "Purity V2 Summary", purity_rows, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "purity_v2_error_last_mean"])
    write_md("stateful_matching_summary.md", "Stateful Matching Summary", stateful_rows, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "stateful_signature_drift_last_mean"])
    write_md("candidate_source_summary.md", "Candidate Source Summary", candidate_diag, ["dataset", "method", "ratio", "seed", "candidate_source", "selected_support_merges"])
    write_md("ratio_matched_baseline_summary.md", "Ratio Matched Baseline Summary", gaps, ["method", "baseline", "comparison_status", "ratio_gap", "delta_macro_f1", "delta_accuracy"])
    write_md("validation_selection_summary.md", "Validation Selection Summary", _read_csv(out / "gate14_validation_selected_test.csv"), ["dataset", "seed", "method", "ratio", "validation_macro_f1", "macro_f1", "accuracy"])
    if not (out / "full_graph_lite_ceiling_summary.md").exists():
        full_rows = [row for row in by_method if row.get("method") == "full-graph-hettree-lite-tuned"]
        write_md("full_graph_lite_ceiling_summary.md", "Full Graph Lite Ceiling Summary", full_rows, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean"])

    final_report = f"""# Gate14 Final Report

Git commit: `{git_commit_hash()}`

## Decision

See `gate14_decision.md`.

## Main Tables

- `gate14_all_runs.csv`
- `gate14_final_by_method.csv`
- `gate14_ratio_matched_gaps.csv`
- `gate14_recovery_vs_ceiling.csv`
- `gate14_validation_selected_test.csv`
- `gate14_oracle_appendix.csv`

## Aggregate Evidence

- Mean ratio-matched macro gap over comparable rows: `{_mean(gaps, 'delta_macro_f1'):.6f}`
- Mean macro recovery vs full graph lite: `{_mean(recovery, 'recovery_vs_full_graph_lite_macro'):.6f}`
- Mean coverage-v2 selected error: `{_mean(merge_diag, 'coverage_v2_error_last'):.6f}`

## Scope

The evaluator remains `diagnostic_lite_only`; official SeHGNN/HETTREE/FreeHGC are not integrated.
"""
    (out / "final_report.md").write_text(final_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
