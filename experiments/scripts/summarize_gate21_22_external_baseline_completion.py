from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_OUTPUTS = (
    "gate21_22_external_repo_audit.csv",
    "gate21_22_freehgc_standard_protocol.csv",
    "gate21_22_condensation_score_tp_results.csv",
    "gate21_22_condensation_score_selector_results.csv",
    "gate21_22_official_training_runs.csv",
    "gate21_22_training_failures.csv",
    "gate21_22_final_compact_table.csv",
    "gate21_22_final_compact_table.md",
    "gate21_22_all_methods_experiment_table.csv",
    "gate21_22_all_methods_experiment_table.md",
    "gate21_22_best_method_comparison.csv",
    "gate21_22_frontiers.csv",
    "gate21_22_decision_flags.csv",
    "gate21_22_decision.json",
    "gate21_22_external_baseline_status.json",
    "gate21_22_summary.md",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.22 external baseline completion outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = _read_json(input_dir / "gate21_22_decision.json")
    status = _read_json(input_dir / "gate21_22_external_baseline_status.json")
    compact = _read_csv(input_dir / "gate21_22_final_compact_table.csv")
    training = _read_csv(input_dir / "gate21_22_official_training_runs.csv")
    failures = _read_csv(input_dir / "gate21_22_training_failures.csv")
    manifest = [
        {
            "filename": name,
            "exists": (input_dir / name).exists(),
            "bytes": (input_dir / name).stat().st_size if (input_dir / name).exists() else 0,
            "recommended_for_chatgpt": name
            in {
                "gate21_22_final_compact_table.csv",
                "gate21_22_final_compact_table.md",
                "gate21_22_best_method_comparison.csv",
                "gate21_22_all_methods_experiment_table.csv",
                "gate21_22_all_methods_experiment_table.md",
                "gate21_22_decision_flags.csv",
                "gate21_22_decision.json",
                "gate21_22_summary.md",
                "gate21_22_condensation_score_tp_results.csv",
                "gate21_22_condensation_score_selector_results.csv",
            },
        }
        for name in REQUIRED_OUTPUTS
    ]
    _write_csv(output_dir / "gate21_22_summary_manifest.csv", manifest)
    _write_csv(output_dir / "gate21_22_decision_flags_compact.csv", _flag_rows(decision))
    (output_dir / "gate21_22_summary_digest.md").write_text(_digest(decision, status, compact, training, failures, manifest), encoding="utf-8")
    print(f"Gate21.22 summary written to {output_dir}")


def _digest(
    decision: Mapping[str, Any],
    status: Mapping[str, Any],
    compact: list[dict[str, str]],
    training: list[dict[str, str]],
    failures: list[dict[str, str]],
    manifest: list[dict[str, Any]],
) -> str:
    lines = [
        "# Gate21.22 Summary Digest",
        "",
        f"- paper_final_external_baselines_ready: {decision.get('PAPER_FINAL_EXTERNAL_BASELINES_READY')}",
        f"- paper_final_table_ready: {decision.get('PAPER_FINAL_TABLE_READY')}",
        f"- official training rows: {len(training)}",
        f"- official training failures: {len(failures)}",
        f"- compact rows: {len(compact)}",
        f"- proxy success rows: {status.get('proxy_success_rows', '')}",
        "",
        "## Compact Rows",
    ]
    for row in compact:
        lines.append(f"- {row.get('dataset')} | {row.get('row_category')} | {row.get('method')} | micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}")
    lines.extend(["", "## Decision Flags"])
    for row in _flag_rows(decision):
        lines.append(f"- {row['flag']}: {row['value']}")
    lines.extend(["", "## Required Files"])
    for row in manifest:
        lines.append(f"- [{'PASS' if row['exists'] else 'FAIL'}] {row['filename']} ({row['bytes']} bytes)")
    blockers = decision.get("PAPER_FINAL_TABLE_BLOCKERS", [])
    lines.extend(["", "## Blockers"])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _flag_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"flag": key, "value": value}
        for key, value in decision.items()
        if key.isupper() and not isinstance(value, (dict, list))
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
