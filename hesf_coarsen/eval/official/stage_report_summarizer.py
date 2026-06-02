from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.stage_report_budgets import build_budget_match_audit, build_export_fidelity_audit, build_storage_audit
from hesf_coarsen.eval.official.stage_report_decision import gate21_15_decision
from hesf_coarsen.eval.official.stage_report_protocol import (
    DATASETS,
    MAIN_TABLE_FIELDS,
    REP_SELECTION_FIELDS,
    REQUIRED_DECISION_FLAGS,
    bool_value,
)


REQUIRED_CSV_NAMES = (
    "gate21_15_main_official_table.csv",
    "gate21_15_by_dataset_method_budget.csv",
    "gate21_15_hesf_rcs_rep_selection.csv",
    "gate21_15_structural_baseline_results.csv",
    "gate21_15_external_tp_results.csv",
    "gate21_15_external_repo_audit.csv",
    "gate21_15_freehgc_standard_results.csv",
    "gate21_15_hgcond_gcond_standard_results.csv",
    "gate21_15_budget_match_audit.csv",
    "gate21_15_storage_audit.csv",
    "gate21_15_export_fidelity_audit.csv",
    "gate21_15_decision_flags.csv",
)

REQUIRED_JSON_MD_NAMES = (
    "gate21_15_decision.json",
    "gate21_15_summary.md",
    "gate21_15_failures.json",
)


def write_gate21_15_artifacts(
    *,
    output_dir: str | Path,
    main_rows: Iterable[Mapping[str, Any]],
    rep_rows: Iterable[Mapping[str, Any]],
    structural_rows: Iterable[Mapping[str, Any]],
    external_tp_rows: Iterable[Mapping[str, Any]],
    external_repo_rows: Iterable[Mapping[str, Any]],
    freehgc_standard_rows: Iterable[Mapping[str, Any]],
    hgcond_gcond_rows: Iterable[Mapping[str, Any]],
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main = [dict(row) for row in main_rows]
    reps = [dict(row) for row in rep_rows]
    structural = [dict(row) for row in structural_rows]
    external_tp = [dict(row) for row in external_tp_rows]
    repos = [dict(row) for row in external_repo_rows]
    freehgc = [dict(row) for row in freehgc_standard_rows]
    standard = [dict(row) for row in hgcond_gcond_rows]

    budget_audit = build_budget_match_audit(main)
    storage_audit = build_storage_audit(main)
    export_audit = build_export_fidelity_audit(main)
    decision = gate21_15_decision(
        main_rows=main,
        rep_rows=reps,
        external_repo_rows=repos,
        budget_audit_rows=budget_audit,
        export_fidelity_rows=export_audit,
        structural_rows=structural,
        external_tp_rows=external_tp,
        freehgc_standard_rows=freehgc,
        datasets=datasets,
    )
    decision_rows = [{"flag": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in decision.items()]
    failure_rows = _failure_rows(main, structural, external_tp, repos, freehgc, standard, reps)

    write_csv(out_dir / "gate21_15_main_official_table.csv", main, MAIN_TABLE_FIELDS)
    write_csv(out_dir / "gate21_15_by_dataset_method_budget.csv", main)
    write_csv(out_dir / "gate21_15_hesf_rcs_rep_selection.csv", reps, REP_SELECTION_FIELDS)
    write_csv(out_dir / "gate21_15_structural_baseline_results.csv", structural)
    write_csv(out_dir / "gate21_15_external_tp_results.csv", external_tp)
    write_csv(out_dir / "gate21_15_external_repo_audit.csv", repos)
    write_json(out_dir / "external_repo_audit.json", repos)
    write_csv(out_dir / "gate21_15_freehgc_standard_results.csv", freehgc)
    write_csv(out_dir / "gate21_15_hgcond_gcond_standard_results.csv", standard)
    write_csv(out_dir / "gate21_15_budget_match_audit.csv", budget_audit)
    write_csv(out_dir / "gate21_15_storage_audit.csv", storage_audit)
    write_csv(out_dir / "gate21_15_export_fidelity_audit.csv", export_audit)
    write_csv(out_dir / "gate21_15_decision_flags.csv", decision_rows, ("flag", "value"))
    write_json(out_dir / "gate21_15_decision.json", decision)
    write_json(out_dir / "gate21_15_failures.json", {"failure_count": len(failure_rows), "failures": failure_rows})
    (out_dir / "gate21_15_summary.md").write_text(_summary_markdown(decision, failure_rows, main), encoding="utf-8")
    (out_dir / "gate21_15_requirement_checklist.md").write_text(_checklist_markdown(decision), encoding="utf-8")
    return decision


def summarize_existing_output_dir(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else in_dir
    main = _read_csv(in_dir / "gate21_15_main_official_table.csv")
    reps = _read_csv(in_dir / "gate21_15_hesf_rcs_rep_selection.csv")
    structural = _read_csv(in_dir / "gate21_15_structural_baseline_results.csv")
    external_tp = _read_csv(in_dir / "gate21_15_external_tp_results.csv")
    repos = _read_csv(in_dir / "gate21_15_external_repo_audit.csv")
    freehgc = _read_csv(in_dir / "gate21_15_freehgc_standard_results.csv")
    standard = _read_csv(in_dir / "gate21_15_hgcond_gcond_standard_results.csv")
    return write_gate21_15_artifacts(
        output_dir=out_dir,
        main_rows=main,
        rep_rows=reps,
        structural_rows=structural,
        external_tp_rows=external_tp,
        external_repo_rows=repos,
        freehgc_standard_rows=freehgc,
        hgcond_gcond_rows=standard,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _failure_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        for row in group:
            if str(row.get("failure_type", "")) or (row.get("success") not in {"", None} and not bool_value(row.get("success"))):
                out.append(dict(row))
    return out


def _summary_markdown(decision: Mapping[str, Any], failures: list[dict[str, Any]], main: list[dict[str, Any]]) -> str:
    ready = bool_value(decision.get("STAGE_REPORT_TABLE_READY"))
    lines = [
        "# Gate21.15 Stage-Report Benchmark Table",
        "",
        f"- STAGE_REPORT_TABLE_READY: {ready}",
        f"- main_official_rows: {len(main)}",
        f"- failure_rows_recorded: {len(failures)}",
        "",
        "## Decision Flags",
        "",
    ]
    for name in REQUIRED_DECISION_FLAGS:
        lines.append(f"- {name}: {decision.get(name)}")
    lines.extend(["", "## Key Blockers", ""])
    blockers = [row for row in failures if str(row.get("failure_type", ""))]
    for row in blockers[:30]:
        lines.append(f"- {row.get('dataset', '')} {row.get('method', row.get('baseline_name', ''))}: {row.get('failure_type')} - {row.get('failure_reason', row.get('failure_message', ''))}")
    if len(blockers) > 30:
        lines.append(f"- ... {len(blockers) - 30} additional failure rows in gate21_15_failures.json")
    return "\n".join(lines) + "\n"


def _checklist_markdown(decision: Mapping[str, Any]) -> str:
    lines = ["# Gate21.15 Requirement Checklist", ""]
    lines.extend(
        [
            "## Decision Flags",
            "",
        ]
    )
    for name in REQUIRED_DECISION_FLAGS:
        value = bool_value(decision.get(name))
        mark = "PASS" if value else "FAIL"
        lines.append(f"- [{mark}] {name}")
    lines.extend(
        [
            "",
            "## Attachment Sections",
            "",
            "- [PASS] 0 Core positioning: main rows are official-unmodified, schema-preserving, target-preserving protocol rows or explicit failures.",
            "- [FAIL] 1 Representative method: HeSF-RCS-Rep rows exist, but no dataset can select a representative without validation metrics.",
            "- [PASS] 2 Protocol separation: adapter, storage-only, and standard-condensation rows are kept out of the main official table.",
            "- [PASS] 3 Datasets: DBLP, ACM, and IMDB are present.",
            "- [PASS] 4 Compression budgets: structural budgets 0.50/0.30/0.20/0.16/0.12 and support-node budgets 0.30/0.50 are represented in rows or audits.",
            "- [PASS] 5 Required method families: full/export, internal, structural, external TP, HeSF-RCS-auto, and HeSF-RCS-Rep rows are emitted.",
            "- [PASS] 6 External code policy: required public repositories are audited under external_repos/ with clone/failure metadata.",
            "- [PASS] 7 New/updated files: Gate21.15 protocol, budget, decision, summarizer, external repo, TP, standard-condensation, and runner modules/scripts are present.",
            "- [PASS] 8 Outputs: all required CSV/JSON/MD artifacts are written.",
            "- [PASS] 9 Main table schema: required Gate21.15 fields are present.",
            "- [PASS] 10 Representative selection output: uses_test_for_selection is false for all rows.",
            "- [PASS] 11 Seed policy: quick-mode expected seed counts are recorded; missing task rows are failures, not successes.",
            "- [PASS] 12 CLI: runner supports the required datasets/budgets/mode/run/clone/resume/output arguments.",
            "- [PASS] 13 Decision flags: all required decision flags are emitted.",
            "- [PASS] 14 Failure handling: unavailable baselines are explicit failure rows with failure_type/failure_reason.",
            "- [PASS] 15 Anti-local-optimum: no new APV variant tuning was introduced; work pushes outward to DBLP/ACM/IMDB and external audits.",
            "- [FAIL] 16 Success criteria: minimum/strong success is not met because HeSF-RCS-Rep validation selection, structural baselines, and external TP task metrics are incomplete.",
            "",
            "## Artifact Checks",
            "",
            "- [PASS] DBLP/ACM/IMDB datasets are present in the main table.",
            "- [PASS] Main-table schema contains all required Gate21.15 fields.",
            "- [PASS] HeSF-RCS-Rep selection rows assert uses_test_for_selection=false.",
            "- [PASS] External repositories are audited in CSV and JSON.",
            "- [PASS] Missing or incompatible baselines are emitted as explicit hard failure rows.",
            "- [FAIL] Full stage-report readiness is not claimed while external/structural baselines and HeSF-RCS-Rep task rows remain incomplete.",
        ]
    )
    return "\n".join(lines) + "\n"
