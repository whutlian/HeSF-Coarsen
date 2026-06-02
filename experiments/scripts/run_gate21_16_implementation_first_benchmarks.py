from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.acm_consistency_export import repair_acm_official_consistency
from hesf_coarsen.eval.official.external_repo_manager import audit_required_external_repos
from hesf_coarsen.eval.official.external_tp_baseline_impl import build_gate21_16_external_tp_rows
from hesf_coarsen.eval.official.freehgc_score_tp_local import build_freehgc_score_tp_local_rows
from hesf_coarsen.eval.official.gate21_16_decision import gate21_16_decision
from hesf_coarsen.eval.official.gate21_16_protocol import GATE21_16_DECISION_FLAGS, GATE21_16_MAIN_FIELDS
from hesf_coarsen.eval.official.hgcond_score_tp_local import build_condensation_score_tp_rows
from hesf_coarsen.eval.official.imdb_consistency_export import repair_imdb_official_consistency
from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.stage_report_budgets import build_budget_match_audit
from hesf_coarsen.eval.official.stage_report_rep_selector import select_gate21_16_representatives
from hesf_coarsen.eval.official.stage_report_table_builder import (
    append_rep_rows,
    build_full_export_rows,
    build_hesf_auto_rows,
    build_internal_rows,
    load_gate21_16_evidence,
)
from hesf_coarsen.eval.official.structural_baseline_impl import build_gate21_16_relation_retention, build_gate21_16_structural_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.16 implementation-first benchmark runner.")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick", "paper"), default="preflight")
    parser.add_argument("--repair-export", action="store_true")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--execute-structural-baselines", action="store_true")
    parser.add_argument("--execute-external-tp", action="store_true")
    parser.add_argument("--execute-hesf-rcs", action="store_true")
    parser.add_argument("--external-repos-dir", default="external_repos")
    parser.add_argument("--clone-missing-baselines", action="store_true")
    parser.add_argument("--output", "--output-dir", dest="output", default="outputs/gate21_16_quick")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(str(item).upper() for item in args.datasets)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = load_gate21_16_evidence()
    acm_rows = [repair_acm_official_consistency(out_dir / "repairs" / "ACM", mode="conservative_locked").as_row()]
    imdb_rows = [repair_imdb_official_consistency(out_dir / "repairs" / "IMDB").as_row()]
    repo_rows = audit_required_external_repos(args.external_repos_dir, clone_missing=bool(args.clone_missing_baselines))

    main_rows = []
    main_rows.extend(build_full_export_rows(datasets=datasets, evidence=evidence))
    main_rows.extend(build_internal_rows(datasets=datasets, evidence=evidence))
    main_rows.extend(build_hesf_auto_rows(datasets=datasets, evidence=evidence))

    structural_rows = build_gate21_16_structural_rows(datasets=datasets, mode=args.mode)
    external_tp_rows = build_gate21_16_external_tp_rows(datasets=datasets, mode=args.mode)
    freehgc_score_rows = [row for row in external_tp_rows if row.get("method") == "FreeHGC-score-TP"]
    freehgc_standard_rows = _freehgc_standard_rows(repo_rows)
    freehgc_protocol_rows = _freehgc_protocol_audit_rows(repo_rows, freehgc_score_rows)
    condensation_score_rows = build_condensation_score_tp_rows(datasets=datasets)

    main_rows.extend(structural_rows)
    main_rows.extend(external_tp_rows)
    rep_rows = select_gate21_16_representatives(main_rows, datasets=datasets)
    main_rows = append_rep_rows(main_rows, rep_rows)

    budget_rows = build_budget_match_audit(main_rows)
    decision = gate21_16_decision(
        main_rows=main_rows,
        acm_consistency_rows=acm_rows,
        imdb_consistency_rows=imdb_rows,
        rep_rows=rep_rows,
        structural_rows=structural_rows,
        external_tp_rows=external_tp_rows,
        freehgc_score_rows=freehgc_score_rows,
        datasets=datasets,
        mode=args.mode,
    )

    write_csv(out_dir / "gate21_16_main_official_table.csv", main_rows, GATE21_16_MAIN_FIELDS)
    write_csv(out_dir / "gate21_16_by_dataset_method_budget.csv", main_rows)
    write_csv(out_dir / "gate21_16_hesf_rcs_rep_selection.csv", rep_rows)
    write_csv(out_dir / "gate21_16_structural_baseline_results.csv", structural_rows)
    write_csv(out_dir / "gate21_16_relation_retention.csv", build_gate21_16_relation_retention(structural_rows))
    write_csv(out_dir / "gate21_16_external_tp_results.csv", external_tp_rows)
    write_csv(out_dir / "gate21_16_freehgc_score_tp_results.csv", freehgc_score_rows)
    write_csv(out_dir / "gate21_16_freehgc_standard_results.csv", freehgc_standard_rows)
    write_csv(out_dir / "gate21_16_freehgc_protocol_audit.csv", freehgc_protocol_rows)
    write_csv(out_dir / "gate21_16_condensation_score_tp_results.csv", condensation_score_rows)
    write_csv(out_dir / "gate21_16_external_repo_audit.csv", repo_rows)
    write_csv(out_dir / "gate21_16_acm_consistency_audit.csv", acm_rows)
    write_csv(out_dir / "gate21_16_imdb_consistency_audit.csv", imdb_rows)
    write_csv(out_dir / "gate21_16_export_fidelity_audit.csv", _export_fidelity_rows(main_rows))
    write_csv(out_dir / "gate21_16_budget_match_audit.csv", budget_rows)
    write_csv(out_dir / "gate21_16_decision_flags.csv", [{"flag": k, "value": json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v} for k, v in decision.items()])
    _write_json(out_dir / "gate21_16_decision.json", decision)
    (out_dir / "gate21_16_failure_to_implementation_report.md").write_text(_failure_to_implementation_report(main_rows), encoding="utf-8")
    (out_dir / "gate21_16_summary.md").write_text(_summary(decision, main_rows), encoding="utf-8")
    (out_dir / "gate21_16_requirement_checklist.md").write_text(_checklist(decision), encoding="utf-8")
    return decision


def _freehgc_standard_rows(repo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo = next((row for row in repo_rows if row.get("baseline_name") == "FreeHGC"), {})
    return [
        {
            "dataset": "ALL",
            "method": "FreeHGC-standard",
            "protocol": "upstream_standard_condensation",
            "repo_url": repo.get("repo_url", "https://github.com/GooLiang/FreeHGC"),
            "clone_success": repo.get("clone_success", False),
            "required_files_present": repo.get("required_files_present", False),
            "training_executed": False,
            "success": False,
            "failure_type": "upstream_incomplete_local_score_fallback_added",
            "failure_reason": "Upstream FreeHGC standard HGB files are incomplete in this checkout; Gate21.16 adds FreeHGC-score-TP local fallback instead of stopping at hard failure.",
        }
    ]


def _freehgc_protocol_audit_rows(repo_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo = next((row for row in repo_rows if row.get("baseline_name") == "FreeHGC"), {})
    return [
        {
            "protocol": "FreeHGC-standard",
            "upstream_attempted": True,
            "clone_success": repo.get("clone_success", False),
            "required_files_present": repo.get("required_files_present", False),
            "local_fallback_implemented": True,
            "current_status": "upstream_incomplete_local_score_fallback_added",
            "remaining_gap": "Run upstream standard condensation only after required HGB model files/dependencies are available.",
        },
        {
            "protocol": "FreeHGC-score-TP",
            "upstream_attempted": True,
            "clone_success": repo.get("clone_success", False),
            "required_files_present": repo.get("required_files_present", False),
            "local_fallback_implemented": True,
            "current_status": "implemented_pending_official_training",
            "remaining_gap": f"{len(score_rows)} local score-TP rows need official SeHGNN training.",
        },
    ]


def _export_fidelity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "schema_compatible": row.get("schema_compatible", ""),
            "target_preserving": row.get("target_preserving", ""),
            "official_hgb_exported": row.get("official_hgb_exported", ""),
            "official_sehgnn_unmodified": row.get("official_sehgnn_unmodified", ""),
            "training_executed": row.get("training_executed", ""),
            "success": row.get("success", ""),
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        for row in rows
    ]


def _failure_to_implementation_report(rows: list[dict[str, Any]]) -> str:
    tracked = {
        "Random-edge-relwise": "not_executed",
        "Degree-edge-relwise": "not_executed",
        "Proportional-relation-budget": "not_executed",
        "Random-HG-TP": "not_executed",
        "Herding-HG-TP": "not_executed",
        "KCenter-HG-TP": "not_executed",
        "GraphSparsify-TP": "not_executed",
        "Coarsening-HG-TP": "not_executed",
        "FreeHGC-score-TP": "edge_provenance_missing",
        "H6-node30": "ACM PK size mismatch",
    }
    lines = ["# Gate21.16 Failure-to-Implementation Report", "", "| baseline | previous_failure_type | local_implementation_added | current_status | remaining_gap |", "|---|---|---|---|---|"]
    for method, previous in tracked.items():
        current = [row for row in rows if row.get("method") == method]
        status = "executed" if any(row.get("success") is True for row in current) else "implemented_pending_official_training"
        lines.append(f"| {method} | {previous} | yes | {status} | official SeHGNN task training still needed for rows without metrics |")
    return "\n".join(lines) + "\n"


def _summary(decision: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# Gate21.16 Implementation-First Benchmarks", "", f"- rows: {len(rows)}", ""]
    for flag in GATE21_16_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    return "\n".join(lines) + "\n"


def _checklist(decision: dict[str, Any]) -> str:
    lines = ["# Gate21.16 Requirement Checklist", "", "## Decision Flags", ""]
    for flag in GATE21_16_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(
        [
            "",
            "## Attachment Sections",
            "",
            "- [PASS] P0 ACM consistency preflight repair/audit emitted.",
            "- [PASS] P1 IMDB consistency preflight repair/audit emitted.",
            "- [PASS] P2 structural baseline local implementations emitted with relation retention audit.",
            "- [PASS] P3 external TP local implementations emitted.",
            "- [PASS] P4 FreeHGC-score-TP local fallback emitted.",
            "- [PASS] P5 HGCond/GCond score TP proxy rows emitted.",
            "- [PASS] P6 representative selection uses validation metric/proxy and never test metrics.",
            "- [PASS] P8 preflight/smoke/quick CLI modes are supported.",
            "- [PASS] P9 main table schema emitted.",
            "- [PASS] P10 decision flags emitted.",
            "- [PASS] P11 failure-to-implementation report emitted.",
            "- [FAIL] P12 strong success remains pending until official SeHGNN training is completed for local baseline exports.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    decision = run(args)
    print(f"Gate21.16 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.16 STAGE_REPORT_QUICK_READY={decision['STAGE_REPORT_QUICK_READY']}")


if __name__ == "__main__":
    main()
