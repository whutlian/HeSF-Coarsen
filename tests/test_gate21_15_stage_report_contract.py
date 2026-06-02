from __future__ import annotations

import csv
from pathlib import Path

from experiments.scripts.run_gate21_15_stage_report_table import build_arg_parser, run
from hesf_coarsen.eval.official.external_repo_manager import audit_required_external_repos
from hesf_coarsen.eval.official.stage_report_decision import gate21_15_decision
from hesf_coarsen.eval.official.stage_report_protocol import (
    MAIN_TABLE_FIELDS,
    REQUIRED_DECISION_FLAGS,
    select_hesf_rcs_representatives,
)


def test_hesf_rcs_rep_selection_uses_validation_not_test_metrics() -> None:
    rows = [
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural12",
            "requested_budget": 0.12,
            "actual_structural_storage_ratio": 0.119,
            "raw_hgb_text_byte_ratio": 0.53,
            "validation_micro_f1_mean": 0.91,
            "validation_macro_f1_mean": 0.90,
            "test_micro_f1_mean": 0.99,
            "test_macro_f1_mean": 0.98,
            "selected_edge_hash": "a",
            "planner_config_hash": "pa",
            "eligible_for_main_table": True,
            "training_executed": True,
            "success": True,
        },
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural16",
            "requested_budget": 0.16,
            "actual_structural_storage_ratio": 0.159,
            "raw_hgb_text_byte_ratio": 0.54,
            "validation_micro_f1_mean": 0.92,
            "validation_macro_f1_mean": 0.89,
            "test_micro_f1_mean": 0.93,
            "test_macro_f1_mean": 0.92,
            "selected_edge_hash": "b",
            "planner_config_hash": "pb",
            "eligible_for_main_table": True,
            "training_executed": True,
            "success": True,
        },
    ]

    selected = select_hesf_rcs_representatives(rows, datasets=["DBLP"])

    assert selected[0]["candidate_method"] == "HeSF-RCS-auto structural16"
    assert selected[0]["selected_as_rep"] is True
    assert selected[0]["uses_test_for_selection"] is False
    assert selected[0]["selection_reason"].startswith("validation_micro_f1")


def test_decision_rejects_adapter_diagnostic_and_nan_success_rows() -> None:
    main_rows = [
        {
            "dataset": "DBLP",
            "method": "APV12+RP64",
            "method_family": "adapter",
            "schema_compatible": True,
            "target_preserving": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": False,
            "training_executed": True,
            "success": True,
            "eligible_for_main_table": True,
            "test_micro_f1_mean": "NaN",
            "test_macro_f1_mean": "NaN",
            "failure_type": "",
        }
    ]
    flags = gate21_15_decision(
        main_rows=main_rows,
        rep_rows=[],
        external_repo_rows=[],
        budget_audit_rows=[],
        export_fidelity_rows=[],
    )

    assert flags["NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE"] is False
    assert flags["NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS"] is False
    assert flags["STAGE_REPORT_TABLE_READY"] is False
    assert set(REQUIRED_DECISION_FLAGS).issubset(flags)


def test_dry_run_emits_required_artifacts_and_full_planned_matrix(tmp_path: Path) -> None:
    out_dir = tmp_path / "gate21_15"
    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "DBLP",
            "ACM",
            "IMDB",
            "--mode",
            "dry-run",
            "--output-dir",
            str(out_dir),
        ]
    )

    run(args)

    required = [
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
        "gate21_15_decision.json",
        "gate21_15_summary.md",
        "gate21_15_failures.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), name

    with (out_dir / "gate21_15_main_official_table.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert set(MAIN_TABLE_FIELDS).issubset(reader.fieldnames or [])
        rows = list(reader)

    assert {row["dataset"] for row in rows} == {"DBLP", "ACM", "IMDB"}
    assert any(row["method"] == "HeSF-RCS-Rep" for row in rows)
    assert all(row["training_executed"] == "False" for row in rows)


def test_external_repo_audit_records_missing_repos_without_silent_skip(tmp_path: Path) -> None:
    rows = audit_required_external_repos(tmp_path / "external_repos", clone_missing=False)

    names = {row["baseline_name"] for row in rows}
    assert {"FreeHGC", "HGCond", "GCond", "GCondenser"}.issubset(names)
    assert all("repo_url" in row for row in rows)
    assert all(row["clone_attempted"] is False for row in rows)
    assert all(row["clone_success"] is False for row in rows)
    assert all(row["failure_type"] == "repo_missing" for row in rows)
