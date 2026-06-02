from __future__ import annotations

import csv
from pathlib import Path

from experiments.scripts.run_gate21_16_implementation_first_benchmarks import build_arg_parser, run
from hesf_coarsen.eval.official.acm_consistency_export import repair_acm_official_consistency
from hesf_coarsen.eval.official.gate21_16_decision import gate21_16_decision
from hesf_coarsen.eval.official.gate21_16_protocol import GATE21_16_DECISION_FLAGS, GATE21_16_MAIN_FIELDS
from hesf_coarsen.eval.official.imdb_consistency_export import repair_imdb_official_consistency
from hesf_coarsen.eval.official.stage_report_rep_selector import select_gate21_16_representatives


def test_gate21_16_protocol_contains_required_main_fields_and_flags() -> None:
    required_fields = {
        "dataset",
        "method",
        "requested_budget_type",
        "requested_budget",
        "validation_proxy_score",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "success",
        "selected_edge_hash",
        "planner_config_hash",
        "source_path",
        "repo_url",
    }
    required_flags = {
        "ACM_EXPORT_CONSISTENCY_PASS",
        "IMDB_EXPORT_CONSISTENCY_PASS",
        "STRUCTURAL_BASELINES_EXECUTED_BY_DATASET",
        "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET",
        "FREEHGC_SCORE_TP_EXECUTED",
        "HESF_RCS_AUTO_EXECUTED_BY_DATASET",
        "STAGE_REPORT_SMOKE_READY",
        "STAGE_REPORT_QUICK_READY",
    }

    assert required_fields.issubset(set(GATE21_16_MAIN_FIELDS))
    assert required_flags.issubset(set(GATE21_16_DECISION_FLAGS))


def test_rep_selector_uses_validation_proxy_without_test_leakage() -> None:
    rows = [
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural12",
            "requested_budget": 0.12,
            "actual_structural_storage_ratio": 0.119,
            "validation_proxy_score": 0.91,
            "test_micro_f1_mean": 0.99,
            "test_macro_f1_mean": 0.98,
            "training_executed": True,
            "success": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "schema_compatible": True,
            "target_preserving": True,
        },
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural16",
            "requested_budget": 0.16,
            "actual_structural_storage_ratio": 0.159,
            "validation_proxy_score": 0.93,
            "test_micro_f1_mean": 0.95,
            "test_macro_f1_mean": 0.94,
            "training_executed": True,
            "success": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "schema_compatible": True,
            "target_preserving": True,
        },
    ]

    selected = select_gate21_16_representatives(rows, datasets=["DBLP"])

    rep = next(row for row in selected if row["selected_as_rep"])
    assert rep["candidate_method"] == "HeSF-RCS-auto structural16"
    assert rep["selection_source"] == "validation_proxy"
    assert rep["uses_test_for_selection"] is False


def test_consistency_repairs_report_official_loader_preflight(tmp_path: Path) -> None:
    acm_report = repair_acm_official_consistency(tmp_path / "ACM", mode="conservative_locked")
    imdb_report = repair_imdb_official_consistency(tmp_path / "IMDB")

    assert acm_report.dataset == "ACM"
    assert acm_report.mode == "conservative_locked"
    assert acm_report.P_matches_PK is True
    assert acm_report.official_loader_preflight_pass is True
    assert imdb_report.dataset == "IMDB"
    assert imdb_report.MD_DM_reciprocal is True
    assert imdb_report.movie_single_director_constraint_pass is True
    assert imdb_report.official_loader_preflight_pass is True


def test_preflight_runner_emits_required_gate21_16_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "gate21_16_preflight"
    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "DBLP",
            "ACM",
            "IMDB",
            "--mode",
            "preflight",
            "--repair-export",
            "--output",
            str(out_dir),
        ]
    )

    decision = run(args)

    required = [
        "gate21_16_main_official_table.csv",
        "gate21_16_by_dataset_method_budget.csv",
        "gate21_16_hesf_rcs_rep_selection.csv",
        "gate21_16_structural_baseline_results.csv",
        "gate21_16_external_tp_results.csv",
        "gate21_16_freehgc_score_tp_results.csv",
        "gate21_16_freehgc_standard_results.csv",
        "gate21_16_freehgc_protocol_audit.csv",
        "gate21_16_external_repo_audit.csv",
        "gate21_16_acm_consistency_audit.csv",
        "gate21_16_imdb_consistency_audit.csv",
        "gate21_16_budget_match_audit.csv",
        "gate21_16_failure_to_implementation_report.md",
        "gate21_16_decision.json",
        "gate21_16_summary.md",
        "gate21_16_requirement_checklist.md",
    ]
    for name in required:
        assert (out_dir / name).exists(), name

    with (out_dir / "gate21_16_main_official_table.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert set(GATE21_16_MAIN_FIELDS).issubset(reader.fieldnames or [])
    assert {row["dataset"] for row in rows} == {"DBLP", "ACM", "IMDB"}
    assert decision["ACM_EXPORT_CONSISTENCY_PASS"] is True
    assert decision["IMDB_EXPORT_CONSISTENCY_PASS"] is True


def test_decision_requires_quick_dataset_coverage() -> None:
    rows = [
        {
            "dataset": "DBLP",
            "method": "Full-native-SeHGNN",
            "method_family": "full_fidelity_baseline",
            "success": True,
            "training_executed": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "schema_compatible": True,
            "target_preserving": True,
            "test_micro_f1_mean": 0.95,
            "test_macro_f1_mean": 0.94,
        }
    ]

    flags = gate21_16_decision(
        main_rows=rows,
        acm_consistency_rows=[],
        imdb_consistency_rows=[],
        rep_rows=[],
        structural_rows=[],
        external_tp_rows=[],
        freehgc_score_rows=[],
    )

    assert flags["STAGE_REPORT_QUICK_READY"] is False
    assert flags["STRUCTURAL_BASELINES_EXECUTED_BY_DATASET"] is False
    assert flags["EXTERNAL_TP_QUICK_READY_BY_DATASET"] is False
