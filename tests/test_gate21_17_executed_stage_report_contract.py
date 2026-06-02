from __future__ import annotations

import csv
from pathlib import Path

from experiments.scripts.run_gate21_17_executed_stage_report import build_arg_parser, run
from hesf_coarsen.eval.official.gate21_17_decision import GATE21_17_DECISION_FLAGS, gate21_17_decision
from hesf_coarsen.eval.official.official_training_queue import build_training_queue, verify_hgb_export_dir
from hesf_coarsen.eval.official.stage_report_executor import ensure_type_max_id_coverage, sort_hgb_link_lines
from hesf_coarsen.eval.official.stage_report_table import GATE21_17_MAIN_FIELDS
from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_17_representatives


def test_gate21_17_protocol_contains_required_main_fields_and_flags() -> None:
    required_fields = {
        "dataset",
        "method",
        "method_family",
        "requested_budget_type",
        "requested_budget",
        "actual_structural_storage_ratio",
        "support_node_ratio",
        "support_edge_ratio",
        "raw_hgb_text_byte_ratio",
        "graph_seed_count",
        "training_seed_count",
        "test_micro_f1_mean",
        "test_micro_f1_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "validation_micro_f1_mean",
        "validation_macro_f1_mean",
        "validation_proxy_score",
        "recovery_vs_native_full_micro",
        "recovery_vs_native_full_macro",
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "eligible_for_main_table",
        "success",
        "failure_type",
        "failure_reason",
        "selected_edge_hash",
        "planner_config_hash",
        "source_path",
        "repo_url",
        "stdout_path",
        "stderr_path",
    }
    required_flags = {
        "FULL_NATIVE_READY_BY_DATASET",
        "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
        "ACM_EXPORT_CONSISTENCY_PASS",
        "IMDB_EXPORT_CONSISTENCY_PASS",
        "STRUCTURAL_BASELINES_SMOKE_EXECUTED_BY_DATASET",
        "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET",
        "FREEHGC_SCORE_TP_SMOKE_EXECUTED_BY_DATASET",
        "CONDENSATION_SCORE_TP_SMOKE_EXECUTED_BY_DATASET",
        "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE",
        "HESF_RCS_REP_ACTUAL_VALIDATION_READY",
        "STAGE_REPORT_SMOKE_READY",
        "STAGE_REPORT_QUICK_READY",
        "NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE",
        "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE",
        "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS",
    }

    assert required_fields.issubset(set(GATE21_17_MAIN_FIELDS))
    assert required_flags.issubset(set(GATE21_17_DECISION_FLAGS))


def test_training_queue_selects_eligible_pending_rows_and_verifies_hgb_export(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports" / "DBLP"
    export_dir.mkdir(parents=True)
    for name in ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat"):
        (export_dir / name).write_text("", encoding="utf-8")

    rows = [
        {
            "dataset": "DBLP",
            "method": "Random-edge-relwise",
            "requested_budget_type": "structural_storage_ratio",
            "requested_budget": "0.20",
            "actual_structural_storage_ratio": "0.20",
            "export_dir": str(export_dir),
            "schema_compatible": "true",
            "target_preserving": "true",
            "official_hgb_exported": "true",
            "official_sehgnn_unmodified": "true",
            "training_executed": "false",
            "failure_type": "implemented_pending_official_training",
            "selected_edge_hash": "edge-hash",
            "planner_config_hash": "planner-hash",
        },
        {
            "dataset": "DBLP",
            "method": "Full-native-SeHGNN",
            "schema_compatible": "true",
            "target_preserving": "true",
            "official_hgb_exported": "true",
            "official_sehgnn_unmodified": "true",
            "training_executed": "true",
            "success": "true",
        },
    ]

    queue = build_training_queue(rows, graph_seeds=[1], training_seeds=[2])
    audit = verify_hgb_export_dir(export_dir)

    assert len(queue) == 1
    assert queue[0]["source_row_id"] == 0
    assert queue[0]["graph_seed"] == 1
    assert queue[0]["training_seed"] == 2
    assert audit["export_dir_ready"] is True


def test_rep_selector_prefers_actual_validation_and_marks_test_oracle_diagnostic_only() -> None:
    rows = [
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural12",
            "method_family": "schema_preserving_rcs",
            "success": True,
            "training_executed": True,
            "test_micro_f1_mean": 0.99,
            "test_macro_f1_mean": 0.98,
            "validation_micro_f1_mean": 0.90,
            "validation_macro_f1_mean": 0.89,
        },
        {
            "dataset": "DBLP",
            "method": "HeSF-RCS-auto structural16",
            "method_family": "schema_preserving_rcs",
            "success": True,
            "training_executed": True,
            "test_micro_f1_mean": 0.95,
            "test_macro_f1_mean": 0.94,
            "validation_micro_f1_mean": 0.93,
            "validation_macro_f1_mean": 0.92,
        },
    ]

    reps = select_gate21_17_representatives(rows, datasets=["DBLP"])

    main_rep = next(row for row in reps if row["method"] == "HeSF-RCS-Rep")
    oracle = next(row for row in reps if row["method"] == "HeSF-RCS-TestOracleRep")
    assert main_rep["source_method"] == "HeSF-RCS-auto structural16"
    assert main_rep["selection_source"] == "actual_validation"
    assert main_rep["uses_test_for_selection"] is False
    assert oracle["source_method"] == "HeSF-RCS-auto structural12"
    assert oracle["eligible_for_main_table"] is False
    assert oracle["eligible_for_decision"] is False
    assert oracle["selection_source"] == "test_oracle_diagnostic_only"


def test_decision_rejects_implemented_pending_final_rows() -> None:
    rows = [
        {
            "dataset": "DBLP",
            "method": "Random-edge-relwise",
            "method_family": "relation_structural_baseline",
            "requested_budget_type": "structural_storage_ratio",
            "requested_budget": 0.20,
            "eligible_for_main_table": True,
            "failure_type": "implemented_pending_official_training",
            "success": False,
        }
    ]

    flags = gate21_17_decision(main_rows=rows, datasets=["DBLP", "ACM", "IMDB"], mode="smoke")

    assert flags["NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE"] is False
    assert flags["STAGE_REPORT_SMOKE_READY"] is False


def test_export_coverage_adds_edges_touching_each_type_max_id() -> None:
    selected = ["0\t2\t0\t1.0\n"]
    source = [
        "0\t2\t0\t1.0\n",
        "1\t4\t0\t1.0\n",
        "2\t0\t1\t1.0\n",
        "4\t1\t1\t1.0\n",
    ]
    node_type_by_id = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}

    covered = ensure_type_max_id_coverage(selected, source, node_type_by_id)

    touched = {int(part) for line in covered for part in line.split("\t")[:2]}
    assert 1 in touched
    assert 4 in touched
    assert selected[0] in covered


def test_hgb_link_sort_preserves_relation_id_insertion_order() -> None:
    lines = ["9\t1\t2\t1.0\n", "0\t4\t0\t1.0\n", "1\t3\t1\t1.0\n", "0\t2\t0\t1.0\n"]

    sorted_lines = sort_hgb_link_lines(lines)

    assert [line.split("\t")[2] for line in sorted_lines] == ["0", "0", "1", "2"]


def test_hgb_link_sort_can_follow_source_first_seen_relation_order() -> None:
    lines = ["9\t1\t1\t1.0\n", "0\t4\t0\t1.0\n", "1\t3\t3\t1.0\n"]

    sorted_lines = sort_hgb_link_lines(lines, relation_order=["0", "3", "1"])

    assert [line.split("\t")[2] for line in sorted_lines] == ["0", "3", "1"]


def test_preflight_runner_emits_required_artifacts_and_no_pending_rows(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--mode",
            "preflight",
            "--datasets",
            "DBLP",
            "ACM",
            "IMDB",
            "--output",
            str(tmp_path),
        ]
    )

    decision = run(args)

    required_files = [
        "gate21_17_main_official_table.csv",
        "gate21_17_by_dataset_method_budget.csv",
        "gate21_17_training_queue.csv",
        "gate21_17_training_runs.csv",
        "gate21_17_training_failures.csv",
        "gate21_17_external_tp_runs.csv",
        "gate21_17_external_tp_by_method.csv",
        "gate21_17_external_tp_budget_audit.csv",
        "gate21_17_acm_consistency_audit.csv",
        "gate21_17_imdb_consistency_audit.csv",
        "gate21_17_hesf_rcs_rep_selection.csv",
        "gate21_17_decision.json",
        "gate21_17_summary.md",
        "gate21_17_failure_to_execution_report.md",
        "gate21_17_requirement_checklist.md",
    ]
    for name in required_files:
        assert (tmp_path / name).exists(), name

    with (tmp_path / "gate21_17_main_official_table.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert set(GATE21_17_MAIN_FIELDS).issubset(reader.fieldnames or [])
    assert {row["dataset"] for row in rows if row["eligible_for_main_table"].lower() == "true"} >= {"DBLP", "ACM", "IMDB"}
    assert "implemented_pending_official_training" not in {row["failure_type"] for row in rows}
    assert decision["NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE"] is True
