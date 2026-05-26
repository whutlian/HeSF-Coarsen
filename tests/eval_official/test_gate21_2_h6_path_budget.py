from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _tiny_dblp_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    relations = {
        0: RelationAdj(np.array([0, 1, 0]), np.array([2, 3, 3]), np.ones(3, dtype=np.float32), 0, 1, 0),
        1: RelationAdj(np.array([2, 3, 3]), np.array([0, 1, 0]), np.ones(3, dtype=np.float32), 1, 0, 1),
        2: RelationAdj(np.array([2, 3, 2]), np.array([4, 5, 5]), np.ones(3, dtype=np.float32), 1, 2, 2),
        3: RelationAdj(np.array([2, 3, 3]), np.array([6, 7, 7]), np.ones(3, dtype=np.float32), 1, 3, 3),
        4: RelationAdj(np.array([4, 5, 5]), np.array([2, 3, 2]), np.ones(3, dtype=np.float32), 2, 1, 4),
        5: RelationAdj(np.array([6, 7, 7]), np.array([2, 3, 2]), np.ones(3, dtype=np.float32), 3, 1, 5),
    }
    relation_specs = {
        0: RelationSpec(0, "AP", 0, 1),
        1: RelationSpec(1, "PA", 1, 0),
        2: RelationSpec(2, "PT", 1, 2),
        3: RelationSpec(3, "PV", 1, 3),
        4: RelationSpec(4, "TP", 2, 1),
        5: RelationSpec(5, "VP", 3, 1),
    }
    return HeteroGraph(
        num_nodes=8,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features={
            0: np.eye(2, dtype=np.float32),
            1: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            2: np.eye(2, dtype=np.float32),
            3: np.eye(2, dtype=np.float32),
        },
        labels=np.array([0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int64),
    )


def _write_hgb_dir(root: Path, *, link_lines: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "node.dat").write_text("0\t0\t0\t1,0\n1\t1\t1\t0,1\n", encoding="utf-8")
    (root / "link.dat").write_text("".join(f"0\t1\t0\t1.0\n" for _ in range(link_lines)), encoding="utf-8")
    (root / "label.dat").write_text("0\t0\t0\t1\n", encoding="utf-8")
    (root / "label.dat.test").write_text("1\t1\t0\t0\n", encoding="utf-8")
    (root / "info.dat").write_text("DBLP\n", encoding="utf-8")
    (root / "mapping.json").write_text(json.dumps({"sidecar": True}), encoding="utf-8")


def test_storage_audit_separates_structural_ratio_from_raw_hgb_bytes(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.storage_audit import audit_hgb_directory

    native = tmp_path / "native" / "DBLP"
    export = tmp_path / "export" / "DBLP"
    _write_hgb_dir(native, link_lines=8)
    _write_hgb_dir(export, link_lines=2)

    result = audit_hgb_directory(
        dataset="DBLP",
        method="H6-struct50-pathaware",
        seed=1,
        export_dir=export,
        native_full_dir=native,
        semantic_structural_storage_ratio=0.50,
        support_node_ratio=0.30,
        support_edge_ratio=0.25,
        total_node_ratio=0.80,
        total_edge_ratio=0.25,
        structural_budget=0.50,
        raw_byte_budget=0.50,
    )

    assert result.semantic_structural_storage_ratio == 0.50
    assert result.hgb_raw_file_byte_ratio != result.semantic_structural_storage_ratio
    assert result.structural_storage_budget_pass is True
    assert result.raw_hgb_byte_budget_pass == (result.hgb_raw_file_byte_ratio <= 0.50)
    assert result.preprocessed_cache_byte_ratio is None
    assert result.link_dat_bytes < result.native_full_total_bytes
    assert result.metadata_sidecar_bytes > 0


def test_relation_mapping_audit_records_source_and_official_relation_ids() -> None:
    from hesf_coarsen.eval.official.relation_mapping_audit import audit_relation_mapping

    rows = audit_relation_mapping(
        graph=_tiny_dblp_graph(),
        dataset="DBLP",
        method="H6-struct50-pathaware",
        seed=1,
        candidate_edge_counts={0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3},
        retained_edge_counts={0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
    )

    ap = next(row for row in rows if row.source_relation_name == "AP")
    pa = next(row for row in rows if row.source_relation_name == "PA")
    assert ap.official_relation_id == 0
    assert ap.official_relation_name == "AP"
    assert ap.reciprocal_relation_name == "PA"
    assert ap.reciprocal_count_consistent is True
    assert pa.relation_dropped_flag is False


def test_relation_budget_allocator_is_deterministic_and_respects_min_edges() -> None:
    from hesf_coarsen.eval.official.relation_budget_allocator import RelationBudgetAllocator, RelationStats

    stats = [
        RelationStats(i, name, pair, "A", "P", 100, candidate, 1)
        for i, name, pair, candidate in [
            (0, "AP", "AP_PA", 20),
            (1, "PA", "AP_PA", 20),
            (2, "PT", "PT_TP", 10),
            (3, "PV", "PV_VP", 10),
            (4, "TP", "PT_TP", 10),
            (5, "VP", "PV_VP", 10),
        ]
    ]
    allocator = RelationBudgetAllocator()

    first = allocator.allocate(relation_stats=stats, total_edge_budget=24, strategy="random_relationwise", seed=7)
    second = allocator.allocate(relation_stats=stats, total_edge_budget=24, strategy="random_relationwise", seed=7)
    manual = allocator.allocate(
        relation_stats=stats,
        total_edge_budget=24,
        strategy="manual_pair",
        relation_pair_weights={"AP_PA": 0.50, "PT_TP": 0.30, "PV_VP": 0.20},
        seed=7,
    )

    assert [row.actual_edges for row in first] == [row.actual_edges for row in second]
    assert sum(row.actual_edges for row in first) <= 24
    assert all(row.actual_edges >= 1 for row in first)
    ap_budget = sum(row.actual_edges for row in manual if row.relation_pair_name == "AP_PA")
    pv_budget = sum(row.actual_edges for row in manual if row.relation_pair_name == "PV_VP")
    assert ap_budget > pv_budget


def test_path_aware_edge_scorer_reports_no_test_label_usage() -> None:
    from hesf_coarsen.eval.official.path_aware_edge_scorer import PathAwareEdgeScorer

    graph = _tiny_dblp_graph()
    scores, diag = PathAwareEdgeScorer().score_edges(
        dataset="DBLP",
        relation_id=0,
        relation_name="AP",
        src_ids=graph.relations[0].src,
        dst_ids=graph.relations[0].dst,
        graph_context={"node_type": graph.node_type},
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
        labels=graph.labels,
        features_by_type=graph.features,
        seed=3,
    )

    assert scores.shape == (3,)
    assert float(scores.max()) > float(scores.min())
    assert diag.trainval_label_used is True
    assert diag.test_label_used is False
    assert diag.no_test_label_usage is True


def test_gate21_2_runner_dry_run_writes_schema_outputs(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_2_h6_path_budget import main as run_main
    from experiments.scripts.summarize_gate21_2_h6_path_budget import summarize_gate21_2

    out = tmp_path / "gate21_2"
    rc = run_main(
        [
            "--dataset",
            "DBLP",
            "--seeds",
            "1",
            "2",
            "--storage-budgets",
            "0.50",
            "0.30",
            "--budget-strategies",
            "proportional",
            "path_aware",
            "random_relationwise",
            "degree_topk_relationwise",
            "--edge-score-strategies",
            "current_heuristic",
            "path_aware",
            "random",
            "degree",
            "--output-root",
            str(out),
            "--dry-run",
        ]
    )

    assert rc == 0
    required = [
        "gate21_2_plan.json",
        "gate21_2_raw_rows.csv",
        "gate21_2_storage_audit.csv",
        "gate21_2_relation_mapping_audit.csv",
        "gate21_2_relation_edge_retention.csv",
        "gate21_2_edge_score_diagnostics.csv",
        "gate21_2_label_graph_ablation.csv",
        "gate21_2_decision.json",
        "gate21_2_decision.md",
        "gate21_2_requirement_checklist.md",
    ]
    assert all((out / name).exists() for name in required)
    summary = summarize_gate21_2(out, out)
    assert "RAW_HGB_BYTE_STORAGE30_NOT_VALIDATED" in summary["decisions"]
    assert summary["target_only_schema_stub_diagnostic_only"] is True
    raw_text = (out / "gate21_2_raw_rows.csv").read_text(encoding="utf-8")
    assert "error_type" in raw_text
    assert "target-only-schema-stub" in raw_text
