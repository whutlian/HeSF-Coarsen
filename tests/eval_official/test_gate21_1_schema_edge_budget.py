from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _tiny_dblp_graph(*, weighted: bool = False) -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    weights = np.array([1.0, 10.0, 100.0], dtype=np.float32) if weighted else np.ones(3, dtype=np.float32)
    relations = {
        0: RelationAdj(np.array([0, 1, 0]), np.array([2, 3, 3]), weights, 0, 1, 0),
        1: RelationAdj(np.array([2, 3, 3]), np.array([0, 1, 0]), weights, 1, 0, 1),
        2: RelationAdj(np.array([2, 3, 2]), np.array([4, 5, 5]), weights, 1, 2, 2),
        3: RelationAdj(np.array([2, 3, 3]), np.array([6, 7, 7]), weights, 1, 3, 3),
        4: RelationAdj(np.array([4, 5, 5]), np.array([2, 3, 2]), weights, 2, 1, 4),
        5: RelationAdj(np.array([6, 7, 7]), np.array([2, 3, 2]), weights, 3, 1, 5),
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
            1: np.eye(2, dtype=np.float32),
            2: np.eye(2, dtype=np.float32),
            3: np.eye(2, dtype=np.float32),
        },
        labels=np.array([0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int64),
    )


def test_schema_stable_edge_budget_prunes_relation_wise_and_preserves_schema() -> None:
    from hesf_coarsen.eval.official.schema_stable_pruning import EdgeBudgetConfig, build_schema_stable_edge_budget_graph

    graph = _tiny_dblp_graph()
    pruned, audit = build_schema_stable_edge_budget_graph(
        graph=graph,
        selected_support_nodes=np.array([2, 3, 4, 5, 6, 7], dtype=np.int64),
        dataset_name="DBLP",
        target_type="A",
        config=EdgeBudgetConfig(
            requested_support_node_ratio=1.0,
            requested_edge_ratio=0.50,
            min_edges_per_relation_fraction=0.01,
            seed=7,
        ),
    )

    assert set(pruned.relations) == set(graph.relations)
    assert all(rel.num_edges >= 1 for rel in pruned.relations.values())
    for type_id in (0, 1, 2, 3):
        max_node = int(np.flatnonzero(pruned.node_type == type_id)[-1])
        assert any(max_node in rel.src.tolist() or max_node in rel.dst.tolist() for rel in pruned.relations.values())
    assert audit["schema_complete"] is True
    assert audit["relation_order_matches_official"] is True
    assert audit["actual_support_edge_ratio"] <= 0.70
    assert {row["relation_name"] for row in audit["relation_retention"]} == {"AP", "PA", "PT", "PV", "TP", "VP"}


def test_storage_budget_uses_total_storage_reference_not_support_node_ratio() -> None:
    from hesf_coarsen.eval.official.schema_stable_pruning import EdgeBudgetConfig, build_schema_stable_edge_budget_graph

    graph = _tiny_dblp_graph()
    pruned, audit = build_schema_stable_edge_budget_graph(
        graph=graph,
        selected_support_nodes=np.array([2, 3, 4, 5, 6, 7], dtype=np.int64),
        dataset_name="DBLP",
        target_type="A",
        config=EdgeBudgetConfig(
            requested_support_node_ratio=1.0,
            requested_storage_ratio=0.65,
            reference_num_nodes=graph.num_nodes,
            reference_num_edges=sum(rel.num_edges for rel in graph.relations.values()),
            min_edges_per_relation_fraction=0.01,
            seed=3,
        ),
    )

    retained_edges = sum(rel.num_edges for rel in pruned.relations.values())
    total_storage_ratio = (pruned.num_nodes + retained_edges) / (graph.num_nodes + sum(rel.num_edges for rel in graph.relations.values()))

    assert audit["actual_total_storage_ratio_vs_full_graph"] == pytest.approx(total_storage_ratio)
    assert total_storage_ratio <= 0.65
    assert audit["actual_support_node_ratio"] == 1.0


def test_target_only_schema_stub_preserves_dblp_loader_schema_but_is_diagnostic() -> None:
    from hesf_coarsen.eval.official.schema_stable_pruning import build_target_only_schema_stub_graph

    graph = _tiny_dblp_graph()
    stub, audit = build_target_only_schema_stub_graph(graph=graph, dataset_name="DBLP", target_type="A")

    assert set(int(v) for v in np.unique(stub.node_type)) == {0, 1, 2, 3}
    assert set(stub.relations) == {0, 1, 2, 3, 4, 5}
    assert all(rel.num_edges >= 1 for rel in stub.relations.values())
    target_max = int(np.flatnonzero(stub.node_type == 0)[-1])
    assert any(target_max in rel.src.tolist() or target_max in rel.dst.tolist() for rel in stub.relations.values())
    assert audit["method_family"] == "schema_stub_diagnostic"
    assert audit["eligible_for_main_decision"] is False
    assert audit["schema_stub_dummy_node_count"] == 3
    assert audit["schema_stub_expected_loader_compatible"] is True


def test_weighted_edge_audit_detects_official_sehgnn_sparse_tensor_drops_values(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
    from hesf_coarsen.eval.official.weighted_edge_audit import audit_sehgnn_edge_weight_semantics

    graph = _tiny_dblp_graph(weighted=True)
    manifest = export_graph_to_sehgnn_hgb(
        graph=graph,
        dataset_name="DBLP",
        target_type="A",
        output_dir=tmp_path / "hgb",
        split_mode="official_trainval",
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
        test_idx=np.array([1], dtype=np.int64),
        labels=graph.labels,
        method_name="weighted_probe",
        seed=1,
    )

    result = audit_sehgnn_edge_weight_semantics(
        export_dir=Path(manifest["export_dir"]),
        dataset_name="DBLP",
        sehgnn_repo_dir=Path("external/SeHGNN"),
        output_dir=tmp_path / "diag",
    )

    assert result["exported_link_weight_nonunit_count"] > 0
    assert result["official_preprocess_accepts_edge_values"] is False
    assert result["official_preprocess_drops_edge_values"] is True
    assert result["weighted_superedge_main_table_allowed"] is False
    assert (tmp_path / "diag" / "gate21_1_weighted_edge_audit.csv").exists()


def test_gate21_1_summary_uses_storage_budget_decisions(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_1_sehgnn_schema_edge_budget import summarize_gate21_1

    rows = [
        {
            "dataset": "DBLP",
            "seed": seed,
            "model_name": "official-SeHGNN",
            "method": "H6-storage50",
            "method_family": "schema_compatible_subgraph",
            "schema_compatible": "True",
            "weighted_coarse_graph": "False",
            "uses_weighted_superedges": "False",
            "weighted_edge_preserved": "False",
            "eligible_for_main_decision": "True",
            "status": "success",
            "actual_total_storage_ratio_vs_full_graph": "0.49",
            "support_node_ratio": "0.30",
            "support_edge_ratio": "0.42",
            "test_micro_f1": "0.94",
            "test_macro_f1": "0.93",
            "native_full_test_micro_f1": "0.95",
            "native_full_test_macro_f1": "0.94",
            "recovery_vs_native_full_micro": "0.989",
            "recovery_vs_native_full_macro": "0.989",
            "no_test_label_export_leakage": "True",
            "schema_complete": "True",
        }
        for seed in range(1, 6)
    ]
    raw_path = tmp_path / "gate21_1_raw_rows.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "diagnostics").mkdir()
    with (tmp_path / "diagnostics" / "gate21_1_weighted_edge_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "seed",
                "official_preprocess_preserves_edge_values",
                "official_preprocess_drops_edge_values",
                "weighted_superedge_main_table_allowed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "DBLP",
                "method": "weighted_probe",
                "seed": 1,
                "official_preprocess_preserves_edge_values": "False",
                "official_preprocess_drops_edge_values": "True",
                "weighted_superedge_main_table_allowed": "False",
            }
        )

    result = summarize_gate21_1(tmp_path, tmp_path)

    assert "COMPRESSED_SEHGNN_VALIDATION_READY" not in result["decisions"]
    assert "SEHGNN_SCHEMA_COMPATIBLE_STORAGE50_PASS" in result["decisions"]
    assert "GENERIC_COARSE_GRAPH_NOT_VALIDATED" in result["decisions"]
    assert result["edge_storage_budget_pass"] is True
    assert result["best_storage50_method"] == "H6-storage50"
    assert result["native_full_accuracy"] == pytest.approx(0.95)
