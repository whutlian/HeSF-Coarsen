from __future__ import annotations

from pathlib import Path


def test_official_main_eligibility_rejects_adapter_missing_targets_and_test_data() -> None:
    from hesf_coarsen.eval.official.icde_protocol import build_protocol_row

    adapter = build_protocol_row(
        baseline_name="adapter",
        protocol="schema_preserving_tp_workload",
        schema_compatible=True,
        official_sehgnn_unmodified=False,
        uses_feature_adapter=True,
        keeps_all_target_nodes=True,
    )
    drops_targets = build_protocol_row(
        baseline_name="drop-targets",
        protocol="schema_preserving_tp_workload",
        schema_compatible=True,
        official_sehgnn_unmodified=True,
        keeps_all_target_nodes=False,
    )
    leaks_test = build_protocol_row(
        baseline_name="leaks-test",
        protocol="schema_preserving_tp_workload",
        schema_compatible=True,
        official_sehgnn_unmodified=True,
        keeps_all_target_nodes=True,
        used_test_data=True,
    )

    assert adapter["eligible_for_official_main_table"] is False
    assert drops_targets["eligible_for_official_main_table"] is False
    assert leaks_test["eligible_for_official_main_table"] is False
    assert leaks_test["eligibility_failure_reasons"] == "used_test_data"


def test_protocol_tables_keep_distinct_cost_ratio_fields() -> None:
    from hesf_coarsen.eval.official.icde_protocol import build_protocol_row

    row = build_protocol_row(
        baseline_name="ratio-check",
        protocol="schema_preserving_tp_workload",
        schema_compatible=True,
        official_sehgnn_unmodified=True,
        keeps_all_target_nodes=True,
        support_node_ratio=0.3,
        structural_storage_ratio=0.12,
        raw_hgb_text_byte_ratio=0.53,
    )

    assert row["support_node_ratio"] == 0.3
    assert row["structural_storage_ratio"] == 0.12
    assert row["raw_hgb_text_byte_ratio"] == 0.53


def test_deterministic_and_stochastic_graph_seed_stability_flags() -> None:
    from hesf_coarsen.eval.official.icde_protocol import assess_graph_seed_stability

    deterministic = assess_graph_seed_stability(
        deterministic_graph_method=True,
        graph_seeds=[1],
        export_hashes=["abc"],
    )
    stochastic_under_sampled = assess_graph_seed_stability(
        deterministic_graph_method=False,
        graph_seeds=[1, 2],
        export_hashes=["a", "b"],
    )

    assert deterministic["actual_export_hash_unique_count"] == 1
    assert deterministic["graph_sampling_stability_pass"] is True
    assert stochastic_under_sampled["graph_sampling_stability_pass"] is False
    assert "graph_seed_count_lt_5" in stochastic_under_sampled["graph_sampling_stability_warnings"]


def test_freehgc_missing_dependency_is_failure_row_not_skip(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.external_baselines_tp import plan_external_tp_rows

    rows = plan_external_tp_rows(
        dataset="DBLP",
        methods=["Random-HG-TP", "FreeHGC-TP"],
        budgets=[0.3],
        graph_seeds=[1],
        training_seeds=[1],
        freehgc_root=tmp_path / "missing-freehgc",
    )

    freehgc = [row for row in rows if row["baseline_name"] == "FreeHGC-TP"]
    random_tp = [row for row in rows if row["baseline_name"] == "Random-HG-TP"]
    assert len(freehgc) == 1
    assert freehgc[0]["success"] is False
    assert freehgc[0]["failure_type"] == "missing_external_dependency"
    assert freehgc[0]["eligible_for_official_main_table"] is False
    assert len(random_tp) == 1
    assert random_tp[0]["success"] is True
    assert random_tp[0]["construction_status"] == "constructed_estimate"
    assert random_tp[0]["failure_type"] == ""


def test_storage_and_resource_rows_are_explicit(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.storage_only_baselines import build_storage_only_row
    from hesf_coarsen.eval.official.system_resource_logger import conservative_resource_row

    storage = build_storage_only_row(
        dataset="DBLP",
        artifact_name="binary_csr_plus_int8_features",
        native_full_text_bytes=1000,
        total_artifact_bytes=250,
        requires_loader_adapter=True,
        changes_training_semantics=False,
        binary_relation_bytes=100,
        binary_feature_bytes=120,
        metadata_bytes=30,
    )
    resource = conservative_resource_row(
        stage_name="compression",
        input_paths=[tmp_path / "missing-input"],
        output_paths=[tmp_path / "missing-output"],
    )

    assert storage["changes_training_semantics"] is False
    assert storage["requires_loader_adapter"] is True
    assert storage["artifact_ratio_vs_native_full_text"] == 0.25
    assert resource["io_bytes_estimated"] is True
    assert resource["input_bytes_read"] == 0
    assert resource["output_bytes_written"] == 0
