from __future__ import annotations


def test_effective_total_bytes_includes_sidecar_and_metadata() -> None:
    from hesf_coarsen.eval.official.feature_cache_storage import compute_effective_total_bytes

    total = compute_effective_total_bytes(
        link_dat_bytes=10,
        label_dat_bytes=20,
        label_test_dat_bytes=30,
        info_dat_bytes=40,
        sidecar_feature_bytes=50,
        sidecar_metadata_bytes=6,
        preprocessed_cache_bytes=7,
        adapter_config_bytes=8,
    )

    assert total == 171


def test_adapter_storage_row_keeps_raw_and_effective_ratios_separate() -> None:
    from hesf_coarsen.eval.official.feature_cache_storage import adapter_storage_row

    row = adapter_storage_row(
        dataset="DBLP",
        method="SeHGNN-feature-compressed-adapter",
        base_graph_method="H6-APV-skeleton",
        graph_seed=1,
        training_seed=1,
        native_full_total_bytes=1000,
        export_total_bytes=700,
        node_dat_bytes=600,
        link_dat_bytes=10,
        label_dat_bytes=20,
        label_test_dat_bytes=30,
        info_dat_bytes=40,
        sidecar_feature_bytes=100,
        sidecar_metadata_bytes=5,
    )

    assert row["raw_hgb_byte_ratio"] == 0.7
    assert row["effective_total_byte_ratio"] == 0.205
    assert row["official_sehgnn_unmodified"] is False
    assert row["eligible_for_main_decision"] is False
    assert row["adapter_family"] == "feature_cache_compression"
