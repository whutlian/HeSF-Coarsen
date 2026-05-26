from __future__ import annotations

import numpy as np


def test_feature_loader_audit_confirms_zero_paper_loaded_tensor() -> None:
    from hesf_coarsen.eval.official.feature_loader_audit import feature_loader_audit_rows

    rows = feature_loader_audit_rows(
        dataset="DBLP",
        method="feature_channel_ablation",
        canonical_method="H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
        graph_seed=1,
        training_seed=1,
        feature_transform_name="zero-paper",
        before_features={1: np.ones((3, 4), dtype=np.float32)},
        after_features={1: np.zeros((3, 4), dtype=np.float32)},
        loaded_features={1: np.zeros((3, 4), dtype=np.float32)},
    )

    paper = rows[0]
    assert paper["feature_transform_applied_flag"] is True
    assert paper["feature_l2_norm_after_loader"] <= 1e-8
    assert paper["feature_zero_fraction_after_loader"] >= 0.999999


def test_feature_loader_audit_confirms_pca_dim_after_loader() -> None:
    from hesf_coarsen.eval.official.feature_loader_audit import feature_loader_audit_rows

    rows = feature_loader_audit_rows(
        dataset="DBLP",
        method="feature_channel_ablation",
        canonical_method="H6-APV-skeleton",
        graph_seed=1,
        training_seed=1,
        feature_transform_name="pca-paper-128",
        before_features={1: np.ones((5, 4231), dtype=np.float32)},
        after_features={1: np.ones((5, 128), dtype=np.float32)},
        loaded_features={1: np.ones((5, 128), dtype=np.float32)},
    )

    assert rows[0]["feature_dim_after_loader"] == 128
