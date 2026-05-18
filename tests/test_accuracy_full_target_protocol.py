import numpy as np

from hesf_coarsen.accuracy.full_target_protocol import (
    make_protocol_row,
    target_preserve_protocol_report,
)
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=5,
        node_type=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1], dtype=np.int64),
                dst=np.array([2, 3], dtype=np.int64),
                weight=None,
                src_type=0,
                dst_type=1,
                relation_id=0,
            ),
        },
        relation_specs={0: RelationSpec(0, "target__to__support", 0, 1)},
        labels=np.array([0, 1, -1, -1, -1], dtype=np.int64),
    )


def test_real_full_target_protocol_requires_original_target_prediction_domain() -> None:
    original = _graph()
    hybrid = HeteroGraph(
        num_nodes=4,
        node_type=np.array([0, 0, 1, 1], dtype=np.int32),
        relations={},
        relation_specs={},
        labels=np.array([0, 1, -1, -1], dtype=np.int64),
    )
    mapping = np.array([0, 1, 2, 2, 3], dtype=np.int64)

    report = target_preserve_protocol_report(original, hybrid, mapping, target_node_type=0)

    assert report["target_mapping_one_to_one"] is True
    assert report["target_domain"] == "original_target_nodes"
    assert report["support_domain"] == "compressed_support_nodes"
    assert report["inference_domain"] == "full_original_target_set"


def test_real_full_target_protocol_uses_explicit_hybrid_target_metrics_not_projected_alias() -> None:
    metrics = {
        "projected_original_macro_f1": 0.1,
        "transfer_original_macro_f1": 0.2,
        "hybrid_target_original_macro_f1": 0.3,
        "hybrid_target_original_micro_f1": 0.4,
        "hybrid_target_original_accuracy": 0.5,
    }

    row = make_protocol_row(
        metrics,
        eval_mode="real_full_target_inference",
        model_name="sehgnn_local",
        model_fidelity="lite_adapter",
        official_repo="no",
        official_preprocess="no",
        adapter_mode="target_preserve_direct",
        path_set="lite",
        split_policy="synthetic_stratified",
        max_hops=2,
    )

    assert row["macro_f1"] == 0.3
    assert row["metric_source"] == "hybrid_target_original"
    assert row["metric_source"] != "projected_original"
    for field in ["target_domain", "support_domain", "inference_domain"]:
        assert row[field]


def test_approx_protocol_is_separate_from_real_full_target_inference() -> None:
    metrics = {
        "projected_original_macro_f1": 0.1,
        "projected_original_micro_f1": 0.2,
        "projected_original_accuracy": 0.25,
    }

    row = make_protocol_row(
        metrics,
        eval_mode="approx_full_target_adapter",
        model_name="hettree_lite",
        model_fidelity="lite_adapter",
        official_repo="no",
        official_preprocess="no",
        adapter_mode="approximate",
        path_set="lite",
        split_policy="synthetic_stratified",
        max_hops=2,
    )

    assert row["eval_mode"] == "approx_full_target_adapter"
    assert row["metric_source"] == "projected_original"
    assert row["inference_domain"] == "projected_original_targets"
