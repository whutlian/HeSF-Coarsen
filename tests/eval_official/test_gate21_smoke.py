from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.eval_official.test_graph_export import make_tiny_official_graph


def test_gate21_smoke_export_calibrate_and_summarize_without_external_repos(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_open_sota import summarize_rows
    from hesf_coarsen.eval.official.calibration_adapter import calibrate_logits_nested
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph

    graph = make_tiny_official_graph()
    export = export_hgb_graph(
        graph,
        dataset_name="Tiny",
        method_name="H6",
        seed=23456,
        support_ratio=0.30,
        output_dir=tmp_path / "exports",
        target_type="type_0",
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4, 5], dtype=np.int64),
        labels=graph.labels,
    )
    assert export["export_status"] == "success"

    cal = calibrate_logits_nested(
        np.array([[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.2, 0.1, 2.0]], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([[1.2, 0.3, 0.1], [0.1, 0.3, 1.4]], dtype=np.float32),
        split_seeds=(11,),
    )
    assert cal["calibration_uses_test_labels"] is False

    result = summarize_rows(
        raw_rows=[
            {
                "dataset": "DBLP",
                "seed": 23456,
                "model_name": "SeHGNN-official",
                "method": "full",
                "support_ratio": "",
                "status": "failed_dependency",
                "validation_macro_f1": "",
                "validation_accuracy": "",
                "test_macro_f1": "",
                "test_accuracy": "",
                "val_logits_path": "",
                "test_logits_path": "",
                "calibrated": False,
                "calibration_uses_test_labels": False,
                "selector_uses_test_labels": False,
            }
        ],
        export_rows=[export],
        calibration_rows=[],
    )
    assert result["stage"] == "Gate21-OpenSOTA"
    assert result["official_bridge_pass"] is False
    assert result["decision"] == "FIX_OFFICIAL_BRIDGE"
    assert result["no_test_leakage"] is True
