from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from tests.eval_official.test_graph_export import make_tiny_official_graph


def test_calibration_function_signature_does_not_accept_test_labels() -> None:
    from hesf_coarsen.eval.official.calibration_adapter import calibrate_logits_nested

    signature = inspect.signature(calibrate_logits_nested)
    assert "test_labels" not in signature.parameters
    assert "labels_test" not in signature.parameters


def test_export_does_not_write_test_labels_into_training_or_calibration_artifacts(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph

    graph = make_tiny_official_graph()
    export = export_hgb_graph(
        graph,
        dataset_name="Tiny",
        method_name="full",
        seed=23456,
        support_ratio=None,
        output_dir=tmp_path,
        target_type="type_0",
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4, 5], dtype=np.int64),
        labels=graph.labels,
    )
    export_dir = Path(export["export_dir"])
    assert export["no_test_label_export_leakage"] is True
    assert (export_dir / "splits" / "train_labels.npy").exists()
    assert (export_dir / "splits" / "val_labels.npy").exists()
    assert not (export_dir / "splits" / "test_labels_for_training.npy").exists()
    assert not (export_dir / "calibration" / "test_labels.npy").exists()


def test_runner_dry_run_marks_calibration_as_validation_only(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_open_sota_bridge import dry_run_row

    row = dry_run_row(
        dataset="DBLP",
        seed=23456,
        method="H6",
        support_ratio=0.30,
        model="sehgnn_official",
        status="dry_run",
        error_message="not executed",
    )
    assert row["calibration_uses_test_labels"] is False
    assert row["selector_uses_test_labels"] is False
