from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.eval_official.test_graph_export import make_tiny_official_graph


def test_sehgnn_feature_blocks_include_target_self_and_neighbor_means(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph
    from hesf_coarsen.eval.official.sehgnn_export_runner import build_target_feature_blocks

    graph = make_tiny_official_graph()
    result = export_hgb_graph(
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
        original_target_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    )

    blocks = build_target_feature_blocks(Path(result["export_dir"]), "type_0")

    assert set(blocks) == {"type_0", "paper__to__author__dst_mean", "author__to__paper__src_mean"}
    np.testing.assert_allclose(blocks["type_0"], graph.features[0])
    expected = np.array(
        [
            [1.0, 0.2],
            [1.0, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.4, 0.6],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(blocks["paper__to__author__dst_mean"], expected)
    np.testing.assert_allclose(blocks["author__to__paper__src_mean"], expected)


def test_openhgnn_bridge_reports_dependency_probe_failure_for_dgl_graph_models(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph
    from hesf_coarsen.eval.official.openhgnn_bridge import run_openhgnn_model

    fake_repo = tmp_path / "OpenHGNN"
    (fake_repo / "openhgnn").mkdir(parents=True)
    (fake_repo / "openhgnn" / "__init__.py").write_text("raise ImportError('broken dgl sparse')\n", encoding="utf-8")
    graph = make_tiny_official_graph()
    export = export_hgb_graph(
        graph,
        dataset_name="Tiny",
        method_name="full",
        seed=23456,
        support_ratio=None,
        output_dir=tmp_path / "out",
        target_type="type_0",
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4, 5], dtype=np.int64),
        labels=graph.labels,
    )

    result = run_openhgnn_model(
        Path(export["export_dir"]),
        fake_repo,
        "openhgnn_hgt",
        "Tiny",
        "type_0",
        23456,
        {"method": "full"},
        tmp_path / "run",
    )

    assert result["status"] == "failed_dependency"
    assert "broken dgl sparse" in result["error_message"]
