from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.eval_official.test_graph_export import make_tiny_official_graph


OPENHGNN_SEHGNN_STUB = """
from . import BaseModel, register_model
import torch
import torch.nn as nn


@register_model("SeHGNN")
class SeHGNN(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.key = sorted(args.data_size)[0]
        self.linear = nn.Linear(args.data_size[self.key], args.nclass)

    def forward(self, fk):
        return self.linear(fk["0"][self.key])
"""


def test_openhgnn_sehgnn_runner_trains_from_gate21_export_without_top_level_openhgnn_import(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph
    from hesf_coarsen.eval.official.openhgnn_export_runner import load_openhgnn_sehgnn_class, train_export

    fake_repo = tmp_path / "OpenHGNN"
    model_dir = fake_repo / "openhgnn" / "models"
    model_dir.mkdir(parents=True)
    (fake_repo / "openhgnn" / "__init__.py").write_text("raise ImportError('top level must not import')\n", encoding="utf-8")
    (model_dir / "SeHGNN.py").write_text(OPENHGNN_SEHGNN_STUB, encoding="utf-8")

    model_cls = load_openhgnn_sehgnn_class(fake_repo)
    assert model_cls.__name__ == "SeHGNN"

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
    result = train_export(
        export_dir=Path(export["export_dir"]),
        repo_dir=fake_repo,
        dataset_name="Tiny",
        target_type="type_0",
        seed=23456,
        result_json=tmp_path / "result.json",
        logits_dir=tmp_path / "logits",
        epochs=1,
        embed_size=8,
        hidden=8,
        batch_size=3,
        device_name="cpu",
    )

    assert result["status"] == "success"
    assert Path(result["val_logits_path"]).exists()
    assert Path(result["test_logits_path"]).exists()
    assert np.load(result["val_logits_path"]).shape == (1, 3)
    assert np.load(result["test_logits_path"]).shape == (2, 3)


def test_openhgnn_bridge_runs_sehgnn_without_top_level_openhgnn_import(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph
    from hesf_coarsen.eval.official.openhgnn_bridge import run_openhgnn_model

    fake_repo = tmp_path / "OpenHGNN"
    model_dir = fake_repo / "openhgnn" / "models"
    model_dir.mkdir(parents=True)
    (fake_repo / "openhgnn" / "__init__.py").write_text("raise ImportError('top level must not import')\n", encoding="utf-8")
    (model_dir / "SeHGNN.py").write_text(OPENHGNN_SEHGNN_STUB, encoding="utf-8")
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
        "openhgnn_sehgnn",
        "Tiny",
        "type_0",
        23456,
        {"method": "full", "epochs": 1, "embed_size": 8, "hidden": 8, "batch_size": 3, "device": "cpu"},
        tmp_path / "run",
    )

    assert result["status"] == "success"
    assert result["model_name"] == "OpenHGNN-SeHGNN"
    assert Path(result["val_logits_path"]).exists()
    assert Path(result["test_logits_path"]).exists()
