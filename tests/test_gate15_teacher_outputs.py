import numpy as np

from hesf_coarsen.task_first.selection.config import TeacherConfig
from hesf_coarsen.task_first.selection.teacher import train_full_graph_lite_teacher
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_teacher_outputs_metrics_and_cache_files(tmp_path):
    graph = make_gate15_graph()
    train_mask, val_mask, test_mask = split_masks()

    outputs = train_full_graph_lite_teacher(
        graph,
        np.asarray(graph.labels),
        train_mask,
        val_mask,
        test_mask,
        TeacherConfig(enabled=True),
        output_dir=tmp_path,
        seed=7,
        epochs=2,
        hidden_dim=8,
        device="cpu",
    )

    assert outputs["logits"].shape[0] == graph.num_nodes
    assert outputs["predictions"].shape == (graph.num_nodes,)
    assert outputs["metrics"]["full_graph_teacher_macro_f1"] >= 0.0
    assert outputs["metrics"]["validation_accuracy"] >= 0.0
    assert outputs["metrics"]["test_labels_used_for_training"] is False
    assert (tmp_path / "teacher_logits.npy").exists()
    assert (tmp_path / "teacher_pred.npy").exists()
    assert (tmp_path / "teacher_metrics.csv").exists()
