import numpy as np
import pytest

from hesf_coarsen.eval.task_gnn import compose_assignments, evaluate_rgcn_task, f1_scores, refine_curve_summary
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_compose_assignments_maps_original_to_final_level(tmp_path):
    level_1 = tmp_path / "assignment1.npz"
    level_2 = tmp_path / "assignment2.npz"
    np.savez_compressed(level_1, assignment=np.array([0, 0, 1, 2], dtype=np.int64))
    np.savez_compressed(level_2, assignment=np.array([0, 1, 1], dtype=np.int64))

    mapping = compose_assignments(4, [str(level_1), str(level_2)])

    assert mapping.tolist() == [0, 0, 1, 1]


def test_f1_scores_reports_micro_and_macro():
    scores = f1_scores(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 1], dtype=np.int64),
    )

    assert np.isclose(scores["micro_f1"], 0.75)
    assert 0.0 <= scores["macro_f1"] <= 1.0


def test_refine_curve_summary_reports_best_epoch_auc_and_time_mapping():
    summary = refine_curve_summary(
        {
            0: {"macro_f1": 0.60, "micro_f1": 0.61, "refine_time": 0.0},
            1: {"macro_f1": 0.55, "micro_f1": 0.56, "refine_time": 0.2},
            3: {"macro_f1": 0.70, "micro_f1": 0.71, "refine_time": 0.7},
            5: {"macro_f1": 0.68, "micro_f1": 0.69, "refine_time": 1.1},
        }
    )

    assert summary["best_refined_macro_f1"] == 0.70
    assert summary["best_refined_epoch"] == 3
    assert np.isclose(summary["refine_auc_macro_f1"], 0.641)
    assert summary["refine_time_by_epoch"] == {"0": 0.0, "1": 0.2, "3": 0.7, "5": 1.1}


def test_task_eval_reports_named_full_graph_baselines():
    pytest.importorskip("torch")
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=707)
    mapping = np.arange(graph.num_nodes, dtype=np.int64)

    metrics = evaluate_rgcn_task(
        graph,
        graph,
        mapping,
        seed=707,
        hidden_dim=8,
        epochs=1,
        refine_epochs=0,
        refine_epochs_list=[0],
        device="cpu",
        full_graph_baselines=[
            "full_graph_rgcn_lite_default",
            "full_graph_rgcn_lite_tuned",
            "full_graph_han_small",
            "full_graph_hgt_small",
        ],
    ).metrics

    for name in (
        "full_graph_rgcn_lite_default",
        "full_graph_rgcn_lite_tuned",
        "full_graph_han_small",
        "full_graph_hgt_small",
    ):
        assert f"{name}_macro_f1" in metrics
        assert f"{name}_micro_f1" in metrics
        assert f"{name}_train_time" in metrics
        assert metrics.get(f"{name}_skipped", False) is False
