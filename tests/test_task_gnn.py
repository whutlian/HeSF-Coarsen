import numpy as np

from hesf_coarsen.eval.task_gnn import compose_assignments, f1_scores


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
