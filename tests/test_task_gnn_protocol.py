import numpy as np

from hesf_coarsen.eval.task_gnn import train_only_coarse_labels


def test_train_only_coarse_labels_do_not_use_test_labels():
    labels = np.array([0, 1, 1, 1], dtype=np.int64)
    original_to_coarse = np.array([0, 0, 0, 1], dtype=np.int64)
    train_nodes = np.array([0], dtype=np.int64)
    test_nodes = np.array([1, 2, 3], dtype=np.int64)

    coarse_labels, diagnostics = train_only_coarse_labels(
        labels,
        original_to_coarse,
        train_nodes,
        num_coarse_nodes=2,
        test_nodes=test_nodes,
    )

    assert coarse_labels.tolist() == [0, -1]
    assert diagnostics["train_only_label_coverage"] == 0.5
    assert diagnostics["test_label_leakage_check"] == "passed"
    assert diagnostics["cluster_train_label_entropy"] == 0.0
