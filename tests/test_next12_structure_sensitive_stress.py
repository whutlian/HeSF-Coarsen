import numpy as np
from pathlib import Path

from experiments.scripts._common import write_csv
from hesf_coarsen.io.schema import HeteroGraph


def test_feature_mask_noise_and_structure_only_transforms_are_deterministic():
    from experiments.scripts.run_next12_structure_sensitive_stress import transform_graph_features

    graph = HeteroGraph(
        num_nodes=3,
        node_type=np.array([0, 0, 1], dtype=np.int32),
        relations={},
        features={
            0: np.ones((2, 4), dtype=np.float32),
            1: np.ones((1, 3), dtype=np.float32),
        },
    )

    masked_a = transform_graph_features(graph, mode="feature_mask", value=0.5, seed=3)
    masked_b = transform_graph_features(graph, mode="feature_mask", value=0.5, seed=3)
    noisy = transform_graph_features(graph, mode="feature_noise", value=0.1, seed=3)
    struct = transform_graph_features(graph, mode="structure_only", value=1.0, seed=3)

    assert np.array_equal(masked_a.features[0], masked_b.features[0])
    assert not np.array_equal(noisy.features[0], graph.features[0])
    assert np.allclose(struct.features[0], 0.0)
    assert np.allclose(struct.features[1], 0.0)


def test_structure_stress_summary_computes_win_rates(tmp_path: Path):
    from experiments.scripts.summarize_next12_structure_sensitive_stress import summarize_next12_structure_sensitive_stress

    inp = tmp_path / "stress"
    out = tmp_path / "summary"
    write_csv(
        inp / "structure_sensitive_stress_runs.csv",
        [
            {"dataset": "ACM", "seed": 1, "stress_name": "feature_mask_0.25", "method": "HeSF-LVC-P", "best_macro_f1": 0.8, "run_status": "available"},
            {"dataset": "ACM", "seed": 1, "stress_name": "feature_mask_0.25", "method": "flatten-sum", "best_macro_f1": 0.7, "run_status": "available"},
            {"dataset": "ACM", "seed": 1, "stress_name": "feature_mask_0.25", "method": "H6-no-spec", "best_macro_f1": 0.75, "run_status": "available"},
        ],
    )

    summarize_next12_structure_sensitive_stress(input=inp, output=out)

    text = (out / "structure_stress_win_rates.csv").read_text(encoding="utf-8")
    assert "win_rate_vs_flatten_sum" in text
    assert "mean_delta_best_vs_H6" in text
