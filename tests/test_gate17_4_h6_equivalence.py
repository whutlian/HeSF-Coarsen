from __future__ import annotations

from experiments.scripts.gate17_4_h6 import compute_h6_equivalence_fields
from experiments.scripts.run_gate17_4_h6_equivalence import (
    DEFAULT_METHODS,
    DIAGNOSTIC_ONLY_METHODS,
    GATE17_4_SINGLE_SEED_BY_DATASET,
    H6_CONSTRUCTION_CONTROL_METHOD,
    H6_SELECTED_SET_CONTROL_METHOD,
    _semantic_fields_for_raw_row,
    parse_dataset_seeds,
)


def test_gate17_4_dataset_seeds_are_exact_pairs_not_cartesian_product():
    pairs = parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])

    assert GATE17_4_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    assert pairs == [("ACM", 23456), ("DBLP", 23456), ("IMDB", 45678)]


def test_gate17_4_methods_separate_h6_construction_and_selected_set_controls():
    assert H6_CONSTRUCTION_CONTROL_METHOD == "HeSF-SS-H6-equivalence-control"
    assert H6_SELECTED_SET_CONTROL_METHOD == "HeSF-SS-H6-selected-set-control"
    assert H6_CONSTRUCTION_CONTROL_METHOD in DEFAULT_METHODS
    assert H6_SELECTED_SET_CONTROL_METHOD in DEFAULT_METHODS
    assert H6_SELECTED_SET_CONTROL_METHOD in DIAGNOSTIC_ONLY_METHODS
    assert H6_CONSTRUCTION_CONTROL_METHOD not in DIAGNOSTIC_ONLY_METHODS


def test_gate17_4_raw_semantic_fields_include_required_hash_alias():
    fields = _semantic_fields_for_raw_row(
        {
            "coarse_tree_hash": "abc123",
            "tree_tensor_l2_delta_vs_full": 1.5,
            "tree_tensor_cosine_delta_vs_full": 0.25,
            "target_path_feature_changed_fraction": 0.5,
            "allclose_to_full": False,
        }
    )

    assert fields == {
        "semantic_tree_hash": "abc123",
        "tree_tensor_l2_delta_vs_full": 1.5,
        "tree_tensor_cosine_delta_vs_full": 0.25,
        "target_path_feature_changed_fraction": 0.5,
        "allclose_to_full": False,
    }


def test_h6_equivalence_fields_pass_for_identical_construction():
    fields = compute_h6_equivalence_fields(
        mode="construction",
        h6_macro_f1=0.8118,
        control_macro_f1=0.8117,
        h6_accuracy=0.90,
        control_accuracy=0.9001,
        h6_validation_macro_f1=0.80,
        control_validation_macro_f1=0.8002,
        tree_l2_delta_vs_h6=0.0,
        tree_cosine_delta_vs_h6=0.0,
        tree_hash_equal_to_h6=True,
        coarse_graph_hash_equal_to_h6=True,
        edge_mass_l1_delta_vs_h6=0.0,
        edge_mass_linf_delta_vs_h6=0.0,
        feature_mean_l2_delta_vs_h6=0.0,
        assignment_equivalent_to_h6=True,
        selected_jaccard_with_H6=1.0,
        selected_recall_of_H6=1.0,
        selected_precision_vs_H6=1.0,
    )

    assert fields["macro_gap_vs_h6"] == -0.0001
    assert fields["construction_equivalence_pass"] is True
    assert fields["h6_construction_gap_detected"] is False


def test_h6_equivalence_fields_fail_when_assignment_or_tree_differs():
    fields = compute_h6_equivalence_fields(
        mode="selected_set",
        h6_macro_f1=0.8118,
        control_macro_f1=0.70,
        h6_accuracy=0.90,
        control_accuracy=0.80,
        h6_validation_macro_f1=0.80,
        control_validation_macro_f1=0.70,
        tree_l2_delta_vs_h6=0.5,
        tree_cosine_delta_vs_h6=0.1,
        tree_hash_equal_to_h6=False,
        coarse_graph_hash_equal_to_h6=False,
        edge_mass_l1_delta_vs_h6=2.0,
        edge_mass_linf_delta_vs_h6=1.0,
        feature_mean_l2_delta_vs_h6=0.2,
        assignment_equivalent_to_h6=False,
        selected_jaccard_with_H6=1.0,
        selected_recall_of_H6=1.0,
        selected_precision_vs_H6=1.0,
    )

    assert fields["construction_equivalence_pass"] is False
    assert fields["h6_construction_gap_detected"] is True
