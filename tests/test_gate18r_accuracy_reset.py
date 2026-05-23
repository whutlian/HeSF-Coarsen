from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scripts.run_gate18r_accuracy_first_reset import (
    DEFAULT_METHODS,
    GATE18R_SINGLE_SEED_BY_DATASET,
    NEW_CANDIDATE_METHODS,
    OLD_ABLATION_METHODS,
    parse_dataset_seeds,
)
from experiments.scripts.summarize_gate18r import summarize
from hesf_coarsen.eval.calibration import (
    apply_calibrator,
    fit_macro_constrained_accuracy_calibrator,
    temperature_scale_logits,
)
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj
from hesf_coarsen.task_first.feature_condensation.path_prototype import class_path_prototype_cache
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache
from hesf_coarsen.task_first.units.h6_units import extract_h6_units
from hesf_coarsen.task_first.units.scoring import score_units


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    method: str,
    dataset: str,
    ratio: float,
    actual_ratio: float,
    macro: float,
    accuracy: float,
    *,
    eligible: bool = False,
    baseline: bool = False,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": 45678 if dataset == "IMDB" else 23456,
        "method": method,
        "requested_support_ratio": ratio,
        "actual_support_ratio": actual_ratio,
        "effective_support_node_ratio": actual_ratio,
        "represented_support_context_ratio": actual_ratio,
        "macro_f1": macro,
        "accuracy": accuracy,
        "validation_macro_f1": macro - 0.01,
        "validation_accuracy": accuracy - 0.01,
        "status": "success",
        "primary_eval_mode": "compressed_projected",
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "calibrator_uses_test_labels": False,
        "no_test_leakage": True,
        "typedhash_included": True,
        "eligible_for_main_decision": eligible,
        "diagnostic_only": not eligible and not baseline,
        "full_residual_upperbound": False,
        "compression_axis": "support_node",
    }


def _summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["ACM", "DBLP", "IMDB"]:
        for ratio in [0.30, 0.50, 0.70]:
            rows.append(_row("random-support-only", dataset, ratio, ratio, 0.50, 0.60, baseline=True))
            rows.append(_row("H6-no-spec-support-only", dataset, ratio, ratio, 0.60, 0.70, baseline=True))
            rows.append(_row("flatten-sum-support-only", dataset, ratio, ratio, 0.62, 0.72, baseline=True))
            rows.append(_row("TypedHash-ChebHeat-support-only", dataset, ratio, ratio, 0.61, 0.73 if dataset == "IMDB" else 0.71, baseline=True))
            rows.append(_row("HeSF-SS-H6-fill-only", dataset, ratio, ratio, 0.80, 0.80, eligible=False))
            rows.append(_row("HeSF-ClusterGate-UnionUnits", dataset, ratio, min(ratio, 0.55), 0.63, 0.721, eligible=True))
            rows.append(_row("HeSF-STC-feature-cache-distill-logit-calibrated", dataset, ratio, 0.0, 0.64, 0.725, eligible=True))
    return rows


def _toy_graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=6,
        node_type=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64),
                dst=np.asarray([2, 3, 4, 0, 1, 1], dtype=np.int64),
                weight=np.ones(6, dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        features={0: np.ones((2, 2), dtype=np.float32), 1: np.ones((4, 2), dtype=np.float32)},
        labels=np.asarray([0, 1, -1, -1, -1, -1], dtype=np.int64),
    )


def test_gate18r_dataset_seeds_and_method_sets_match_prompt():
    assert GATE18R_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    assert parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"]) == [
        ("ACM", 23456),
        ("DBLP", 23456),
        ("IMDB", 45678),
    ]
    for method in [
        "full-graph-hettree-lite-tuned",
        "target-only-empty-support",
        "random-support-only",
        "H6-no-spec-support-only",
        "flatten-sum-support-only",
        "TypedHash-ChebHeat-support-only",
        "HeSF-SS-validation-underfill-pareto",
        "HeSF-SS-validation-underfill-pareto-logit-calibrated",
        "HeSF-SS-validation-H6-fill-logit-calibrated",
        "HeSF-ClusterGate-H6-units",
        "HeSF-ClusterGate-TypedHash-units",
        "HeSF-ClusterGate-Flatten-units",
        "HeSF-ClusterGate-UnionUnits",
        "HeSF-ClusterGate-UnionUnits-logit-calibrated",
        "HeSF-STC-path-prune",
        "HeSF-STC-path-prototype",
        "HeSF-STC-feature-cache-distill",
        "HeSF-STC-feature-cache-distill-logit-calibrated",
    ]:
        assert method in DEFAULT_METHODS
        assert method in NEW_CANDIDATE_METHODS or not method.startswith("HeSF-")
    assert "HeSF-SS-validation-H6-fill-acc0.50" in OLD_ABLATION_METHODS


def test_macro_constrained_calibrator_uses_validation_only_and_applies_bias():
    val_logits = np.asarray([[2.0, 0.0], [0.2, 1.1], [1.2, 0.9], [0.0, 1.8]], dtype=np.float32)
    val_labels = np.asarray([0, 1, 1, 1], dtype=np.int64)

    scaled = temperature_scale_logits(val_logits, 2.0)
    fit = fit_macro_constrained_accuracy_calibrator(
        val_logits,
        val_labels,
        baseline_macro=0.50,
        macro_epsilon=0.20,
        temperatures=(0.5, 1.0, 2.0),
        class_bias_values=(-0.5, 0.0, 0.5),
    )
    calibrated = apply_calibrator(val_logits, fit)

    assert scaled[0, 0] == pytest.approx(1.0)
    assert fit["calibrator_uses_test_labels"] is False
    assert fit["constraint_satisfied"] is True
    assert calibrated.shape == val_logits.shape
    assert set(fit["class_bias"].keys()) == {"0", "1"}


def test_h6_units_report_nonzero_structure_and_accuracy_first_scores():
    graph = _toy_graph()
    assignment = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64)
    split = {"train": np.asarray([0, 1], dtype=np.int64), "val": np.asarray([0, 1], dtype=np.int64)}

    units = extract_h6_units(graph, assignment, target_type=0, labels=graph.labels, splits=split)
    scored = score_units(units, lambda_acc=1.0, lambda_macro=0.5)

    assert len(units) == 3
    assert sum(unit.edge_mass for unit in units) > 0.0
    assert sum(unit.target_anchor_coverage for unit in units) > 0.0
    assert all("score" in unit.metadata for unit in scored)


def test_path_prototype_does_not_assign_by_test_labels():
    cache = SemanticTreeCache(
        tensor=np.asarray([[[0.0, 0.0]], [[10.0, 10.0]], [[0.1, 0.1]]], dtype=np.float32),
        target_nodes=np.asarray([0, 1, 2], dtype=np.int64),
        paths=[()],
        feature_width=2,
        type_ids=(0,),
    )
    labels = np.asarray([0, 1, 1], dtype=np.int64)

    proto = class_path_prototype_cache(cache, labels=labels, train_nodes=np.asarray([0, 1], dtype=np.int64))

    assert np.allclose(proto.tensor[2], proto.tensor[0])
    assert not np.allclose(proto.tensor[2], proto.tensor[1])


def test_gate18r_summary_writes_pareto_frontier_and_excludes_old_diagnostics(tmp_path):
    _write_rows(tmp_path / "gate18r_raw_rows.csv", _summary_rows())

    result = summarize(tmp_path, tmp_path)

    assert result["stage"] == "Gate18R"
    assert result["typedhash_included"] is True
    assert result["best_method"] != "HeSF-SS-H6-fill-only"
    assert result["best_method"] in {"HeSF-ClusterGate-UnionUnits", "HeSF-STC-feature-cache-distill-logit-calibrated"}
    assert result["primary_eval_mode"] == "compressed_projected"
    assert result["no_test_leakage"] is True
    assert (tmp_path / "gate18r_pareto_frontier.csv").exists()
    assert (tmp_path / "gate18r_result.json").exists()
    saved = json.loads((tmp_path / "gate18r_result.json").read_text(encoding="utf-8"))
    assert "full_graph_lite_accuracy_by_dataset" in saved
