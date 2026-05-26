from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_relation_edge_retention_required_columns_not_nan_on_mock(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_3_relation_channel import main as run_main

    out = tmp_path / "gate21_3"
    assert run_main(["--dataset", "DBLP", "--preset", "quick", "--graph-seeds", "1", "--training-seeds", "1", "--dry-run", "--output-dir", str(out)]) == 0

    rows = _read_csv(out / "gate21_3_relation_edge_retention.csv")
    required = [
        "official_relation_id",
        "official_relation_name",
        "relation_pair_name",
        "original_full_edge_count",
        "candidate_edge_count_after_node_pruning",
        "retained_edge_count",
        "retention_vs_candidate",
        "retention_vs_full",
        "requested_relation_budget",
        "actual_relation_budget",
    ]
    assert rows
    for row in rows:
        assert all(row[column] not in {"", "NaN", "nan"} for column in required)


def test_storage_audit_required_columns_present(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_3_relation_channel import main as run_main

    out = tmp_path / "gate21_3"
    run_main(["--dataset", "DBLP", "--preset", "quick", "--graph-seeds", "1", "--training-seeds", "1", "--dry-run", "--output-dir", str(out)])

    header = (out / "gate21_3_storage_audit.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "node_dat_fraction_of_export" in header
    assert "link_dat_fraction_of_export" in header
    assert "feature_dominates_raw_bytes_flag" in header
    assert "edge_pruning_raw_byte_savings_estimated" in header


def test_decision_json_contains_required_flags(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_3_relation_channel import main as run_main
    from experiments.scripts.summarize_gate21_3_relation_channel import main as summarize_main

    out = tmp_path / "gate21_3"
    run_main(["--dataset", "DBLP", "--preset", "quick", "--graph-seeds", "1", "--training-seeds", "1", "--dry-run", "--output-dir", str(out)])
    summarize_main(["--results-dir", str(out), "--output-dir", str(out / "summary")])
    decision = json.loads((out / "summary" / "gate21_3_decision.json").read_text(encoding="utf-8"))

    for key in [
        "relation_mapping_audit_pass",
        "relation_retention_audit_pass",
        "pairgrid_implementation_pass",
        "graph_seed_stability_validated",
        "pathaware_v2_beats_random_at_matched_budget",
        "weighted_edge_semantics_supported_for_unmodified_official",
    ]:
        assert key in decision
    assert "RELATION_MAPPING_AUDIT_PASS" in decision["decisions"]
