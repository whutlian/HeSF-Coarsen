from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_pathaware_export_uses_tristate_gain_rule(tmp_path: Path) -> None:
    from experiments.scripts.export_gate21_4_pathaware_v2_validation import main

    fields = [
        "dataset",
        "method",
        "edge_score_strategy",
        "relation_channel_spec",
        "graph_seed",
        "training_seed",
        "semantic_structural_storage_ratio",
        "hgb_raw_file_byte_ratio",
        "support_edge_ratio",
        "validation_micro_f1",
        "validation_macro_f1",
        "test_micro_f1",
        "test_macro_f1",
        "success",
        "status",
        "failed_reason",
    ]
    rows = []
    for method, micro in [
        ("H6-struct40-relgrid-best-random", 0.9490),
        ("H6-struct40-relgrid-best-pathaware-v2-stratified", 0.9495),
    ]:
        for graph_seed in (1, 2, 3):
            for training_seed in (1, 2, 3):
                rows.append(
                    {
                        "dataset": "DBLP",
                        "method": method,
                        "edge_score_strategy": "random_edge_within_relation",
                        "relation_channel_spec": "APPA100-PVVP100-PTTP30",
                        "graph_seed": graph_seed,
                        "training_seed": training_seed,
                        "semantic_structural_storage_ratio": 0.372,
                        "hgb_raw_file_byte_ratio": 0.538,
                        "support_edge_ratio": 0.34,
                        "validation_micro_f1": micro - 0.001,
                        "validation_macro_f1": micro - 0.004,
                        "test_micro_f1": micro,
                        "test_macro_f1": micro - 0.004,
                        "success": True,
                        "status": "success",
                        "failed_reason": "",
                    }
                )
    _write_csv(tmp_path / "gate21_3_raw_rows.csv", rows, fields)
    _write_csv(tmp_path / "gate21_3_edge_score_diagnostics.csv", [], ["dataset", "method", "graph_seed", "score_component_hub_penalty_mean"])
    _write_csv(tmp_path / "gate21_3_coverage_diagnostics.csv", [], ["dataset", "method", "graph_seed", "paper_coverage_ratio"])

    assert main(["--input-dir", str(tmp_path), "--output-dir", str(tmp_path)]) == 0

    decision = json.loads((tmp_path / "gate21_4_decision.json").read_text(encoding="utf-8"))
    assert "PATHAWARE_V2_GAIN_FAIL" in decision["decisions"]
    with (tmp_path / "gate21_4_pathaware_v2_validation.csv").open(newline="", encoding="utf-8") as handle:
        out_rows = list(csv.DictReader(handle))
    assert len(out_rows) == 18
    assert out_rows[0]["edge_score_diagnostics_path"].endswith("gate21_4_edge_score_diagnostics.csv")
