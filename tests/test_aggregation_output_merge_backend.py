from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=4,
        node_type=np.zeros(4, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 2, 3, 0, 1], dtype=np.int64),
                dst=np.array([1, 2, 3, 0, 1, 2], dtype=np.int64),
                weight=np.ones(6, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "r", 0, 0)},
    )


def test_output_merge_backends_are_valid_and_preserve_edge_weight(tmp_path: Path) -> None:
    assignment = Assignment(np.array([0, 0, 1, 1], dtype=np.int64), np.array([0, 0], dtype=np.int32))
    for reducer in ["sort", "direct_relation_writer", "parallel_relation_output_writer", "shard_count_chunk_sweep"]:
        diagnostics: dict[str, object] = {}
        coarse = coarsen_graph_chunked(_graph(), assignment, chunk_size=2, output_dir=tmp_path / reducer, reducer=reducer, aggregation_diagnostics=diagnostics)
        assert reducer == "sort" or diagnostics[f"aggregation_{reducer}_backend"] is True
        assert abs(sum(float(rel.weight.sum()) for rel in coarse.relations.values()) - 6.0) < 1.0e-6
        assert "exclusive_timing_residual_frac" in diagnostics
        assert float(diagnostics["exclusive_timing_residual_frac"]) <= 0.05
        assert diagnostics["num_shards"] >= 1
        assert diagnostics["bytes_written"] >= 0


def test_output_merge_summary_never_recommends_incorrect_backend(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_ogbn_output_merge_backend import summarize_next14_ogbn_output_merge_backend

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = [
        {
            "size": "full-local",
            "method": "HeSF-LVC-P",
            "backend": "A0_current_sort_reducer",
            "run_status": "available",
            "aggregation_total_sec": 10.0,
            "peak_rss_gb": 1.0,
            "correctness_passed": "true",
            "edge_weight_preservation_checks": "passed",
            "exclusive_timing_residual_frac": 0.01,
        },
        {
            "size": "full-local",
            "method": "HeSF-LVC-P",
            "backend": "A6_direct_relation_writer",
            "run_status": "available",
            "aggregation_total_sec": 1.0,
            "peak_rss_gb": 1.0,
            "correctness_passed": "false",
            "edge_weight_preservation_checks": "failed",
            "exclusive_timing_residual_frac": 0.01,
        },
    ]
    with (input_dir / "aggregation_backend_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for name in ["aggregation_backend_exclusive_timing.csv", "aggregation_backend_by_relation.csv", "aggregation_output_merge_diagnostics.csv", "edge_weight_preservation_checks.csv"]:
        with (input_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["size", "method", "backend"])
            writer.writeheader()
    output = tmp_path / "out"
    summarize_next14_ogbn_output_merge_backend(input=input_dir, output=output)
    with (output / "aggregation_backend_speedup_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    bad = next(row for row in summary if row["backend"] == "A6_direct_relation_writer")
    assert bad["recommended"] == "False"
