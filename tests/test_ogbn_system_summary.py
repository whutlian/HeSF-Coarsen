import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_ogbn_system_summary_reports_required_scale_columns(tmp_path):
    from experiments.scripts.summarize_ogbn_system_scale import summarize_ogbn_system_scale

    summary = tmp_path / "ogbn_mag_next6_stage_200k_20260516_summary"
    _write_csv(
        summary / "run_final_summary.csv",
        [
            {
                "generated_candidates_by_source.bucket": 90,
                "generated_candidates_by_source.onehop": 10,
                "candidate_retained_pair_count": 80,
                "matched_merges": 30,
                "coarse_edge_count_by_relation.0": 1000,
                "coarse_edge_count_by_relation.1": 500,
                "runtime_by_stage.matching": 0.8,
                "runtime_by_stage.aggregation": 1.9,
                "candidate_pairs_scored_per_sec": 400.0,
                "edges_aggregated_per_sec": 500.0,
                "peak_rss_gb": 0.92,
            }
        ],
    )
    _write_csv(
        summary / "resource_summary_runlevel.csv",
        [{"aggregation_shard_bytes": 1073741824}],
    )

    summarize_ogbn_system_scale(["200k=" + str(summary)], tmp_path / "system")

    rows = _read_csv(tmp_path / "system" / "ogbn_system_scale_table.csv")
    assert rows == [
        {
            "size": "200k",
            "candidate_pairs": "100",
            "scored_pairs": "80",
            "selected_merges": "30",
            "coarse_edges": "1500",
            "matching_sec": "0.8",
            "aggregation_sec": "1.9",
            "edges_per_sec": "500",
            "pairs_per_sec": "400",
            "rss_gb": "0.92",
            "shard_gb": "1",
        }
    ]
    report = (tmp_path / "system" / "ogbn_system_scale_report.md").read_text(encoding="utf-8")
    assert "OGBN-MAG system-scale summary" in report
    assert "aggregation" in report

