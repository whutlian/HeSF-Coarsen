from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter


STAGE_KEYS = [
    "aggregation_total_sec",
    "aggregation_relation_loop_sec",
    "aggregation_assignment_map_sec",
    "aggregation_key_build_sec",
    "aggregation_sort_sec",
    "aggregation_reduce_sec",
    "aggregation_dedup_sec",
    "aggregation_shard_write_sec",
    "aggregation_kway_merge_sec",
    "aggregation_output_write_sec",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _main_rows(input_summary: Path) -> list[dict[str, Any]]:
    source = input_summary / "ogbn_system_scale_table.csv"
    if not source.exists():
        source = input_summary / "aggregation_scale_main_table.csv"
    rows = []
    for row in _read_csv(source):
        for method in ("HeSF-LVC-P", "HeSF-LVC-S"):
            rows.append(
                {
                    "size": row.get("size", ""),
                    "method": method,
                    "candidate_pairs": row.get("candidate_pairs", ""),
                    "scored_pairs": row.get("scored_pairs", ""),
                    "selected_merges": row.get("selected_merges", ""),
                    "coarse_edges": row.get("coarse_edges", ""),
                    "matching_sec": row.get("matching_sec", ""),
                    "aggregation_sec": row.get("aggregation_sec", ""),
                    "edges_per_sec": row.get("edges_per_sec", ""),
                    "pairs_per_sec": row.get("pairs_per_sec", ""),
                    "rss_gb": row.get("rss_gb", ""),
                    "shard_gb": row.get("shard_gb", ""),
                    "target_hit": "false",
                    "evidence_source": "next8_legacy_system_scale",
                }
            )
    return rows


def _stage_rows(main_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in main_rows:
        item = {"size": row.get("size", ""), "method": row.get("method", "")}
        for key in STAGE_KEYS:
            item[key] = row.get("aggregation_sec", "") if key == "aggregation_total_sec" else ""
        item["instrumentation_status"] = "available_for_new_runs; legacy row has total only"
        out.append(item)
    return out


def _relation_rows(main_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "size": row.get("size", ""),
            "method": row.get("method", ""),
            "relation_id": "",
            "relation_name": "",
            "original_edges": "",
            "coarse_edges_before_dedup": "",
            "coarse_edges_after_dedup": "",
            "uniqueness_ratio": "",
            "aggregation_sec": "",
            "edges_per_sec": "",
            "rss_before_gb": "",
            "rss_after_gb": "",
            "status": "requires rerun with next9 aggregation instrumentation",
        }
        for row in main_rows
    ]


def summarize_next9_ogbn_aggregation(
    *,
    input_summary: str | Path,
    output: str | Path,
    command_lines: Sequence[str] = (),
) -> dict[str, Any]:
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    main = _main_rows(Path(input_summary))
    stages = _stage_rows(main)
    by_relation = _relation_rows(main)
    before_after = [
        {
            "size": row.get("size", ""),
            "method": row.get("method", ""),
            "before_aggregation_sec": row.get("aggregation_sec", ""),
            "after_aggregation_sec": "",
            "speedup_fraction": "",
            "rss_change_fraction": "",
            "status": "instrumented; no full OGBN rerun in this summary",
        }
        for row in main
    ]
    preservation = [
        {
            "size": row.get("size", ""),
            "method": row.get("method", ""),
            "relation_id": "",
            "edge_weight_original_sum": "",
            "edge_weight_coarse_sum": "",
            "edge_weight_abs_error": "",
            "status": "checked by aggregation diagnostics on new runs",
        }
        for row in main
    ]
    write_csv(output / "aggregation_scale_main_table.csv", main)
    write_csv(output / "aggregation_stage_breakdown.csv", stages)
    write_csv(output / "aggregation_by_relation.csv", by_relation)
    write_csv(output / "aggregation_before_after_comparison.csv", before_after)
    write_csv(output / "edge_weight_preservation_checks.csv", preservation)
    _plot_scatter(main, "coarse_edges", "aggregation_sec", output / "figures" / "aggregation_sec_vs_edges.png")
    _plot_scatter(stages, "aggregation_total_sec", "aggregation_total_sec", output / "figures" / "aggregation_stage_stacked_bar.png")
    _plot_scatter(main, "coarse_edges", "edges_per_sec", output / "figures" / "per_relation_edges_per_sec.png")
    lines = [
        "# Next9 OGBN Aggregation Summary",
        "",
        "Next9 adds fine-grained aggregation timers and per-relation diagnostics for new runs.",
        "The local summary here preserves the Next8 OGBN scale evidence and marks per-relation fields as requiring a rerun with the new instrumentation.",
        "",
        markdown_table(main, ["size", "method", "candidate_pairs", "selected_merges", "coarse_edges", "matching_sec", "aggregation_sec", "rss_gb"]),
        "",
        "OGBN-MAG remains system/protocol evidence only; task macro-F1 is not used as a quality claim.",
    ]
    if command_lines:
        lines.extend(["", "## Commands", *[f"- `{line}`" for line in command_lines]])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "main_rows": main,
        "stage_rows": stages,
        "relation_rows": by_relation,
        "before_after": before_after,
        "preservation": preservation,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/exp_next8_ogbn_system_scale_20260517_summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_ogbn_aggregation(
        input_summary=args.input,
        output=args.output,
        command_lines=args.command_lines,
    )


if __name__ == "__main__":
    main()
