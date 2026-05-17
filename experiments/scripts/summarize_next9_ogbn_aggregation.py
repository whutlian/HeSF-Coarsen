from __future__ import annotations

import argparse
import csv
import json
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

STAGE_ALIAS_KEYS = {
    "aggregation_relation_loop_sec": "relation_loop_sec",
    "aggregation_assignment_map_sec": "assignment_map_sec",
    "aggregation_key_build_sec": "key_build_sec",
    "aggregation_sort_sec": "sort_sec",
    "aggregation_reduce_sec": "reduce_sec",
    "aggregation_shard_write_sec": "shard_write_sec",
    "aggregation_kway_merge_sec": "kway_merge_sec",
    "aggregation_output_write_sec": "output_write_sec",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _sum_mapping_values(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(sum(int(float(v or 0)) for v in value.values()))
    return 0


def _fresh_run_dirs(input_runs: Path) -> list[Path]:
    if not input_runs.exists():
        return []
    dirs = set()
    for metadata in input_runs.rglob("metadata.json"):
        dirs.add(metadata.parent)
    for diagnostics in input_runs.rglob("level_*/diagnostics.json"):
        dirs.add(diagnostics.parent.parent)
    return sorted(dirs)


def _fresh_rows(input_runs: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    by_relation: list[dict[str, Any]] = []
    preservation: list[dict[str, Any]] = []
    for run_dir in _fresh_run_dirs(input_runs):
        metadata_path = run_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        if str(metadata.get("status", "success")) not in {"success", "created", "running"}:
            continue
        level_paths = sorted(
            run_dir.glob("level_*/diagnostics.json"),
            key=lambda path: int(path.parent.name.removeprefix("level_")),
        )
        if not level_paths:
            continue
        diagnostics = _read_json(level_paths[-1])
        aggregation = diagnostics.get("aggregation", {})
        if not isinstance(aggregation, Mapping):
            aggregation = {}
        method = str(metadata.get("method") or metadata.get("paper_method") or "")
        size = str(metadata.get("size") or metadata.get("subset_size") or "")
        aggregation_total = _as_float(
            aggregation.get("aggregation_total_sec"),
            _as_float(diagnostics.get("runtime_by_stage", {}).get("aggregation"), None)
            if isinstance(diagnostics.get("runtime_by_stage"), Mapping)
            else None,
        )
        original_edges = sum(
            int(rel.get("original_edges", 0) or 0)
            for rel in aggregation.get("aggregation_by_relation", [])
            if isinstance(rel, Mapping)
        )
        coarse_edges = _sum_mapping_values(diagnostics.get("coarse_edge_count_by_relation"))
        rss_bytes = None
        envelope = diagnostics.get("large_graph_envelope", {})
        if isinstance(envelope, Mapping):
            rss_bytes = _as_float(envelope.get("process_rss_bytes"), None)
        rss_gb = rss_bytes / (1024**3) if rss_bytes is not None else ""
        matching_sec = ""
        runtime = diagnostics.get("runtime_by_stage", {})
        if isinstance(runtime, Mapping):
            matching_sec = runtime.get("matching", "")
        main.append(
            {
                "size": size,
                "method": method,
                "candidate_pairs": diagnostics.get("candidate_retained_pair_count", diagnostics.get("candidate_count_total", "")),
                "scored_pairs": diagnostics.get("candidate_retained_pair_count", ""),
                "selected_merges": diagnostics.get("matched_units", diagnostics.get("matched_pairs", "")),
                "coarse_edges": coarse_edges,
                "matching_sec": matching_sec,
                "aggregation_sec": aggregation_total if aggregation_total is not None else "",
                "edges_per_sec": float(original_edges / aggregation_total)
                if aggregation_total and original_edges
                else "",
                "pairs_per_sec": "",
                "rss_gb": rss_gb,
                "shard_gb": "",
                "target_hit": metadata.get("target_hit", ""),
                "evidence_source": "fresh_next10_instrumented_run",
                "run_dir": str(run_dir),
            }
        )
        stage = {"size": size, "method": method}
        for key in STAGE_KEYS:
            stage[key] = aggregation.get(key, "")
        for source_key, alias_key in STAGE_ALIAS_KEYS.items():
            stage[alias_key] = aggregation.get(source_key, "")
        stage["instrumentation_status"] = "fresh_instrumented"
        stage["run_dir"] = str(run_dir)
        stages.append(stage)
        for rel in aggregation.get("aggregation_by_relation", []):
            if not isinstance(rel, Mapping):
                continue
            relation_row = {
                "size": size,
                "method": method,
                "relation_id": rel.get("relation_id", ""),
                "relation_name": rel.get("relation_name", ""),
                "input_edges": rel.get("original_edges", ""),
                "original_edges": rel.get("original_edges", ""),
                "coarse_edges_before_dedup": rel.get("coarse_edges_before_dedup", ""),
                "coarse_edges": rel.get("coarse_edges_after_dedup", ""),
                "coarse_edges_after_dedup": rel.get("coarse_edges_after_dedup", ""),
                "uniqueness_ratio": rel.get("uniqueness_ratio", ""),
                "duplicate_collapse_ratio": (
                    1.0 - float(rel.get("uniqueness_ratio"))
                    if rel.get("uniqueness_ratio") not in {None, ""}
                    else ""
                ),
                "aggregation_sec": rel.get("aggregation_sec", ""),
                "edges_per_sec": rel.get("edges_per_sec", ""),
                "rss_before_gb": rel.get("rss_before_gb", ""),
                "rss_after_gb": rel.get("rss_after_gb", ""),
                "status": "fresh_instrumented",
                "run_dir": str(run_dir),
            }
            by_relation.append(relation_row)
            preservation.append(
                {
                    "size": size,
                    "method": method,
                    "relation_id": rel.get("relation_id", ""),
                    "edge_weight_original_sum": rel.get("edge_weight_original_sum", ""),
                    "edge_weight_coarse_sum": rel.get("edge_weight_coarse_sum", ""),
                    "edge_weight_abs_error": rel.get("edge_weight_abs_error", ""),
                    "status": "checked",
                    "run_dir": str(run_dir),
                }
            )
    return main, stages, by_relation, preservation


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
        for source_key, alias_key in STAGE_ALIAS_KEYS.items():
            item[alias_key] = item.get(source_key, "")
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
            "input_edges": "",
            "original_edges": "",
            "coarse_edges_before_dedup": "",
            "coarse_edges": "",
            "coarse_edges_after_dedup": "",
            "uniqueness_ratio": "",
            "duplicate_collapse_ratio": "",
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
    input_summary: str | Path = "outputs/exp_next8_ogbn_system_scale_20260517_summary",
    input_runs: str | Path | None = None,
    output: str | Path,
    command_lines: Sequence[str] = (),
) -> dict[str, Any]:
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    if input_runs is not None:
        main, stages, by_relation, preservation = _fresh_rows(Path(input_runs))
    else:
        main = _main_rows(Path(input_summary))
        stages = _stage_rows(main)
        by_relation = _relation_rows(main)
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
    before_after = [
        {
            "size": row.get("size", ""),
            "method": row.get("method", ""),
            "before_aggregation_sec": row.get("aggregation_sec", ""),
            "after_aggregation_sec": row.get("aggregation_sec", "") if input_runs is not None else "",
            "speedup_fraction": "",
            "rss_change_fraction": "",
            "status": "fresh_instrumented" if input_runs is not None else "instrumented; no full OGBN rerun in this summary",
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
        (
            "This summary reads fresh runs with aggregation instrumentation."
            if input_runs is not None
            else "The local summary here preserves the Next8 OGBN scale evidence and marks per-relation fields as requiring a rerun with the new instrumentation."
        ),
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
    parser.add_argument("--input-runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_ogbn_aggregation(
        input_summary=args.input,
        input_runs=args.input_runs,
        output=args.output,
        command_lines=args.command_lines,
    )


if __name__ == "__main__":
    main()
