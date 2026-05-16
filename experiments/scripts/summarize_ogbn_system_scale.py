from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv


SYSTEM_COLUMNS = [
    "size",
    "candidate_pairs",
    "scored_pairs",
    "selected_merges",
    "coarse_edges",
    "matching_sec",
    "aggregation_sec",
    "edges_per_sec",
    "pairs_per_sec",
    "rss_gb",
    "shard_gb",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    number = _as_float(value, None)
    if number is None:
        return default
    return int(number)


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _sum_prefix(row: Mapping[str, Any], prefix: str) -> int:
    total = 0
    for key, value in row.items():
        if key.startswith(prefix):
            total += int(float(value or 0))
    return total


def _parse_input(spec: str | Path) -> tuple[str, Path]:
    raw = str(spec)
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    path = Path(raw)
    match = re.search(r"stage_([^_]+(?:_[^_]+)?)_\d{8}", path.name)
    label = match.group(1).replace("_", "-") if match else path.name
    return label, path


def _load_summary_row(summary_dir: Path) -> dict[str, str]:
    candidates = [
        summary_dir / "run_final_summary.csv",
        summary_dir / "final_summary.csv",
        summary_dir / "all_runs.csv",
    ]
    for path in candidates:
        for row in _read_csv(path):
            if (
                _first(row, ("candidate_retained_pair_count", "candidate_count_total"), "") != ""
                or _sum_prefix(row, "generated_candidates_by_source.") > 0
            ):
                return row
    return {}


def _load_resource_row(summary_dir: Path) -> dict[str, str]:
    for path in (
        summary_dir / "resource_summary_runlevel.csv",
        summary_dir / "resource_summary.csv",
    ):
        rows = _read_csv(path)
        if rows:
            return rows[0]
    return {}


def _system_row(label: str, summary_dir: Path) -> dict[str, str]:
    row = _load_summary_row(summary_dir)
    resource = _load_resource_row(summary_dir)
    candidate_pairs = _sum_prefix(row, "generated_candidates_by_source.")
    if candidate_pairs <= 0:
        candidate_pairs = _as_int(_first(row, ("candidate_count_total",), 0), 0) or 0
    scored_pairs = _as_int(
        _first(row, ("candidate_retained_pair_count", "candidate_count_total"), 0),
        0,
    ) or 0
    selected_merges = _as_int(_first(row, ("matched_merges", "matched_units", "matched_pairs"), 0), 0) or 0
    coarse_edges = _sum_prefix(row, "coarse_edge_count_by_relation.")
    aggregation_shard_bytes = _as_float(
        _first(
            resource,
            ("aggregation_shard_bytes", "large_graph_envelope.artifact_bytes_by_name.aggregation_shards"),
            "",
        ),
        None,
    )
    shard_gb = "" if aggregation_shard_bytes is None else aggregation_shard_bytes / (1024.0**3)
    return {
        "size": label,
        "candidate_pairs": str(candidate_pairs),
        "scored_pairs": str(scored_pairs),
        "selected_merges": str(selected_merges),
        "coarse_edges": str(coarse_edges),
        "matching_sec": _fmt(_first(row, ("runtime_by_stage.matching", "large_graph_envelope.runtime_by_stage.matching"))),
        "aggregation_sec": _fmt(
            _first(row, ("runtime_by_stage.aggregation", "large_graph_envelope.runtime_by_stage.aggregation"))
        ),
        "edges_per_sec": _fmt(_first(row, ("edges_aggregated_per_sec",))),
        "pairs_per_sec": _fmt(_first(row, ("candidate_pairs_scored_per_sec", "candidate_pairs_per_sec"))),
        "rss_gb": _fmt(_first(row, ("peak_rss_gb", "peak_cpu_memory_gb"))),
        "shard_gb": _fmt(shard_gb),
    }


def summarize_ogbn_system_scale(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    command_lines: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    output = Path(output)
    rows = [_system_row(label, path) for label, path in (_parse_input(spec) for spec in inputs)]
    write_csv(output / "ogbn_system_scale_table.csv", rows)
    if command_lines:
        (output / "run_commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    report = [
        "# OGBN-MAG system-scale summary",
        "",
        "This table treats OGBN-MAG as protocol/system evidence rather than a task-quality claim.",
        "",
        markdown_table(rows, SYSTEM_COLUMNS),
        "",
        "The bottleneck columns separate matching from aggregation so relation-wise aggregation, "
        "sort-reduce/dedup, per-relation parallel aggregation, edge uniqueness ratio, and aggregation "
        "memory traffic can be discussed directly.",
        "",
    ]
    (output / "ogbn_system_scale_report.md").write_text("\n".join(report), encoding="utf-8")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Entries like 200k=path/to/summary")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_ogbn_system_scale(args.inputs, args.output, command_lines=args.command_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
