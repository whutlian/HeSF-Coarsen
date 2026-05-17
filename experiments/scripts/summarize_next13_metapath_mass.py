from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png
from hesf_coarsen.eval.metapath_mass import classify_metapath_mass_evidence


METRICS = [
    "metapath_mass_relative_error",
    "metapath_mass_rmse",
    "metapath_mass_mae",
    "metapath_probe_cosine_similarity",
    "metapath_probe_correlation",
    "metapath_energy_error",
    "schema_path_mass_js_or_l1",
    "start_node_topk_overlap",
    "terminal_mass_conservation_error",
    "collapse_adjusted_path_error",
]


def _baseline_index(rows: Sequence[Mapping[str, Any]], baseline: str) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("schema_path", ""))): row
        for row in rows
        if row.get("method", "") == baseline
    }


def _gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flatten = _baseline_index(rows, "flatten-sum")
    h6 = _baseline_index(rows, "H6-no-spec")
    out = []
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("schema_path", "")))
        value = as_float(row.get("metapath_mass_relative_error"), None)
        if value is None:
            continue
        ref_flatten = as_float(flatten.get(key, {}).get("metapath_mass_relative_error"), None)
        ref_h6 = as_float(h6.get(key, {}).get("metapath_mass_relative_error"), None)
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "schema_path": row.get("schema_path", ""),
                "method": row.get("method", ""),
                "delta_error_vs_flatten_sum": "" if ref_flatten is None else float(value - ref_flatten),
                "delta_error_vs_H6": "" if ref_h6 is None else float(value - ref_h6),
                "win_vs_flatten_sum": "" if ref_flatten is None else bool(value <= ref_flatten),
                "win_vs_H6": "" if ref_h6 is None else bool(value <= ref_h6),
            }
        )
    return out


def _secondary_rows(next12_metapath_summary: Path) -> list[dict[str, Any]]:
    path = next12_metapath_summary / "metapath_retention_by_method.csv"
    rows = read_csv(path)
    out = []
    for row in rows:
        log_error = as_float(row.get("log_path_count_error_mean"), None)
        collapse = as_float(row.get("endpoint_pair_collapse_rate_mean"), None)
        adjusted = "" if log_error is None or collapse is None else float(log_error * (1.0 + collapse))
        out.append(
            {
                "method": row.get("method", ""),
                "endpoint_pair_collapse_rate": row.get("endpoint_pair_collapse_rate_mean", ""),
                "any_consecutive_collapse_rate": row.get("any_consecutive_collapse_rate_mean", ""),
                "unique_cluster_ratio": row.get("unique_cluster_ratio_mean", ""),
                "log_path_count_error": row.get("log_path_count_error_mean", ""),
                "collapse_adjusted_path_error": adjusted,
            }
        )
    return out


def summarize_next13_metapath_mass(*, input: str | Path, output: str | Path, next12_metapath_summary: str | Path = "outputs/exp_next12_metapath_retention_20260517_summary") -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "metapath_mass_by_run.csv")
    secondary = _secondary_rows(Path(next12_metapath_summary))
    adjusted_by_method = {row.get("method", ""): row.get("collapse_adjusted_path_error", "") for row in secondary}
    for row in rows:
        row["collapse_adjusted_path_error"] = adjusted_by_method.get(row.get("method", ""), "")
    by_method = aggregate(rows, ["method"], METRICS)
    by_dataset = aggregate(rows, ["dataset", "method"], METRICS)
    by_schema = aggregate(rows, ["dataset", "method", "schema_path"], METRICS)
    gaps = _gap_rows(rows)
    gap_summary = aggregate(gaps, ["method"], ["delta_error_vs_flatten_sum", "delta_error_vs_H6"])
    verdict = classify_metapath_mass_evidence(by_method)
    write_csv(output / "metapath_mass_by_run.csv", rows)
    write_csv(output / "metapath_mass_by_method.csv", by_method)
    write_csv(output / "metapath_mass_by_dataset.csv", by_dataset)
    write_csv(output / "metapath_mass_by_schema_path.csv", by_schema)
    write_csv(output / "metapath_collapse_count_secondary.csv", secondary)
    write_csv(output / "metapath_mass_gap_vs_flatten_h6.csv", gaps)
    write_png(output / "figures" / "metapath_mass_error_by_method.png", by_method, "method", "metapath_mass_relative_error_mean")
    write_png(output / "figures" / "metapath_probe_cosine_by_method.png", by_method, "method", "metapath_probe_cosine_similarity_mean")
    write_png(output / "figures" / "metapath_mass_error_by_schema_path.png", by_schema, "schema_path", "metapath_mass_relative_error_mean")
    write_png(output / "figures" / "metapath_mass_gap_vs_flatten_h6.png", gap_summary, "method", "delta_error_vs_flatten_sum_mean")
    write_png(output / "figures" / "collapse_adjusted_path_error_by_method.png", secondary, "method", "collapse_adjusted_path_error")
    lines = [
        "# Next13 Metapath Mass",
        "",
        f"Paper location: {verdict['paper_location']}.",
        f"Reason: {verdict['reason']}.",
        "",
        markdown_table(by_method, ["method", "metapath_mass_relative_error_mean", "metapath_probe_cosine_similarity_mean", "metapath_energy_error_mean", "collapse_adjusted_path_error_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"paper_location": verdict["paper_location"], "rows": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--next12-metapath-summary", type=Path, default=Path("outputs/exp_next12_metapath_retention_20260517_summary"))
    args = parser.parse_args(argv)
    summarize_next13_metapath_mass(input=args.input, output=args.output, next12_metapath_summary=args.next12_metapath_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
