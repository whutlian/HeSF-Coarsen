from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


METRICS = [
    "typed_exact_step_survival_rate",
    "untyped_step_survival_rate",
    "schema_path_survival_gap",
    "endpoint_pair_collapse_rate",
    "any_consecutive_collapse_rate",
    "unique_cluster_ratio",
    "log_path_count_error",
    "path_weight_missing_step_rate",
]


def _norm_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("run_status", "available")) != "failed"]


def _variation_by_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    datasets = sorted({str(row.get("dataset", "")) for row in rows})
    result: dict[str, bool] = {}
    for dataset in datasets:
        sub = [row for row in rows if str(row.get("dataset", "")) == dataset]
        varied = False
        for metric in METRICS:
            values = {
                round(float(value), 8)
                for value in (as_float(row.get(metric), None) for row in sub)
                if value is not None
            }
            if len(values) > 1:
                varied = True
                break
        result[dataset] = varied
    return result


def _varies(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> bool:
    for metric in metrics:
        values = {
            round(float(value), 8)
            for value in (as_float(row.get(metric), None) for row in rows)
            if value is not None
        }
        if len(values) > 1:
            return True
    return False


def summarize_next12_metapath_retention(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    per_sample = _norm_rows(read_csv(input / "path_retention_per_sample.csv"))
    schema_rows = read_csv(input / "method_schema_path_retention.csv")
    method_dataset = read_csv(input / "method_dataset_retention.csv")
    count_rows = read_csv(input / "path_count_preservation_per_sample.csv")

    by_method = aggregate(per_sample, ["method"], METRICS)
    by_dataset = aggregate(per_sample, ["dataset", "method"], METRICS)
    by_schema = aggregate(per_sample, ["dataset", "method", "schema_path"], METRICS)
    gap_table = aggregate(per_sample, ["dataset", "method"], ["schema_path_survival_gap", "typed_exact_step_survival_rate", "untyped_step_survival_rate"])
    collapse_table = aggregate(per_sample, ["dataset", "method"], ["endpoint_pair_collapse_rate", "any_consecutive_collapse_rate", "unique_cluster_ratio"])
    count_table = aggregate(count_rows, ["dataset", "method"], ["log_path_count_error", "path_count_ratio"])

    paper_rows = []
    by_method_index = {row.get("method", ""): row for row in by_method}
    for method, row in sorted(by_method_index.items()):
        paper_rows.append(
            {
                "method": method,
                "typed_exact_step_survival_rate": row.get("typed_exact_step_survival_rate_mean", ""),
                "untyped_step_survival_rate": row.get("untyped_step_survival_rate_mean", ""),
                "survival_gap_untyped_minus_typed": row.get("schema_path_survival_gap_mean", ""),
                "endpoint_pair_collapse_rate": row.get("endpoint_pair_collapse_rate_mean", ""),
                "log_path_count_error": row.get("log_path_count_error_mean", ""),
            }
        )

    write_csv(output / "metapath_retention_by_method.csv", by_method)
    write_csv(output / "metapath_retention_by_dataset.csv", by_dataset)
    write_csv(output / "metapath_retention_by_schema_path.csv", by_schema or schema_rows)
    write_csv(output / "metapath_survival_gap_table.csv", gap_table)
    write_csv(output / "metapath_endpoint_collapse_table.csv", collapse_table)
    write_csv(output / "metapath_path_count_drift_table.csv", count_table)
    write_csv(output / "paper_metapath_rebuttal_table.csv", paper_rows)
    write_png(output / "figures" / "typed_vs_untyped_survival_by_method.png", paper_rows, "typed_exact_step_survival_rate", "untyped_step_survival_rate")
    write_png(output / "figures" / "survival_gap_by_dataset.png", gap_table, "method", "schema_path_survival_gap_mean")
    write_png(output / "figures" / "endpoint_collapse_by_method.png", collapse_table, "method", "endpoint_pair_collapse_rate_mean")
    write_png(output / "figures" / "path_count_error_by_method.png", count_table, "method", "log_path_count_error_mean")

    variation = _variation_by_dataset(per_sample)
    diagnostic = bool(variation) and all(variation.values())
    typed_gap_varies = _varies(per_sample, ["typed_exact_step_survival_rate", "schema_path_survival_gap"])
    diagnostic_scope = (
        "typed survival/gap plus collapse/count diagnostics"
        if typed_gap_varies
        else "collapse/count diagnostics; typed survival and untyped gap stayed flat under this protocol"
    )
    lines = [
        "# Next12 Metapath Retention",
        "",
        "This metric maps shared original typed path samples through method-specific assignments and tests typed coarse survival.",
        "No dense adjacency or explicit relation-product materialization is used.",
        (
            f"Conclusion: diagnostic enough for main text ({diagnostic_scope}); at least one metapath metric varies across methods in every dataset."
            if diagnostic
            else "Conclusion: appendix only; bounded metapath retention did not separate methods in every dataset."
        ),
        "",
        "Dataset variation flags: " + ", ".join(f"{key}={value}" for key, value in sorted(variation.items())),
        "",
        markdown_table(paper_rows[:12], ["method", "typed_exact_step_survival_rate", "untyped_step_survival_rate", "survival_gap_untyped_minus_typed", "endpoint_pair_collapse_rate", "log_path_count_error"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"diagnostic": diagnostic, "variation": variation}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next12_metapath_retention(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
