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
    "holdout_operator_relative_error",
    "holdout_operator_lifted_relative_error",
    "holdout_operator_cosine_similarity",
    "holdout_operator_energy_error",
    "holdout_operator_dirichlet_error",
    "holdout_operator_relation_energy_error",
]


def _baseline_index(rows: Sequence[Mapping[str, Any]], baseline: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row.get("dataset", "")), str(row.get("seed", ""))): row
        for row in rows
        if row.get("method") == baseline
    }


def _gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flatten = _baseline_index(rows, "flatten-sum")
    h6 = _baseline_index(rows, "H6-no-spec")
    out = []
    for row in rows:
        value = as_float(row.get("holdout_operator_relative_error"), None)
        if value is None:
            continue
        key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        ref_flatten = as_float(flatten.get(key, {}).get("holdout_operator_relative_error"), None)
        ref_h6 = as_float(h6.get(key, {}).get("holdout_operator_relative_error"), None)
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "delta_error_vs_flatten_sum": "" if ref_flatten is None else float(value - ref_flatten),
                "delta_error_vs_H6": "" if ref_h6 is None else float(value - ref_h6),
                "win_vs_flatten_sum": "" if ref_flatten is None else bool(value <= ref_flatten),
                "win_vs_H6": "" if ref_h6 is None else bool(value <= ref_h6),
            }
        )
    return out


def _interpret(by_method: Sequence[Mapping[str, Any]]) -> str:
    lookup = {row.get("method", ""): as_float(row.get("holdout_operator_relative_error_mean"), None) for row in by_method}
    p = lookup.get("HeSF-LVC-P")
    s = lookup.get("HeSF-LVC-S")
    flatten = lookup.get("flatten-sum")
    h6 = lookup.get("H6-no-spec")
    if p is None or s is None or flatten is None or h6 is None:
        return "insufficient baseline coverage; keep claim limited to existing operator/relation evidence."
    if p <= flatten and p <= h6 and s <= flatten and s <= h6:
        return "P/S beat flatten-sum and H6 on held-out fused-operator probes; use as main generalization evidence."
    return "P/S do not both beat flatten-sum/H6 on held-out probes; report honestly and keep this diagnostic bounded."


def summarize_next14_operator_holdout(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "holdout_operator_figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "holdout_operator_runs.csv")
    available = [row for row in rows if row.get("run_status", "available") == "available"]
    by_method = aggregate(available, ["method"], METRICS)
    by_dataset = aggregate(available, ["dataset", "method"], METRICS)
    gaps = _gap_rows(available)
    gap_summary = aggregate(gaps, ["method"], ["delta_error_vs_flatten_sum", "delta_error_vs_H6"])
    write_csv(output / "holdout_operator_runs.csv", rows)
    write_csv(output / "holdout_operator_by_method.csv", by_method)
    write_csv(output / "holdout_operator_by_dataset.csv", by_dataset)
    write_csv(output / "holdout_operator_gap_vs_flatten_h6.csv", gaps)
    write_png(output / "holdout_operator_figures" / "holdout_error_by_method.png", by_method, "method", "holdout_operator_relative_error_mean")
    write_png(output / "holdout_operator_figures" / "holdout_cosine_by_method.png", by_method, "method", "holdout_operator_cosine_similarity_mean")
    interpretation = _interpret(by_method)
    lines = [
        "# Next14 Held-Out Fused-Operator Probe",
        "",
        interpretation,
        "",
        markdown_table(by_method, ["method", "holdout_operator_relative_error_mean", "holdout_operator_cosine_similarity_mean", "holdout_operator_energy_error_mean", "holdout_operator_relation_energy_error_mean"]),
        "",
        "## Gap vs Flatten/H6",
        "",
        markdown_table(gap_summary, ["method", "delta_error_vs_flatten_sum_mean", "delta_error_vs_H6_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "available": len(available), "interpretation": interpretation}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next14_operator_holdout(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
