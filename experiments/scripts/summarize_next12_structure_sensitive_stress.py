from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


def _relation_energy_index(paper_tables_summary: Path | None) -> dict[str, str]:
    if paper_tables_summary is None:
        return {}
    table2 = paper_tables_summary / "table2_flatten_h6_rebuttal_with_metapath.csv"
    if not table2.exists():
        return {}
    out: dict[str, str] = {}
    for row in read_csv(table2):
        method = str(row.get("method", ""))
        value = row.get("relation_energy_error", "")
        if method and value not in {"", None}:
            out[method] = str(value)
    return out


def _compare(rows: Sequence[Mapping[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("method", "")) == baseline and str(row.get("run_status", "")) == "available":
            by_key[(str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("stress_name", "")))] = row
    groups: dict[tuple[str, str], list[float]] = {}
    wins: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method == baseline or str(row.get("run_status", "")) != "available":
            continue
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("stress_name", "")))
        ref = by_key.get(key)
        delta = as_float(row.get("best_macro_f1"), None)
        ref_value = as_float(ref.get("best_macro_f1") if ref else None, None)
        if delta is None or ref_value is None:
            continue
        gkey = (str(row.get("stress_name", "")), method)
        groups.setdefault(gkey, []).append(float(delta - ref_value))
        wins.setdefault(gkey, []).append(float(delta >= ref_value))
    out = []
    for (stress_name, method), values in sorted(groups.items()):
        suffix = "flatten_sum" if baseline == "flatten-sum" else "H6"
        out.append(
            {
                "stress_name": stress_name,
                "method": method,
                f"win_rate_vs_{suffix}": sum(wins[(stress_name, method)]) / max(len(wins[(stress_name, method)]), 1),
                f"mean_delta_best_vs_{suffix}": sum(values) / max(len(values), 1),
            }
        )
    return out


def summarize_next12_structure_sensitive_stress(
    *,
    input: str | Path,
    output: str | Path,
    paper_tables_summary: str | Path | None = Path("outputs/exp_next12_paper_tables_20260517_summary"),
) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "structure_sensitive_stress_runs.csv")
    relation_energy = _relation_energy_index(Path(paper_tables_summary) if paper_tables_summary else None)
    for row in rows:
        if row.get("relation_energy_error", "") in {"", None}:
            method = str(row.get("method", ""))
            row["relation_energy_error"] = relation_energy.get(method, "")
    available = [row for row in rows if str(row.get("run_status", "")) == "available"]
    by_method = aggregate(available, ["stress_name", "method"], ["projected_macro_f1", "refined@0", "refined@1", "refined@3", "refined@5", "best_macro_f1", "AUC", "relation_energy_error", "metapath_typed_survival"])
    vs_flatten = _compare(available, "flatten-sum")
    vs_h6 = _compare(available, "H6-no-spec")
    merged = {(row["stress_name"], row["method"]): dict(row) for row in vs_flatten}
    for row in vs_h6:
        merged.setdefault((row["stress_name"], row["method"]), {"stress_name": row["stress_name"], "method": row["method"]}).update(row)
    win_rows = list(merged.values())
    write_csv(output / "structure_stress_by_method.csv", by_method)
    write_csv(output / "structure_stress_win_rates.csv", win_rows)
    write_csv(output / "structure_stress_runs.csv", rows)
    write_png(output / "figures" / "stress_delta_vs_flatten_sum.png", win_rows, "method", "mean_delta_best_vs_flatten_sum")
    write_png(output / "figures" / "stress_delta_vs_h6.png", win_rows, "method", "mean_delta_best_vs_H6")
    robust = [
        row
        for row in win_rows
        if str(row.get("method", "")) in {"HeSF-LVC-P", "HeSF-LVC-S"}
        and as_float(row.get("win_rate_vs_flatten_sum"), 0.0) >= 0.6
        and as_float(row.get("win_rate_vs_H6"), 0.0) >= 0.6
    ]
    lines = [
        "# Next12 Structure-Sensitive Stress",
        "",
        (
            "Robustness finding: at least one P/S stress setting beats both flatten-sum and H6 by the configured win-rate rule."
            if robust
            else "Task-superiority remains unsupported: P/S did not meet the configured stress win-rate rule."
        ),
        "",
        markdown_table(win_rows[:20], ["stress_name", "method", "win_rate_vs_flatten_sum", "mean_delta_best_vs_flatten_sum", "win_rate_vs_H6", "mean_delta_best_vs_H6"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"robust_rows": len(robust)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-tables-summary", type=Path, default=Path("outputs/exp_next12_paper_tables_20260517_summary"))
    args = parser.parse_args(argv)
    summarize_next12_structure_sensitive_stress(input=args.input, output=args.output, paper_tables_summary=args.paper_tables_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
