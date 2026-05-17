from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import read_csv


def _index(rows: Sequence[Mapping[str, Any]], key: str = "method") -> dict[str, Mapping[str, Any]]:
    return {str(row.get(key, "")): row for row in rows}


def _first(row: Mapping[str, Any] | None, *keys: str) -> Any:
    if row is None:
        return ""
    for key in keys:
        value = row.get(key, "")
        if value not in {"", None}:
            return value
    return ""


def summarize_next12_paper_tables(
    *,
    rebuttal: str | Path,
    external: str | Path,
    metapath: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    rebuttal = Path(rebuttal)
    external = Path(external)
    metapath = Path(metapath)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hgb = read_csv(rebuttal / "paper_rebuttal_table_aggregate.csv")
    hgb_by_method = _index(hgb)
    external_rows = read_csv(external / "external_baseline_by_method.csv")
    metapath_rows = read_csv(metapath / "metapath_retention_by_method.csv")
    meta_by_method = _index(metapath_rows)

    table1 = []
    for method, row in sorted(hgb_by_method.items()):
        table1.append(
            {
                "method": method,
                "paper_final_dee": _first(row, "cumulative_dee_or_audited_dee_mean", "paper_final_dee_mean"),
                "relation_energy_error": _first(row, "relation_energy_error_mean_mean", "relation_energy_error_mean"),
                "relation_mass_drift": _first(row, "relation_js_drift_mean_mean", "relation_mass_drift_mean"),
                "best_macro_f1": _first(row, "best_macro_f1_mean", "best_mean"),
            }
        )

    table2 = []
    for method, row in sorted(hgb_by_method.items()):
        meta = meta_by_method.get(method)
        table2.append(
            {
                "method": method,
                "paper_final_dee": _first(row, "cumulative_dee_or_audited_dee_mean", "paper_final_dee_mean"),
                "relation_energy_error": _first(row, "relation_energy_error_mean_mean", "relation_energy_error_mean"),
                "relation_mass_drift": _first(row, "relation_js_drift_mean_mean", "relation_mass_drift_mean"),
                "typed_exact_step_survival_rate": _first(meta, "typed_exact_step_survival_rate_mean"),
                "survival_gap_untyped_minus_typed": _first(meta, "schema_path_survival_gap_mean"),
                "endpoint_pair_collapse_rate": _first(meta, "endpoint_pair_collapse_rate_mean"),
                "log_path_count_error": _first(meta, "log_path_count_error_mean"),
                "best_macro_f1": _first(row, "best_macro_f1_mean", "best_mean"),
                "refined_macro_f1@5": _first(row, "refined_macro_f1@5_mean", "refined@5_mean"),
            }
        )

    table3 = [dict(row) for row in external_rows]
    table4 = [
        {
            "method": row.get("method", ""),
            "typed_exact_step_survival_rate": row.get("typed_exact_step_survival_rate_mean", ""),
            "untyped_step_survival_rate": row.get("untyped_step_survival_rate_mean", ""),
            "survival_gap_untyped_minus_typed": row.get("schema_path_survival_gap_mean", ""),
            "endpoint_pair_collapse_rate": row.get("endpoint_pair_collapse_rate_mean", ""),
            "log_path_count_error": row.get("log_path_count_error_mean", ""),
        }
        for row in metapath_rows
    ]
    table5 = [
        {"claim": "P/S preservation-first relation/path structure", "status": "supported_if_metapath_and_operator_tables_agree"},
        {"claim": "P/S task-F1 dominance over flatten/H6", "status": "unsupported"},
        {"claim": "P/S beat full tuned RGCN", "status": "unsupported"},
        {"claim": "AH-UGC-style is official AH-UGC", "status": "unsupported"},
        {"claim": "OGBN-MAG task quality", "status": "unsupported"},
    ]

    write_csv(output / "table1_hgb_main_operator_task.csv", table1)
    write_csv(output / "table2_flatten_h6_rebuttal_with_metapath.csv", table2)
    write_csv(output / "table3_external_baselines_with_ahugc_tuned_if_available.csv", table3)
    write_csv(output / "table4_metapath_diagnostics.csv", table4)
    write_csv(output / "table5_claim_boundary.csv", table5)

    summary_text = (metapath / "summary.md").read_text(encoding="utf-8") if (metapath / "summary.md").exists() else ""
    diagnostic = "diagnostic enough for main text" in summary_text
    note = (
        "Metapath retention is diagnostic enough for main text under the bounded sample protocol; in this run the method separation is in collapse/count distortion, not typed-vs-untyped survival gap."
        if diagnostic
        else "Metapath retention did not provide method-separating evidence under the bounded sample protocol; operator/relation-energy metrics remain the primary rebuttal evidence."
    )
    lines = [
        "# Next12 Paper Tables",
        "",
        note,
        "",
        markdown_table(table2[:10], list(table2[0].keys()) if table2 else ["method"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"diagnostic": diagnostic, "table2_rows": len(table2)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuttal", type=Path, default=Path("outputs/exp_next11_hgb_rebuttal_paper_table_20260517"))
    parser.add_argument("--external", type=Path, default=Path("outputs/exp_next11_hgb_external_baselines_20260517_summary"))
    parser.add_argument("--metapath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next12_paper_tables(rebuttal=args.rebuttal, external=args.external, metapath=args.metapath, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
