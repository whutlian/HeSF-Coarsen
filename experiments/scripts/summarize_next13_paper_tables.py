from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import read_csv
from experiments.scripts.summarize_next13_ahugc_fair_baseline import _global_fixed_row


def _assert_no_bare_dee(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if "DEE" in row:
            raise ValueError("paper-facing tables must not contain bare DEE")


def summarize_next13_paper_tables(*, next12_paper: str | Path, next12_ahugc: str | Path, output: str | Path) -> dict[str, int]:
    next12_paper = Path(next12_paper)
    next12_ahugc = Path(next12_ahugc)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table1 = [dict(row) for row in read_csv(next12_paper / "table1_hgb_main_operator_task.csv")]
    table2_src = read_csv(next12_paper / "table2_flatten_h6_rebuttal_with_metapath.csv")
    table2 = [
        {
            "method": row.get("method", ""),
            "paper_final_dee": row.get("paper_final_dee", ""),
            "relation_energy_error": row.get("relation_energy_error", ""),
            "relation_mass_drift": row.get("relation_mass_drift", ""),
            "best_macro_f1": row.get("best_macro_f1", ""),
            "refined_macro_f1@5": row.get("refined_macro_f1@5", ""),
            "metapath_note": "path-mass metrics reported separately in Next13",
        }
        for row in table2_src
    ] or table1
    best = read_csv(next12_ahugc / "ahugc_style_best_overall.csv")
    if not best:
        raise FileNotFoundError(next12_ahugc / "ahugc_style_best_overall.csv")
    tuned = _global_fixed_row(best[0])
    table3 = [
        {
            "method": "AH-UGC-style tuned-global",
            "result_class": "global_fixed",
            "appendix_only": False,
            "hash_bits": tuned.get("hash_bits", ""),
            "bucket_topk": tuned.get("bucket_topk", ""),
            "assignment_source": tuned.get("assignment_source", ""),
            "best_macro_f1_mean": tuned.get("best_macro_f1_mean", ""),
            "best_macro_f1_std": tuned.get("best_macro_f1_std", ""),
            "target_hit_rate": tuned.get("target_hit_rate", ""),
            "resource_logged_cumulative_dee": tuned.get("resource_logged_cumulative_dee_mean", ""),
            "relation_energy_error": tuned.get("relation_energy_error_mean", ""),
            "coarsening_wall_clock_sec": tuned.get("coarsening_wall_clock_sec_mean", ""),
            "notes": tuned.get("notes", ""),
        }
    ]
    table3.extend(
        {
            "method": "AH-UGC-style validation-selected",
            "result_class": "validation_selected_by_dataset",
            "appendix_only": True,
            **row,
        }
        for row in read_csv(next12_ahugc / "ahugc_style_best_config_by_dataset.csv")
    )
    oracle = max(read_csv(next12_ahugc / "ahugc_style_sweep_by_config.csv"), key=lambda row: float(row.get("best_macro_f1_mean", -1) or -1), default={})
    if oracle:
        table3.append({"method": "AH-UGC-style oracle-max", "result_class": "oracle_appendix_only", "appendix_only": True, **oracle})
    old_external = read_csv(next12_paper / "table3_external_baselines_with_ahugc_tuned_if_available.csv")
    table3.extend(_external_table3_row(row) for row in old_external if row.get("method", "") != "AH-UGC-style")
    table4 = [
        {
            "method": row.get("method", ""),
            "quality_metric": "best_macro_f1",
            "cost_metric": "coarsening_wall_clock_sec",
            "best_macro_f1": row.get("best_macro_f1", row.get("best_mean", "")),
            "coarsening_wall_clock_sec": row.get("coarsening_wall_clock_sec_mean", row.get("coarsening_wall_clock_sec", "")),
        }
        for row in table3
    ]
    table5 = [
        {"claim": "P/S operator-preserving and task-competitive", "status": "supported"},
        {"claim": "P/S task-F1 dominance over flatten/H6", "status": "unsupported"},
        {"claim": "AH-UGC-style is official AH-UGC", "status": "unsupported"},
        {"claim": "metapath survival proves P/S superiority", "status": "unsupported"},
        {"claim": "OGBN-MAG task quality", "status": "unsupported"},
    ]
    for rows in (table1, table2, table3, table4, table5):
        _assert_no_bare_dee(rows)
    write_csv(output / "table1_hgb_main_operator_task.csv", table1)
    write_csv(output / "table2_flatten_h6_rebuttal.csv", table2)
    write_csv(output / "table3_external_baselines_fair.csv", table3)
    write_csv(output / "table4_quality_cost.csv", table4)
    write_csv(output / "table5_claim_boundary.csv", table5)
    lines = [
        "# Next13 Paper Tables",
        "",
        "AH-UGC-style tuned-global is sourced from the full Next12 sweep, not the older weaker default row.",
        "Metapath survival fields are removed from main evidence unless path-mass diagnostics support them.",
        "",
        markdown_table(table3[:10], list(table3[0].keys()) if table3 else ["method"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"table3_rows": len(table3)}


def _external_table3_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "method": row.get("method", ""),
        "result_class": "global_fixed",
        "appendix_only": False,
        "hash_bits": "",
        "bucket_topk": "",
        "assignment_source": "",
        "best_macro_f1_mean": row.get("best_mean", ""),
        "best_macro_f1_std": row.get("best_std", ""),
        "target_hit_rate": row.get("target_hit_rate", ""),
        "resource_logged_cumulative_dee": row.get("cumulative_dee_or_audited_dee_mean", ""),
        "relation_energy_error": row.get("relation_energy_error_mean", ""),
        "coarsening_wall_clock_sec": row.get("coarsening_wall_clock_sec_mean", ""),
        "notes": "Next12 external-style baseline table row",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next12-paper", type=Path, required=True)
    parser.add_argument("--next12-ahugc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next13_paper_tables(next12_paper=args.next12_paper, next12_ahugc=args.next12_ahugc, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
