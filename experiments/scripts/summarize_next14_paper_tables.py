from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv


MAIN_ROWS = [
    "HeSF-LVC-P",
    "HeSF-LVC-S",
    "flatten-sum",
    "H6-no-spec",
    "H0-mutual-best",
    "random",
    "GraphZoom-style",
    "ConvMatch-style",
    "TypedHash-ChebHeat tuned-global",
    "TypedHash-raw global",
    "full RGCN default",
    "full RGCN tuned",
    "HAN-small",
    "HGT-lite",
]


def _require_sources(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if path is not None and not path.exists()]
    if missing:
        raise FileNotFoundError("missing required input sources: " + ", ".join(missing))


def _parse_pm(value: Any) -> tuple[float | str, float | str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    match = re.match(r"\s*([-+0-9.eE]+)(?:\s*\+/-\s*([-+0-9.eE]+))?", text)
    if not match:
        return "", ""
    return float(match.group(1)), float(match.group(2)) if match.group(2) is not None else ""


def _no_bare_dee(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if "DEE" in row:
            raise ValueError("paper-facing table contains bare DEE column")


def _bool_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", ""}


def _typedhash_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": "TypedHash-ChebHeat tuned-global",
        "result_class": "global_fixed",
        "appendix_only": False,
        "hash_bits": row.get("hash_bits", ""),
        "bucket_topk": row.get("bucket_topk", ""),
        "assignment_source": row.get("assignment_source", ""),
        "best_macro_f1_mean": row.get("best_macro_f1_mean", ""),
        "best_macro_f1_std": row.get("best_macro_f1_std", ""),
        "target_hit_rate": row.get("target_hit_rate", ""),
        "resource_logged_cumulative_dee": row.get("resource_logged_cumulative_dee", row.get("resource_logged_cumulative_dee_mean", "")),
        "relation_energy_error": row.get("relation_energy_error", row.get("relation_energy_error_mean", "")),
        "coarsening_wall_clock_sec": row.get("coarsening_wall_clock_sec", row.get("coarsening_wall_clock_sec_mean", "")),
        "notes": "TypedHash-ChebHeat; protocol-matched type-isolated hash baseline; not official AH-UGC",
    }


def _external_row(row: Mapping[str, Any]) -> dict[str, Any]:
    best, best_std = _parse_pm(row.get("best_macro_f1_mean_pm_std", row.get("best_macro_f1_mean", "")))
    projected, _ = _parse_pm(row.get("projected_macro_f1_mean_pm_std", row.get("projected_macro_f1_mean", "")))
    dee, dee_std = _parse_pm(row.get("resource_logged_cumulative_dee_mean_pm_std", row.get("resource_logged_cumulative_dee", "")))
    paper_dee, _ = _parse_pm(row.get("paper_final_dee_mean_pm_std_if_available", row.get("paper_final_dee", "")))
    rel, _ = _parse_pm(row.get("relation_energy_error_mean_pm_std", row.get("relation_energy_error", "")))
    sec, _ = _parse_pm(row.get("coarsening_wall_clock_sec_mean_pm_std", row.get("coarsening_wall_clock_sec", "")))
    rss, _ = _parse_pm(row.get("peak_rss_gb_mean_pm_std", row.get("peak_rss_gb", "")))
    return {
        "method": row.get("method", ""),
        "result_class": "global_fixed",
        "appendix_only": False,
        "best_macro_f1_mean": best,
        "best_macro_f1_std": best_std,
        "projected_macro_f1_mean": projected,
        "resource_logged_cumulative_dee": dee,
        "resource_logged_cumulative_dee_std": dee_std,
        "paper_final_dee": paper_dee,
        "relation_energy_error": rel,
        "coarsening_wall_clock_sec": sec,
        "peak_rss_gb": rss,
        "notes": row.get("notes", ""),
    }


def _resource_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row.get("method", ""),
        "result_class": "task_reference",
        "appendix_only": False,
        "best_macro_f1_mean": row.get("best_macro_f1_mean", ""),
        "best_macro_f1_std": row.get("best_macro_f1_std", ""),
        "projected_macro_f1_mean": row.get("best_macro_f1_mean", ""),
        "resource_logged_cumulative_dee": row.get("DEE_mean", ""),
        "resource_logged_cumulative_dee_std": row.get("DEE_std", ""),
        "paper_final_dee": "",
        "relation_energy_error": "",
        "coarsening_wall_clock_sec": row.get("coarsening_wall_clock_sec_mean", row.get("total_wall_clock_sec_mean", "")),
        "peak_rss_gb": row.get("peak_rss_gb_mean", ""),
        "notes": "full graph task reference; not a coarse baseline",
    }


def _quality_cost_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(row.get("result_class", "global_fixed")) != "global_fixed":
        return None
    if str(row.get("appendix_only", "False")).lower() == "true":
        return None
    best = as_float(row.get("best_macro_f1_mean", row.get("best_macro_f1", "")), None)
    sec = as_float(row.get("coarsening_wall_clock_sec", row.get("coarsening_wall_clock_sec_mean", "")), None)
    if best is None or sec is None:
        return None
    return {
        "method": row.get("method", ""),
        "result_class": "global_fixed",
        "appendix_only": False,
        "best_macro_f1_mean": best,
        "resource_logged_cumulative_dee": row.get("resource_logged_cumulative_dee", row.get("resource_logged_cumulative_dee_mean", "")),
        "coarsening_wall_clock_sec": sec,
        "peak_rss_gb": row.get("peak_rss_gb", row.get("peak_rss_gb_mean", "")),
        "point_role": "coarsening" if not str(row.get("method", "")).startswith("full RGCN") else "task_reference",
    }


def summarize_next14_paper_tables(
    *,
    next13_paper: str | Path,
    next13_ahugc: str | Path,
    next13_metapath: str | Path,
    next13_structure: str | Path,
    output: str | Path,
    next13_ogbn: str | Path | None = None,
    next12_paper: str | Path | None = None,
    next10_rebuttal: str | Path | None = None,
    next10_resource: str | Path | None = None,
) -> dict[str, int]:
    next13_paper = Path(next13_paper)
    next13_ahugc = Path(next13_ahugc)
    next13_metapath = Path(next13_metapath)
    next13_structure = Path(next13_structure)
    optional_sources = [Path(path) for path in (next13_ogbn, next12_paper, next10_rebuttal, next10_resource) if path is not None]
    _require_sources([next13_paper, next13_ahugc, next13_metapath, next13_structure, *optional_sources])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    table1_src = read_csv(next13_paper / "table1_hgb_main_operator_task.csv")
    table2_src = read_csv(next13_paper / "table2_flatten_h6_rebuttal.csv")
    typed_source = read_csv(next13_paper / "table3_external_baselines_fair.csv")
    external_source = read_csv(next13_ahugc / "external_baseline_main_table.csv")
    resource_rows = read_csv(Path(next10_resource) / "hgb_resource_logged_by_method.csv") if next10_resource is not None else []

    external_index = {
        _external_row(row)["method"]: _external_row(row)
        for row in external_source
        if "AH-UGC-style" not in str(row.get("method", ""))
    }
    typedhash_source = next((row for row in typed_source if row.get("method") == "AH-UGC-style tuned-global" and row.get("result_class") == "global_fixed"), {})
    typedhash = _typedhash_row(typedhash_source) if typedhash_source else {}
    resource_index = {_resource_row(row)["method"]: _resource_row(row) for row in resource_rows}

    table1 = []
    for row in table1_src:
        table1.append(
            {
                "method": row.get("method", ""),
                "paper_final_dee": row.get("paper_final_dee", ""),
                "relation_energy_error": row.get("relation_energy_error", ""),
                "relation_mass_drift": row.get("relation_mass_drift", ""),
                "best_macro_f1": row.get("best_macro_f1", ""),
                "evidence_role": "main_operator_task",
            }
        )
    for method in ("random", "GraphZoom-style", "ConvMatch-style"):
        if method in external_index:
            row = external_index[method]
            table1.append(
                {
                    "method": method,
                    "paper_final_dee": row.get("paper_final_dee", ""),
                    "relation_energy_error": row.get("relation_energy_error", ""),
                    "relation_mass_drift": "",
                    "best_macro_f1": row.get("best_macro_f1_mean", ""),
                    "evidence_role": "external_style_reference",
                }
            )
    if typedhash:
        table1.append(
            {
                "method": "TypedHash-ChebHeat tuned-global",
                "paper_final_dee": "",
                "relation_energy_error": typedhash.get("relation_energy_error", ""),
                "relation_mass_drift": "",
                "best_macro_f1": typedhash.get("best_macro_f1_mean", ""),
                "evidence_role": "protocol_matched_hash_reference",
            }
        )
    for method in ("full RGCN default", "full RGCN tuned"):
        if method in resource_index:
            row = resource_index[method]
            table1.append(
                {
                    "method": method,
                    "paper_final_dee": "",
                    "relation_energy_error": "",
                    "relation_mass_drift": "",
                    "best_macro_f1": row.get("best_macro_f1_mean", ""),
                    "evidence_role": "full_graph_task_reference_not_coarse_baseline",
                }
            )

    table2 = [
        {
            "method": row.get("method", ""),
            "paper_final_dee": row.get("paper_final_dee", ""),
            "relation_energy_error": row.get("relation_energy_error", ""),
            "relation_mass_drift": row.get("relation_mass_drift", ""),
            "best_macro_f1": row.get("best_macro_f1", ""),
            "refined_macro_f1@5": row.get("refined_macro_f1@5", ""),
            "interpretation": "operator/relation rebuttal; metapath is appendix only",
        }
        for row in table2_src
    ]

    table3 = []
    if typedhash:
        table3.append(typedhash)
    for row in typed_source:
        if row.get("method", "") == "AH-UGC-style tuned-global":
            continue
        if "AH-UGC-style" in row.get("method", ""):
            converted = dict(row)
            converted["method"] = row.get("method", "").replace("AH-UGC-style", "TypedHash")
            converted["appendix_only"] = True
            converted["notes"] = "appendix-only TypedHash sweep result"
            table3.append(converted)
        elif row.get("method"):
            table3.append(_external_row(row))

    quality_sources = [typedhash] if typedhash else []
    quality_sources.extend(external_index.values())
    quality_sources.extend(resource_index.get(method, {}) for method in ("full RGCN default", "full RGCN tuned"))
    table4 = [row for row in (_quality_cost_row(row) for row in quality_sources if row) if row is not None]
    table5 = [
        {"claim": "P/S preservation-first and task-competitive", "status": "supported", "paper_location": "main"},
        {"claim": "P/S task-F1 dominance over flatten-sum/H6", "status": "unsupported", "paper_location": "not used"},
        {"claim": "TypedHash-ChebHeat is official AH-UGC", "status": "unsupported", "paper_location": "not used"},
        {"claim": "metapath/path-mass proves P/S superiority over flatten-sum/H6", "status": "unsupported", "paper_location": "appendix only"},
        {"claim": "OGBN-MAG task quality", "status": "unsupported", "paper_location": "system profiling only"},
    ]

    metapath_appendix = read_csv(next13_metapath / "metapath_mass_by_method.csv")
    structure_appendix = read_csv(next13_structure / "structure_task_by_method.csv")
    missing_rows = []
    table_methods = {str(row.get("method", "")) for row in table1 + table3 + table4}
    for method in MAIN_ROWS:
        if method not in table_methods:
            missing_rows.append({"method": method, "reason": "not available in local Next10/12/13 summaries or appendix-only by design"})

    for rows in (table1, table2, table3, table4, table5):
        _no_bare_dee(rows)
    write_csv(output / "table1_hgb_main_operator_task_final.csv", table1)
    write_csv(output / "table2_flatten_h6_rebuttal_final.csv", table2)
    write_csv(output / "table3_external_typedhash_baselines_final.csv", table3)
    write_csv(output / "table4_quality_cost_final.csv", table4)
    write_csv(output / "table5_claim_boundary_final.csv", table5)
    write_csv(output / "appendix_metapath_mass_diagnostics.csv", metapath_appendix)
    write_csv(output / "appendix_structure_critical_diagnostics.csv", structure_appendix)
    write_csv(output / "missing_rows.csv", missing_rows)
    (output / "missing_rows.md").write_text(
        "# Missing Rows\n\n" + markdown_table(missing_rows, ["method", "reason"]) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Next14 Paper Table Hygiene",
        "",
        "Main evidence: paper_final_dee, relation_energy_error, flatten/H6 rebuttal, quality-cost, and TypedHash fairness.",
        "Appendix-only evidence: metapath/path-mass diagnostics and structure-critical diagnostics.",
        "TypedHash-ChebHeat is a protocol-matched hash baseline and not official AH-UGC.",
        "Full tuned RGCN is a task reference, not an oracle coarse baseline.",
        "",
        "## Main Quality-Cost Rows",
        "",
        markdown_table(table4, ["method", "best_macro_f1_mean", "resource_logged_cumulative_dee", "coarsening_wall_clock_sec", "point_role"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "table1_rows": len(table1),
        "table3_rows": len(table3),
        "table4_rows": len(table4),
        "missing_rows": len(missing_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next13-paper", type=Path, required=True)
    parser.add_argument("--next13-ahugc", type=Path, required=True)
    parser.add_argument("--next13-metapath", type=Path, required=True)
    parser.add_argument("--next13-structure", type=Path, required=True)
    parser.add_argument("--next13-ogbn", type=Path, default=Path("outputs/exp_next13_ogbn_aggregation_backend_20260517_summary"))
    parser.add_argument("--next12-paper", type=Path, default=Path("outputs/exp_next12_paper_tables_20260517_summary"))
    parser.add_argument("--next10-rebuttal", type=Path, default=Path("outputs/exp_next10_hgb_rebuttal_tables_20260517_summary"))
    parser.add_argument("--next10-resource", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next14_paper_tables(
        next13_paper=args.next13_paper,
        next13_ahugc=args.next13_ahugc,
        next13_metapath=args.next13_metapath,
        next13_structure=args.next13_structure,
        next13_ogbn=args.next13_ogbn,
        next12_paper=args.next12_paper,
        next10_rebuttal=args.next10_rebuttal,
        next10_resource=args.next10_resource,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
