from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv


MAIN_METHODS = ["AH-UGC-style tuned-global", "HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H0-mutual-best", "GraphZoom-style", "ConvMatch-style", "random"]


def _pm(row: Mapping[str, Any], mean_key: str, std_key: str) -> str:
    mean = as_float(row.get(mean_key), None)
    std = as_float(row.get(std_key), None)
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f}"


def _require_replicated(row: Mapping[str, Any], label: str) -> None:
    if int(float(row.get("run_count", 0) or 0)) <= 1:
        raise ValueError(f"{label} run_count must be > 1 for mean/std paper rows")


def _global_fixed_row(best: Mapping[str, Any]) -> dict[str, Any]:
    _require_replicated(best, "AH-UGC-style tuned-global")
    return {
        "method": "AH-UGC-style tuned-global",
        "result_class": "global_fixed",
        "appendix_only": False,
        "hash_bits": best.get("hash_bits", ""),
        "bucket_topk": best.get("bucket_topk", ""),
        "assignment_source": best.get("assignment_source", ""),
        "target_hit_rate": best.get("target_hit_rate", ""),
        "best_macro_f1_mean": best.get("best_macro_f1_mean", ""),
        "best_macro_f1_std": best.get("best_macro_f1_std", ""),
        "projected_macro_f1_mean": best.get("projected_macro_f1_mean", ""),
        "projected_macro_f1_std": best.get("projected_macro_f1_std", ""),
        "resource_logged_cumulative_dee_mean": best.get("resource_logged_cumulative_dee_mean", ""),
        "resource_logged_cumulative_dee_std": best.get("resource_logged_cumulative_dee_std", ""),
        "relation_energy_error_mean": best.get("relation_energy_error_mean", ""),
        "relation_energy_error_std": best.get("relation_energy_error_std", ""),
        "coarsening_wall_clock_sec_mean": best.get("coarsening_wall_clock_sec_mean", ""),
        "coarsening_wall_clock_sec_std": best.get("coarsening_wall_clock_sec_std", ""),
        "peak_rss_gb_mean": best.get("peak_rss_gb_mean", ""),
        "peak_rss_gb_std": best.get("peak_rss_gb_std", ""),
        "notes": "protocol-matched type-isolated hash/LSH baseline; not official AH-UGC",
    }


def summarize_next13_ahugc_fair_baseline(*, next12_ahugc: str | Path, next12_paper: str | Path, output: str | Path) -> dict[str, Any]:
    next12_ahugc = Path(next12_ahugc)
    next12_paper = Path(next12_paper)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = read_csv(next12_ahugc / "ahugc_style_best_overall.csv")
    if not best:
        raise FileNotFoundError(next12_ahugc / "ahugc_style_best_overall.csv")
    global_row = _global_fixed_row(best[0])
    validation_rows = [
        {
            "method": "AH-UGC-style validation-selected",
            "result_class": "validation_selected_by_dataset",
            "appendix_only": True,
            **row,
            "notes": "selected independently by dataset; appendix only",
        }
        for row in read_csv(next12_ahugc / "ahugc_style_best_config_by_dataset.csv")
    ]
    oracle_source = read_csv(next12_ahugc / "ahugc_style_sweep_by_config.csv")
    oracle = max(oracle_source, key=lambda row: as_float(row.get("best_macro_f1_mean"), -1.0) or -1.0, default={})
    oracle_rows = [
        {
            "method": "AH-UGC-style oracle-max",
            "result_class": "oracle_appendix_only",
            "appendix_only": True,
            **oracle,
            "notes": "oracle over sweep configs; appendix only",
        }
    ] if oracle else []
    external_rows = read_csv(next12_paper / "table3_external_baselines_with_ahugc_tuned_if_available.csv")
    converted = [_main_row_from_global(global_row)]
    for method in MAIN_METHODS[1:]:
        source = next((row for row in external_rows if row.get("method", "") == method), None)
        if source is None:
            continue
        converted.append(_main_row_from_external(source))
    write_csv(output / "ahugc_global_fixed.csv", [global_row])
    write_csv(output / "ahugc_validation_selected_by_dataset.csv", validation_rows)
    write_csv(output / "ahugc_oracle_appendix_only.csv", oracle_rows)
    write_csv(output / "external_baseline_main_table.csv", converted)
    lines = [
        "# Next13 AH-UGC-Style Fair Baseline",
        "",
        "AH-UGC-style is a protocol-matched type-isolated hash/LSH baseline, not an official AH-UGC reproduction.",
        "P/S achieve stronger operator preservation and better task recovery at higher coarsening cost.",
        "",
        markdown_table(converted, ["method", "target_hit_rate", "best_macro_f1_mean_pm_std", "resource_logged_cumulative_dee_mean_pm_std", "relation_energy_error_mean_pm_std", "coarsening_wall_clock_sec_mean_pm_std", "notes"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"main_rows": len(converted)}


def _main_row_from_global(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row.get("method", ""),
        "target_hit_rate": row.get("target_hit_rate", ""),
        "best_macro_f1_mean_pm_std": _pm(row, "best_macro_f1_mean", "best_macro_f1_std"),
        "projected_macro_f1_mean_pm_std": _pm(row, "projected_macro_f1_mean", "projected_macro_f1_std"),
        "resource_logged_cumulative_dee_mean_pm_std": _pm(row, "resource_logged_cumulative_dee_mean", "resource_logged_cumulative_dee_std"),
        "paper_final_dee_mean_pm_std_if_available": "",
        "relation_energy_error_mean_pm_std": _pm(row, "relation_energy_error_mean", "relation_energy_error_std"),
        "coarsening_wall_clock_sec_mean_pm_std": _pm(row, "coarsening_wall_clock_sec_mean", "coarsening_wall_clock_sec_std"),
        "peak_rss_gb_mean_pm_std": _pm(row, "peak_rss_gb_mean", "peak_rss_gb_std"),
        "notes": row.get("notes", ""),
    }


def _main_row_from_external(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row.get("method", ""),
        "target_hit_rate": row.get("target_hit_rate", ""),
        "best_macro_f1_mean_pm_std": _pm(row, "best_mean", "best_std"),
        "projected_macro_f1_mean_pm_std": _pm(row, "projected_mean", "projected_std"),
        "resource_logged_cumulative_dee_mean_pm_std": _pm(row, "cumulative_dee_or_audited_dee_mean", "cumulative_dee_or_audited_dee_std"),
        "paper_final_dee_mean_pm_std_if_available": _pm(row, "paper_final_dee_mean", "paper_final_dee_std"),
        "relation_energy_error_mean_pm_std": _pm(row, "relation_energy_error_mean", "relation_energy_error_std"),
        "coarsening_wall_clock_sec_mean_pm_std": _pm(row, "coarsening_wall_clock_sec_mean", "coarsening_wall_clock_sec_std"),
        "peak_rss_gb_mean_pm_std": _pm(row, "peak_rss_gb_mean", "peak_rss_gb_std"),
        "notes": "Next12 external-style baseline table row",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next12-ahugc", type=Path, required=True)
    parser.add_argument("--next12-paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next13_ahugc_fair_baseline(next12_ahugc=args.next12_ahugc, next12_paper=args.next12_paper, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
