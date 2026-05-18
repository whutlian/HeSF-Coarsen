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


MAIN_METHODS = [
    "TypedHash-ChebHeat tuned-global",
    "TypedHash-raw global",
    "HeSF-LVC-P",
    "HeSF-LVC-S",
    "flatten-sum",
    "H0-mutual-best",
    "GraphZoom-style",
    "ConvMatch-style",
    "random",
]


def _parse_pm(value: Any) -> tuple[float | str, float | str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    match = re.match(r"\s*([-+0-9.eE]+)(?:\s*\+/-\s*([-+0-9.eE]+))?", text)
    if not match:
        return "", ""
    return float(match.group(1)), float(match.group(2)) if match.group(2) is not None else ""


def _typedhash_global(best: Mapping[str, Any]) -> dict[str, Any]:
    method = "TypedHash-ChebHeat tuned-global" if best.get("assignment_source") == "chebheat_sketch" else "TypedHash-raw global"
    return {
        "method": method,
        "result_class": "global_fixed",
        "appendix_only": False,
        "hash_bits": best.get("hash_bits", ""),
        "bucket_topk": best.get("bucket_topk", ""),
        "assignment_source": best.get("assignment_source", ""),
        "target_hit_rate": best.get("target_hit_rate", ""),
        "best_macro_f1_mean": best.get("best_macro_f1_mean", ""),
        "best_macro_f1_std": best.get("best_macro_f1_std", ""),
        "projected_macro_f1_mean": best.get("projected_macro_f1_mean", ""),
        "resource_logged_cumulative_dee_mean": best.get("resource_logged_cumulative_dee_mean", ""),
        "resource_logged_cumulative_dee_std": best.get("resource_logged_cumulative_dee_std", ""),
        "paper_final_dee_mean_if_available": "",
        "relation_energy_error_mean": best.get("relation_energy_error_mean", ""),
        "coarsening_wall_clock_sec_mean": best.get("coarsening_wall_clock_sec_mean", ""),
        "peak_rss_gb_mean": best.get("peak_rss_gb_mean", ""),
        "notes": "protocol-matched type-isolated hash baseline; not official AH-UGC",
    }


def _external_row(row: Mapping[str, Any]) -> dict[str, Any]:
    best_mean, best_std = _parse_pm(row.get("best_macro_f1_mean_pm_std", row.get("best_macro_f1_mean", "")))
    projected_mean, _projected_std = _parse_pm(row.get("projected_macro_f1_mean_pm_std", row.get("projected_macro_f1_mean", "")))
    dee_mean, dee_std = _parse_pm(row.get("resource_logged_cumulative_dee_mean_pm_std", row.get("resource_logged_cumulative_dee", "")))
    paper_dee_mean, _paper_dee_std = _parse_pm(row.get("paper_final_dee_mean_pm_std_if_available", row.get("paper_final_dee_mean_if_available", "")))
    rel_mean, _rel_std = _parse_pm(row.get("relation_energy_error_mean_pm_std", row.get("relation_energy_error", "")))
    sec_mean, _sec_std = _parse_pm(row.get("coarsening_wall_clock_sec_mean_pm_std", row.get("coarsening_wall_clock_sec", "")))
    rss_mean, _rss_std = _parse_pm(row.get("peak_rss_gb_mean_pm_std", row.get("peak_rss_gb", "")))
    return {
        "method": row.get("method", ""),
        "result_class": "global_fixed",
        "appendix_only": False,
        "target_hit_rate": row.get("target_hit_rate", ""),
        "best_macro_f1_mean": best_mean,
        "best_macro_f1_std": best_std,
        "projected_macro_f1_mean": projected_mean,
        "resource_logged_cumulative_dee_mean": dee_mean,
        "resource_logged_cumulative_dee_std": dee_std,
        "paper_final_dee_mean_if_available": paper_dee_mean,
        "relation_energy_error_mean": rel_mean,
        "coarsening_wall_clock_sec_mean": sec_mean,
        "peak_rss_gb_mean": rss_mean,
        "notes": row.get("notes", "paper table source row"),
    }


def summarize_next14_typedhash_baseline(*, next12_ahugc: str | Path, next13_paper: str | Path, output: str | Path) -> dict[str, int]:
    next12_ahugc = Path(next12_ahugc)
    next13_paper = Path(next13_paper)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best_rows = read_csv(next12_ahugc / "ahugc_style_best_overall.csv")
    if not best_rows:
        raise FileNotFoundError(next12_ahugc / "ahugc_style_best_overall.csv")
    main_rows = [_typedhash_global(best_rows[0])]
    raw_candidates = [row for row in read_csv(next12_ahugc / "ahugc_style_sweep_by_config.csv") if row.get("assignment_source") in {"raw_feature", "raw"}]
    if raw_candidates:
        raw = max(raw_candidates, key=lambda row: as_float(row.get("best_macro_f1_mean"), -1.0) or -1.0)
        raw_row = _typedhash_global(raw)
        raw_row["method"] = "TypedHash-raw global"
        main_rows.append(raw_row)
    external_source = read_csv(next13_paper / "table3_external_baselines_fair.csv")
    if not external_source:
        external_source = read_csv(next13_paper / "external_baseline_main_table.csv")
    for method in MAIN_METHODS:
        if method.startswith("TypedHash"):
            continue
        source = next((row for row in external_source if row.get("method") == method), None)
        if source is not None:
            main_rows.append(_external_row(source))
    validation = [
        {
            "method": "TypedHash validation-selected-by-dataset",
            "result_class": "validation_selected_by_dataset",
            "appendix_only": True,
            **row,
            "notes": "appendix only; selected by dataset",
        }
        for row in read_csv(next12_ahugc / "ahugc_style_best_config_by_dataset.csv")
    ]
    oracle_source = read_csv(next12_ahugc / "ahugc_style_sweep_by_config.csv")
    oracle_rows: list[dict[str, Any]] = []
    if oracle_source:
        oracle = max(oracle_source, key=lambda row: as_float(row.get("best_macro_f1_mean"), -1.0) or -1.0)
        oracle_rows.append({"method": "TypedHash oracle-max", "result_class": "oracle_appendix_only", "appendix_only": True, **oracle, "notes": "appendix-only oracle over sweep"})
    hesf_p = next((row for row in main_rows if row.get("method") == "HeSF-LVC-P"), {})
    hesf_s = next((row for row in main_rows if row.get("method") == "HeSF-LVC-S"), {})
    gaps = []
    for row in main_rows:
        if str(row.get("method", "")).startswith("TypedHash"):
            typed_best = as_float(row.get("best_macro_f1_mean"), None)
            typed_dee = as_float(row.get("resource_logged_cumulative_dee_mean"), None)
            gaps.append(
                {
                    "method": row.get("method", ""),
                    "delta_best_vs_HeSF_LVC_P": typed_best - as_float(hesf_p.get("best_macro_f1_mean"), typed_best) if typed_best is not None else "",
                    "delta_best_vs_HeSF_LVC_S": typed_best - as_float(hesf_s.get("best_macro_f1_mean"), typed_best) if typed_best is not None else "",
                    "delta_resource_logged_cumulative_dee_vs_HeSF_LVC_P": typed_dee - as_float(hesf_p.get("resource_logged_cumulative_dee_mean"), typed_dee) if typed_dee is not None else "",
                }
            )
    quality = [
        {
            "method": row.get("method", ""),
            "result_class": row.get("result_class", ""),
            "appendix_only": row.get("appendix_only", ""),
            "best_macro_f1_mean": row.get("best_macro_f1_mean", ""),
            "resource_logged_cumulative_dee_mean": row.get("resource_logged_cumulative_dee_mean", ""),
            "coarsening_wall_clock_sec_mean": row.get("coarsening_wall_clock_sec_mean", ""),
            "peak_rss_gb_mean": row.get("peak_rss_gb_mean", ""),
        }
        for row in main_rows
    ]
    write_csv(output / "typedhash_main_table.csv", main_rows)
    write_csv(output / "typedhash_appendix_validation_selected.csv", validation)
    write_csv(output / "typedhash_appendix_oracle.csv", oracle_rows)
    write_csv(output / "typedhash_gap_vs_hesf.csv", gaps)
    write_csv(output / "typedhash_quality_cost.csv", quality)
    lines = [
        "# Next14 TypedHash Fair Baseline",
        "",
        "TypedHash-ChebHeat is a protocol-matched type-isolated hash baseline using ChebHeat sketches; it is not official AH-UGC.",
        "Validation-selected and oracle rows are appendix only.",
        "",
        markdown_table(main_rows, list(main_rows[0].keys()) if main_rows else ["method"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"main_rows": len(main_rows), "validation_rows": len(validation), "oracle_rows": len(oracle_rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next12-ahugc", type=Path, required=True)
    parser.add_argument("--next13-paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next14_typedhash_baseline(next12_ahugc=args.next12_ahugc, next13_paper=args.next13_paper, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
