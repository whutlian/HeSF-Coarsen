from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


METRICS = ["cumulative_dee_or_audited_dee", "FSE", "REEmax", "SIPE", "projected", "refined@0", "refined@1", "refined@3", "refined@5", "best", "AUC", "coarsening_wall_clock_sec", "task_train_wall_clock_sec", "refine_wall_clock_sec", "peak_rss_gb", "peak_vram_allocated_gb"]


def _normalize_rows(rows: list[Mapping[str, str]]) -> list[dict]:
    normalized = []
    for row in rows:
        item = dict(row)
        aliases = {
            "cumulative_dee_or_audited_dee": ("DEE", "dee"),
            "projected": ("projected_macro_f1",),
            "refined@0": ("refined_macro_f1@0",),
            "refined@1": ("refined_macro_f1@1",),
            "refined@3": ("refined_macro_f1@3",),
            "refined@5": ("refined_macro_f1@5",),
            "best": ("best_macro_f1",),
            "AUC": ("refine_auc_macro_f1",),
            "task_train_wall_clock_sec": ("coarse_train_sec",),
        }
        for canonical, sources in aliases.items():
            if item.get(canonical) in {None, ""}:
                for source in sources:
                    if item.get(source) not in {None, ""}:
                        item[canonical] = item[source]
                        break
        normalized.append(item)
    return normalized


def _gap_vs_hesf(rows: list[Mapping[str, str]]) -> list[dict]:
    base = {(row.get("dataset", ""), row.get("seed", "")): row for row in rows if row.get("method") == "HeSF-LVC-P" and row.get("run_status") == "available"}
    out = []
    for row in rows:
        if row.get("method") == "HeSF-LVC-P" or row.get("run_status") != "available":
            continue
        ref = base.get((row.get("dataset", ""), row.get("seed", "")))
        if not ref:
            continue
        out.append(
            {
                "method": row.get("method", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "delta_best_vs_hesf_p": as_float(row.get("best"), as_float(row.get("best_macro_f1"), 0.0)) - as_float(ref.get("best"), as_float(ref.get("best_macro_f1"), 0.0)),
                "delta_dee_vs_hesf_p": as_float(row.get("cumulative_dee_or_audited_dee"), as_float(row.get("DEE"), 0.0)) - as_float(ref.get("cumulative_dee_or_audited_dee"), as_float(ref.get("DEE"), 0.0)),
            }
        )
    return out


def summarize_next11_external_baselines(*, input: str | Path, output: str | Path) -> dict:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = _normalize_rows(read_csv(input / "external_baseline_runs.csv"))
    by_method = aggregate([row for row in rows if row.get("run_status") == "available"], ["method"], METRICS)
    by_dataset = aggregate([row for row in rows if row.get("run_status") == "available"], ["dataset", "method"], METRICS)
    gap = _gap_vs_hesf(rows)
    write_csv(output / "external_baseline_runs.csv", rows)
    write_csv(output / "external_baseline_by_method.csv", by_method)
    write_csv(output / "external_baseline_by_dataset.csv", by_dataset)
    write_csv(output / "external_baseline_gap_vs_hesf.csv", gap)
    write_png(output / "figures" / "external_baseline_task.png", by_method, "cumulative_dee_or_audited_dee_mean", "best_mean")
    write_png(output / "figures" / "external_baseline_operator.png", by_method, "cumulative_dee_or_audited_dee_mean", "REEmax_mean")
    lines = [
        "# Next11 External Baselines",
        "",
        "This is an AH-UGC-style protocol-matched baseline, not an official AH-UGC result.",
        "",
        markdown_table(by_method, ["method", "run_count", "best_mean", "cumulative_dee_or_audited_dee_mean", "peak_rss_gb_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"by_method": by_method}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_external_baselines(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
