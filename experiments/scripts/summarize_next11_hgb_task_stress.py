from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


TASK_METRICS = ["projected_macro_f1", "refined_macro_f1@0", "refined_macro_f1@1", "refined_macro_f1@3", "refined_macro_f1@5", "best_macro_f1", "refine_auc_macro_f1"]


def _delta(rows: Sequence[Mapping[str, Any]], baseline: str) -> list[dict[str, Any]]:
    base = {
        (row.get("stress_block", ""), row.get("stress_name", ""), row.get("dataset", ""), row.get("seed", "")): row
        for row in rows
        if row.get("method") == baseline and row.get("run_status") == "available"
    }
    out = []
    for row in rows:
        if row.get("method") == baseline or row.get("run_status") != "available":
            continue
        ref = base.get((row.get("stress_block", ""), row.get("stress_name", ""), row.get("dataset", ""), row.get("seed", "")))
        if not ref:
            continue
        item = {
            "stress_block": row.get("stress_block", ""),
            "stress_name": row.get("stress_name", ""),
            "method": row.get("method", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "baseline": baseline,
        }
        for metric in TASK_METRICS:
            item[f"delta_{metric}"] = (
                as_float(row.get(metric), 0.0) - as_float(ref.get(metric), 0.0)
                if as_float(row.get(metric), None) is not None and as_float(ref.get(metric), None) is not None
                else ""
            )
        out.append(item)
    return out


def _win_rate(delta_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in delta_rows:
        groups.setdefault((str(row.get("stress_block", "")), str(row.get("stress_name", "")), str(row.get("method", "")), str(row.get("baseline", ""))), []).append(row)
    out = []
    for (block, name, method, baseline), group in sorted(groups.items()):
        deltas = [as_float(row.get("delta_best_macro_f1"), None) for row in group]
        deltas = [value for value in deltas if value is not None]
        out.append({"stress_block": block, "stress_name": name, "method": method, "baseline": baseline, "win_rate": sum(v > 0 for v in deltas) / len(deltas) if deltas else "", "mean_delta_best_macro_f1": sum(deltas) / len(deltas) if deltas else ""})
    return out


def summarize_next11_hgb_task_stress(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "task_stress_runs.csv")
    available = [row for row in rows if row.get("run_status") == "available"]
    by_method_dataset = aggregate(available, ["stress_block", "stress_name", "method", "dataset"], TASK_METRICS)
    low = [row for row in by_method_dataset if row.get("stress_block") == "low_label"]
    early = [row for row in by_method_dataset if row.get("stress_block") == "early_refine"]
    cross = [row for row in by_method_dataset if row.get("stress_block") == "cross_model"]
    mask = [row for row in by_method_dataset if row.get("stress_block") == "relation_mask"]
    delta_flatten = _delta(rows, "flatten-sum")
    delta_h6 = _delta(rows, "H6-no-spec")
    wins = _win_rate(delta_flatten) + _win_rate(delta_h6)
    write_csv(output / "task_stress_by_method_dataset.csv", by_method_dataset)
    write_csv(output / "low_label_summary.csv", low)
    write_csv(output / "early_refine_summary.csv", early)
    write_csv(output / "cross_model_summary.csv", cross)
    write_csv(output / "relation_mask_summary.csv", mask)
    write_csv(output / "delta_vs_flatten_sum_by_stress.csv", delta_flatten)
    write_csv(output / "delta_vs_h6_by_stress.csv", delta_h6)
    write_csv(output / "win_rate_by_stress.csv", wins)
    write_png(output / "figures" / "low_label_curves.png", low, "stress_name", "best_macro_f1_mean")
    write_png(output / "figures" / "early_refine_curves.png", early, "stress_name", "best_macro_f1_mean")
    write_png(output / "figures" / "cross_model_bars.png", cross, "stress_name", "best_macro_f1_mean")
    write_png(output / "figures" / "relation_mask_heatmap.png", mask, "stress_name", "best_macro_f1_mean")
    p_wins = [row for row in wins if row.get("method") == "HeSF-LVC-P" and row.get("baseline") == "flatten-sum"]
    task_claim = any(as_float(row.get("mean_delta_best_macro_f1"), -1.0) >= 0.005 and as_float(row.get("win_rate"), 0.0) >= 0.6 for row in p_wins)
    conclusion = (
        "task-superiority stress evidence is present under the configured acceptance rule."
        if task_claim
        else "Task recovery remains competitive, but the main advantage remains operator preservation."
    )
    lines = [
        "# Next11 HGB Task Stress",
        "",
        conclusion,
        "",
        "- Does HeSF-LVC-P beat flatten-sum under low-label? See `low_label_summary.csv` and `delta_vs_flatten_sum_by_stress.csv`.",
        "- Does HeSF-LVC-P beat flatten-sum in projected/refined@0/refined@1? See `early_refine_summary.csv`.",
        "- Does HeSF-LVC-P transfer better to HAN/HGT training? See `cross_model_summary.csv`.",
        "- Does HeSF-LVC-P degrade less under relation masking? See `relation_mask_summary.csv`.",
        "",
        markdown_table(wins[:20], ["stress_block", "stress_name", "method", "baseline", "win_rate", "mean_delta_best_macro_f1"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "win_rate": wins}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_hgb_task_stress(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
