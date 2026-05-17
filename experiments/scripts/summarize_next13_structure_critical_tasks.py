from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


METRICS = ["signal_mse", "signal_mae", "signal_correlation", "signal_cosine", "projected", "refined@0", "refined@1", "refined@3", "refined@5", "best", "AUC"]


def _gap(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        baseline: {
            (row.get("dataset", ""), row.get("seed", ""), row.get("task", "")): row
            for row in rows
            if row.get("method", "") == baseline
        }
        for baseline in ("flatten-sum", "H6-no-spec")
    }
    out = []
    for row in rows:
        task = str(row.get("task", ""))
        metric = "signal_mse" if task == "lowpass_signal_reconstruction" else "best"
        value = as_float(row.get(metric), None)
        if value is None:
            continue
        item: dict[str, Any] = {"dataset": row.get("dataset", ""), "seed": row.get("seed", ""), "task": task, "method": row.get("method", ""), "metric": metric}
        for baseline, index in baselines.items():
            ref = as_float(index.get((row.get("dataset", ""), row.get("seed", ""), task), {}).get(metric), None)
            suffix = "flatten_sum" if baseline == "flatten-sum" else "H6"
            if ref is None:
                item[f"delta_vs_{suffix}"] = ""
                item[f"win_vs_{suffix}"] = ""
            elif metric == "signal_mse":
                item[f"delta_vs_{suffix}"] = float(value - ref)
                item[f"win_vs_{suffix}"] = bool(value <= ref)
            else:
                item[f"delta_vs_{suffix}"] = float(value - ref)
                item[f"win_vs_{suffix}"] = bool(value >= ref)
        out.append(item)
    return out


def summarize_next13_structure_critical_tasks(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "structure_task_runs.csv")
    available = [row for row in rows if row.get("run_status", "available") == "available"]
    by_method = aggregate(available, ["task", "method"], METRICS)
    gaps = _gap(available)
    gap_summary = aggregate(gaps, ["task", "method"], ["delta_vs_flatten_sum", "delta_vs_H6"])
    write_csv(output / "structure_task_runs.csv", rows)
    write_csv(output / "structure_task_by_method.csv", by_method)
    write_csv(output / "structure_task_gap_vs_flatten_h6.csv", gaps)
    write_png(output / "figures" / "lowpass_signal_mse_by_method.png", [row for row in by_method if row.get("task") == "lowpass_signal_reconstruction"], "method", "signal_mse_mean")
    write_png(output / "figures" / "feature_free_best_by_method.png", [row for row in by_method if row.get("task") == "feature_free_label_propagation"], "method", "best_mean")
    lines = [
        "# Next13 Structure-Critical Tasks",
        "",
        "These are synthetic structure-critical diagnostics, not official HGB task performance.",
        "",
        markdown_table(gap_summary, ["task", "method", "delta_vs_flatten_sum_mean", "delta_vs_H6_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next13_structure_critical_tasks(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
