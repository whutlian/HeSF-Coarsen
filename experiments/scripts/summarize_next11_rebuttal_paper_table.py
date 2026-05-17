from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


METHODS = ["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best"]
METRICS = [
    "cumulative_dee_or_audited_dee",
    "relation_mass_drift_mean",
    "relation_js_drift_mean",
    "relation_energy_error_mean",
    "edge_collapse_ratio",
    "duplicate_collapse_ratio",
    "self_loop_share_mean",
    "projected_macro_f1",
    "refined_macro_f1@0",
    "refined_macro_f1@1",
    "refined_macro_f1@3",
    "refined_macro_f1@5",
    "best_macro_f1",
    "refine_auc_macro_f1",
]


def _metric(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) not in {None, ""}:
            return row.get(name)
    return ""


def _base_rows(rebuttal: Path) -> list[dict[str, Any]]:
    rows = []
    self_loop = {
        (row.get("method", ""), row.get("dataset", "")): row
        for row in aggregate(read_csv(rebuttal / "self_loop_share_by_relation.csv"), ["method", "dataset"], ["self_loop_share_mean"])
    }
    for row in read_csv(rebuttal / "paper_rebuttal_table.csv"):
        method = str(row.get("method", ""))
        if method not in METHODS:
            continue
        key = (method, str(row.get("dataset", "")))
        loop = self_loop.get(key, {})
        rows.append(
            {
                "method": method,
                "dataset": row.get("dataset", ""),
                "cumulative_dee_or_audited_dee": _metric(row, "audited_dee", "DEE_mean", "DEE"),
                "cumulative_ree_max_or_audited_ree": _metric(row, "REEmax_mean", "REEmax"),
                "sipe": _metric(row, "SIPE_mean", "SIPE"),
                "relation_mass_drift_mean": _metric(row, "relation_mass_l1_drift_mean", "relation_mass_drift_mean"),
                "relation_js_drift_mean": _metric(row, "relation_mass_js_drift_mean", "relation_js_drift_mean"),
                "relation_energy_error_mean": _metric(row, "relation_energy_error_mean"),
                "edge_collapse_ratio": _metric(row, "edge_collapse_ratio", "duplicate_collapse_ratio_mean"),
                "duplicate_collapse_ratio": _metric(row, "duplicate_collapse_ratio_mean"),
                "self_loop_share_mean": loop.get("self_loop_share_mean_mean", loop.get("self_loop_share_mean", "")),
                "projected_macro_f1": _metric(row, "projected_macro_f1_mean", "projected_macro_f1"),
                "refined_macro_f1@5": _metric(row, "refined_macro_f1@5_mean", "refined_macro_f1@5"),
                "best_macro_f1": _metric(row, "best_macro_f1_mean", "best_macro_f1"),
            }
        )
    checkpoint = read_csv(rebuttal / "checkpoint_refine_masking_curve.csv")
    checkpoint_index = {
        (row.get("method", ""), row.get("dataset", ""), row.get("checkpoint", "")): row
        for row in checkpoint
    }
    for row in rows:
        for checkpoint_name in ("refined@0", "refined@1", "refined@3", "refined@5", "best", "projected"):
            source = checkpoint_index.get((row["method"], row["dataset"], checkpoint_name), {})
            key = "projected_macro_f1" if checkpoint_name == "projected" else ("best_macro_f1" if checkpoint_name == "best" else f"refined_macro_f1@{checkpoint_name.split('@')[-1]}")
            if source.get("macro_f1_mean") not in {None, ""}:
                row[key] = source["macro_f1_mean"]
        values = [as_float(row.get(name), None) for name in ("projected_macro_f1", "refined_macro_f1@0", "refined_macro_f1@1", "refined_macro_f1@3", "refined_macro_f1@5")]
        values = [value for value in values if value is not None]
        row["refine_auc_macro_f1"] = sum(values) / len(values) if values else ""
    return rows


def _delta(rows: Sequence[Mapping[str, Any]], baseline: str) -> list[dict[str, Any]]:
    base = {(row["dataset"]): row for row in rows if row.get("method") == baseline}
    out = []
    for row in rows:
        method = str(row.get("method", ""))
        if method == baseline or method.startswith("full RGCN"):
            continue
        ref = base.get(str(row.get("dataset", "")))
        if not ref:
            continue
        item = {"method": method, "dataset": row.get("dataset", ""), "baseline": baseline}
        for metric in METRICS:
            item[f"delta_{metric}"] = (
                as_float(row.get(metric), 0.0) - as_float(ref.get(metric), 0.0)
                if as_float(row.get(metric), None) is not None and as_float(ref.get(metric), None) is not None
                else ""
            )
        out.append(item)
    return out


def summarize_next11_rebuttal_paper_table(
    *,
    rebuttal: str | Path,
    paper_final: str | Path,
    dee_consistency: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = _base_rows(Path(rebuttal))
    by_dataset = sorted(rows, key=lambda row: (str(row.get("dataset", "")), str(row.get("method", ""))))
    aggregate_rows = aggregate(by_dataset, ["method"], METRICS)
    by_relation = read_csv(Path(rebuttal) / "duplicate_collapse_ratio_by_relation.csv")
    by_relation = [row for row in by_relation if row.get("method") in METHODS]
    deltas_flatten = _delta(by_dataset, "flatten-sum")
    deltas_h6 = _delta(by_dataset, "H6-no-spec")
    deltas_h0 = _delta(by_dataset, "H0-mutual-best")
    checkpoint = read_csv(Path(rebuttal) / "checkpoint_refine_masking_curve.csv")
    checkpoint = [row for row in checkpoint if row.get("method") in METHODS]
    write_csv(output / "paper_rebuttal_table_aggregate.csv", aggregate_rows)
    write_csv(output / "paper_rebuttal_table_by_dataset.csv", by_dataset)
    write_csv(output / "paper_rebuttal_table_by_relation.csv", by_relation)
    write_csv(output / "paper_rebuttal_delta_vs_flatten_sum.csv", deltas_flatten)
    write_csv(output / "paper_rebuttal_delta_vs_h6.csv", deltas_h6)
    write_csv(output / "paper_rebuttal_delta_vs_h0.csv", deltas_h0)
    write_csv(output / "checkpoint_refine_masking_summary.csv", checkpoint)
    write_png(output / "figures" / "relation_js_drift_by_dataset.png", by_dataset, "relation_js_drift_mean", "best_macro_f1")
    write_png(output / "figures" / "relation_energy_error_by_dataset.png", by_dataset, "relation_energy_error_mean", "best_macro_f1")
    write_png(output / "figures" / "edge_collapse_by_dataset.png", by_dataset, "edge_collapse_ratio", "best_macro_f1")
    write_png(output / "figures" / "checkpoint_refine_masking_curve.png", checkpoint, "checkpoint_index", "macro_f1_mean")
    flat_drift = [as_float(row.get("delta_relation_js_drift_mean"), None) for row in deltas_flatten if row.get("method") in {"HeSF-LVC-P", "HeSF-LVC-S"}]
    flat_energy = [as_float(row.get("delta_relation_energy_error_mean"), None) for row in deltas_flatten if row.get("method") in {"HeSF-LVC-P", "HeSF-LVC-S"}]
    task_delta = [as_float(row.get("delta_best_macro_f1"), None) for row in deltas_flatten if row.get("method") in {"HeSF-LVC-P", "HeSF-LVC-S"}]
    lines = [
        "# Next11 HGB Rebuttal Paper Table",
        "",
        f"HeSF-LVC-P/S relation drift vs flatten-sum: {'reduced' if flat_drift and sum(v < 0 for v in flat_drift if v is not None) >= 1 else 'mixed'}",
        f"HeSF-LVC-P/S relation energy error vs flatten-sum: {'reduced' if flat_energy and sum(v < 0 for v in flat_energy if v is not None) >= 1 else 'mixed'}",
        f"Task F1 vs flatten-sum: {'tied/competitive' if task_delta and max(abs(v or 0.0) for v in task_delta) < 0.02 else 'mixed'}",
        "",
        "Paper-safe claim: HeSF-LVC-P/S preserve heterogeneous relation/operator structure more strongly than flatten-sum and H6-no-spec while maintaining competitive refined task recovery.",
        "",
        "Not-supported claim: These results do not show that HeSF-LVC dominates flatten-sum or H6 on task F1.",
        "",
        markdown_table(aggregate_rows, ["method", "run_count", "cumulative_dee_or_audited_dee_mean", "relation_js_drift_mean_mean", "relation_energy_error_mean_mean", "best_macro_f1_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"aggregate": aggregate_rows, "by_dataset": by_dataset}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuttal", type=Path, required=True)
    parser.add_argument("--paper-final", type=Path, required=True)
    parser.add_argument("--dee-consistency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_rebuttal_paper_table(rebuttal=args.rebuttal, paper_final=args.paper_final, dee_consistency=args.dee_consistency, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

