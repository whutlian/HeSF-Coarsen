from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from hesf_coarsen.task_first.selection.diagnostics import aggregate_rows, ratio_matched_gaps


NEW_METHOD_PREFIX = "HeSF-SS"
STRONG_BASELINES = {
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
}
BASELINE_METHODS = STRONG_BASELINES | {
    "random-support-only",
    "A0-current-all-type-coarse-transfer-reference",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_md(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + markdown_table(rows, columns) + "\n", encoding="utf-8")


def _select_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "success") == "success":
            groups.setdefault((row.get("dataset"), row.get("seed"), row.get("method")), []).append(row)
    selected = []
    for _key, group in groups.items():
        best = max(
            group,
            key=lambda row: (_float(row.get("validation_macro_f1"), -1.0), _float(row.get("validation_accuracy"), -1.0)),
        )
        item = dict(best)
        item["selected_by_validation"] = True
        selected.append(item)
    return sorted(selected, key=lambda row: (str(row.get("dataset")), str(row.get("method")), str(row.get("seed"))))


def _recovery_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ceilings = {
        (row.get("dataset"), row.get("seed")): row
        for row in rows
        if row.get("method") == "full-graph-hettree-lite-tuned" and row.get("status", "success") == "success"
    }
    out = []
    for row in rows:
        if row.get("method") == "full-graph-hettree-lite-tuned" or row.get("status", "success") != "success":
            continue
        ceiling = ceilings.get((row.get("dataset"), row.get("seed")))
        if ceiling is None:
            continue
        cm = _float(ceiling.get("macro_f1"))
        ca = _float(ceiling.get("accuracy"))
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "requested_support_ratio": row.get("requested_support_ratio"),
                "realized_support_ratio": row.get("realized_support_ratio"),
                "macro_recovery_vs_full_graph": _float(row.get("macro_f1")) / cm if cm else 0.0,
                "accuracy_recovery_vs_full_graph": _float(row.get("accuracy")) / ca if ca else 0.0,
            }
        )
    return out


def _wins_by_dataset(gaps: list[dict[str, Any]]) -> dict[str, int]:
    wins: dict[str, set[str]] = {}
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in gaps:
        if row.get("comparison_status") not in {"matched", "nearest_flagged"}:
            continue
        if row.get("baseline") not in STRONG_BASELINES:
            continue
        method = str(row.get("method"))
        dataset = str(row.get("dataset"))
        grouped.setdefault((method, dataset), []).append(_float(row.get("delta_macro_f1")))
    for (method, dataset), values in grouped.items():
        if values and float(np.mean(values)) > 0.0:
            wins.setdefault(method, set()).add(dataset)
    return {method: len(datasets) for method, datasets in wins.items()}


def _decision(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    new_selected = [row for row in selected if str(row.get("method", "")).startswith(NEW_METHOD_PREFIX)]
    refs = [row for row in selected if str(row.get("method", "")).startswith("HeSF-TC")]
    best_new = max(new_selected, key=lambda row: _float(row.get("macro_f1"), -1.0), default={})
    best_ref = max(refs, key=lambda row: _float(row.get("macro_f1"), -1.0), default={})
    gap_values = [
        _float(row.get("delta_macro_f1"))
        for row in gaps
        if str(row.get("method", "")).startswith(NEW_METHOD_PREFIX)
        and row.get("baseline") in STRONG_BASELINES
        and row.get("comparison_status") in {"matched", "nearest_flagged"}
        and row.get("delta_macro_f1") not in {"", None}
    ]
    mean_gap = float(np.mean(gap_values)) if gap_values else -1.0
    recovery_values = [
        _float(row.get("macro_recovery_vs_full_graph"))
        for row in recovery
        if str(row.get("method", "")).startswith(NEW_METHOD_PREFIX)
    ]
    acc_recovery_values = [
        _float(row.get("accuracy_recovery_vs_full_graph"))
        for row in recovery
        if str(row.get("method", "")).startswith(NEW_METHOD_PREFIX)
    ]
    mean_recovery = float(np.mean(recovery_values)) if recovery_values else 0.0
    mean_acc_recovery = float(np.mean(acc_recovery_values)) if acc_recovery_values else 0.0
    dataset_wins = _wins_by_dataset(gaps).get(str(best_new.get("method", "")), 0)
    improves_gate14 = bool(best_new) and _float(best_new.get("macro_f1")) > _float(best_ref.get("macro_f1"))
    no_leakage = all(str(row.get("selector_uses_test_labels", "False")).lower() in {"false", "0", ""} for row in new_selected)
    continue_branch = (
        (mean_gap >= 0.02 or dataset_wins >= 2)
        and mean_recovery >= 0.85
        and mean_acc_recovery >= 0.90
        and improves_gate14
        and no_leakage
    )
    stop_branch = (
        mean_gap < 0.0
        and mean_recovery < 0.80
        and not improves_gate14
    )
    decision = "CONTINUE_TASK_FIRST_HESF_SELECTION" if continue_branch else "DROP_HESF_TASK_FIRST_SELECTION_BRANCH"
    if not continue_branch and not stop_branch:
        decision = "DROP_HESF_TASK_FIRST_SELECTION_BRANCH"
    evidence = {
        "best_new_method": best_new.get("method", ""),
        "best_new_macro_f1": _float(best_new.get("macro_f1")),
        "best_new_accuracy": _float(best_new.get("accuracy")),
        "best_gate14_reference": best_ref.get("method", ""),
        "best_gate14_reference_macro_f1": _float(best_ref.get("macro_f1")),
        "mean_ratio_matched_macro_gap_vs_strong_baselines": mean_gap,
        "mean_macro_recovery_vs_full_graph": mean_recovery,
        "mean_accuracy_recovery_vs_full_graph": mean_acc_recovery,
        "dataset_wins_for_best_new": dataset_wins,
        "improves_gate14_reference": improves_gate14,
        "no_test_leakage": no_leakage,
    }
    return decision, evidence


def _plot_outputs(output: Path, by_curve: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    fig_dir = output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        for name in (
            "accuracy_vs_support_ratio.png",
            "macro_f1_vs_support_ratio.png",
            "recovery_vs_support_ratio.png",
            "importance_distribution.png",
            "anchor_coverage_before_after.png",
        ):
            (fig_dir / name).write_bytes(b"")
        return

    def line_plot(metric: str, path: Path, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
        methods = sorted({str(row.get("method")) for row in by_curve})
        for method in methods:
            group = [row for row in by_curve if row.get("method") == method]
            group = sorted(group, key=lambda row: _float(row.get("requested_support_ratio")))
            if not group:
                continue
            ax.plot(
                [_float(row.get("requested_support_ratio")) for row in group],
                [_float(row.get(f"{metric}_mean")) for row in group],
                marker="o",
                linewidth=1.2,
                label=method,
            )
        ax.set_xlabel("requested support ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    line_plot("accuracy", fig_dir / "accuracy_vs_support_ratio.png", "accuracy")
    line_plot("macro_f1", fig_dir / "macro_f1_vs_support_ratio.png", "macro-F1")
    line_plot("macro_recovery_vs_full_graph", fig_dir / "recovery_vs_support_ratio.png", "macro recovery")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    values = [_float(row.get("selected_importance_mean")) for row in rows if row.get("selected_importance_mean") not in {"", None}]
    ax.hist(values or [0.0], bins=20, color="#386cb0", alpha=0.85)
    ax.set_xlabel("selected support importance mean")
    ax.set_ylabel("runs")
    fig.tight_layout()
    fig.savefig(fig_dir / "importance_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    before = np.mean([_float(row.get("anchor_coverage_before")) for row in rows]) if rows else 0.0
    after = np.mean([_float(row.get("anchor_coverage_after")) for row in rows]) if rows else 0.0
    ax.bar(["before", "after"], [before, after], color=["#7fc97f", "#fdc086"])
    ax.set_ylabel("anchor coverage")
    fig.tight_layout()
    fig.savefig(fig_dir / "anchor_coverage_before_after.png")
    plt.close(fig)


def summarize(output: Path) -> dict[str, Any]:
    runs_path = output / "runs" / "gate15_all_runs.csv"
    rows = read_csv(runs_path)
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    metrics = (
        "realized_support_ratio",
        "realized_full_ratio",
        "selected_support_count",
        "background_node_count",
        "dropped_support_count",
        "macro_f1",
        "micro_f1",
        "accuracy",
        "validation_macro_f1",
        "validation_accuracy",
        "macro_recovery_vs_full_graph",
        "accuracy_recovery_vs_full_graph",
        "selected_importance_mean",
        "context_collision_rate",
        "zero_footprint_share",
    )
    by_dataset = aggregate_rows(rows, ["dataset", "method", "requested_support_ratio"], metrics)
    by_method = aggregate_rows(rows, ["method", "requested_support_ratio"], metrics)
    selected = _select_validation(rows)
    recovery = _recovery_rows(rows)
    method_rows = [row for row in rows if str(row.get("method", "")).startswith(NEW_METHOD_PREFIX)]
    baseline_rows = [row for row in rows if row.get("method") in BASELINE_METHODS]
    gaps = ratio_matched_gaps(method_rows, baseline_rows, baseline_names=STRONG_BASELINES)
    curve = by_method
    write_csv(summary / "gate15_by_method_ratio_dataset.csv", by_dataset)
    write_csv(summary / "gate15_final_by_method.csv", by_method)
    write_csv(summary / "gate15_ratio_matched_gaps.csv", gaps)
    write_csv(summary / "gate15_recovery_vs_ceiling.csv", recovery)
    write_csv(summary / "gate15_validation_selected_test.csv", selected)
    write_csv(summary / "gate15_accuracy_budget_curve.csv", curve)
    decision, evidence = _decision(rows, selected, gaps, recovery)
    decision_text = "# Gate15 Decision\n\n"
    decision_text += f"Decision: `{decision}`\n\n"
    decision_text += "## Evidence\n\n"
    for key, value in evidence.items():
        decision_text += f"- {key}: `{value}`\n"
    decision_text += "\n## Boundary\n\nThis is a downstream-task-first decision. It does not recommend returning to preservation-first as an accuracy solution.\n"
    (summary / "gate15_decision.md").write_text(decision_text, encoding="utf-8")
    report = "# Gate15 Final Report\n\n"
    report += f"Verdict: `{decision}`\n\n"
    report += "1. Supervised support selection vs ratio-matched H6/flatten/TypedHash: see `gate15_ratio_matched_gaps.csv`.\n"
    report += "2. Gate14 handcrafted HeSF-TC comparison: see validation-selected rows and decision evidence.\n"
    report += "3. Recovery vs full-graph-lite ceiling: see `gate15_recovery_vs_ceiling.csv`.\n"
    report += "4. Dataset failures are visible in `gate15_by_method_ratio_dataset.csv`.\n"
    report += "5. Teacher contribution is represented by teacher-topk/diverse/hybrid rows.\n"
    report += "6. Response regularization is auxiliary in `HeSF-SS-hybrid-teacher-response`.\n"
    report += "7. Pareto knee should be read from `gate15_accuracy_budget_curve.csv` and figures.\n"
    report += "8. Evaluator status remains `diagnostic_lite_only`.\n"
    report += f"9. Final action: `{decision}`.\n"
    (summary / "final_report.md").write_text(report, encoding="utf-8")
    _plot_outputs(output, curve, rows)
    return {"decision": decision, **evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate15 supervised support selection outputs.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
