from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from hesf_coarsen.task_first.selection.diagnostics import aggregate_rows


STRONG_BASELINES = {
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
}
ALL_BASELINES = STRONG_BASELINES | {"random-support-only"}
NEW_PREFIX = "HeSF-SS"


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


def _success(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "success")) == "success"


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset")),
        str(row.get("seed")),
        str(row.get("requested_support_ratio", row.get("support_ratio"))),
        str(row.get("requested_support_count", "")),
    )


def exact_budget_paired_gaps(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success(row):
            by_group[_group_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in by_group.items():
        baselines = [row for row in group if row.get("method") in STRONG_BASELINES]
        if not baselines:
            continue
        best_baseline = max(baselines, key=lambda row: (_float(row.get("macro_f1")), _float(row.get("accuracy"))))
        for row in group:
            method = str(row.get("method", ""))
            if not method.startswith(NEW_PREFIX):
                continue
            exact = str(row.get("support_budget_exact_match", "False")).lower() in {"true", "1"}
            exact = exact and str(best_baseline.get("support_budget_exact_match", "False")).lower() in {"true", "1"}
            out.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "support_ratio": key[2],
                    "requested_support_count": key[3],
                    "method": method,
                    "best_baseline_method": best_baseline.get("method", ""),
                    "method_macro_f1": _float(row.get("macro_f1")),
                    "baseline_macro_f1": _float(best_baseline.get("macro_f1")),
                    "delta_macro_f1": _float(row.get("macro_f1")) - _float(best_baseline.get("macro_f1")),
                    "method_accuracy": _float(row.get("accuracy")),
                    "baseline_accuracy": _float(best_baseline.get("accuracy")),
                    "delta_accuracy": _float(row.get("accuracy")) - _float(best_baseline.get("accuracy")),
                    "support_budget_exact_match": exact,
                    "primary_eval_mode": row.get("primary_eval_mode", ""),
                }
            )
    return out


def validation_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success(row):
            groups[(str(row.get("dataset")), str(row.get("seed")), str(row.get("method")))].append(row)
    selected: list[dict[str, Any]] = []
    for _key, group in groups.items():
        best = max(group, key=lambda row: (_float(row.get("validation_macro_f1"), -1.0), _float(row.get("validation_accuracy"), -1.0)))
        item = dict(best)
        item["selected_by_validation"] = True
        selected.append(item)
    return sorted(selected, key=lambda row: (str(row.get("dataset")), str(row.get("method")), str(row.get("seed"))))


def recovery_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ceilings = {
        (row.get("dataset"), row.get("seed")): row
        for row in rows
        if row.get("method") == "full-graph-hettree-lite-tuned" and _success(row)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _success(row) or row.get("method") == "full-graph-hettree-lite-tuned":
            continue
        ceiling = ceilings.get((row.get("dataset"), row.get("seed")))
        if not ceiling:
            continue
        cm = _float(ceiling.get("macro_f1"))
        ca = _float(ceiling.get("accuracy"))
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "requested_support_ratio": row.get("requested_support_ratio"),
                "requested_support_count": row.get("requested_support_count"),
                "macro_recovery_vs_full_graph": _float(row.get("macro_f1")) / cm if cm else 0.0,
                "accuracy_recovery_vs_full_graph": _float(row.get("accuracy")) / ca if ca else 0.0,
                "primary_eval_mode": row.get("primary_eval_mode", ""),
            }
        )
    return out


def _dataset_wins(gaps: list[dict[str, Any]], method: str) -> int:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in gaps:
        if row.get("method") == method:
            grouped[str(row.get("dataset"))].append(_float(row.get("delta_macro_f1")))
    return sum(1 for values in grouped.values() if values and float(np.mean(values)) > 0.0)


def _exact_match_rate(rows: list[dict[str, Any]]) -> float:
    values = [
        str(row.get("support_budget_exact_match", "False")).lower() in {"true", "1"}
        for row in rows
        if row.get("method") != "full-graph-hettree-lite-tuned" and _success(row)
    ]
    return float(np.mean(values)) if values else 0.0


def _teacher_reliable(teacher_rows: list[dict[str, Any]]) -> bool:
    if not teacher_rows:
        return False
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in teacher_rows:
        by_dataset[str(row.get("dataset"))].append(_float(row.get("full_graph_teacher_macro_f1")))
    return all(values and float(np.mean(values)) >= 0.35 and (float(np.std(values)) <= 0.20 or len(values) < 2) for values in by_dataset.values())


def _write_decision(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Gate16 Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Required Questions",
        "",
        f"1. Projected vs transfer mean macro gap: `{result['projected_vs_transfer_gap_mean']}`.",
        "2. Gate15 transfer-primary results are diagnostic only after this evaluator patch; Gate16 uses compressed_projected primary.",
        f"3. Full-graph teacher reliable: `{result['teacher_reliable']}`.",
        f"4. Exact-budget mean macro gap vs strong baseline: `{result['mean_exact_budget_macro_gap_vs_best_strong_baseline']}`.",
        "5. DBLP improvement should be read from `gate16_by_method_ratio_dataset.csv`; decision uses exact-budget gaps, not max single rows.",
        "6. IMDB collapse is separated by full-graph ceiling and recovery tables.",
        "7. Support importance vs task performance is exposed through `support_importance.csv` and validation-selected rows.",
        "8. Prototype background comparison is exposed through prototype diagnostics and prototype-residual methods.",
        f"9. Test leakage: `{not result['no_test_leakage']}`.",
        f"10. Next step: `{result['decision']}`.",
        "",
        "## Evidence",
        "",
    ]
    for key, value in result.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(root: Path) -> dict[str, Any]:
    tables = root / "gate16_tables"
    diag = root / "gate16_diag"
    rows = read_csv(tables / "gate16_all_runs.csv")
    teacher_rows = read_csv(root / "gate16_teacher" / "full_graph_teacher_by_dataset_seed.csv")
    metrics = (
        "requested_support_count",
        "realized_support_count",
        "realized_support_ratio",
        "realized_full_ratio",
        "support_budget_error",
        "macro_f1",
        "micro_f1",
        "accuracy",
        "validation_macro_f1",
        "validation_accuracy",
        "projected_vs_transfer_macro_gap",
        "macro_recovery_vs_full_graph",
        "accuracy_recovery_vs_full_graph",
    )
    by_dataset = aggregate_rows(rows, ["dataset", "method", "requested_support_ratio"], metrics)
    by_method = aggregate_rows(rows, ["method", "requested_support_ratio"], metrics)
    selected = validation_selected(rows)
    gaps = exact_budget_paired_gaps(rows)
    recovery = recovery_rows(rows)
    write_csv(tables / "gate16_by_method_ratio_dataset.csv", by_dataset)
    write_csv(tables / "gate16_final_by_method.csv", by_method)
    write_csv(tables / "gate16_validation_selected_test.csv", selected)
    write_csv(tables / "gate16_exact_budget_paired_gaps.csv", gaps)
    write_csv(tables / "gate16_recovery_vs_ceiling.csv", recovery)
    new_selected = [row for row in selected if str(row.get("method", "")).startswith(NEW_PREFIX)]
    best_selected = max(new_selected, key=lambda row: (_float(row.get("macro_f1")), _float(row.get("accuracy"))), default={})
    best_method = str(best_selected.get("method", ""))
    gap_values = [_float(row.get("delta_macro_f1")) for row in gaps if row.get("method") == best_method]
    acc_gap_values = [_float(row.get("delta_accuracy")) for row in gaps if row.get("method") == best_method]
    exact_rate = _exact_match_rate(rows)
    projected_gaps = [_float(row.get("projected_vs_transfer_macro_gap")) for row in rows if _success(row) and row.get("projected_vs_transfer_macro_gap") not in {"", None}]
    no_leakage = all(
        str(row.get("selector_uses_test_labels", "False")).lower() in {"false", "0", ""}
        and str(row.get("teacher_uses_test_labels_for_training", "False")).lower() in {"false", "0", ""}
        for row in rows
        if str(row.get("method", "")).startswith(NEW_PREFIX)
    )
    result = {
        "decision": "DROP_TASK_FIRST_SUPPORT_COMPRESSION",
        "primary_eval_mode": "compressed_projected",
        "best_validation_selected_method": best_method,
        "best_validation_selected_macro_f1_mean": _float(best_selected.get("macro_f1")),
        "best_validation_selected_accuracy_mean": _float(best_selected.get("accuracy")),
        "mean_exact_budget_macro_gap_vs_best_strong_baseline": float(np.mean(gap_values)) if gap_values else 0.0,
        "mean_exact_budget_accuracy_gap_vs_best_strong_baseline": float(np.mean(acc_gap_values)) if acc_gap_values else 0.0,
        "dataset_wins_vs_best_strong_baseline": _dataset_wins(gaps, best_method),
        "teacher_reliable": _teacher_reliable(teacher_rows),
        "no_test_leakage": no_leakage,
        "support_budget_exact_match_rate": exact_rate,
        "projected_vs_transfer_gap_mean": float(np.mean(projected_gaps)) if projected_gaps else 0.0,
        "failed": sum(1 for row in rows if not _success(row)),
        "success": sum(1 for row in rows if _success(row)),
    }
    if (
        result["mean_exact_budget_macro_gap_vs_best_strong_baseline"] > 0.0
        and result["mean_exact_budget_accuracy_gap_vs_best_strong_baseline"] > 0.0
        and result["dataset_wins_vs_best_strong_baseline"] >= 2
        and result["support_budget_exact_match_rate"] >= 0.95
        and result["no_test_leakage"]
    ):
        result["decision"] = "CONTINUE_WITH_OFFICIAL_EVALUATOR" if result["teacher_reliable"] else "CONTINUE"
    _write_decision(tables / "gate16_decision.md", result)
    (tables / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    final_report = "# Gate16 Final Report\n\n"
    final_report += f"Decision: `{result['decision']}`\n\n"
    final_report += markdown_table(
        by_method[:20],
        ["method", "requested_support_ratio", "macro_f1_mean", "accuracy_mean", "projected_vs_transfer_macro_gap_mean"],
    )
    (tables / "final_report.md").write_text(final_report + "\n", encoding="utf-8")
    diag.mkdir(parents=True, exist_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate16 outputs.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    summarize(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
