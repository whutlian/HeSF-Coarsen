from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
GATE17_PREFIX = "HeSF-SS"
PRIMARY_GATE17_METHODS = {
    "HeSF-SS-sensitivity-plus-prototype",
    "HeSF-SS-real-occlusion-block",
    "HeSF-SS-real-validation-block-greedy",
    "HeSF-SS-dblp-aware-prototype",
    "HeSF-SS-occlusion-plus-dblp-prototype",
}
REQUIRED_RESULT_FIELDS = {
    "decision": "DROP_AFTER_GATE17",
    "primary_eval_mode": "compressed_projected",
    "best_validation_selected_method": "",
    "best_validation_selected_macro_f1_mean": 0.0,
    "best_validation_selected_macro_f1_std": 0.0,
    "best_validation_selected_accuracy_mean": 0.0,
    "best_validation_selected_accuracy_std": 0.0,
    "best_single_run_macro_f1": 0.0,
    "best_single_run_method": "",
    "best_single_run_dataset": "",
    "mean_exact_budget_macro_gap_vs_best_strong_baseline": 0.0,
    "mean_exact_budget_accuracy_gap_vs_best_strong_baseline": 0.0,
    "dataset_wins_vs_best_strong_baseline": 0,
    "dblp_exact_budget_macro_gap": 0.0,
    "acm_exact_budget_macro_gap": 0.0,
    "imdb_exact_budget_macro_gap": 0.0,
    "no_test_leakage": True,
    "teacher_reliable": False,
    "support_budget_exact_match_rate_for_best_method": 0.0,
    "projected_vs_transfer_gap_mean": 0.0,
    "true_validation_trial_count_total": 0,
    "occlusion_trial_count_total": 0,
    "large_prototype_count_total": 0,
    "failed": 0,
    "success": 0,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_raw_rows(input_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [
        input_dir if input_dir.is_file() else input_dir / "gate17_raw_rows.csv",
        input_dir / "gate17_all_runs.csv",
        input_dir / "gate17_tables" / "gate17_raw_rows.csv",
        input_dir / "gate17_tables" / "gate17_all_runs.csv",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path, read_csv(path)
    return candidates[0], []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _success(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "success")) == "success"


def _method_is_gate17(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")).startswith(GATE17_PREFIX)


def _same_budget_bucket(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_ratio = str(left.get("requested_support_ratio", left.get("support_ratio", "")))
    right_ratio = str(right.get("requested_support_ratio", right.get("support_ratio", "")))
    left_count = str(left.get("requested_support_count", ""))
    right_count = str(right.get("requested_support_count", ""))
    return (left_ratio != "" and left_ratio == right_ratio) or (left_count != "" and left_count == right_count)


def _same_requested_ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_ratio = str(left.get("requested_support_ratio", left.get("support_ratio", "")))
    right_ratio = str(right.get("requested_support_ratio", right.get("support_ratio", "")))
    if left_ratio != "" and right_ratio != "":
        return left_ratio == right_ratio
    return _same_budget_bucket(left, right)


def _round_delta(value: float) -> float:
    return float(round(float(value), 12))


def validation_selected(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success(row):
            groups[(str(row.get("dataset")), str(row.get("seed")), str(row.get("method")))].append(row)
    selected: list[dict[str, Any]] = []
    for _key, group in groups.items():
        best = max(
            group,
            key=lambda row: (
                _float(row.get("validation_macro_f1"), -1.0),
                _float(row.get("validation_accuracy"), -1.0),
                -_float(row.get("requested_support_ratio"), 1.0e9),
                -_float(row.get("requested_support_count"), 1.0e9),
                _float(row.get("macro_f1"), -1.0),
                _float(row.get("accuracy"), -1.0),
            ),
        )
        item = dict(best)
        item["selected_by_validation"] = True
        selected.append(item)
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("method")),
            str(row.get("seed")),
            _float(row.get("requested_support_ratio")),
        ),
    )


def _best_baseline_for(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    exact_only: bool,
    requested_ratio_only: bool,
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    for baseline in rows:
        if not _success(baseline) or str(baseline.get("method", "")) not in STRONG_BASELINES:
            continue
        if str(baseline.get("dataset")) != str(row.get("dataset")) or str(baseline.get("seed")) != str(row.get("seed")):
            continue
        if exact_only and (not _bool(row.get("support_budget_exact_match")) or not _bool(baseline.get("support_budget_exact_match"))):
            continue
        same_bucket = _same_requested_ratio(row, baseline) if requested_ratio_only else _same_budget_bucket(row, baseline)
        if not same_bucket:
            continue
        candidates.append(baseline)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))))


def _paired_gap_row(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    return {
        "comparison_scope": scope,
        "dataset": row.get("dataset"),
        "seed": row.get("seed"),
        "requested_support_ratio": row.get("requested_support_ratio", row.get("support_ratio", "")),
        "requested_support_count": row.get("requested_support_count", ""),
        "method": row.get("method"),
        "best_baseline_method": baseline.get("method", ""),
        "method_budget_exact": _bool(row.get("support_budget_exact_match")),
        "baseline_budget_exact": _bool(baseline.get("support_budget_exact_match")),
        "method_realized_support_count": row.get("realized_support_count", ""),
        "baseline_realized_support_count": baseline.get("realized_support_count", ""),
        "method_macro_f1": _float(row.get("macro_f1")),
        "baseline_macro_f1": _float(baseline.get("macro_f1")),
        "delta_macro_f1": _round_delta(_float(row.get("macro_f1")) - _float(baseline.get("macro_f1"))),
        "method_accuracy": _float(row.get("accuracy")),
        "baseline_accuracy": _float(baseline.get("accuracy")),
        "delta_accuracy": _round_delta(_float(row.get("accuracy")) - _float(baseline.get("accuracy"))),
        "primary_eval_mode": row.get("primary_eval_mode", ""),
    }


def exact_only_paired_gaps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _success(row) or not _method_is_gate17(row) or not _bool(row.get("support_budget_exact_match")):
            continue
        baseline = _best_baseline_for(row, rows, exact_only=True, requested_ratio_only=False)
        if baseline is not None:
            out.append(_paired_gap_row(row, baseline, scope="exact_only"))
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), str(item.get("seed")), str(item.get("requested_support_ratio"))))


def requested_ratio_paired_gaps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _success(row) or not _method_is_gate17(row):
            continue
        baseline = _best_baseline_for(row, rows, exact_only=False, requested_ratio_only=True)
        if baseline is not None:
            out.append(_paired_gap_row(row, baseline, scope="requested_ratio"))
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), str(item.get("seed")), str(item.get("requested_support_ratio"))))


def recovery_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ceilings = {
        (str(row.get("dataset")), str(row.get("seed"))): row
        for row in rows
        if row.get("method") == "full-graph-hettree-lite-tuned" and _success(row)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _success(row) or row.get("method") == "full-graph-hettree-lite-tuned":
            continue
        ceiling = ceilings.get((str(row.get("dataset")), str(row.get("seed"))))
        if ceiling is None:
            continue
        ceiling_macro = _float(ceiling.get("macro_f1"))
        ceiling_acc = _float(ceiling.get("accuracy"))
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "requested_support_ratio": row.get("requested_support_ratio"),
                "requested_support_count": row.get("requested_support_count"),
                "macro_recovery_vs_full_graph": _float(row.get("macro_f1")) / ceiling_macro if ceiling_macro else 0.0,
                "accuracy_recovery_vs_full_graph": _float(row.get("accuracy")) / ceiling_acc if ceiling_acc else 0.0,
                "primary_eval_mode": row.get("primary_eval_mode", ""),
            }
        )
    return out


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.mean(items)) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.std(items, ddof=1)) if len(items) > 1 else 0.0


def _aggregate_validation_selected_by_method(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row.get("method"))].append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(groups.items()):
        item = {
            "method": method,
            "runs": int(len(group)),
            "macro_f1_mean": _mean(_float(row.get("macro_f1")) for row in group),
            "macro_f1_std": _std(_float(row.get("macro_f1")) for row in group),
            "accuracy_mean": _mean(_float(row.get("accuracy")) for row in group),
            "accuracy_std": _std(_float(row.get("accuracy")) for row in group),
            "validation_macro_f1_mean": _mean(_float(row.get("validation_macro_f1")) for row in group),
            "validation_macro_f1_std": _std(_float(row.get("validation_macro_f1")) for row in group),
            "requested_support_ratio_mean": _mean(_float(row.get("requested_support_ratio")) for row in group),
            "realized_support_ratio_mean": _mean(_float(row.get("realized_support_ratio")) for row in group),
            "support_budget_exact_match_mean": _mean(1.0 if _bool(row.get("support_budget_exact_match")) else 0.0 for row in group),
            "projected_vs_transfer_macro_gap_mean": _mean(_float(row.get("projected_vs_transfer_macro_gap")) for row in group),
            "macro_recovery_vs_full_graph_mean": _mean(_float(row.get("macro_recovery_vs_full_graph")) for row in group),
        }
        out.append(item)
    return sorted(out, key=lambda item: (-float(item["macro_f1_mean"]), -float(item["accuracy_mean"]), str(item["method"])))


def _dataset_gap(gaps: Sequence[dict[str, Any]], method: str, dataset: str) -> float:
    values = [
        _float(row.get("delta_macro_f1"))
        for row in gaps
        if str(row.get("method")) == method and str(row.get("dataset")).upper() == dataset.upper()
    ]
    return _mean(values)


def _dataset_wins(gaps: Sequence[dict[str, Any]], method: str) -> int:
    return sum(
        1
        for dataset in {"ACM", "DBLP", "IMDB"}
        if _dataset_gap(gaps, method, dataset) > 0.0
    )


def _trial_count(rows: Sequence[Mapping[str, Any]], *keys: str) -> int:
    total = 0
    for row in rows:
        for key in keys:
            total += _int(row.get(key), 0)
    return int(total)


def _no_test_leakage(rows: Sequence[Mapping[str, Any]]) -> bool:
    relevant = [row for row in rows if _method_is_gate17(row)]
    if not relevant:
        return True
    return all(
        not _bool(row.get("selector_uses_test_labels"))
        and not _bool(row.get("teacher_uses_test_labels_for_training"))
        for row in relevant
    )


def _teacher_reliable(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset:
            by_dataset[dataset].append(_float(row.get("full_graph_teacher_macro_f1")))
    if not by_dataset:
        return False
    acm = by_dataset.get("ACM", [])
    dblp = by_dataset.get("DBLP", [])
    imdb = by_dataset.get("IMDB", [])
    return (
        bool(acm) and _mean(acm) > 0.80 and _std(acm) < 0.10
        and bool(dblp) and _mean(dblp) > 0.70
        and (bool(imdb) and _mean(imdb) > 0.35)
    )


def _decision(result: dict[str, Any]) -> str:
    if (
        result["mean_exact_budget_macro_gap_vs_best_strong_baseline"] > 0.02
        and result["mean_exact_budget_accuracy_gap_vs_best_strong_baseline"] > 0.02
        and result["acm_exact_budget_macro_gap"] > 0.0
        and result["dblp_exact_budget_macro_gap"] >= 0.0
        and result["no_test_leakage"]
    ):
        return "CONTINUE_NARROWED_TO_REAL_OCCLUSION_PROTOTYPE"
    if (
        result["mean_exact_budget_macro_gap_vs_best_strong_baseline"] > 0.0
        and result["dblp_exact_budget_macro_gap"] < 0.0
    ):
        return "PARTIAL_DBLP_BLOCKER"
    if (
        result["mean_exact_budget_macro_gap_vs_best_strong_baseline"] <= 0.0
        and result["dblp_exact_budget_macro_gap"] < -0.03
    ):
        return "DROP_AFTER_GATE17"
    return "PARTIAL_DBLP_BLOCKER" if result["best_validation_selected_method"] else "DROP_AFTER_GATE17"


def _write_decision(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Gate17 Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Checks",
        "",
        f"- primary_eval_mode: `{result['primary_eval_mode']}`",
        f"- best method: `{result['best_validation_selected_method']}`",
        f"- method-level macro mean: `{result['best_validation_selected_macro_f1_mean']}`",
        f"- best single run macro: `{result['best_single_run_macro_f1']}`",
        f"- exact-only macro gap: `{result['mean_exact_budget_macro_gap_vs_best_strong_baseline']}`",
        f"- DBLP exact gap: `{result['dblp_exact_budget_macro_gap']}`",
        f"- no_test_leakage: `{result['no_test_leakage']}`",
        f"- true validation trials: `{result['true_validation_trial_count_total']}`",
        f"- occlusion trials: `{result['occlusion_trial_count_total']}`",
        "",
        "## Result JSON",
        "",
    ]
    for key, value in result.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    path: Path,
    result: dict[str, Any],
    selected_by_method: Sequence[Mapping[str, Any]],
    exact_gaps: Sequence[Mapping[str, Any]],
) -> None:
    gap_summary = [row for row in exact_gaps if row.get("method") == result.get("best_validation_selected_method")]
    lines = [
        "# Gate17 Final Report",
        "",
        "## Decision",
        "",
        f"- `{result['decision']}`",
        "",
        "## Main Claim",
        "",
        "Gate17 uses compressed/projected task metrics as primary and separates method-level validation-selected aggregates from best single-run results.",
        "",
        "## Validation-Selected Method Aggregates",
        "",
        markdown_table(
            selected_by_method,
            [
                "method",
                "runs",
                "macro_f1_mean",
                "macro_f1_std",
                "accuracy_mean",
                "accuracy_std",
                "support_budget_exact_match_mean",
            ],
        ),
        "",
        "## Exact-Budget Paired Gaps",
        "",
        markdown_table(
            gap_summary[:20],
            [
                "dataset",
                "seed",
                "method",
                "best_baseline_method",
                "delta_macro_f1",
                "delta_accuracy",
            ],
        ),
        "",
        "## Leakage / Budget / Metric Checks",
        "",
        f"- no_test_leakage: `{result['no_test_leakage']}`",
        f"- primary_eval_mode: `{result['primary_eval_mode']}`",
        f"- support_budget_exact_match_rate_for_best_method: `{result['support_budget_exact_match_rate_for_best_method']}`",
        f"- projected_vs_transfer_gap_mean: `{result['projected_vs_transfer_gap_mean']}`",
        "",
        "## Next Action",
        "",
        f"- `{result['decision']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "gate17_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, rows = _find_raw_rows(input_dir)
    if rows and raw_path.resolve() != (output_dir / "gate17_raw_rows.csv").resolve():
        write_csv(output_dir / "gate17_raw_rows.csv", rows)

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
    by_method_ratio = aggregate_rows(rows, ["method", "requested_support_ratio"], metrics)
    selected = validation_selected(rows)
    selected_by_method = _aggregate_validation_selected_by_method(selected)
    exact_gaps = exact_only_paired_gaps(rows)
    requested_gaps = requested_ratio_paired_gaps(rows)
    recovery = recovery_rows(rows)

    write_csv(output_dir / "gate17_final_by_method.csv", by_method_ratio)
    write_csv(output_dir / "gate17_by_method_ratio_dataset.csv", by_dataset)
    write_csv(output_dir / "gate17_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_validation_selected_by_dataset.csv", selected)
    write_csv(output_dir / "gate17_exact_only_paired_gaps.csv", exact_gaps)
    write_csv(output_dir / "gate17_requested_ratio_paired_gaps.csv", requested_gaps)
    write_csv(output_dir / "gate17_recovery_vs_ceiling.csv", recovery)

    best_agg = max(
        (row for row in selected_by_method if str(row.get("method", "")).startswith(GATE17_PREFIX)),
        key=lambda row: (_float(row.get("macro_f1_mean")), _float(row.get("accuracy_mean"))),
        default={},
    )
    best_method = str(best_agg.get("method", ""))
    best_single = max(
        (row for row in rows if _success(row) and _method_is_gate17(row)),
        key=lambda row: (_float(row.get("macro_f1")), _float(row.get("accuracy"))),
        default={},
    )
    method_exact_gaps = [row for row in exact_gaps if str(row.get("method")) == best_method]
    selected_best_rows = [row for row in selected if str(row.get("method")) == best_method]
    teacher_rows = read_csv(input_dir / "gate17_diagnostics" / "full_graph_teacher_by_dataset_seed.csv")
    if not teacher_rows:
        teacher_rows = read_csv(input_dir / "full_graph_teacher_by_dataset_seed.csv")
    projected_gaps = [
        _float(row.get("projected_vs_transfer_macro_gap"))
        for row in rows
        if _success(row) and row.get("projected_vs_transfer_macro_gap") not in {"", None}
    ]
    result = dict(REQUIRED_RESULT_FIELDS)
    result.update(
        {
            "best_validation_selected_method": best_method,
            "best_validation_selected_macro_f1_mean": _float(best_agg.get("macro_f1_mean")),
            "best_validation_selected_macro_f1_std": _float(best_agg.get("macro_f1_std")),
            "best_validation_selected_accuracy_mean": _float(best_agg.get("accuracy_mean")),
            "best_validation_selected_accuracy_std": _float(best_agg.get("accuracy_std")),
            "best_single_run_macro_f1": _float(best_single.get("macro_f1")),
            "best_single_run_method": str(best_single.get("method", "")),
            "best_single_run_dataset": str(best_single.get("dataset", "")),
            "mean_exact_budget_macro_gap_vs_best_strong_baseline": _mean(_float(row.get("delta_macro_f1")) for row in method_exact_gaps),
            "mean_exact_budget_accuracy_gap_vs_best_strong_baseline": _mean(_float(row.get("delta_accuracy")) for row in method_exact_gaps),
            "dataset_wins_vs_best_strong_baseline": _dataset_wins(exact_gaps, best_method) if best_method else 0,
            "dblp_exact_budget_macro_gap": _dataset_gap(exact_gaps, best_method, "DBLP") if best_method else 0.0,
            "acm_exact_budget_macro_gap": _dataset_gap(exact_gaps, best_method, "ACM") if best_method else 0.0,
            "imdb_exact_budget_macro_gap": _dataset_gap(exact_gaps, best_method, "IMDB") if best_method else 0.0,
            "no_test_leakage": _no_test_leakage(rows),
            "teacher_reliable": _teacher_reliable(teacher_rows),
            "support_budget_exact_match_rate_for_best_method": _mean(1.0 if _bool(row.get("support_budget_exact_match")) else 0.0 for row in selected_best_rows),
            "projected_vs_transfer_gap_mean": _mean(projected_gaps),
            "true_validation_trial_count_total": _trial_count(rows, "validation_trial_count", "validation_greedy_trial_count"),
            "occlusion_trial_count_total": _trial_count(rows, "occlusion_trial_count"),
            "large_prototype_count_total": _trial_count(rows, "large_prototype_count"),
            "failed": sum(1 for row in rows if not _success(row)),
            "success": sum(1 for row in rows if _success(row)),
        }
    )
    result["decision"] = _decision(result)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_decision(output_dir / "gate17_decision.md", result)
    _write_final_report(output_dir / "final_report.md", result, selected_by_method, exact_gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17 outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate17_tables"))
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
