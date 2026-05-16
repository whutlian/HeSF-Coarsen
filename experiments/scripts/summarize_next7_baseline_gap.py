from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv


BASELINE_LABELS = {
    "random": "random",
    "graphzoom_style": "GraphZoom-style",
    "convmatch_style": "ConvMatch-style",
}
COARSE_BASELINE_METHODS = {
    "random",
    "GraphZoom-style",
    "ConvMatch-style",
    "H0-mutual-best",
    "flatten-sum",
}
METHOD_ORDER = [
    "HeSF-LVC-P",
    "HeSF-LVC-S",
    "HeSF-LVC-T",
    "H0-mutual-best",
    "flatten-sum",
    "random",
    "GraphZoom-style",
    "ConvMatch-style",
    "H2-old",
    "H3-old",
    "H4-no-conv",
    "H6-no-spec",
]
METRIC_COLUMNS = [
    "dee",
    "ree_max",
    "sipe",
    "projected_macro_f1",
    "refined_macro_f1@0",
    "refined_macro_f1@1",
    "refined_macro_f1@3",
    "refined_macro_f1@5",
    "best_macro_f1",
    "refine_auc_macro_f1",
    "runtime_sec",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _close(value: Any, expected: float, tol: float = 1e-9) -> bool:
    number = _as_float(value, None)
    return number is not None and abs(number - expected) <= tol


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value, None) for value in values]
    clean = [float(value) for value in numbers if value is not None]
    return None if not clean else float(mean(clean))


def _std(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value, None) for value in values]
    clean = [float(value) for value in numbers if value is not None]
    return None if len(clean) <= 1 else float(pstdev(clean))


def _lambda_value(row: Mapping[str, Any], name: str) -> Any:
    return _first(row, (name, f"config.scoring.{name}", f"scoring.{name}"), "")


def _final_method(row: Mapping[str, Any]) -> str:
    paper_method = _first(row, ("paper.method", "config.paper.method"), "")
    if paper_method:
        return str(paper_method)
    baseline_method = _first(row, ("paper.baseline_method", "config.paper.baseline_method"), "")
    if baseline_method:
        return BASELINE_LABELS.get(str(baseline_method), str(baseline_method))
    variant = str(_first(row, ("paper.variant", "config.paper.variant", "variant"), ""))
    lambda_spec = _lambda_value(row, "lambda_spec")
    lambda_conv = _lambda_value(row, "lambda_conv")
    lambda_rel = _lambda_value(row, "lambda_rel")
    if variant == "H2" and _close(lambda_spec, 0.25) and _close(lambda_conv, 0.0) and _close(lambda_rel, 0.0):
        return "HeSF-LVC-P"
    if variant == "H3" and _close(lambda_spec, 0.5) and _close(lambda_conv, 0.0) and _close(lambda_rel, 0.0):
        return "HeSF-LVC-S"
    if variant == "H2" and _close(lambda_spec, 2.0) and _close(lambda_conv, 0.25) and _close(lambda_rel, 0.0):
        return "HeSF-LVC-T"
    if variant == "H0":
        return "H0-mutual-best"
    if variant == "H2-single-relation-sum":
        return "flatten-sum"
    if variant == "H2":
        return "H2-old"
    if variant == "H3":
        return "H3-old"
    if variant == "H4":
        return "H4-no-conv"
    if variant == "H6":
        return "H6-no-spec"
    return variant or str(row.get("run_name", "unknown"))


def _final_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = _final_method(row)
    return {
        "method": method,
        "source": "final_summary",
        "dataset": row.get("dataset", ""),
        "variant": row.get("variant", ""),
        "seed": str(row.get("seed", "")),
        "target_ratio": row.get("target_ratio", ""),
        "run_name": row.get("run_name", ""),
        "run_dir": row.get("run_dir", ""),
        "lambda_spec": _lambda_value(row, "lambda_spec"),
        "lambda_conv": _lambda_value(row, "lambda_conv"),
        "lambda_rel": _lambda_value(row, "lambda_rel"),
        "dee": _first(row, ("cumulative_dee", "cumulative_spectral.dirichlet_energy_relative_error"), ""),
        "ree_max": _first(row, ("cumulative_ree_max", "cumulative_spectral.relation_energy_relative_error_max"), ""),
        "sipe": _first(
            row,
            (
                "cumulative_sipe",
                "cumulative_spectral.chebheat_sketch_inner_product_relative_error",
            ),
        ),
        "projected_macro_f1": _first(
            row,
            ("task_projected_macro_f1", "task.projected_original_macro_f1"),
        ),
        "refined_macro_f1@0": _first(
            row,
            ("task_refined_macro_f1@0", "task.refined_original_macro_f1@0", "task_refined_macro_f1"),
        ),
        "refined_macro_f1@1": _first(
            row,
            ("task_refined_macro_f1@1", "task.refined_original_macro_f1@1", "task_refined_macro_f1"),
        ),
        "refined_macro_f1@3": _first(
            row,
            ("task_refined_macro_f1@3", "task.refined_original_macro_f1@3", "task_refined_macro_f1"),
        ),
        "refined_macro_f1@5": _first(
            row,
            ("task_refined_macro_f1@5", "task.refined_original_macro_f1@5", "task_refined_macro_f1"),
        ),
        "best_macro_f1": _first(
            row,
            (
                "task_best_refined_macro_f1",
                "task.best_refined_macro_f1",
                "task_refined_macro_f1@5",
                "task_refined_macro_f1",
            ),
        ),
        "refine_auc_macro_f1": _first(row, ("task_refine_auc_macro_f1", "task.refine_auc_macro_f1"), ""),
        "train_time_sec": _first(row, ("task_train_time", "task.train_time", "runtime_by_stage.task_train"), ""),
        "refine_time_sec": _first(row, ("task_refine_time", "task.refine_time", "runtime_by_stage.task_refine"), ""),
        "total_time_sec": _first(row, ("task_total_time", "task.total_time"), ""),
        "runtime_sec": _first(row, ("runtime_total_run", "runtime_total", "task.total_time"), ""),
        "full_graph_rgcn_lite_default_macro_f1": _first(
            row,
            ("task_full_graph_rgcn_lite_default_macro_f1", "task.full_graph_rgcn_lite_default_macro_f1"),
        ),
        "full_graph_rgcn_lite_tuned_macro_f1": _first(
            row,
            ("task_full_graph_rgcn_lite_tuned_macro_f1", "task.full_graph_rgcn_lite_tuned_macro_f1"),
        ),
        "full_graph_han_small_macro_f1": _first(
            row,
            ("task_full_graph_han_small_macro_f1", "task.full_graph_han_small_macro_f1"),
        ),
        "full_graph_hgt_small_macro_f1": _first(
            row,
            ("task_full_graph_hgt_small_macro_f1", "task.full_graph_hgt_small_macro_f1"),
        ),
    }


def _baseline_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(row.get("comparison_status", "included")) != "included":
        return None
    method = BASELINE_LABELS.get(str(row.get("baseline", "")))
    if method is None:
        return None
    return {
        "method": method,
        "source": "baseline_summary",
        "dataset": row.get("dataset", ""),
        "variant": row.get("variant", ""),
        "seed": str(row.get("seed", "")),
        "target_ratio": row.get("target_ratio", ""),
        "run_name": row.get("run_name", ""),
        "run_dir": row.get("run_dir", ""),
        "lambda_spec": "",
        "lambda_conv": "",
        "lambda_rel": "",
        "dee": row.get("baseline_cumulative_dee", ""),
        "ree_max": row.get("baseline_cumulative_ree_max", ""),
        "sipe": row.get("baseline_cumulative_sipe", ""),
        "projected_macro_f1": row.get("baseline_projected_macro_f1", ""),
        "refined_macro_f1@0": row.get("baseline_refined_macro_f1@0", ""),
        "refined_macro_f1@1": row.get("baseline_refined_macro_f1@1", ""),
        "refined_macro_f1@3": row.get("baseline_refined_macro_f1@3", ""),
        "refined_macro_f1@5": _first(row, ("baseline_refined_macro_f1@5", "baseline_refined_macro_f1"), ""),
        "best_macro_f1": _first(
            row,
            ("baseline_task_best_refined_macro_f1", "baseline_refined_macro_f1@5", "baseline_refined_macro_f1"),
        ),
        "refine_auc_macro_f1": row.get("baseline_task_refine_auc_macro_f1", ""),
        "train_time_sec": row.get("baseline_task_train_time", ""),
        "refine_time_sec": row.get("baseline_task_refine_time", ""),
        "total_time_sec": row.get("baseline_task_total_time", ""),
        "runtime_sec": row.get("baseline_runtime_total", ""),
        "full_graph_rgcn_lite_default_macro_f1": "",
        "full_graph_rgcn_lite_tuned_macro_f1": "",
        "full_graph_han_small_macro_f1": "",
        "full_graph_hgt_small_macro_f1": "",
    }


def _dedupe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("method", "")),
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("target_ratio", "")),
        )
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def _best_by_key(rows: Sequence[dict[str, Any]], methods: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("method", "")) not in methods:
            continue
        best = _as_float(row.get("best_macro_f1"), None)
        if best is None:
            continue
        key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        current = result.get(key)
        if current is None or best > float(_as_float(current.get("best_macro_f1"), -math.inf) or -math.inf):
            result[key] = row
    return result


def _full_refs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        if key in refs:
            continue
        if not any(str(row.get(name, "")) for name in (
            "full_graph_rgcn_lite_default_macro_f1",
            "full_graph_rgcn_lite_tuned_macro_f1",
            "full_graph_han_small_macro_f1",
            "full_graph_hgt_small_macro_f1",
        )):
            continue
        refs[key] = {
            "dataset": key[0],
            "seed": key[1],
            "full_graph_rgcn_lite_default_macro_f1": row.get("full_graph_rgcn_lite_default_macro_f1", ""),
            "full_graph_rgcn_lite_tuned_macro_f1": row.get("full_graph_rgcn_lite_tuned_macro_f1", ""),
            "full_graph_han_small_macro_f1": row.get("full_graph_han_small_macro_f1", ""),
            "full_graph_hgt_small_macro_f1": row.get("full_graph_hgt_small_macro_f1", ""),
        }
    return sorted(refs.values(), key=lambda item: (str(item["dataset"]), str(item["seed"])))


def _attach_gaps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best_baselines = _best_by_key(rows, COARSE_BASELINE_METHODS)
    h0_rows = _best_by_key(rows, {"H0-mutual-best"})
    full_refs_by_key = {
        (str(row["dataset"]), str(row["seed"])): row
        for row in _full_refs(rows)
    }
    enriched: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        best_baseline = best_baselines.get(key)
        h0 = h0_rows.get(key)
        full_ref = full_refs_by_key.get(key)
        for prefix, comparator in (
            ("best_baseline", best_baseline),
            ("h0", h0),
        ):
            output[f"{prefix}_method"] = comparator.get("method", "") if comparator else ""
            output[f"delta_best_vs_{prefix}"] = ""
            output[f"dee_reduction_vs_{prefix}"] = ""
            if comparator:
                own_best = _as_float(row.get("best_macro_f1"), None)
                comp_best = _as_float(comparator.get("best_macro_f1"), None)
                if own_best is not None and comp_best is not None:
                    output[f"delta_best_vs_{prefix}"] = own_best - comp_best
                own_dee = _as_float(row.get("dee"), None)
                comp_dee = _as_float(comparator.get("dee"), None)
                if own_dee is not None and comp_dee not in (None, 0.0):
                    output[f"dee_reduction_vs_{prefix}"] = (comp_dee - own_dee) / comp_dee
        output["delta_best_vs_full_tuned"] = ""
        if full_ref:
            own_best = _as_float(row.get("best_macro_f1"), None)
            full_tuned = _as_float(full_ref.get("full_graph_rgcn_lite_tuned_macro_f1"), None)
            if own_best is not None and full_tuned is not None:
                output["delta_best_vs_full_tuned"] = own_best - full_tuned
        enriched.append(output)
    return enriched


def _group_rows(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        summary = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        summary["run_count"] = len(group)
        summary["seed_count"] = len({str(row.get("seed", "")) for row in group if str(row.get("seed", ""))})
        if "dataset" not in group_keys:
            summary["datasets"] = ",".join(sorted({str(row.get("dataset", "")) for row in group if row.get("dataset")}))
        for column in METRIC_COLUMNS:
            summary[f"{column}_mean"] = _mean(row.get(column) for row in group) or ""
            summary[f"{column}_std"] = _std(row.get(column) for row in group) or ""
            mean_value = _as_float(summary[f"{column}_mean"], None)
            std_value = _as_float(summary[f"{column}_std"], None)
            summary[f"{column}_mean_pm_std"] = (
                ""
                if mean_value is None
                else f"{_fmt(mean_value)} +/- {_fmt(std_value or 0.0)}"
            )
        for gap_col in (
            "delta_best_vs_best_baseline",
            "delta_best_vs_h0",
            "delta_best_vs_full_tuned",
            "dee_reduction_vs_best_baseline",
            "dee_reduction_vs_h0",
        ):
            summary[f"{gap_col}_mean"] = _mean(row.get(gap_col) for row in group) or ""
        output.append(summary)
    return sorted(output, key=lambda row: _sort_key(row.get("method", "")))


def _sort_key(method: Any) -> tuple[int, str]:
    text = str(method)
    try:
        return (METHOD_ORDER.index(text), text)
    except ValueError:
        return (len(METHOD_ORDER), text)


def _win_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") in COARSE_BASELINE_METHODS:
            continue
        if _as_float(row.get("delta_best_vs_best_baseline"), None) is None:
            continue
        groups[str(row.get("method", ""))].append(row)
    output = []
    for method, group in groups.items():
        wins = sum(1 for row in group if float(_as_float(row.get("delta_best_vs_best_baseline"), 0.0) or 0.0) > 0.0)
        ties = sum(1 for row in group if float(_as_float(row.get("delta_best_vs_best_baseline"), 0.0) or 0.0) == 0.0)
        total = len(group)
        output.append(
            {
                "method": method,
                "compared_seeds": total,
                "wins_vs_best_baseline": wins,
                "ties_vs_best_baseline": ties,
                "win_rate_vs_best_baseline": "" if total == 0 else wins / total,
            }
        )
    return sorted(output, key=lambda row: _sort_key(row["method"]))


def _pareto_rows(aggregate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in aggregate_rows:
        points.append(
            {
                "method": row.get("method", ""),
                "dee_mean": row.get("dee_mean", ""),
                "best_macro_f1_mean": row.get("best_macro_f1_mean", ""),
                "runtime_sec_mean": row.get("runtime_sec_mean", ""),
            }
        )
    for point in points:
        point["pareto_frontier"] = "true"
        dee = _as_float(point.get("dee_mean"), None)
        best = _as_float(point.get("best_macro_f1_mean"), None)
        runtime = _as_float(point.get("runtime_sec_mean"), None)
        if dee is None or best is None:
            continue
        for other in points:
            if other is point:
                continue
            other_dee = _as_float(other.get("dee_mean"), None)
            other_best = _as_float(other.get("best_macro_f1_mean"), None)
            other_runtime = _as_float(other.get("runtime_sec_mean"), None)
            if other_dee is None or other_best is None:
                continue
            runtime_ok = runtime is None or other_runtime is None or other_runtime <= runtime
            strict = other_dee < dee or other_best > best or (
                runtime is not None and other_runtime is not None and other_runtime < runtime
            )
            if other_dee <= dee and other_best >= best and runtime_ok and strict:
                point["pareto_frontier"] = "false"
                break
    return sorted(points, key=lambda row: _sort_key(row["method"]))


def summarize_next7_baseline_gap(
    *,
    input_summaries: Sequence[str | Path],
    baseline_summaries: Sequence[str | Path] = (),
    output: str | Path,
    command_lines: Sequence[str] = (),
) -> None:
    rows: list[dict[str, Any]] = []
    for path in input_summaries:
        rows.extend(_final_row(row) for row in _read_csv(Path(path)))
    for path in baseline_summaries:
        for row in _read_csv(Path(path)):
            converted = _baseline_row(row)
            if converted is not None:
                rows.append(converted)
    rows = _attach_gaps(_dedupe(rows))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    aggregate_rows = _group_rows(rows, ["method"])
    per_dataset_rows = _group_rows(rows, ["dataset", "method"])
    gap_rows = _group_rows(
        [row for row in rows if row.get("method") not in COARSE_BASELINE_METHODS],
        ["method"],
    )

    write_csv(output / "per_seed_table.csv", rows)
    write_csv(output / "aggregate_main_table.csv", aggregate_rows)
    write_csv(output / "per_dataset_main_table.csv", per_dataset_rows)
    write_csv(output / "gap_summary.csv", gap_rows)
    write_csv(output / "win_rate_by_seed.csv", _win_rows(rows))
    write_csv(output / "pareto_points.csv", _pareto_rows(aggregate_rows))
    write_csv(output / "full_graph_reference_table.csv", _full_refs(rows))
    (output / "run_commands.txt").write_text("\n".join(command_lines) + ("\n" if command_lines else ""), encoding="utf-8")

    report_rows = [
        row
        for row in aggregate_rows
        if row.get("method") in {
            "HeSF-LVC-P",
            "HeSF-LVC-S",
            "HeSF-LVC-T",
            "H0-mutual-best",
            "flatten-sum",
            "random",
            "GraphZoom-style",
            "ConvMatch-style",
        }
    ]
    report = [
        "# Next7 HGB Baseline-Gap Report",
        "",
        f"Rows: {len(rows)}",
        f"Methods: {len(aggregate_rows)}",
        "",
        "## Main Table",
        "",
        markdown_table(
            report_rows,
            [
                "method",
                "run_count",
                "seed_count",
                "datasets",
                "dee_mean_pm_std",
                "ree_max_mean_pm_std",
                "sipe_mean_pm_std",
                "refined_macro_f1@5_mean_pm_std",
                "best_macro_f1_mean_pm_std",
                "refine_auc_macro_f1_mean_pm_std",
                "delta_best_vs_best_baseline_mean",
                "delta_best_vs_full_tuned_mean",
            ],
        ),
        "",
        "## Notes",
        "",
        "- Best baseline is selected per dataset/seed from random, GraphZoom-style, ConvMatch-style, H0-mutual-best, and flatten-sum.",
        "- Spectral reductions use DEE, where lower is better.",
        "- Full graph references are read from cached task-eval columns when available.",
        "",
        "## Commands",
        "",
        "```powershell",
        *command_lines,
        "```",
    ]
    (output / "next7_hgb_baseline_gap_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Next7 HGB baseline-gap tables from experiment summaries.")
    parser.add_argument("--input-summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-summaries", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-lines", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_next7_baseline_gap(
        input_summaries=args.input_summaries,
        baseline_summaries=args.baseline_summaries,
        output=args.output,
        command_lines=args.command_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
