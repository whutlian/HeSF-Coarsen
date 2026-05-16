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


ERROR_METRICS = {"DEE", "FSE", "REEmax", "SIPE"}
TASK_METRICS = {
    "best_macro_f1",
    "refined_macro_f1@5",
    "projected_macro_f1",
}
FIXED_BASELINES = [
    "H0-mutual-best",
    "flatten-sum",
    "H6-no-spec",
    "random",
    "GraphZoom-style",
    "ConvMatch-style",
    "full RGCN tuned",
]
ORACLE_COARSE_BASELINES = {
    "H0-mutual-best",
    "flatten-sum",
    "random",
    "GraphZoom-style",
    "ConvMatch-style",
    "H6-no-spec",
}
METHOD_GROUP = {
    "HeSF-LVC-P": "ours",
    "HeSF-LVC-S": "ours",
    "HeSF-LVC-T": "ours",
    "H0-mutual-best": "coarse_baseline",
    "flatten-sum": "coarse_baseline",
    "random": "coarse_baseline",
    "GraphZoom-style": "coarse_baseline",
    "ConvMatch-style": "coarse_baseline",
    "full RGCN default": "full_graph_reference",
    "full RGCN tuned": "full_graph_reference",
    "HAN-small": "full_graph_reference",
    "HGT-lite": "full_graph_reference",
    "H6-no-spec": "ablation",
}
METHOD_ROLE = {
    "HeSF-LVC-P": "default",
    "HeSF-LVC-S": "spectral_safe",
    "HeSF-LVC-T": "appendix",
    "H0-mutual-best": "negative_control",
    "flatten-sum": "negative_control",
    "random": "negative_control",
    "GraphZoom-style": "negative_control",
    "ConvMatch-style": "negative_control",
    "full RGCN default": "full_graph",
    "full RGCN tuned": "full_graph",
    "HAN-small": "full_graph",
    "HGT-lite": "full_graph",
    "H6-no-spec": "no_spec",
}
BY_SEED_COLUMNS = [
    "method",
    "method_group",
    "role",
    "seed",
    "dataset",
    "target_ratio",
    "final_ratio",
    "target_hit",
    "original_nodes",
    "coarse_nodes",
    "original_edges",
    "coarse_edges",
    "coarse_graph_ratio",
    "DEE",
    "FSE",
    "REEmax",
    "SIPE",
    "projected_macro_f1",
    "refined_macro_f1@0",
    "refined_macro_f1@1",
    "refined_macro_f1@3",
    "refined_macro_f1@5",
    "best_macro_f1",
    "refine_auc_macro_f1",
    "projected_micro_f1",
    "refined_micro_f1@5",
    "best_micro_f1",
    "coarsen_sec",
    "coarse_train_sec",
    "refine_sec",
    "total_wall_clock_sec",
    "peak_rss_gb",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "candidate_sec",
    "scoring_sec",
    "matching_sec",
    "aggregation_sec",
]
NUMERIC_BY_SEED_COLUMNS = set(BY_SEED_COLUMNS) - {"method", "method_group", "role", "seed", "dataset", "target_hit"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_from_run_row(row: Mapping[str, Any]) -> str:
    method = str(row.get("method", "") or "")
    if method:
        return method
    variant = str(row.get("variant", "") or "")
    if variant in {"H0", "H0-mutual-best"}:
        return "H0-mutual-best"
    if variant in {"H6", "H6-no-spec"}:
        return "H6-no-spec"
    if variant in {"flatten-sum", "flatten_sum"}:
        return "flatten-sum"
    lambda_spec = _as_float(row.get("lambda_spec", row.get("config.scoring.lambda_spec")), None)
    lambda_conv = _as_float(row.get("lambda_conv", row.get("config.scoring.lambda_conv")), None)
    lambda_rel = _as_float(row.get("lambda_rel", row.get("config.scoring.lambda_rel")), None)
    if lambda_conv == 0.0 and lambda_rel == 0.0:
        if lambda_spec == 0.25:
            return "HeSF-LVC-P"
        if lambda_spec == 0.5:
            return "HeSF-LVC-S"
    return variant


def _sum_prefix(row: Mapping[str, Any], prefix: str) -> str:
    total = 0.0
    found = False
    for key, value in row.items():
        if key.startswith(prefix + "."):
            number = _as_float(value, None)
            if number is not None:
                total += number
                found = True
    return _fmt(total) if found else ""


def _run_summary_index(run_summary_dirs: Sequence[str | Path] | None) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for summary_dir in run_summary_dirs or []:
        path = Path(summary_dir)
        rows = _read_csv(path / "run_final_summary.csv") or _read_csv(path / "final_summary.csv")
        for row in rows:
            method = _method_from_run_row(row)
            key = (method, str(row.get("dataset", "")), str(row.get("seed", "")))
            if method and key not in index:
                index[key] = row
    return index


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


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (_as_float(value, None) for value in values) if value is not None]
    return None if not clean else float(mean(clean))


def _std(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (_as_float(value, None) for value in values) if value is not None]
    return None if len(clean) <= 1 else float(pstdev(clean))


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = str(row.get("method", ""))
    peak_rss = _first(row, ("peak_rss_gb", "peak_cpu_memory_gb", "peak_memory_gb"), "")
    peak_reserved = _first(row, ("peak_vram_reserved_gb", "peak_gpu_memory_gb", "peak_memory_gb"), "")
    total_wall = _first(row, ("total_wall_clock_sec", "runtime_sec", "total_time_sec"), "")
    coarsen = _first(row, ("coarsen_sec", "runtime_sec"), "")
    out = {
        "method": method,
        "method_group": METHOD_GROUP.get(method, "other"),
        "role": METHOD_ROLE.get(method, "other"),
        "seed": row.get("seed", ""),
        "dataset": row.get("dataset", ""),
        "target_ratio": row.get("target_ratio", ""),
        "final_ratio": _first(row, ("final_ratio", "coarse_graph_ratio"), ""),
        "target_hit": row.get("target_hit", ""),
        "original_nodes": _first(row, ("original_nodes", "initial_nodes", "task.original_nodes"), ""),
        "coarse_nodes": _first(row, ("coarse_nodes", "final_nodes", "task.coarse_nodes"), ""),
        "original_edges": _first(row, ("original_edges",), ""),
        "coarse_edges": _first(row, ("coarse_edges",), ""),
        "coarse_graph_ratio": _first(row, ("coarse_graph_ratio", "final_ratio"), ""),
        "DEE": _first(row, ("DEE", "dee"), ""),
        "FSE": _first(row, ("FSE", "fse"), ""),
        "REEmax": _first(row, ("REEmax", "ree_max"), ""),
        "SIPE": _first(row, ("SIPE", "sipe"), ""),
        "projected_macro_f1": row.get("projected_macro_f1", ""),
        "refined_macro_f1@0": row.get("refined_macro_f1@0", ""),
        "refined_macro_f1@1": row.get("refined_macro_f1@1", ""),
        "refined_macro_f1@3": row.get("refined_macro_f1@3", ""),
        "refined_macro_f1@5": row.get("refined_macro_f1@5", ""),
        "best_macro_f1": row.get("best_macro_f1", ""),
        "refine_auc_macro_f1": row.get("refine_auc_macro_f1", ""),
        "projected_micro_f1": _first(row, ("projected_micro_f1", "task_projected_micro_f1"), ""),
        "refined_micro_f1@5": _first(row, ("refined_micro_f1@5", "task_refined_micro_f1@5"), ""),
        "best_micro_f1": _first(row, ("best_micro_f1", "task_best_refined_micro_f1"), ""),
        "coarsen_sec": coarsen,
        "coarse_train_sec": _first(row, ("coarse_train_sec", "train_time_sec"), ""),
        "refine_sec": _first(row, ("refine_sec", "refine_time_sec"), ""),
        "total_wall_clock_sec": total_wall,
        "peak_rss_gb": peak_rss,
        "peak_vram_allocated_gb": _first(row, ("peak_vram_allocated_gb",), ""),
        "peak_vram_reserved_gb": peak_reserved,
        "candidate_sec": _first(row, ("candidate_sec",), ""),
        "scoring_sec": _first(row, ("scoring_sec",), ""),
        "matching_sec": _first(row, ("matching_sec",), ""),
        "aggregation_sec": _first(row, ("aggregation_sec",), ""),
    }
    for key in NUMERIC_BY_SEED_COLUMNS:
        out[key] = _fmt(out.get(key, ""))
    return out


def _enrich_row(row: dict[str, Any], run_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if run_row is None:
        return row
    enrich = {
        "original_nodes": _first(run_row, ("original_nodes", "initial_nodes", "task.original_nodes"), ""),
        "coarse_nodes": _first(run_row, ("coarse_nodes", "final_nodes", "task.coarse_nodes"), ""),
        "original_edges": _sum_prefix(run_row, "original_edge_count_by_relation"),
        "coarse_edges": _sum_prefix(run_row, "coarse_edge_count_by_relation"),
        "projected_micro_f1": _first(run_row, ("task_projected_micro_f1", "task.projected_original_micro_f1"), ""),
        "refined_micro_f1@5": _first(run_row, ("task_refined_micro_f1@5", "task.refined_original_micro_f1@5"), ""),
        "best_micro_f1": _first(run_row, ("task_best_refined_micro_f1", "task.best_refined_micro_f1"), ""),
        "coarsen_sec": _first(run_row, ("runtime_total_run", "runtime_sec"), ""),
        "coarse_train_sec": _first(run_row, ("task.train_time", "task_train_time", "train_time_sec"), ""),
        "refine_sec": _first(run_row, ("task.refine_time", "task_refine_time", "refine_time_sec"), ""),
        "peak_rss_gb": _first(run_row, ("peak_rss_gb", "peak_cpu_memory_gb"), ""),
        "peak_vram_allocated_gb": _first(run_row, ("peak_vram_allocated_gb", "peak_gpu_memory_allocated_gb"), ""),
        "peak_vram_reserved_gb": _first(run_row, ("peak_vram_reserved_gb", "peak_gpu_memory_reserved_gb"), ""),
        "candidate_sec": _first(run_row, ("runtime_by_stage.candidates", "candidate_generation_time"), ""),
        "scoring_sec": _first(run_row, ("runtime_by_stage.scoring",), ""),
        "matching_sec": _first(run_row, ("runtime_by_stage.matching",), ""),
        "aggregation_sec": _first(run_row, ("runtime_by_stage.aggregation",), ""),
    }
    if row.get("total_wall_clock_sec") in {None, ""}:
        coarsen = _as_float(enrich.get("coarsen_sec"), 0.0) or 0.0
        train = _as_float(enrich.get("coarse_train_sec"), 0.0) or 0.0
        refine = _as_float(enrich.get("refine_sec"), 0.0) or 0.0
        enrich["total_wall_clock_sec"] = _fmt(coarsen + train + refine) if coarsen or train or refine else ""
    for key, value in enrich.items():
        if row.get(key) in {None, ""} and value not in {None, ""}:
            row[key] = _fmt(value) if key in NUMERIC_BY_SEED_COLUMNS else value
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    metrics = [
        "DEE",
        "FSE",
        "REEmax",
        "SIPE",
        "projected_macro_f1",
        "refined_macro_f1@5",
        "best_macro_f1",
        "refine_auc_macro_f1",
        "coarsen_sec",
        "coarse_train_sec",
        "refine_sec",
        "total_wall_clock_sec",
        "peak_rss_gb",
        "peak_vram_reserved_gb",
    ]
    output = []
    for key, group in sorted(groups.items()):
        row = {name: value for name, value in zip(group_keys, key)}
        row["run_count"] = len(group)
        row["seed_count"] = len({str(item.get("seed", "")) for item in group if item.get("seed", "") != ""})
        for metric in metrics:
            values = [item.get(metric, "") for item in group]
            row[f"{metric}_mean"] = _fmt(_mean(values))
            row[f"{metric}_std"] = _fmt(_std(values) or 0.0)
        output.append(row)
    return output


def _value_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row.get("method", "")), str(row.get("dataset", "")), str(row.get("seed", ""))): row
        for row in rows
    }


def _gap(ours: Any, baseline: Any, metric: str) -> tuple[str, str]:
    ours_value = _as_float(ours, None)
    baseline_value = _as_float(baseline, None)
    if ours_value is None or baseline_value is None:
        return "", ""
    if metric in ERROR_METRICS:
        absolute = baseline_value - ours_value
        reduction = "" if baseline_value == 0 else 1.0 - ours_value / baseline_value
        return _fmt(absolute), _fmt(reduction)
    return _fmt(ours_value - baseline_value), ""


def _fixed_gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _value_lookup(rows)
    output = []
    for ours_method in ("HeSF-LVC-P", "HeSF-LVC-S"):
        for row in rows:
            if row.get("method") != ours_method:
                continue
            dataset = str(row.get("dataset", ""))
            seed = str(row.get("seed", ""))
            for baseline in FIXED_BASELINES:
                baseline_row = lookup.get((baseline, dataset, seed))
                if baseline_row is None:
                    continue
                for metric in [*TASK_METRICS, *ERROR_METRICS]:
                    absolute, reduction = _gap(row.get(metric, ""), baseline_row.get(metric, ""), metric)
                    output.append(
                        {
                            "method": ours_method,
                            "dataset": dataset,
                            "seed": seed,
                            "baseline": baseline,
                            "metric": metric,
                            "absolute_gap": absolute,
                            "relative_error_reduction": reduction,
                        }
                    )
    return output


def _oracle_gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset_seed: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset_seed[(str(row.get("dataset", "")), str(row.get("seed", "")))].append(row)
    output = []
    for ours_method in ("HeSF-LVC-P", "HeSF-LVC-S"):
        for key, group in sorted(by_dataset_seed.items()):
            ours = next((row for row in group if row.get("method") == ours_method), None)
            if ours is None:
                continue
            coarse = [
                row for row in group
                if row.get("method") in ORACLE_COARSE_BASELINES and _as_float(row.get("best_macro_f1"), None) is not None
            ]
            if not coarse:
                continue
            oracle = max(coarse, key=lambda row: float(row.get("best_macro_f1") or 0.0))
            for metric in [*TASK_METRICS, *ERROR_METRICS]:
                absolute, reduction = _gap(ours.get(metric, ""), oracle.get(metric, ""), metric)
                output.append(
                    {
                        "method": ours_method,
                        "dataset": key[0],
                        "seed": key[1],
                        "oracle_coarse_baseline": oracle.get("method", ""),
                        "metric": metric,
                        "absolute_gap": absolute,
                        "relative_error_reduction": reduction,
                    }
                )
    return output


def _win_rate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fixed = _fixed_gap_rows(rows)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fixed:
        grouped[(row["method"], row["dataset"], row["baseline"], row["metric"])].append(row)
    output = []
    for (method, dataset, baseline, metric), group in sorted(grouped.items()):
        gaps = [_as_float(row.get("absolute_gap"), None) for row in group]
        clean = [gap for gap in gaps if gap is not None]
        wins = sum(1 for gap in clean if gap > 0)
        output.append(
            {
                "method": method,
                "dataset": dataset,
                "baseline": baseline,
                "metric": metric,
                "run_count": len(clean),
                "win_rate": _fmt(wins / len(clean)) if clean else "",
            }
        )
    return output


def _checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = ["projected_macro_f1", "refined_macro_f1@0", "refined_macro_f1@1", "refined_macro_f1@3", "refined_macro_f1@5"]
    output = []
    for row in rows:
        for checkpoint in checkpoints:
            output.append(
                {
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "checkpoint": checkpoint,
                    "macro_f1": row.get(checkpoint, ""),
                }
            )
    return output


def _cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": row.get("method", ""),
            "method_group": row.get("method_group", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "coarsen_sec": row.get("coarsen_sec", ""),
            "coarse_train_sec": row.get("coarse_train_sec", ""),
            "refine_sec": row.get("refine_sec", ""),
            "total_wall_clock_sec": row.get("total_wall_clock_sec", ""),
            "peak_rss_gb": row.get("peak_rss_gb", ""),
            "peak_vram_allocated_gb": row.get("peak_vram_allocated_gb", ""),
            "peak_vram_reserved_gb": row.get("peak_vram_reserved_gb", ""),
        }
        for row in rows
    ]


def _plot_scatter(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc````\x00"
            b"\x00\x00\x05\x00\x01\xa5\xf6E@\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return
    points = [
        (float(x), float(y), str(row.get("method", "")))
        for row in rows
        for x, y in [(_as_float(row.get(x_key), None), _as_float(row.get(y_key), None))]
        if x is not None and y is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if points:
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=36)
        for x, y, label in points:
            ax.annotate(label, (x, y), fontsize=7, alpha=0.75)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_next9_hgb_paper_final(
    *,
    next8_summary_dir: str | Path,
    output: str | Path,
    run_summary_dirs: Sequence[str | Path] | None = None,
    command_lines: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    next8_summary_dir = Path(next8_summary_dir)
    output = Path(output)
    raw_rows = _read_csv(next8_summary_dir / "per_seed_table.csv")
    run_index = _run_summary_index(run_summary_dirs)
    by_seed = []
    for raw_row in raw_rows:
        normalized = _normalize_row(raw_row)
        key = (
            str(normalized.get("method", "")),
            str(normalized.get("dataset", "")),
            str(normalized.get("seed", "")),
        )
        by_seed.append(_enrich_row(normalized, run_index.get(key)))
    aggregate = _aggregate(by_seed, ("method",))
    by_dataset = _aggregate(by_seed, ("dataset", "method"))
    fixed = _fixed_gap_rows(by_seed)
    oracle = _oracle_gap_rows(by_seed)
    win_rate = _win_rate_rows(by_seed)
    checkpoint = _checkpoint_rows(by_seed)
    costs = _cost_rows(by_seed)

    write_csv(output / "final_main_table_by_seed.csv", by_seed)
    write_csv(output / "final_main_table_aggregate.csv", aggregate)
    write_csv(output / "final_main_table_by_dataset.csv", by_dataset)
    write_csv(output / "final_gap_vs_fixed_baselines.csv", fixed)
    write_csv(output / "final_gap_vs_oracle_coarse_baseline.csv", oracle)
    write_csv(output / "final_win_rate_by_dataset_seed.csv", win_rate)
    write_csv(output / "final_checkpoint_curve.csv", checkpoint)
    write_csv(output / "final_cost_memory_table.csv", costs)
    if command_lines:
        (output / "run_commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    figure_rows = aggregate
    figure_dir = output / "figures"
    _plot_scatter(figure_rows, "DEE_mean", "best_macro_f1_mean", figure_dir / "dee_vs_best_macro_f1.png")
    _plot_scatter(figure_rows, "REEmax_mean", "best_macro_f1_mean", figure_dir / "ree_max_vs_best_macro_f1.png")
    _plot_scatter(
        figure_rows,
        "total_wall_clock_sec_mean",
        "best_macro_f1_mean",
        figure_dir / "best_macro_f1_vs_wall_clock.png",
    )
    _plot_scatter(figure_rows, "peak_rss_gb_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_peak_rss.png")
    _plot_scatter(figure_rows, "seed_count", "refined_macro_f1@5_mean", figure_dir / "checkpoint_curve_by_method_dataset.png")

    summary = [
        "# Next9 HGB Paper-ready Final Table",
        "",
        "Per-dataset rows are primary; aggregate rows include standard deviations and must not hide dataset behavior.",
        "",
        markdown_table(
            aggregate,
            ["method", "run_count", "DEE_mean", "REEmax_mean", "SIPE_mean", "refined_macro_f1@5_mean", "best_macro_f1_mean"],
        ),
        "",
        "HeSF-LVC-P/S are positioned as operator-preserving compressed methods. flatten-sum and H6-no-spec may remain task-competitive while showing higher operator distortion.",
        "",
        "The full tuned RGCN remains stronger on pure task F1 when its row is above P/S; the claim is quality-cost and operator preservation, not task-SOTA.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "by_seed": by_seed,
        "aggregate": aggregate,
        "by_dataset": by_dataset,
        "fixed": fixed,
        "oracle": oracle,
        "win_rate": win_rate,
        "checkpoint": checkpoint,
        "costs": costs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next8-summary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-summary-dirs", nargs="*", default=[])
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_hgb_paper_final(
        next8_summary_dir=args.next8_summary_dir,
        output=args.output,
        run_summary_dirs=args.run_summary_dirs,
        command_lines=args.command_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
