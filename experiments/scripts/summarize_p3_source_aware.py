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


RUN_METRICS = [
    "dee",
    "ree_max",
    "sipe",
    "projected_macro_f1",
    "refined_macro_f1@5",
    "best_macro_f1",
    "onehop_retained",
    "onehop_selected",
    "fallback_selected",
    "bucket_selected",
    "onehop_rejected_by_spec",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


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


def _lambda_value(row: Mapping[str, Any], name: str) -> Any:
    return _first(row, (name, f"config.scoring.{name}", f"scoring.{name}"), "")


def _close(value: Any, expected: float, tol: float = 1e-9) -> bool:
    number = _as_float(value, None)
    return number is not None and abs(number - expected) <= tol


def _method(row: Mapping[str, Any]) -> str:
    explicit = _first(row, ("method", "paper.method"), "")
    if explicit:
        return str(explicit)
    variant = str(row.get("variant", ""))
    if variant == "H2" and (_close(_lambda_value(row, "lambda_spec"), 0.25) or _lambda_value(row, "lambda_spec") == ""):
        return "HeSF-LVC-P"
    if variant == "H3" and (_close(_lambda_value(row, "lambda_spec"), 0.5) or _lambda_value(row, "lambda_spec") == ""):
        return "HeSF-LVC-S"
    return variant or "unknown"


def _metric(row: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "dee": ("dee", "cumulative_dee", "cumulative_spectral.dirichlet_energy_relative_error"),
        "ree_max": ("ree_max", "cumulative_ree_max", "cumulative_spectral.relation_energy_relative_error_max"),
        "sipe": ("sipe", "cumulative_sipe", "cumulative_spectral.chebheat_sketch_inner_product_relative_error"),
        "projected_macro_f1": ("task_projected_macro_f1", "task.projected_original_macro_f1"),
        "refined_macro_f1@5": ("task_refined_macro_f1@5", "task.refined_original_macro_f1@5"),
        "best_macro_f1": ("task_best_refined_macro_f1", "task.best_refined_macro_f1"),
        "onehop_retained": ("candidate_source_counts.onehop",),
        "onehop_selected": ("selected_merges_by_source.onehop", "matched_pairs_by_source.onehop"),
        "fallback_selected": ("selected_merges_by_source.fallback", "matched_pairs_by_source.fallback"),
        "bucket_selected": ("selected_merges_by_source.bucket", "matched_pairs_by_source.bucket"),
        "onehop_rejected_by_spec": ("source_policy_filter.onehop_rejected_by_spec",),
    }
    return _first(row, aliases[name], 0 if name.endswith("_selected") or name.endswith("_retained") else "")


def _run_rows(summary_dir: Path, policy: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(summary_dir / "run_final_summary.csv"):
        method = _method(row)
        if method not in {"HeSF-LVC-P", "HeSF-LVC-S"}:
            continue
        item = dict(row)
        item["policy"] = policy
        item["method"] = method
        item["target_hit_numeric"] = 1.0 if str(row.get("target_hit", "")).lower() == "true" else 0.0
        rows.append(item)
    return rows


def _aggregate_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), str(row["method"]), str(row.get("dataset", "")))].append(row)
    out_rows: list[dict[str, str]] = []
    for (policy, method, dataset), group in sorted(groups.items()):
        out = {"policy": policy, "method": method, "dataset": dataset, "run_count": str(len(group))}
        out["target_hit_rate"] = _fmt(_mean(row.get("target_hit_numeric") for row in group))
        for metric in RUN_METRICS:
            values = [_metric(row, metric) for row in group]
            out[f"{metric}_mean"] = _fmt(_mean(values))
            out[f"{metric}_std"] = _fmt(_std(values) or 0.0)
        out_rows.append(out)
    return out_rows


def _source_rows(summary_dir: Path, policy: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(summary_dir / "candidate_source_pareto.csv"):
        method = _method(row)
        if method not in {"HeSF-LVC-P", "HeSF-LVC-S"}:
            continue
        item = dict(row)
        item["policy"] = policy
        item["method"] = method
        rows.append(item)
    return rows


def _aggregate_sources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["policy"]),
                str(row["method"]),
                str(row.get("dataset", "")),
                str(row.get("source", "")),
            )
        ].append(row)
    out_rows: list[dict[str, str]] = []
    for (policy, method, dataset, source), group in sorted(groups.items()):
        out = {
            "policy": policy,
            "method": method,
            "dataset": dataset,
            "source": source,
            "run_count": str(len(group)),
            "candidate_fraction_mean": _fmt(_mean(row.get("candidate_fraction") for row in group)),
            "selected_fraction_mean": _fmt(_mean(row.get("selected_fraction") for row in group)),
            "avg_delta_spec_mean": _fmt(_mean(row.get("avg_delta_spec") for row in group)),
            "avg_delta_conv_mean": _fmt(_mean(row.get("avg_delta_conv") for row in group)),
        }
        out_rows.append(out)
    return out_rows


def summarize_p3_source_aware(
    *,
    baseline_summary_dir: str | Path,
    source_aware_summary_dir: str | Path,
    output: str | Path,
    command_lines: Sequence[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    baseline_summary_dir = Path(baseline_summary_dir)
    source_aware_summary_dir = Path(source_aware_summary_dir)
    output = Path(output)
    run_rows = [
        *_run_rows(baseline_summary_dir, "baseline"),
        *_run_rows(source_aware_summary_dir, "source-aware"),
    ]
    source_rows = [
        *_source_rows(baseline_summary_dir, "baseline"),
        *_source_rows(source_aware_summary_dir, "source-aware"),
    ]
    comparison = _aggregate_runs(run_rows)
    distribution = _aggregate_sources(source_rows)
    write_csv(output / "source_policy_comparison.csv", comparison)
    write_csv(output / "source_distribution_by_policy.csv", distribution)
    if command_lines:
        (output / "run_commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    report = [
        "# P3 source-aware filtering",
        "",
        "This compares baseline candidate policy against source-aware policy for HeSF-LVC-P/S.",
        "",
        "## Policy Metrics",
        "",
        markdown_table(
            comparison,
            [
                "policy",
                "method",
                "dataset",
                "run_count",
                "dee_mean",
                "ree_max_mean",
                "sipe_mean",
                "projected_macro_f1_mean",
                "refined_macro_f1@5_mean",
                "best_macro_f1_mean",
                "target_hit_rate",
                "onehop_retained_mean",
                "onehop_rejected_by_spec_mean",
            ],
        ),
        "",
        "## Source Distribution",
        "",
        markdown_table(
            distribution,
            [
                "policy",
                "method",
                "dataset",
                "source",
                "candidate_fraction_mean",
                "selected_fraction_mean",
                "avg_delta_spec_mean",
            ],
        ),
        "",
    ]
    (output / "p3_source_aware_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"comparison": comparison, "distribution": distribution}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary-dir", required=True, type=Path)
    parser.add_argument("--source-aware-summary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_p3_source_aware(
        baseline_summary_dir=args.baseline_summary_dir,
        source_aware_summary_dir=args.source_aware_summary_dir,
        output=args.output,
        command_lines=args.command_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
