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


CHECKPOINT_METRICS = [
    "dee",
    "fse",
    "ree_max",
    "sipe",
    "projected_macro_f1",
    "refined_macro_f1@0",
    "refined_macro_f1@1",
    "refined_macro_f1@3",
    "refined_macro_f1@5",
    "best_macro_f1",
]
TASK_METRICS = [
    "projected_macro_f1",
    "refined_macro_f1@0",
    "refined_macro_f1@1",
    "refined_macro_f1@3",
    "refined_macro_f1@5",
    "best_macro_f1",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_many(paths: Sequence[str | Path] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths or []:
        rows.extend(_read_csv(path))
    return rows


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


def _metric_value(row: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "projected_macro_f1": ("projected_macro_f1", "task_projected_macro_f1", "projected_original_macro_f1"),
        "refined_macro_f1@0": ("refined_macro_f1@0", "task_refined_macro_f1@0", "refined_original_macro_f1@0"),
        "refined_macro_f1@1": ("refined_macro_f1@1", "task_refined_macro_f1@1", "refined_original_macro_f1@1"),
        "refined_macro_f1@3": ("refined_macro_f1@3", "task_refined_macro_f1@3", "refined_original_macro_f1@3"),
        "refined_macro_f1@5": (
            "refined_macro_f1@5",
            "task_refined_macro_f1@5",
            "refined_original_macro_f1@5",
            "refined_original_macro_f1",
        ),
        "best_macro_f1": ("best_macro_f1", "task_best_refined_macro_f1", "best_refined_macro_f1"),
        "dee": ("dee", "cumulative_dee"),
        "fse": ("fse", "cumulative_fse_unweighted"),
        "ree_max": ("ree_max", "cumulative_ree_max"),
        "sipe": ("sipe", "cumulative_sipe"),
    }
    return _first(row, aliases.get(name, (name,)))


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
    if variant == "H2-single-relation-sum":
        return "flatten-sum"
    if variant == "H0":
        return "H0-mutual-best"
    if variant == "H4":
        return "H4-no-conv"
    if variant == "H6":
        return "H6-no-spec"
    if variant == "H2" and (_close(_lambda_value(row, "lambda_spec"), 0.25) or _lambda_value(row, "lambda_spec") == ""):
        return "HeSF-LVC-P"
    if variant == "H3" and (_close(_lambda_value(row, "lambda_spec"), 0.5) or _lambda_value(row, "lambda_spec") == ""):
        return "HeSF-LVC-S"
    run_name = str(row.get("run_name", "")).lower()
    if "flatten" in run_name:
        return "flatten-sum"
    return variant or "unknown"


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(group_key, "")) for group_key in group_keys)
        groups[key].append(row)
    output_rows: list[dict[str, str]] = []
    for key, group in sorted(groups.items()):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["run_count"] = str(len(group))
        for metric in metrics:
            values = [_metric_value(row, metric) for row in group]
            out[f"{metric}_mean"] = _fmt(_mean(values))
            out[f"{metric}_std"] = _fmt(_std(values) or 0.0)
        output_rows.append(out)
    return output_rows


def _checkpoint_rows(per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for row in per_seed:
        item = dict(row)
        item["method"] = _method(row)
        normalized.append(item)
    return _aggregate(normalized, group_keys=("method",), metrics=CHECKPOINT_METRICS)


def _flatten_failure_rows(per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_dataset_method: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        dataset = str(row.get("dataset", ""))
        if not dataset:
            continue
        by_dataset_method[(dataset, _method(row))].append(row)
    datasets = sorted({dataset for dataset, _ in by_dataset_method})
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        flatten = _mean(_metric_value(row, "best_macro_f1") for row in by_dataset_method.get((dataset, "flatten-sum"), []))
        if flatten is None:
            continue
        out = {"dataset": dataset, "flatten_sum_best_macro_f1": _fmt(flatten)}
        for method in ("HeSF-LVC-P", "HeSF-LVC-S"):
            value = _mean(_metric_value(row, "best_macro_f1") for row in by_dataset_method.get((dataset, method), []))
            out[f"{method}_best_macro_f1"] = _fmt(value)
            out[f"delta_best_vs_{method}"] = _fmt(flatten - value) if value is not None else ""
        rows.append(out)
    return rows


def _task_rows(rows: Sequence[Mapping[str, Any]], *, default_fraction: str = "") -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["method"] = _method(row)
        item["coarse_model"] = _first(row, ("coarse_model", "model"), "rgcn_lite")
        item["train_fraction"] = str(_first(row, ("train_fraction", "task_train_fraction"), default_fraction))
        normalized.append(item)
    return normalized


def summarize_p1_flatten_sum_challenge(
    *,
    final_gap_dir: str | Path,
    output: str | Path,
    cross_model_inputs: Sequence[str | Path] | None = None,
    low_label_inputs: Sequence[str | Path] | None = None,
    command_lines: Sequence[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    final_gap_dir = Path(final_gap_dir)
    output = Path(output)
    per_seed = _read_csv(final_gap_dir / "per_seed_table.csv")
    checkpoint = _checkpoint_rows(per_seed)
    failure = _flatten_failure_rows(per_seed)
    cross_model = _aggregate(
        _task_rows(_read_many(cross_model_inputs)),
        group_keys=("method", "dataset", "coarse_model"),
        metrics=TASK_METRICS,
    )
    low_label = _aggregate(
        _task_rows(_read_many(low_label_inputs), default_fraction="low-label"),
        group_keys=("method", "dataset", "train_fraction"),
        metrics=TASK_METRICS,
    )
    write_csv(output / "checkpoint_comparison.csv", checkpoint)
    write_csv(output / "flatten_sum_failure_by_dataset.csv", failure)
    write_csv(output / "cross_model_transfer.csv", cross_model)
    write_csv(output / "low_label_transfer.csv", low_label)
    if command_lines:
        (output / "run_commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    report = [
        "# P1 flatten-sum challenge",
        "",
        "This summary frames HeSF-LVC as operator-preserving coarsening and isolates whether task recovery "
        "comes from projection, early refinement, low-label transfer, or cross-model transfer.",
        "",
        "## Checkpoints",
        "",
        markdown_table(checkpoint, ["method", "run_count", "projected_macro_f1_mean", "refined_macro_f1@0_mean", "refined_macro_f1@1_mean", "refined_macro_f1@3_mean", "refined_macro_f1@5_mean", "best_macro_f1_mean"]),
        "",
        "## Flatten-sum Failure Cases",
        "",
        markdown_table(failure, ["dataset", "flatten_sum_best_macro_f1", "HeSF-LVC-P_best_macro_f1", "delta_best_vs_HeSF-LVC-P", "HeSF-LVC-S_best_macro_f1", "delta_best_vs_HeSF-LVC-S"]),
        "",
        "## Cross-model Transfer",
        "",
        markdown_table(cross_model, ["method", "dataset", "coarse_model", "run_count", "best_macro_f1_mean", "refined_macro_f1@5_mean"]),
        "",
        "## Low-label Transfer",
        "",
        markdown_table(low_label, ["method", "dataset", "train_fraction", "run_count", "best_macro_f1_mean", "refined_macro_f1@5_mean"]),
        "",
    ]
    (output / "p1_flatten_sum_challenge_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "checkpoint": checkpoint,
        "failure": failure,
        "cross_model": cross_model,
        "low_label": low_label,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-gap-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cross-model-inputs", nargs="*", default=[])
    parser.add_argument("--low-label-inputs", nargs="*", default=[])
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_p1_flatten_sum_challenge(
        final_gap_dir=args.final_gap_dir,
        output=args.output,
        cross_model_inputs=args.cross_model_inputs,
        low_label_inputs=args.low_label_inputs,
        command_lines=args.command_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
