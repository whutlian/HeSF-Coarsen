from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter
from hesf_coarsen.io.edge_list import load_graph


METHODS = {"HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best"}
CHECKPOINTS = [
    ("projected", "projected_macro_f1"),
    ("refined@0", "refined_macro_f1@0"),
    ("refined@1", "refined_macro_f1@1"),
    ("refined@3", "refined_macro_f1@3"),
    ("refined@5", "refined_macro_f1@5"),
    ("best", "best_macro_f1"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fmt(value: Any, digits: int = 6) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return default


def _method_from_row(row: Mapping[str, Any]) -> str:
    method = str(row.get("method", "") or "")
    if method in METHODS:
        return method
    variant = str(row.get("variant", "") or "")
    if variant in {"H0", "H0-mutual-best"}:
        return "H0-mutual-best"
    if variant in {"H6", "H6-no-spec"}:
        return "H6-no-spec"
    if variant in {"flatten-sum", "flatten_sum", "H2-single-relation-sum"}:
        return "flatten-sum"
    operator_mode = str(
        row.get("fusion.relation_operator_mode", row.get("config.fusion.relation_operator_mode", ""))
    )
    if operator_mode == "single_relation_sum":
        return "flatten-sum"
    lambda_spec = _as_float(row.get("lambda_spec", row.get("config.scoring.lambda_spec")), None)
    lambda_conv = _as_float(row.get("lambda_conv", row.get("config.scoring.lambda_conv")), None)
    lambda_rel = _as_float(row.get("lambda_rel", row.get("config.scoring.lambda_rel")), None)
    if lambda_conv == 0.0 and lambda_rel == 0.0:
        if lambda_spec == 0.25:
            return "HeSF-LVC-P"
        if lambda_spec == 0.5:
            return "HeSF-LVC-S"
    return method or variant


def _std(values: Sequence[float]) -> str:
    return _fmt(pstdev(values), 6) if len(values) > 1 else "0"


def _relation_ids(row: Mapping[str, Any], prefix: str) -> list[str]:
    ids = []
    token = prefix + "."
    for key in row:
        if key.startswith(token):
            ids.append(key.removeprefix(token))
    return sorted(set(ids), key=lambda value: int(value) if value.isdigit() else value)


def _distribution_from_counts(row: Mapping[str, Any], prefix: str) -> dict[str, float]:
    values = {rid: _as_float(row.get(f"{prefix}.{rid}"), 0.0) or 0.0 for rid in _relation_ids(row, prefix)}
    total = sum(values.values())
    if total <= 0.0:
        return {rid: 0.0 for rid in values}
    return {rid: value / total for rid, value in values.items()}


def _js(p: Sequence[float], q: Sequence[float]) -> float:
    eps = 1.0e-12
    p_sum = max(sum(p), eps)
    q_sum = max(sum(q), eps)
    pp = [value / p_sum for value in p]
    qq = [value / q_sum for value in q]
    mm = [(a + b) * 0.5 for a, b in zip(pp, qq)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log((x + eps) / (y + eps)) for x, y in zip(a, b) if x > 0.0)

    return 0.5 * kl(pp, mm) + 0.5 * kl(qq, mm)


def _run_rows(run_summary_dirs: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_dir in run_summary_dirs:
        for filename in ("run_final_summary.csv", "final_summary.csv"):
            rows.extend(_read_csv(summary_dir / filename))
            if rows:
                break
    filtered = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        method = _method_from_row(row)
        if method not in METHODS:
            continue
        key = (method, str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("run_dir", "")))
        if key in seen:
            continue
        seen.add(key)
        out = dict(row)
        out["method"] = method
        filtered.append(out)
    return filtered


def _relation_energy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ids = sorted(
            set(_relation_ids(row, "spectral.relation_energy_relative_error"))
            | set(_relation_ids(row, "cumulative_spectral.relation_energy_relative_error")),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        for rid in ids:
            error = row.get(f"cumulative_spectral.relation_energy_relative_error.{rid}") or row.get(
                f"spectral.relation_energy_relative_error.{rid}",
                "",
            )
            out.append(
                {
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "relation_id": rid,
                    "relation_energy_error": _fmt(error),
                    "relation_energy_before": _fmt(
                        row.get(f"cumulative_spectral.relation_energy_before.{rid}")
                        or row.get(f"spectral.relation_energy_before.{rid}")
                    ),
                    "relation_energy_after": _fmt(
                        row.get(f"cumulative_spectral.relation_energy_after.{rid}")
                        or row.get(f"spectral.relation_energy_after.{rid}")
                    ),
                }
            )
    return out


def _relation_distribution_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        original = _distribution_from_counts(row, "original_edge_count_by_relation")
        coarse = _distribution_from_counts(row, "coarse_edge_count_by_relation")
        ids = sorted(set(original) | set(coarse), key=lambda value: int(value) if value.isdigit() else value)
        p = [original.get(rid, 0.0) for rid in ids]
        q = [coarse.get(rid, 0.0) for rid in ids]
        l1 = sum(abs(a - b) for a, b in zip(p, q))
        for rid in ids:
            out.append(
                {
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "relation_id": rid,
                    "relation_edge_mass_original": _fmt(original.get(rid, 0.0)),
                    "relation_edge_mass_coarse": _fmt(coarse.get(rid, 0.0)),
                    "relation_mass_l1_drift": _fmt(l1),
                    "relation_mass_js_drift": _fmt(_js(p, q)),
                }
            )
    return out


def _edge_collapse_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ids = sorted(
            set(_relation_ids(row, "original_edge_count_by_relation"))
            | set(_relation_ids(row, "coarse_edge_count_by_relation")),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        for rid in ids:
            original = _as_float(row.get(f"original_edge_count_by_relation.{rid}"), 0.0) or 0.0
            coarse = _as_float(row.get(f"coarse_edge_count_by_relation.{rid}"), 0.0) or 0.0
            uniqueness = coarse / max(original, 1.0)
            before_w = _as_float(row.get(f"relation_weight_before.{rid}"), 0.0) or 0.0
            after_w = _as_float(row.get(f"relation_weight_after.{rid}"), 0.0) or 0.0
            out.append(
                {
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "relation_id": rid,
                    "original_edges": int(original),
                    "coarse_edges_before_dedup": int(original),
                    "coarse_edges_after_dedup": int(coarse),
                    "coarse_edge_uniqueness_ratio": _fmt(uniqueness),
                    "self_loop_share": "",
                    "duplicate_collapse_ratio": _fmt(max(0.0, 1.0 - uniqueness)),
                    "run_dir": row.get("run_dir", ""),
                    "edge_weight_original_sum": _fmt(before_w),
                    "edge_weight_coarse_sum": _fmt(after_w),
                    "edge_weight_abs_error": _fmt(abs(before_w - after_w)),
                }
            )
    return out


def _aggregate_metric_by_dataset(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    output_metric: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method", "")), str(row.get("dataset", "")))].append(row)
    out = []
    for (method, dataset), group in sorted(groups.items()):
        values = [value for value in (_as_float(row.get(metric), None) for row in group) if value is not None]
        out.append(
            {
                "method": method,
                "dataset": dataset,
                f"{output_metric}_mean": _fmt(mean(values)) if values else "",
                f"{output_metric}_std": _std(values) if values else "",
                "valid_n": len(values),
                "missing_n": len(group) - len(values),
                "n_rows": len(group),
            }
        )
    return out


def _collapse_by_dataset(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method", "")), str(row.get("dataset", "")))].append(row)
    out = []
    for (method, dataset), group in sorted(groups.items()):
        values = [
            value
            for value in (_as_float(row.get("duplicate_collapse_ratio"), None) for row in group)
            if value is not None
        ]
        before = sum(_as_float(row.get("coarse_edges_before_dedup"), 0.0) or 0.0 for row in group)
        after = sum(_as_float(row.get("coarse_edges_after_dedup"), 0.0) or 0.0 for row in group)
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "duplicate_collapse_ratio_mean": _fmt(mean(values)) if values else "",
                "duplicate_collapse_ratio_std": _std(values) if values else "",
                "duplicate_collapse_ratio_from_edge_sums": _fmt(1.0 - after / before) if before > 0 else "",
                "coarse_edges_before_dedup_sum": _fmt(before),
                "coarse_edges_after_dedup_sum": _fmt(after),
                "valid_n": len(values),
                "missing_n": len(group) - len(values),
            }
        )
    return out


def _final_level_dir(run_dir: Path) -> Path | None:
    levels = []
    if not run_dir.exists():
        return None
    for child in run_dir.glob("level_*"):
        if not child.is_dir():
            continue
        try:
            level = int(child.name.removeprefix("level_"))
        except ValueError:
            continue
        if (child / "schema.json").exists():
            levels.append((level, child))
    return max(levels, default=(0, None))[1]


def _self_loop_rows(collapse_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    per_run: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    graph_cache: dict[str, Any | None] = {}
    for row in collapse_rows:
        key = (
            str(row.get("method", "")),
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("relation_id", "")),
        )
        run_dir = str(row.get("run_dir", "") or "")
        relation_id = str(row.get("relation_id", ""))
        status = "not_available"
        share = ""
        count = ""
        denom = _as_float(row.get("coarse_edges_after_dedup"), None)
        if run_dir:
            if run_dir not in graph_cache:
                level_dir = _final_level_dir(Path(run_dir))
                try:
                    graph_cache[run_dir] = load_graph(level_dir) if level_dir is not None else None
                except Exception:
                    graph_cache[run_dir] = None
            graph = graph_cache[run_dir]
            if graph is not None:
                try:
                    rel = graph.relations[int(relation_id)]
                    self_loops = int((rel.src == rel.dst).sum())
                    edge_count = int(rel.num_edges)
                    count = self_loops
                    share = _fmt(self_loops / max(edge_count, 1))
                    status = "computed_from_final_coarse_graph"
                except Exception:
                    status = "relation_not_found"
        per_run[key] = {
            "method": row.get("method", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "relation_id": relation_id,
            "self_loop_count": count,
            "coarse_edges_after_dedup": int(denom) if denom is not None else "",
            "self_loop_share": share,
            "status": status,
        }
    return list(per_run.values())


def _group_by_relation(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    output_metric: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("method", "")),
                str(row.get("dataset", "")),
                str(row.get("relation_id", "")),
            )
        ].append(row)
    out = []
    for (method, dataset, relation_id), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]) if item[0][2].isdigit() else item[0][2],
        ),
    ):
        values = [value for value in (_as_float(row.get(metric), None) for row in group) if value is not None]
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "relation_id": relation_id,
                f"{output_metric}_mean": _fmt(mean(values)) if values else "",
                f"{output_metric}_std": _std(values) if values else "",
                "valid_n": len(values),
                "missing_n": len(group) - len(values),
            }
        )
    return out


def _checkpoint_curve_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("method", "")),
                str(row.get("dataset", "")),
                str(row.get("checkpoint", "")),
                str(row.get("checkpoint_index", "")),
            )
        ].append(row)
    out = []
    for (method, dataset, checkpoint, checkpoint_index), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "method": method,
            "dataset": dataset,
            "checkpoint": checkpoint,
            "checkpoint_index": checkpoint_index,
            "seed_count": len({str(row.get("seed", "")) for row in group}),
        }
        for metric in ("macro_f1", "delta_vs_projected", "delta_vs_best"):
            values = [value for value in (_as_float(row.get(metric), None) for row in group) if value is not None]
            item[f"{metric}_mean"] = _fmt(mean(values)) if values else ""
            item[f"{metric}_std"] = _std(values) if values else ""
        out.append(item)
    return out


def _task_aggregate_rows(next8_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in next8_rows:
        method = str(row.get("method", ""))
        if method in METHODS:
            groups[(method, str(row.get("dataset", "")))].append(row)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for key, group in groups.items():
        item = {}
        for metric in ("DEE", "projected_macro_f1", "refined_macro_f1@5", "best_macro_f1"):
            source_keys = (metric, metric.lower()) if metric == "DEE" else (metric,)
            values = [
                value
                for value in (
                    _as_float(_first(row, source_keys), None)
                    for row in group
                )
                if value is not None
            ]
            item[f"{metric}_mean"] = _fmt(mean(values)) if values else ""
        out[key] = item
    return out


def _paper_rebuttal_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    drift_js_rows: Sequence[Mapping[str, Any]],
    task_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    js_lookup = {
        (str(row.get("method", "")), str(row.get("dataset", ""))): row
        for row in drift_js_rows
    }
    out = []
    for row in summary_rows:
        key = (str(row.get("method", "")), str(row.get("dataset", "")))
        task = task_rows.get(key, {})
        js = js_lookup.get(key, {})
        out.append(
            {
                "method": key[0],
                "dataset": key[1],
                "DEE_mean": task.get("DEE_mean", ""),
                "relation_mass_l1_drift_mean": row.get("relation_mass_l1_drift_mean", ""),
                "relation_mass_js_drift_mean": js.get("relation_mass_js_drift_mean", ""),
                "relation_energy_error_mean": row.get("relation_energy_error_mean", ""),
                "duplicate_collapse_ratio_mean": row.get("duplicate_collapse_ratio_mean", ""),
                "projected_macro_f1_mean": task.get("projected_macro_f1_mean", ""),
                "refined_macro_f1@5_mean": task.get("refined_macro_f1@5_mean", ""),
                "best_macro_f1_mean": row.get("best_macro_f1_mean", task.get("best_macro_f1_mean", "")),
            }
        )
    return out


def _checkpoint_rows(next8_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in next8_rows:
        method = str(row.get("method", ""))
        if method not in METHODS:
            continue
        projected = _as_float(row.get("projected_macro_f1"), None)
        best = _as_float(row.get("best_macro_f1"), None)
        for checkpoint, key in CHECKPOINTS:
            out.append(
                {
                    "method": method,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "checkpoint": checkpoint,
                    "checkpoint_index": len(out) % len(CHECKPOINTS),
                    "macro_f1": _fmt(row.get(key)),
                    "delta_vs_projected": _fmt((_as_float(row.get(key), 0.0) or 0.0) - projected)
                    if projected is not None
                    else "",
                    "delta_vs_best": _fmt((_as_float(row.get(key), 0.0) or 0.0) - best)
                    if best is not None
                    else "",
                }
            )
    return out


def _metapath_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "method": row.get("method", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "metapath_name": "bounded_relation_pair_sample",
                "sampled_pair_count": "",
                "original_connectivity_score": "",
                "coarse_projected_connectivity_score": "",
                "relative_error": "",
                "status": "not_available_in_legacy_run_summary",
                "run_dir": row.get("run_dir", ""),
            }
        )
    return out


def _summary_rows(
    energy_rows: Sequence[Mapping[str, Any]],
    drift_rows: Sequence[Mapping[str, Any]],
    collapse_rows: Sequence[Mapping[str, Any]],
    next8_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in energy_rows:
        value = _as_float(row.get("relation_energy_error"), None)
        if value is not None:
            grouped[(str(row.get("method", "")), str(row.get("dataset", "")))]["relation_energy_error"].append(value)
    for row in drift_rows:
        value = _as_float(row.get("relation_mass_l1_drift"), None)
        if value is not None:
            grouped[(str(row.get("method", "")), str(row.get("dataset", "")))]["relation_mass_l1_drift"].append(value)
    for row in collapse_rows:
        value = _as_float(row.get("duplicate_collapse_ratio"), None)
        if value is not None:
            grouped[(str(row.get("method", "")), str(row.get("dataset", "")))]["duplicate_collapse_ratio"].append(value)
    task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in next8_rows:
        method = str(row.get("method", ""))
        if method in METHODS:
            value = _as_float(row.get("best_macro_f1"), None)
            if value is not None:
                task[(method, str(row.get("dataset", "")))].append(value)
    out = []
    for key, values in sorted(grouped.items()):
        method, dataset = key
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "relation_energy_error_mean": _fmt(mean(values["relation_energy_error"]))
                if values["relation_energy_error"]
                else "",
                "relation_mass_l1_drift_mean": _fmt(mean(values["relation_mass_l1_drift"]))
                if values["relation_mass_l1_drift"]
                else "",
                "duplicate_collapse_ratio_mean": _fmt(mean(values["duplicate_collapse_ratio"]))
                if values["duplicate_collapse_ratio"]
                else "",
                "best_macro_f1_mean": _fmt(mean(task.get(key, []))) if task.get(key) else "",
            }
        )
    return out


def summarize_next9_hgb_rebuttal(
    *,
    next8_summary_dir: str | Path,
    run_summary_dirs: Sequence[str | Path],
    output: str | Path,
    command_lines: Sequence[str] = (),
) -> dict[str, Any]:
    next8_rows = _read_csv(Path(next8_summary_dir) / "per_seed_table.csv")
    run_rows = _run_rows([Path(path) for path in run_summary_dirs])
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    energy_rows = _relation_energy_rows(run_rows)
    drift_rows = _relation_distribution_rows(run_rows)
    collapse_rows = _edge_collapse_rows(run_rows)
    metapath_rows = _metapath_rows(run_rows)
    checkpoint_rows = _checkpoint_rows(next8_rows)
    summary_rows = _summary_rows(energy_rows, drift_rows, collapse_rows, next8_rows)
    relation_mass_by_dataset = _aggregate_metric_by_dataset(
        drift_rows,
        "relation_mass_l1_drift",
        "relation_mass_l1_drift",
    )
    relation_js_by_dataset = _aggregate_metric_by_dataset(
        drift_rows,
        "relation_mass_js_drift",
        "relation_mass_js_drift",
    )
    relation_energy_by_dataset = _aggregate_metric_by_dataset(
        energy_rows,
        "relation_energy_error",
        "relation_energy_error",
    )
    collapse_by_dataset = _collapse_by_dataset(collapse_rows)
    self_loop_rows = _self_loop_rows(collapse_rows)
    self_loop_by_relation = _group_by_relation(
        self_loop_rows,
        "self_loop_share",
        "self_loop_share",
    )
    duplicate_by_relation = _group_by_relation(
        collapse_rows,
        "duplicate_collapse_ratio",
        "duplicate_collapse_ratio",
    )
    checkpoint_curve = _checkpoint_curve_rows(checkpoint_rows)
    paper_rows = _paper_rebuttal_rows(
        summary_rows,
        relation_js_by_dataset,
        _task_aggregate_rows(next8_rows),
    )

    write_csv(output / "relation_energy_error_by_relation.csv", energy_rows)
    write_csv(output / "relation_distribution_drift.csv", drift_rows)
    write_csv(output / "coarse_edge_collapse_by_relation.csv", collapse_rows)
    write_csv(output / "metapath_connectivity_sampled.csv", metapath_rows)
    write_csv(output / "checkpoint_refine_masking.csv", checkpoint_rows)
    write_csv(output / "flatten_h6_rebuttal_summary.csv", summary_rows)
    write_csv(output / "relation_mass_drift_by_dataset.csv", relation_mass_by_dataset)
    write_csv(output / "relation_js_drift_by_dataset.csv", relation_js_by_dataset)
    write_csv(output / "relation_energy_error_by_dataset.csv", relation_energy_by_dataset)
    write_csv(output / "coarse_edge_collapse_by_dataset.csv", collapse_by_dataset)
    write_csv(output / "self_loop_share_by_relation.csv", self_loop_by_relation)
    write_csv(output / "duplicate_collapse_ratio_by_relation.csv", duplicate_by_relation)
    write_csv(output / "checkpoint_refine_masking_curve.csv", checkpoint_curve)
    write_csv(output / "paper_rebuttal_table.csv", paper_rows)

    _plot_scatter(energy_rows, "relation_id", "relation_energy_error", output / "figures" / "relation_energy_error_heatmap.png")
    _plot_scatter(drift_rows, "relation_id", "relation_mass_l1_drift", output / "figures" / "relation_mass_drift_by_method.png")
    _plot_scatter(collapse_rows, "relation_id", "duplicate_collapse_ratio", output / "figures" / "coarse_edge_collapse_by_method.png")
    _plot_scatter(checkpoint_rows, "checkpoint_index", "macro_f1", output / "figures" / "refine_curve_flatten_h6_vs_ps.png")

    summary = [
        "# Next9 HGB flatten-sum / H6 Rebuttal Summary",
        "",
        "This summary compares relation-sensitive diagnostics for HeSF-LVC-P/S, flatten-sum, H6-no-spec, and H0.",
        "Legacy run summaries do not contain bounded metapath samples; the metapath CSV records that limitation explicitly.",
        "",
        markdown_table(
            paper_rows[:20],
            [
                "method",
                "dataset",
                "DEE_mean",
                "relation_mass_l1_drift_mean",
                "relation_mass_js_drift_mean",
                "relation_energy_error_mean",
                "duplicate_collapse_ratio_mean",
                "projected_macro_f1_mean",
                "refined_macro_f1@5_mean",
                "best_macro_f1_mean",
            ],
        ),
        "",
        "Interpretation: use relation energy, distribution drift, and collapse ratios to explain structural damage even when task F1 is competitive.",
        "If flatten-sum or H6 ties/wins task on a dataset, the task win is preserved in `best_macro_f1_mean` rather than hidden.",
        "Self-loop shares are computed from final coarse graph endpoints when the referenced run directories are available; otherwise rows are marked with missing counts.",
    ]
    if command_lines:
        summary.extend(["", "## Commands", *[f"- `{line}`" for line in command_lines]])
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "relation_energy_rows": energy_rows,
        "relation_distribution_rows": drift_rows,
        "coarse_edge_collapse_rows": collapse_rows,
        "checkpoint_rows": checkpoint_rows,
        "summary_rows": summary_rows,
        "paper_rows": paper_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next8-summary-dir", "--input", required=True)
    parser.add_argument("--run-summary-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_hgb_rebuttal(
        next8_summary_dir=args.next8_summary_dir,
        run_summary_dirs=args.run_summary_dirs,
        output=args.output,
        command_lines=args.command_lines,
    )


if __name__ == "__main__":
    main()
