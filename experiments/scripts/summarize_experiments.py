from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import (
    diagnostics_row,
    discover_run_dirs,
    disk_usage_bytes,
    flatten_mapping,
    markdown_table,
    read_json,
    write_csv,
)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _target_ratio_from_name(run_name: str) -> float | None:
    match = re.search(r"_r([0-9]+(?:p[0-9]+)?)", run_name)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def _fmt_metric(value: Any, digits: int = 4) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def _runtime_total(level_rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in level_rows:
        for key in (
            "sketch",
            "candidates",
            "scoring",
            "matching_and_aggregation",
            "spectral_diagnostics",
        ):
            total += float(row.get(f"runtime_by_stage.{key}", 0) or 0)
    return float(total)


def _runtime_stage_total(level_rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(f"runtime_by_stage.{key}", 0) or 0) for row in level_rows))


def _sum_float_key(level_rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0) or 0) for row in level_rows))


def _baseline_runtime_total(row: Mapping[str, Any]) -> float | str:
    total = 0.0
    found = False
    for key, value in row.items():
        if key.startswith("cumulative_spectral.baseline_comparison.") and key.endswith(".runtime_total"):
            number = _as_float(value, None)
            if number is not None:
                total += float(number)
                found = True
    return float(total) if found else ""


def _candidate_pairs_total(level_rows: list[dict[str, Any]]) -> int:
    return int(sum(int(float(row.get("candidate_count_total", 0) or 0)) for row in level_rows))


def _edge_count_total(level_rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in level_rows:
        for key, value in row.items():
            if key.startswith("original_edge_count_by_relation."):
                total += int(float(value or 0))
    return int(total)


def _peak_rss_gb(level_rows: list[dict[str, Any]], metadata: Mapping[str, Any]) -> float | str:
    metadata_peak = _as_float(metadata.get("peak_rss_gb"), None)
    peaks = []
    if metadata_peak is not None:
        peaks.append(metadata_peak)
    for row in level_rows:
        rss_bytes = _as_float(row.get("large_graph_envelope.process_rss_bytes"), None)
        if rss_bytes is not None:
            peaks.append(float(rss_bytes) / (1024.0**3))
    return "" if not peaks else float(max(peaks))


def _peak_vram_gb(
    level_rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    *,
    kind: str,
) -> float | str:
    metadata_key = f"peak_vram_{kind}_gb"
    metadata_peak = _as_float(metadata.get(metadata_key), None)
    peaks = []
    if metadata_peak is not None:
        peaks.append(metadata_peak)
    byte_key = f"large_graph_envelope.cuda_memory.peak_{kind}_bytes"
    for row in level_rows:
        raw_gb = _as_float(row.get(metadata_key), None)
        if raw_gb is not None:
            peaks.append(raw_gb)
        raw_bytes = _as_float(row.get(byte_key), None)
        if raw_bytes is not None:
            peaks.append(float(raw_bytes) / (1024.0**3))
    return "" if not peaks else float(max(peaks))


def _count_computed_baselines(row: Mapping[str, Any], prefix: str = "spectral") -> int:
    needle = f"{prefix}.baseline_comparison."
    count = 0
    for key, value in row.items():
        if key.startswith(needle) and key.endswith(".status") and str(value) == "computed":
            count += 1
    return count


def _task_value(row: Mapping[str, Any], name: str) -> Any:
    return _first(row, (f"task.{name}", name), "")


def _with_task_aliases(row: dict[str, Any]) -> None:
    projected_micro = _task_value(row, "projected_original_micro_f1")
    projected_macro = _task_value(row, "projected_original_macro_f1")
    refined_micro = _task_value(row, "refined_original_micro_f1")
    refined_macro = _task_value(row, "refined_original_macro_f1")
    coarse_micro = _task_value(row, "coarse_train_micro_f1")
    coarse_macro = _task_value(row, "coarse_train_macro_f1")
    primary_name = _task_value(row, "primary_task_metric_name") or (
        "refined_original_macro_f1" if refined_macro != "" else "projected_original_macro_f1"
    )
    primary_metric = _task_value(row, "primary_task_metric")
    if primary_metric == "":
        primary_metric = refined_macro if primary_name == "refined_original_macro_f1" else projected_macro
    primary_macro = refined_macro if primary_name == "refined_original_macro_f1" else primary_metric
    checkpoint_aliases: dict[str, Any] = {}
    for epoch in (0, 1, 3, 5):
        checkpoint_aliases[f"task_refined_micro_f1@{epoch}"] = _task_value(
            row,
            f"refined_original_micro_f1@{epoch}",
        )
        checkpoint_aliases[f"task_refined_macro_f1@{epoch}"] = _task_value(
            row,
            f"refined_original_macro_f1@{epoch}",
        )
        checkpoint_aliases[f"task_refine_time@{epoch}"] = _task_value(
            row,
            f"refine_time@{epoch}",
        )
    row.update(
        {
            "task_projected_micro_f1": projected_micro,
            "task_projected_macro_f1": projected_macro,
            "task_refined_micro_f1": refined_micro,
            "task_refined_macro_f1": refined_macro,
            **checkpoint_aliases,
            "task_best_refined_macro_f1": _task_value(row, "best_refined_macro_f1"),
            "task_best_refined_epoch": _task_value(row, "best_refined_epoch"),
            "task_refine_auc_macro_f1": _task_value(row, "refine_auc_macro_f1"),
            "task_refine_time_by_epoch": _task_value(row, "refine_time_by_epoch"),
            "task_full_graph_macro_f1": (
                _task_value(row, "full_graph_rgcn_lite_tuned_macro_f1")
                or _task_value(row, "full_graph_rgcn_lite_default_macro_f1")
                or _task_value(row, "full_graph_rgcn_lite_macro_f1")
            ),
            "task_full_graph_micro_f1": (
                _task_value(row, "full_graph_rgcn_lite_tuned_micro_f1")
                or _task_value(row, "full_graph_rgcn_lite_default_micro_f1")
                or _task_value(row, "full_graph_rgcn_lite_micro_f1")
            ),
            "task_full_graph_rgcn_lite_default_macro_f1": (
                _task_value(row, "full_graph_rgcn_lite_default_macro_f1")
                or _task_value(row, "full_graph_rgcn_lite_macro_f1")
            ),
            "task_full_graph_rgcn_lite_tuned_macro_f1": _task_value(
                row,
                "full_graph_rgcn_lite_tuned_macro_f1",
            ),
            "task_full_graph_han_small_macro_f1": _task_value(row, "full_graph_han_small_macro_f1"),
            "task_full_graph_hgt_small_macro_f1": _task_value(row, "full_graph_hgt_small_macro_f1"),
            "task_coarse_train_micro_f1": coarse_micro,
            "task_coarse_train_macro_f1": coarse_macro,
            "task_primary_metric_name": primary_name,
            "task_primary_metric": primary_metric,
            "task_primary_macro_f1": primary_macro,
            "task_micro_f1": refined_micro or projected_micro or row.get("task_micro_f1", ""),
            "task_macro_f1": primary_macro or projected_macro or row.get("task_macro_f1", ""),
            "task_target_node_type": _task_value(row, "target_node_type"),
            "task_target_node_type_id": _task_value(row, "target_node_type_id"),
            "task_num_labeled_nodes_train": _task_value(row, "num_labeled_nodes_train"),
            "task_num_labeled_nodes_val": _task_value(row, "num_labeled_nodes_val"),
            "task_num_labeled_nodes_test": _task_value(row, "num_labeled_nodes_test"),
            "task_num_classes_present_train": _task_value(row, "num_classes_present_train"),
            "task_num_classes_present_val": _task_value(row, "num_classes_present_val"),
            "task_num_classes_present_test": _task_value(row, "num_classes_present_test"),
            "task_macro_f1_empty_class_policy": _task_value(row, "macro_f1_empty_class_policy"),
            "task_official_split_consistency": _task_value(row, "official_split_consistency"),
            "task_coarse_train_label_source": _task_value(row, "coarse_train_label_source"),
            "compute_device": _task_value(row, "device") or row.get("compute_device", "cpu"),
        }
    )


def _score_share_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    parts: list[str] = []
    for term in ("spec", "rel", "feat", "conv", "boundary"):
        value = row.get(f"score_contribution_share.{term}", "")
        aliases[f"score_contribution_share_{term}"] = value
        if value is not None and value != "":
            parts.append(f"{term}={_fmt_metric(value, 3)}")
    aliases["score_contribution_share"] = ", ".join(parts)
    return aliases


def _selected_source_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    matched_total = _as_float(row.get("matched_pairs"), 0.0) or 0.0

    def count_for(source: str) -> float:
        return _as_float(
            _first(
                row,
                (
                    f"selected_match_source_distribution_after_quota.{source}",
                    f"selected_merges_by_source.{source}",
                    f"matched_pairs_by_source.{source}",
                ),
                0.0,
            ),
            0.0,
        ) or 0.0

    bucket_count = count_for("bucket")
    twohop_count = count_for("twohop") + count_for("capped_twohop")
    fallback_count = count_for("fallback")
    return {
        "selected_bucket_fraction": "" if matched_total <= 0.0 else float(bucket_count / matched_total),
        "selected_twohop_fraction": "" if matched_total <= 0.0 else float(twohop_count / matched_total),
        "selected_fallback_fraction": "" if matched_total <= 0.0 else float(fallback_count / matched_total),
        "quota_violation_bucket": row.get("quota_violation.bucket", ""),
        "quota_violation_twohop": row.get("quota_violation.twohop", ""),
        "quota_violation_fallback": row.get("quota_violation.fallback", ""),
    }


def _baseline_method_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    prefix = "cumulative_spectral.baseline_comparison."
    methods = sorted(
        {
            key.removeprefix(prefix).split(".", 1)[0]
            for key in row
            if key.startswith(prefix) and "." in key.removeprefix(prefix)
        }
    )
    computed_task_baselines: list[dict[str, Any]] = []
    for method in methods:
        safe = method.replace("-", "_")
        base = f"{prefix}{method}."
        aliases[f"baseline_{safe}_final_cumulative_ratio"] = _first(
            row,
            (
                f"{base}final_cumulative_ratio",
                f"{base}cumulative_ratio",
                f"{base}compression_ratio",
            ),
            "",
        )
        aliases[f"baseline_{safe}_cumulative_dee"] = _first(
            row,
            (
                f"{base}sketch_dirichlet_energy_relative_error",
                f"{base}dirichlet_energy_relative_error",
            ),
            "",
        )
        aliases[f"baseline_{safe}_cumulative_fwe_weighted"] = row.get(
            f"{base}relation_weighted_fused_energy_relative_error",
            "",
        )
        aliases[f"baseline_{safe}_cumulative_fse_unweighted"] = _first(
            row,
            (
                f"{base}fused_sketch_energy_relative_error",
                f"{base}chebheat_sketch_inner_product_relative_error",
            ),
            "",
        )
        aliases[f"baseline_{safe}_cumulative_ree_max"] = row.get(
            f"{base}relation_energy_relative_error_max",
            "",
        )
        aliases[f"baseline_{safe}_cumulative_sipe"] = _first(
            row,
            (
                f"{base}chebheat_sketch_inner_product_relative_error",
                f"{base}sketch_inner_product_relative_error",
            ),
            "",
        )
        aliases[f"baseline_{safe}_cumulative_sampled_eigen_error"] = _first(
            row,
            (
                f"{base}exact_eigenvalue_sanity.relative_error",
                f"{base}sampled_eigen_error",
            ),
            "",
        )
        projected_macro = _first(
            row,
            (
                f"{base}task_projected_macro_f1",
                f"{base}task.projected_original_macro_f1",
            ),
            "",
        )
        refined_macro = _first(
            row,
            (
                f"{base}task_refined_macro_f1",
                f"{base}task.refined_original_macro_f1",
            ),
            "",
        )
        train_time = _first(row, (f"{base}task_train_time", f"{base}task.train_time"), "")
        refine_time = _first(row, (f"{base}task_refine_time", f"{base}task.refine_time"), "")
        total_time = _first(row, (f"{base}task_total_time", f"{base}task.total_time"), "")
        aliases[f"baseline_{safe}_task_projected_macro_f1"] = projected_macro
        aliases[f"baseline_{safe}_task_refined_macro_f1"] = refined_macro
        for epoch in (0, 1, 3, 5):
            aliases[f"baseline_{safe}_refined_macro_f1@{epoch}"] = _first(
                row,
                (
                    f"{base}task_refined_macro_f1@{epoch}",
                    f"{base}task_refined_original_macro_f1@{epoch}",
                    f"{base}task.refined_original_macro_f1@{epoch}",
                ),
                "",
            )
        aliases[f"baseline_{safe}_task_best_refined_macro_f1"] = _first(
            row,
            (
                f"{base}task_best_refined_macro_f1",
                f"{base}task.best_refined_macro_f1",
            ),
            "",
        )
        aliases[f"baseline_{safe}_task_best_refined_epoch"] = _first(
            row,
            (
                f"{base}task_best_refined_epoch",
                f"{base}task.best_refined_epoch",
            ),
            "",
        )
        aliases[f"baseline_{safe}_task_refine_auc_macro_f1"] = _first(
            row,
            (
                f"{base}task_refine_auc_macro_f1",
                f"{base}task.refine_auc_macro_f1",
            ),
            "",
        )
        aliases[f"baseline_{safe}_task_train_time"] = train_time
        aliases[f"baseline_{safe}_task_refine_time"] = refine_time
        aliases[f"baseline_{safe}_task_total_time"] = total_time
        if str(row.get(f"{base}status", "")) == "computed" and (
            projected_macro != "" or refined_macro != ""
        ):
            computed_task_baselines.append(
                {
                    "projected": projected_macro,
                    "refined": refined_macro,
                    "train_time": train_time,
                    "refine_time": refine_time,
                    "total_time": total_time,
                }
            )
        aliases[f"baseline_{safe}_runtime_total"] = _first(
            row,
            (
                f"{base}runtime_total",
                f"{base}runtime_total_run",
            ),
            "",
        )
        aliases[f"baseline_{safe}_target_ratio"] = row.get(f"{base}target_ratio", "")
        aliases[f"baseline_{safe}_target_tolerance"] = row.get(f"{base}target_tolerance", "")
        aliases[f"baseline_{safe}_target_abs_error"] = row.get(f"{base}target_abs_error", "")
        aliases[f"baseline_{safe}_target_hit"] = row.get(f"{base}target_hit", "")
        aliases[f"baseline_{safe}_levels"] = row.get(f"{base}levels", "")
        aliases[f"baseline_{safe}_stopped_by"] = row.get(f"{base}stopped_by", "")
    selected: Mapping[str, Any] | None = None
    if computed_task_baselines:
        with_refined = [
            item for item in computed_task_baselines if _as_float(item.get("refined"), None) is not None
        ]
        selected = max(
            with_refined,
            key=lambda item: float(_as_float(item.get("refined"), 0.0) or 0.0),
        ) if with_refined else computed_task_baselines[0]
    aliases["baseline_projected_macro_f1"] = selected.get("projected", "") if selected else ""
    aliases["baseline_refined_macro_f1"] = selected.get("refined", "") if selected else ""
    aliases["baseline_train_time"] = selected.get("train_time", "") if selected else ""
    aliases["baseline_refine_time"] = selected.get("refine_time", "") if selected else ""
    aliases["baseline_total_time"] = selected.get("total_time", "") if selected else ""
    return aliases


def _quality_row(base: Mapping[str, Any], row: Mapping[str, Any], *, row_type: str) -> dict[str, Any]:
    return {
        "run_name": base["run_name"],
        "dataset": base.get("dataset", ""),
        "variant": base.get("variant", row.get("variant", "")),
        "row_type": row_type,
        "level": row.get("level", ""),
        "compression_ratio": row.get("compression_ratio", ""),
        "matched_pairs": row.get("matched_pairs", ""),
        "singleton_ratio": row.get("singleton_ratio", ""),
        "candidate_count_mean": row.get("candidate_count_mean", ""),
        "spectral_sketch_dirichlet_energy_relative_error": row.get(
            "spectral.sketch_dirichlet_energy_relative_error",
            "",
        ),
        "spectral_relation_weighted_fused_energy_relative_error": row.get(
            "spectral.relation_weighted_fused_energy_relative_error",
            "",
        ),
        "spectral_fused_sketch_energy_relative_error": row.get(
            "spectral.fused_sketch_energy_relative_error",
            "",
        ),
        "spectral_relation_energy_relative_error_max": row.get(
            "spectral.relation_energy_relative_error_max",
            "",
        ),
        "spectral_chebheat_sketch_inner_product_relative_error": row.get(
            "spectral.chebheat_sketch_inner_product_relative_error",
            "",
        ),
        "final_cumulative_ratio": row.get("final_cumulative_ratio", ""),
        "target_abs_error": row.get("target_abs_error", ""),
        "target_hit": row.get("target_hit", ""),
        "final_DEE": row.get("final_DEE", ""),
        "final_FWE_weighted": row.get("final_FWE_weighted", ""),
        "final_FSE_unweighted": row.get("final_FSE_unweighted", ""),
        "final_REE_max": row.get("final_REE_max", ""),
        "final_SIPE": row.get("final_SIPE", ""),
        "task_projected_macro_f1": row.get("task_projected_macro_f1", row.get("task.projected_original_macro_f1", "")),
        "task_refined_macro_f1": row.get("task_refined_macro_f1", row.get("task.refined_original_macro_f1", "")),
        "task_refined_macro_f1@0": row.get("task_refined_macro_f1@0", ""),
        "task_refined_macro_f1@1": row.get("task_refined_macro_f1@1", ""),
        "task_refined_macro_f1@3": row.get("task_refined_macro_f1@3", ""),
        "task_refined_macro_f1@5": row.get("task_refined_macro_f1@5", ""),
        "task_best_refined_macro_f1": row.get("task_best_refined_macro_f1", ""),
        "task_primary_macro_f1": row.get("task_primary_macro_f1", ""),
        "task_micro_f1": row.get("task_micro_f1", row.get("task.micro_f1", "")),
        "task_macro_f1": row.get("task_macro_f1", row.get("task.macro_f1", "")),
        "runtime_total_run": row.get("runtime_total_run", ""),
        "peak_rss_gb": row.get("peak_rss_gb", ""),
        "score_contribution_share_spec": row.get("score_contribution_share_spec", ""),
        "score_contribution_share_rel": row.get("score_contribution_share_rel", ""),
        "score_contribution_share_feat": row.get("score_contribution_share_feat", ""),
        "score_contribution_share_conv": row.get("score_contribution_share_conv", ""),
        "score_contribution_share_boundary": row.get("score_contribution_share_boundary", ""),
    }


def _final_cumulative_row(
    base: Mapping[str, Any],
    level_rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(level_rows, key=lambda row: _as_int(row.get("level"), 0) or 0)
    first = ordered[0]
    last = ordered[-1]
    initial_nodes = _as_int(
        _first(
            first,
            (
                "target_control.original_nodes",
                "original_nodes",
            ),
        ),
        0,
    ) or 0
    final_nodes = _as_int(last.get("coarse_nodes"), 0) or 0
    final_ratio = float(final_nodes / max(initial_nodes, 1))
    run_name = str(base["run_name"])
    target_ratio = _as_float(
        _first(
            last,
            (
                "config.coarsening.target_ratio",
                "target_control.target_ratio",
                "target_ratio",
            ),
            "",
        ),
        None,
    )
    if target_ratio is None:
        target_ratio = _target_ratio_from_name(run_name)
    target_ratio = float(target_ratio if target_ratio is not None else final_ratio)
    target_abs_error = abs(final_ratio - target_ratio)
    hit_tolerance = _as_float(
        _first(last, ("config.coarsening.target_hit_tolerance", "target_hit_tolerance"), ""),
        0.05,
    )
    target_hit = bool(target_abs_error <= float(hit_tolerance or 0.05))

    def level_error(row: Mapping[str, Any]) -> float:
        coarse_nodes = _as_int(row.get("coarse_nodes"), final_nodes) or final_nodes
        return abs(float(coarse_nodes / max(initial_nodes, 1)) - target_ratio)

    best = min(ordered, key=level_error)
    max_levels = _as_int(
        _first(last, ("config.coarsening.max_levels", "target_control.max_levels"), ""),
        len(ordered),
    ) or len(ordered)
    stopped_by = "target_hit" if target_hit else "max_levels"
    if len(ordered) < max_levels and not target_hit:
        input_nodes = _as_int(last.get("original_nodes"), final_nodes) or final_nodes
        stopped_by = "no_decrease" if final_nodes >= input_nodes else "no_match"

    final_dee = last.get("spectral.sketch_dirichlet_energy_relative_error", "")
    final_fwe_weighted = last.get("spectral.relation_weighted_fused_energy_relative_error", "")
    final_fse_unweighted = last.get("spectral.fused_sketch_energy_relative_error", "")
    final_ree_max = last.get("spectral.relation_energy_relative_error_max", "")
    final_sipe = last.get("spectral.chebheat_sketch_inner_product_relative_error", "")
    cumulative_dee = _first(
        last,
        (
            "cumulative_spectral.sketch_dirichlet_energy_relative_error",
            "cumulative_spectral.dirichlet_energy_relative_error",
            "spectral.cumulative_sketch_dirichlet_energy_relative_error",
        ),
        final_dee,
    )
    cumulative_fwe_weighted = _first(
        last,
        (
            "cumulative_spectral.relation_weighted_fused_energy_relative_error",
            "spectral.cumulative_relation_weighted_fused_energy_relative_error",
        ),
        final_fwe_weighted,
    )
    cumulative_fse_unweighted = _first(
        last,
        (
            "cumulative_spectral.fused_sketch_energy_relative_error",
            "spectral.cumulative_fused_sketch_energy_relative_error",
        ),
        final_fse_unweighted,
    )
    cumulative_ree_max = _first(
        last,
        (
            "cumulative_spectral.relation_energy_relative_error_max",
            "spectral.cumulative_relation_energy_relative_error_max",
        ),
        final_ree_max,
    )
    cumulative_sipe = _first(
        last,
        (
            "cumulative_spectral.chebheat_sketch_inner_product_relative_error",
            "cumulative_spectral.sketch_inner_product_relative_error",
            "spectral.cumulative_chebheat_sketch_inner_product_relative_error",
        ),
        final_sipe,
    )
    final = {**base, **last}
    final.update(
        {
            "row_type": "final",
            "level": "final",
            "level_row_count": int(len(ordered)),
            "final_level": str(last.get("level", "")),
            "initial_nodes": int(initial_nodes),
            "final_nodes": int(final_nodes),
            "final_cumulative_ratio": final_ratio,
            "target_ratio": target_ratio,
            "target_abs_error": target_abs_error,
            "target_hit": "true" if target_hit else "false",
            "best_level": str(best.get("level", "")),
            "stopped_by": stopped_by,
            "final_DEE": final_dee,
            "final_FWE_weighted": final_fwe_weighted,
            "final_FSE_unweighted": final_fse_unweighted,
            "final_REE_max": final_ree_max,
            "final_SIPE": final_sipe,
            "cumulative_dee": cumulative_dee,
            "cumulative_fwe_weighted": cumulative_fwe_weighted,
            "cumulative_fse_unweighted": cumulative_fse_unweighted,
            "cumulative_ree_max": cumulative_ree_max,
            "cumulative_sipe": cumulative_sipe,
            "cumulative_sampled_eigen_error": _first(
                last,
                (
                    "cumulative_spectral.exact_eigenvalue_sanity.relative_error",
                    "cumulative_spectral.sampled_eigen_error",
                    "spectral.cumulative_sampled_eigen_error",
                ),
                "",
            ),
            "task_micro_f1": last.get("task.micro_f1", ""),
            "task_macro_f1": last.get("task.macro_f1", ""),
            "runtime_total_run": _runtime_total(ordered),
            "runtime_by_stage.sketch": _runtime_stage_total(ordered, "sketch"),
            "runtime_by_stage.candidates": _runtime_stage_total(ordered, "candidates"),
            "runtime_by_stage.scoring": _runtime_stage_total(ordered, "scoring"),
            "runtime_by_stage.matching": _runtime_stage_total(ordered, "matching"),
            "runtime_by_stage.aggregation": _runtime_stage_total(ordered, "aggregation"),
            "runtime_by_stage.matching_and_aggregation": _runtime_stage_total(
                ordered,
                "matching_and_aggregation",
            ),
            "runtime_by_stage.cumulative_diagnostics": _runtime_stage_total(
                ordered,
                "spectral_diagnostics",
            ),
            "runtime_by_stage.task_train": _first(last, ("task.train_time", "task_train_time"), ""),
            "runtime_by_stage.task_refine": _first(last, ("task.refine_time", "task_refine_time"), ""),
            "runtime_by_stage.baselines": _baseline_runtime_total(last),
            "peak_rss_gb": _peak_rss_gb(ordered, metadata),
            "peak_vram_allocated_gb": _peak_vram_gb(ordered, metadata, kind="allocated"),
            "peak_vram_reserved_gb": _peak_vram_gb(ordered, metadata, kind="reserved"),
        }
    )
    scoring_runtime = _as_float(final.get("runtime_by_stage.scoring"), 0.0) or 0.0
    aggregation_runtime = (
        _as_float(final.get("runtime_by_stage.aggregation"), 0.0)
        or _as_float(final.get("runtime_by_stage.matching_and_aggregation"), 0.0)
        or 0.0
    )
    candidate_generation_time = _sum_float_key(ordered, "candidate_generation_time")
    candidate_retained_pair_count = int(_sum_float_key(ordered, "candidate_retained_pair_count"))
    final["candidate_generation_time"] = candidate_generation_time
    final["candidate_retained_pair_count"] = candidate_retained_pair_count
    final["candidate_pairs_per_sec"] = (
        float(candidate_retained_pair_count / candidate_generation_time)
        if candidate_generation_time > 0.0
        else ""
    )
    for substage in (
        "onehop",
        "incident_index_build",
        "twohop_expansion",
        "simhash",
        "bucket_emit",
        "partition_ann",
        "fallback",
        "store_finalize",
    ):
        final[f"candidate_substage_times.{substage}"] = _sum_float_key(
            ordered,
            f"candidate_substage_times.{substage}",
        )
    final["bucket_coverage"] = _first(last, ("bucket_coverage", "candidate_source_coverage.bucket"), "")
    final["twohop_expansion_time"] = final.get("candidate_substage_times.twohop_expansion", "")
    final["partition_count"] = last.get("partition_imbalance.partition_count", "")
    final["partition_imbalance_max_to_mean"] = last.get("partition_imbalance.max_to_mean", "")
    final["candidate_buffer_bytes"] = last.get("memory_by_candidate_buffers.estimated_total_bytes", "")
    final["candidate_pairs_scored_per_sec"] = (
        float(_candidate_pairs_total(ordered) / scoring_runtime) if scoring_runtime > 0 else ""
    )
    final["edges_aggregated_per_sec"] = (
        float(_edge_count_total(ordered) / aggregation_runtime) if aggregation_runtime > 0 else ""
    )
    final["peak_vram_gb"] = final["peak_vram_allocated_gb"]
    final["peak_cpu_memory_gb"] = final["peak_rss_gb"]
    final["peak_gpu_memory_allocated_gb"] = final["peak_vram_allocated_gb"]
    final["peak_gpu_memory_reserved_gb"] = final["peak_vram_reserved_gb"]
    final["spectral_baseline_computed_count"] = _count_computed_baselines(last, "spectral")
    final["cumulative_spectral_baseline_computed_count"] = _count_computed_baselines(
        last,
        "cumulative_spectral",
    )
    final["spectral_exact_eigenvalue_sanity_status"] = last.get(
        "spectral.exact_eigenvalue_sanity.status",
        "",
    )
    final["spectral_exact_eigenvalue_sanity_mode"] = last.get(
        "spectral.exact_eigenvalue_sanity.mode",
        "",
    )
    final["spectral_exact_eigenvalue_sanity_relative_error"] = last.get(
        "spectral.exact_eigenvalue_sanity.relative_error",
        "",
    )
    peak_allocated = _as_float(final.get("peak_vram_allocated_gb"), 0.0) or 0.0
    peak_reserved = _as_float(final.get("peak_vram_reserved_gb"), 0.0) or 0.0
    cuda_available = any(
        str(row.get("large_graph_envelope.cuda_memory.available", "")).lower() == "true"
        for row in ordered
    ) or max(peak_allocated, peak_reserved) > 0.0
    task_device = _task_value(last, "device")
    if task_device:
        final["compute_device"] = str(task_device)
    elif max(peak_allocated, peak_reserved) > 0.0:
        final["compute_device"] = "cuda"
    else:
        final["compute_device"] = "cpu"
    final["cuda_available"] = "true" if cuda_available else "false"
    final["cpu_only"] = (
        "true" if final["compute_device"] == "cpu" and max(peak_allocated, peak_reserved) <= 0.0 else "false"
    )
    _with_task_aliases(final)
    final.update(_selected_source_aliases(last))
    final.update(_score_share_aliases(last))
    final.update(_baseline_method_aliases(last))
    return final


def _mean_numeric(rows: list[Mapping[str, Any]], key: str) -> float | str:
    values = [
        value
        for value in (_as_float(row.get(key), None) for row in rows)
        if value is not None
    ]
    return "" if not values else float(sum(values) / len(values))


def _core_report_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        variant = str(row.get("variant") or row.get("run_name") or "")
        groups.setdefault(variant, []).append(row)
    report_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(groups.items()):
        report_rows.append(
            {
                "variant": variant,
                "final ratio": _fmt_metric(_mean_numeric(rows, "final_cumulative_ratio")),
                "DEE \u2193": _fmt_metric(_mean_numeric(rows, "cumulative_dee")),
                "FSE-unweighted \u2193": _fmt_metric(_mean_numeric(rows, "cumulative_fse_unweighted")),
                "REE-max \u2193": _fmt_metric(_mean_numeric(rows, "cumulative_ree_max")),
                "SIPE \u2193": _fmt_metric(_mean_numeric(rows, "cumulative_sipe")),
                "macro-F1 \u2191": _fmt_metric(_mean_numeric(rows, "task_primary_macro_f1")),
                "runtime \u2193": _fmt_metric(_mean_numeric(rows, "runtime_total_run")),
                "peak RAM": _fmt_metric(_mean_numeric(rows, "peak_rss_gb")),
            }
        )
    return report_rows


def _run_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_block": row.get("experiment_block", ""),
        "unique_run_key": row.get("unique_run_key", row.get("run_name", "")),
        "run_name": row.get("run_name", ""),
        "dataset": row.get("dataset", ""),
        "variant": row.get("variant", ""),
        "target_ratio": row.get("target_ratio", ""),
        "final_cumulative_ratio": row.get("final_cumulative_ratio", ""),
    }


def _score_term_scale_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in final_rows:
        for term in ("spec", "rel", "feat", "conv", "boundary"):
            item = {
                **_run_identity(row),
                "term": term,
                "share": row.get(f"score_contribution_share_{term}", ""),
            }
            for prefix, label in (
                ("score_terms", "raw"),
                ("score_contributions", "weighted_normalized"),
            ):
                for stat in ("count", "mean", "p50", "p95", "p99"):
                    item[f"{label}_{stat}"] = row.get(f"{prefix}.{term}.{stat}", "")
            rows.append(item)
    return rows


def _candidate_source_pareto_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in final_rows:
        sources: set[str] = set()
        for key in row:
            if key.startswith("candidate_source_counts."):
                sources.add(key.removeprefix("candidate_source_counts."))
            if key.startswith("generated_candidates_by_source."):
                sources.add(key.removeprefix("generated_candidates_by_source."))
            if key.startswith("selected_merges_by_source."):
                sources.add(key.removeprefix("selected_merges_by_source."))
            if key.startswith("matched_pairs_by_source."):
                sources.add(key.removeprefix("matched_pairs_by_source."))
        candidate_total = _as_float(row.get("candidate_count_total"), None)
        matched_total = _as_float(row.get("matched_pairs"), None)
        selected_total = sum(
            _as_float(row.get(f"selected_merges_by_source.{source}"), 0.0) or 0.0
            for source in sources
        ) or matched_total
        for source in sorted(sources):
            candidate_count = (
                _as_float(row.get(f"generated_candidates_by_source.{source}"), None)
                if row.get(f"generated_candidates_by_source.{source}") not in {None, ""}
                else _as_float(row.get(f"candidate_source_counts.{source}"), 0.0)
            ) or 0.0
            selected_count = (
                _as_float(row.get(f"selected_merges_by_source.{source}"), None)
                if row.get(f"selected_merges_by_source.{source}") not in {None, ""}
                else _as_float(row.get(f"matched_pairs_by_source.{source}"), 0.0)
            ) or 0.0
            if candidate_count == 0.0 and selected_count == 0.0:
                continue
            rows.append(
                {
                    **_run_identity(row),
                    "source": source,
                    "candidate_count": candidate_count,
                    "candidate_fraction": ""
                    if not candidate_total
                    else float(candidate_count / candidate_total),
                    "selected_count": selected_count,
                    "selected_fraction": ""
                    if not selected_total
                    else float(selected_count / selected_total),
                    "retained_candidate_count": _as_float(
                        row.get(f"candidate_source_counts.{source}"),
                        0.0,
                    )
                    or 0.0,
                    "matched_pair_count": _as_float(
                        row.get(f"matched_pairs_by_source.{source}"),
                        0.0,
                    )
                    or 0.0,
                    "avg_score": row.get(f"selected_source_avg_score.{source}", ""),
                    "avg_delta_spec": row.get(f"selected_source_avg_delta_spec.{source}", ""),
                    "avg_delta_conv": row.get(f"selected_source_avg_delta_conv.{source}", ""),
                }
            )
    return rows


def _task_summary_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **_run_identity(row),
            "task_model": row.get("task.model", ""),
            "task_primary_metric_name": row.get("task_primary_metric_name", ""),
            "task_primary_metric": row.get("task_primary_metric", ""),
            "task_primary_macro_f1": row.get("task_primary_macro_f1", ""),
            "task_projected_micro_f1": row.get("task_projected_micro_f1", ""),
            "task_projected_macro_f1": row.get("task_projected_macro_f1", ""),
            "task_refined_micro_f1": row.get("task_refined_micro_f1", ""),
            "task_refined_macro_f1": row.get("task_refined_macro_f1", ""),
            "task_refined_macro_f1@0": row.get("task_refined_macro_f1@0", ""),
            "task_refined_macro_f1@1": row.get("task_refined_macro_f1@1", ""),
            "task_refined_macro_f1@3": row.get("task_refined_macro_f1@3", ""),
            "task_refined_macro_f1@5": row.get("task_refined_macro_f1@5", ""),
            "task_best_refined_macro_f1": row.get("task_best_refined_macro_f1", ""),
            "task_best_refined_epoch": row.get("task_best_refined_epoch", ""),
            "task_refine_auc_macro_f1": row.get("task_refine_auc_macro_f1", ""),
            "task_refine_time_by_epoch": row.get("task_refine_time_by_epoch", ""),
            "task_full_graph_micro_f1": row.get("task_full_graph_micro_f1", ""),
            "task_full_graph_macro_f1": row.get("task_full_graph_macro_f1", ""),
            "task_coarse_train_micro_f1": row.get("task_coarse_train_micro_f1", ""),
            "task_coarse_train_macro_f1": row.get("task_coarse_train_macro_f1", ""),
            "task_micro_f1": row.get("task_micro_f1", row.get("task.micro_f1", "")),
            "task_macro_f1": row.get("task_macro_f1", row.get("task.macro_f1", "")),
            "task_labeled_nodes": row.get("task.labeled_nodes", ""),
            "task_skipped": row.get("task.skipped", ""),
            "target_node_type": row.get("task_target_node_type", row.get("task.target_node_type", "")),
            "target_node_type_id": row.get("task_target_node_type_id", row.get("task.target_node_type_id", "")),
            "num_labeled_nodes_train": row.get("task_num_labeled_nodes_train", row.get("task.num_labeled_nodes_train", "")),
            "num_labeled_nodes_val": row.get("task_num_labeled_nodes_val", row.get("task.num_labeled_nodes_val", "")),
            "num_labeled_nodes_test": row.get("task_num_labeled_nodes_test", row.get("task.num_labeled_nodes_test", "")),
            "num_labeled_nodes_total": row.get("task_num_labeled_nodes_total", row.get("task.num_labeled_nodes_total", "")),
            "label_coverage_train": row.get("task_label_coverage_train", row.get("task.label_coverage_train", "")),
            "label_coverage_val": row.get("task_label_coverage_val", row.get("task.label_coverage_val", "")),
            "label_coverage_test": row.get("task_label_coverage_test", row.get("task.label_coverage_test", "")),
            "train_only_label_coverage": row.get(
                "task_train_only_label_coverage",
                row.get("task.train_only_label_coverage", ""),
            ),
            "task_split_policy": row.get("task_task_split_policy", row.get("task.task_split_policy", "")),
            "test_label_leakage_check": row.get(
                "task_test_label_leakage_check",
                row.get("task.test_label_leakage_check", ""),
            ),
            "num_classes": row.get("task_num_classes", row.get("task.num_classes", "")),
            "num_classes_present_train": row.get("task_num_classes_present_train", row.get("task.num_classes_present_train", "")),
            "num_classes_present_val": row.get("task_num_classes_present_val", row.get("task.num_classes_present_val", "")),
            "num_classes_present_test": row.get("task_num_classes_present_test", row.get("task.num_classes_present_test", "")),
            "macro_f1_empty_class_policy": row.get("task_macro_f1_empty_class_policy", row.get("task.macro_f1_empty_class_policy", "")),
            "official_split_consistency": row.get("task_official_split_consistency", row.get("task.official_split_consistency", "")),
            "coarse_train_label_source": row.get("task_coarse_train_label_source", row.get("task.coarse_train_label_source", "")),
            "coarse_train_micro_f1": row.get("task_coarse_train_micro_f1", row.get("task.coarse_train_micro_f1", "")),
            "coarse_train_macro_f1": row.get("task_coarse_train_macro_f1", row.get("task.coarse_train_macro_f1", "")),
            "projected_original_micro_f1": row.get("task_projected_micro_f1", row.get("task.projected_original_micro_f1", "")),
            "projected_original_macro_f1": row.get("task_projected_macro_f1", row.get("task.projected_original_macro_f1", "")),
            "refined_original_micro_f1": row.get("task_refined_micro_f1", row.get("task.refined_original_micro_f1", "")),
            "refined_original_macro_f1": row.get("task_refined_macro_f1", row.get("task.refined_original_macro_f1", "")),
            "best_refined_macro_f1": row.get("task_best_refined_macro_f1", ""),
            "best_refined_epoch": row.get("task_best_refined_epoch", ""),
            "refine_auc_macro_f1": row.get("task_refine_auc_macro_f1", ""),
            "train_time": row.get("task.train_time", ""),
            "refine_time": row.get("task.refine_time", ""),
            "total_time": row.get("task.total_time", ""),
        }
        for row in final_rows
    ]


def _run_resource_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **_run_identity(row),
            "runtime_total_run": row.get("runtime_total_run", ""),
            "runtime_by_stage.sketch": row.get("runtime_by_stage.sketch", ""),
            "runtime_by_stage.candidates": row.get("runtime_by_stage.candidates", ""),
            "runtime_by_stage.scoring": row.get("runtime_by_stage.scoring", ""),
            "runtime_by_stage.matching": row.get("runtime_by_stage.matching", ""),
            "runtime_by_stage.aggregation": row.get("runtime_by_stage.aggregation", ""),
            "runtime_by_stage.matching_and_aggregation": row.get(
                "runtime_by_stage.matching_and_aggregation",
                "",
            ),
            "runtime_by_stage.cumulative_diagnostics": row.get(
                "runtime_by_stage.cumulative_diagnostics",
                "",
            ),
            "runtime_by_stage.task_train": row.get("runtime_by_stage.task_train", ""),
            "runtime_by_stage.task_refine": row.get("runtime_by_stage.task_refine", ""),
            "runtime_by_stage.baselines": row.get("runtime_by_stage.baselines", ""),
            "candidate_generation_time": row.get("candidate_generation_time", ""),
            "candidate_pairs_per_sec": row.get("candidate_pairs_per_sec", ""),
            "candidate_substage_times.onehop": row.get("candidate_substage_times.onehop", ""),
            "candidate_substage_times.incident_index_build": row.get(
                "candidate_substage_times.incident_index_build",
                "",
            ),
            "candidate_substage_times.twohop_expansion": row.get(
                "candidate_substage_times.twohop_expansion",
                "",
            ),
            "candidate_substage_times.simhash": row.get("candidate_substage_times.simhash", ""),
            "candidate_substage_times.bucket_emit": row.get(
                "candidate_substage_times.bucket_emit",
                "",
            ),
            "candidate_substage_times.partition_ann": row.get(
                "candidate_substage_times.partition_ann",
                "",
            ),
            "candidate_substage_times.fallback": row.get("candidate_substage_times.fallback", ""),
            "candidate_substage_times.store_finalize": row.get(
                "candidate_substage_times.store_finalize",
                "",
            ),
            "bucket_coverage": row.get("bucket_coverage", ""),
            "twohop_expansion_time": row.get("twohop_expansion_time", ""),
            "partition_count": row.get("partition_count", ""),
            "partition_imbalance_max_to_mean": row.get("partition_imbalance_max_to_mean", ""),
            "candidate_buffer_bytes": row.get("candidate_buffer_bytes", ""),
            "candidate_pairs_scored_per_sec": row.get("candidate_pairs_scored_per_sec", ""),
            "edges_aggregated_per_sec": row.get("edges_aggregated_per_sec", ""),
            "peak_rss_gb": row.get("peak_rss_gb", ""),
            "peak_cpu_memory_gb": row.get("peak_cpu_memory_gb", row.get("peak_rss_gb", "")),
            "peak_vram_gb": row.get("peak_vram_gb", ""),
            "peak_vram_allocated_gb": row.get("peak_vram_allocated_gb", ""),
            "peak_vram_reserved_gb": row.get("peak_vram_reserved_gb", ""),
            "peak_gpu_memory_allocated_gb": row.get("peak_gpu_memory_allocated_gb", row.get("peak_vram_allocated_gb", "")),
            "peak_gpu_memory_reserved_gb": row.get("peak_gpu_memory_reserved_gb", row.get("peak_vram_reserved_gb", "")),
            "compute_device": row.get("compute_device", ""),
            "cuda_available": row.get("cuda_available", ""),
            "cpu_only": row.get("cpu_only", ""),
            "process_rss_bytes": row.get("large_graph_envelope.process_rss_bytes", ""),
            "peak_vram_allocated_bytes": row.get(
                "large_graph_envelope.cuda_memory.peak_allocated_bytes",
                "",
            ),
            "peak_vram_reserved_bytes": row.get(
                "large_graph_envelope.cuda_memory.peak_reserved_bytes",
                "",
            ),
            "artifact_bytes_total": row.get("large_graph_envelope.artifact_bytes_total", ""),
            "aggregation_shard_bytes": row.get(
                "large_graph_envelope.artifact_bytes_by_name.aggregation_shards",
                "",
            ),
            "candidate_store_estimated_bytes": row.get(
                "large_graph_envelope.candidate_store_estimated_bytes",
                "",
            ),
            "graph_array_bytes": row.get("large_graph_envelope.graph_array_bytes", ""),
        }
        for row in final_rows
    ]


def _baseline_summary_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_names = sorted(
        {
            key.removeprefix("baseline_").split("_target_hit", 1)[0]
            for row in final_rows
            for key in row
            if key.startswith("baseline_") and key.endswith("_target_hit")
        }
    )
    for row in final_rows:
        for baseline in baseline_names:
            prefix = f"baseline_{baseline}_"
            target_hit = row.get(f"{prefix}target_hit", "")
            item = {
                **_run_identity(row),
                "baseline": baseline,
                "baseline_target_hit": target_hit,
                "baseline_target_abs_error": row.get(f"{prefix}target_abs_error", ""),
                "baseline_final_cumulative_ratio": row.get(f"{prefix}final_cumulative_ratio", ""),
                "baseline_cumulative_dee": row.get(f"{prefix}cumulative_dee", ""),
                "baseline_cumulative_fwe_weighted": row.get(f"{prefix}cumulative_fwe_weighted", ""),
                "baseline_cumulative_fse_unweighted": row.get(f"{prefix}cumulative_fse_unweighted", ""),
                "baseline_cumulative_ree_max": row.get(f"{prefix}cumulative_ree_max", ""),
                "baseline_cumulative_sipe": row.get(f"{prefix}cumulative_sipe", ""),
                "baseline_cumulative_sampled_eigen_error": row.get(
                    f"{prefix}cumulative_sampled_eigen_error",
                    "",
                ),
                "baseline_projected_macro_f1": row.get(f"{prefix}task_projected_macro_f1", ""),
                "baseline_refined_macro_f1": row.get(f"{prefix}task_refined_macro_f1", ""),
                "baseline_refined_macro_f1@0": row.get(f"{prefix}refined_macro_f1@0", ""),
                "baseline_refined_macro_f1@1": row.get(f"{prefix}refined_macro_f1@1", ""),
                "baseline_refined_macro_f1@3": row.get(f"{prefix}refined_macro_f1@3", ""),
                "baseline_refined_macro_f1@5": row.get(f"{prefix}refined_macro_f1@5", ""),
                "baseline_runtime_total": row.get(f"{prefix}runtime_total", ""),
                "comparison_status": "included"
                if str(target_hit).lower() == "true"
                else "failed target control",
            }
            rows.append(item)
    return rows


def _target_check_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in final_rows:
        key = (str(row.get("dataset", "")), str(row.get("target_ratio", "")))
        groups.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (dataset, target_ratio), group in sorted(groups.items()):
        hit_values = [1.0 if str(row.get("target_hit", "")).lower() == "true" else 0.0 for row in group]
        rows.append(
            {
                "dataset": dataset,
                "target_ratio": target_ratio,
                "run_count": len(group),
                "final_cumulative_ratio_mean": _mean_numeric(group, "final_cumulative_ratio"),
                "target_abs_error_mean": _mean_numeric(group, "target_abs_error"),
                "target_hit_rate": "" if not hit_values else float(sum(hit_values) / len(hit_values)),
            }
        )
    return rows


def _compare_rows(
    rows: list[Mapping[str, Any]],
    group_by: list[str],
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    metrics = metrics or [
        "final_cumulative_ratio",
        "target_abs_error",
        "cumulative_dee",
        "cumulative_fwe_weighted",
        "cumulative_fse_unweighted",
        "cumulative_ree_max",
        "cumulative_sipe",
        "task_projected_macro_f1",
        "task_refined_macro_f1",
        "task_primary_macro_f1",
        "runtime_total_run",
        "peak_rss_gb",
        "peak_vram_allocated_gb",
    ]
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in group_by)
        groups.setdefault(key, []).append(row)
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        out: dict[str, Any] = {column: value for column, value in zip(group_by, key)}
        out["run_count"] = int(len(group))
        for metric in metrics:
            values = _numeric(row.get(metric) for row in group)
            if not values:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""
                continue
            out[f"{metric}_mean"] = float(sum(values) / len(values))
            out[f"{metric}_min"] = float(min(values))
            out[f"{metric}_max"] = float(max(values))
        out_rows.append(out)
    return out_rows


PAPER_TABLE_METRICS = [
    "final_cumulative_ratio",
    "cumulative_dee",
    "cumulative_fse_unweighted",
    "cumulative_ree_max",
    "cumulative_sipe",
    "cumulative_sampled_eigen_error",
    "task_projected_micro_f1",
    "task_projected_macro_f1",
    "task_refined_micro_f1",
    "task_refined_macro_f1",
    "task_refined_micro_f1@0",
    "task_refined_macro_f1@0",
    "task_refined_micro_f1@1",
    "task_refined_macro_f1@1",
    "task_refined_micro_f1@3",
    "task_refined_macro_f1@3",
    "task_refined_micro_f1@5",
    "task_refined_macro_f1@5",
    "task_best_refined_macro_f1",
    "task_refine_auc_macro_f1",
    "task_full_graph_rgcn_lite_default_macro_f1",
    "task_full_graph_rgcn_lite_tuned_macro_f1",
    "task_full_graph_han_small_macro_f1",
    "task_full_graph_hgt_small_macro_f1",
    "runtime_total_run",
    "candidate_generation_time",
    "candidate_substage_times.twohop_expansion",
    "candidate_substage_times.bucket_emit",
    "peak_rss_gb",
    "peak_vram_gb",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "candidate_buffer_bytes",
]


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = float(sum(values) / len(values))
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def _fmt_mean_pm_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _compute_device_mark(rows: list[Mapping[str, Any]]) -> str:
    for row in rows:
        device = str(row.get("compute_device", row.get("task.device", ""))).lower()
        cuda_available = str(row.get("cuda_available", "")).lower() == "true"
        peak_vram = max(
            _as_float(row.get("peak_vram_gb"), 0.0) or 0.0,
            _as_float(row.get("peak_vram_allocated_gb"), 0.0) or 0.0,
            _as_float(row.get("peak_vram_reserved_gb"), 0.0) or 0.0,
        )
        if "cuda" in device or "gpu" in device or cuda_available or peak_vram > 0.0:
            return "GPU"
    return "CPU"


def _mean_std_rows(final_rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in final_rows:
        key = tuple(str(row.get(column, "")) for column in group_by)
        groups.setdefault(key, []).append(row)
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        out: dict[str, Any] = {column: value for column, value in zip(group_by, key)}
        out["run_count"] = int(len(group))
        seeds = {str(row.get("seed", "")) for row in group if str(row.get("seed", ""))}
        out["seed_count"] = int(len(seeds))
        out["compute_device_mark"] = _compute_device_mark(group)
        if "dataset" not in out:
            datasets = sorted({str(row.get("dataset", "")) for row in group if str(row.get("dataset", ""))})
            out["datasets"] = ",".join(datasets)
        for metric in PAPER_TABLE_METRICS:
            values = _numeric(row.get(metric) for row in group)
            column = metric.replace(".", "_").replace("@", "at")
            if not values:
                out[f"{column}_mean"] = ""
                out[f"{column}_std"] = ""
                out[f"{column}_mean_pm_std"] = ""
                continue
            mean = float(sum(values) / len(values))
            std = _std(values)
            out[f"{column}_mean"] = mean
            out[f"{column}_std"] = std
            out[f"{column}_mean_pm_std"] = _fmt_mean_pm_std(mean, std)
        out_rows.append(out)
    return out_rows


def _numeric(values: Iterable[Any]) -> list[float]:
    return [value for value in (_as_float(item, None) for item in values) if value is not None]


def _group_mean(rows: list[Mapping[str, Any]], group_key: str, metric: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _as_float(row.get(metric), None)
        if value is None:
            continue
        groups.setdefault(str(row.get(group_key, "")), []).append(value)
    return {key: float(sum(values) / len(values)) for key, values in groups.items() if values}


def _maybe_write_figures(
    output: Path,
    final_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=160)
        plt.close()

    if target_rows:
        labels = [f"{row.get('dataset')} r={_fmt_metric(row.get('target_ratio'))}" for row in target_rows]
        values = [_as_float(row.get("target_hit_rate"), 0.0) or 0.0 for row in target_rows]
        plt.figure(figsize=(max(6, len(labels) * 0.7), 3.5))
        plt.bar(labels, values, color="#4C78A8")
        plt.ylim(0.0, 1.05)
        plt.ylabel("hit rate")
        plt.xticks(rotation=35, ha="right")
        plt.title("Target Ratio Hit Rate")
        savefig("target_ratio_hit_rate.png")

    variant_dee = _group_mean(final_rows, "variant", "final_DEE")
    if "base" in variant_dee and len(variant_dee) > 1:
        labels = sorted(variant_dee)
        deltas = [variant_dee[label] - variant_dee["base"] for label in labels]
        plt.figure(figsize=(max(6, len(labels) * 0.75), 3.5))
        plt.bar(labels, deltas, color="#F58518")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.ylabel("DEE minus base")
        plt.xticks(rotation=35, ha="right")
        plt.title("Ablation Delta DEE")
        savefig("ablation_delta_DEE.png")

    variant_f1 = _group_mean(final_rows, "variant", "task_primary_macro_f1")
    if "base" in variant_f1 and len(variant_f1) > 1:
        labels = sorted(variant_f1)
        deltas = [variant_f1[label] - variant_f1["base"] for label in labels]
        plt.figure(figsize=(max(6, len(labels) * 0.75), 3.5))
        plt.bar(labels, deltas, color="#54A24B")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.ylabel("macro-F1 minus base")
        plt.xticks(rotation=35, ha="right")
        plt.title("Ablation Delta Task F1")
        savefig("ablation_delta_task_f1.png")

    source_groups: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        source = str(row.get("candidate_source", ""))
        if source:
            source_groups.setdefault(source, []).append(row)
    if len(source_groups) > 1:
        labels: list[str] = []
        runtimes: list[float] = []
        dees: list[float] = []
        for source, rows in sorted(source_groups.items()):
            runtime_values = _numeric(row.get("runtime_total_run") for row in rows)
            dee_values = _numeric(row.get("final_DEE") for row in rows)
            if not runtime_values or not dee_values:
                continue
            labels.append(source)
            runtimes.append(float(sum(runtime_values) / len(runtime_values)))
            dees.append(float(sum(dee_values) / len(dee_values)))
        if labels:
            plt.figure(figsize=(6, 4))
            plt.scatter(runtimes, dees, color="#B279A2")
            for label, x, y in zip(labels, runtimes, dees):
                plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
            plt.xlabel("runtime_total_run")
            plt.ylabel("DEE")
            plt.title("Source Pareto: DEE vs Runtime")
            savefig("source_pareto_DEE_runtime.png")

    share_terms = ("spec", "rel", "feat", "conv", "boundary")
    share_groups: dict[str, dict[str, float]] = {}
    for variant in sorted({str(row.get("variant", "")) for row in final_rows if row.get("variant")}):
        rows = [row for row in final_rows if str(row.get("variant", "")) == variant]
        share_groups[variant] = {
            term: float(_mean_numeric(rows, f"score_contribution_share_{term}") or 0.0)
            for term in share_terms
        }
    if share_groups:
        labels = list(share_groups)
        bottoms = [0.0] * len(labels)
        colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
        plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
        for term, color in zip(share_terms, colors):
            values = [share_groups[label][term] for label in labels]
            plt.bar(labels, values, bottom=bottoms, label=term, color=color)
            bottoms = [left + value for left, value in zip(bottoms, values)]
        plt.ylabel("share")
        plt.xticks(rotation=35, ha="right")
        plt.legend(ncols=min(5, len(share_terms)), fontsize=8)
        plt.title("Score Contribution Share")
        savefig("score_contribution_share.png")

    gap_rows = [
        row
        for row in final_rows
        if _as_float(row.get("cumulative_dee"), None) is not None
        and _as_float(row.get("final_DEE"), None) is not None
    ]
    if gap_rows:
        labels = [str(row.get("variant") or row.get("run_name", "")) for row in gap_rows]
        values = [
            float(_as_float(row.get("cumulative_dee"), 0.0) or 0.0)
            - float(_as_float(row.get("final_DEE"), 0.0) or 0.0)
            for row in gap_rows
        ]
        plt.figure(figsize=(max(6, len(labels) * 0.8), 3.5))
        plt.bar(labels, values, color="#72B7B2")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.ylabel("cumulative DEE - final DEE")
        plt.xticks(rotation=35, ha="right")
        plt.title("Cumulative vs Final Gap")
        savefig("cumulative_vs_final_gap.png")

    source_distribution: dict[str, dict[str, float]] = {}
    for row in final_rows:
        variant = str(row.get("variant") or row.get("run_name", ""))
        if not variant:
            continue
        matched_total = _as_float(row.get("matched_pairs"), None)
        for key, value in row.items():
            if not key.startswith("matched_pairs_by_source."):
                continue
            source = key.removeprefix("matched_pairs_by_source.")
            count = _as_float(value, None)
            if count is None:
                continue
            denom = float(matched_total) if matched_total not in (None, 0.0) else 1.0
            source_distribution.setdefault(variant, {}).setdefault(source, 0.0)
            source_distribution[variant][source] += float(count / denom)
    if source_distribution:
        labels = sorted(source_distribution)
        sources = sorted({source for values in source_distribution.values() for source in values})
        bottoms = [0.0] * len(labels)
        plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
        palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
        for index, source in enumerate(sources):
            values = [source_distribution[label].get(source, 0.0) for label in labels]
            plt.bar(labels, values, bottom=bottoms, label=source, color=palette[index % len(palette)])
            bottoms = [left + value for left, value in zip(bottoms, values)]
        plt.ylabel("selected source fraction")
        plt.xticks(rotation=35, ha="right")
        plt.legend(ncols=min(4, len(sources)), fontsize=8)
        plt.title("Source Distribution by Variant")
        savefig("source_distribution_by_variant.png")

    task_scatter = [
        row
        for row in final_rows
        if _as_float(row.get("cumulative_dee"), None) is not None
        and _as_float(row.get("task_refined_macro_f1") or row.get("task_projected_macro_f1"), None)
        is not None
    ]
    if task_scatter:
        xs = [float(_as_float(row.get("cumulative_dee"), 0.0) or 0.0) for row in task_scatter]
        ys = [
            float(
                _as_float(
                    row.get("task_refined_macro_f1") or row.get("task_projected_macro_f1"),
                    0.0,
                )
                or 0.0
            )
            for row in task_scatter
        ]
        labels = [str(row.get("variant") or row.get("run_name", "")) for row in task_scatter]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, color="#4C78A8")
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        plt.xlabel("cumulative DEE")
        plt.ylabel("task macro-F1")
        plt.title("Task vs Cumulative DEE")
        savefig("task_vs_cumulative_dee.png")

    dim_groups: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        dim = str(row.get("sketch_dim") or row.get("config.sketch.dim") or "")
        method = str(row.get("sketch_method") or row.get("config.sketch.method") or "")
        if dim:
            label = f"{method or 'sketch'}-d{dim}"
            dim_groups.setdefault(label, []).append(row)
    if len(dim_groups) > 1:
        labels = sorted(dim_groups)
        coverage = [
            float(_mean_numeric(dim_groups[label], "candidate_coverage") or 0.0)
            for label in labels
        ]
        dee = [float(_mean_numeric(dim_groups[label], "final_DEE") or 0.0) for label in labels]
        x = list(range(len(labels)))
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(x, coverage, marker="o", color="#4C78A8", label="candidate coverage")
        ax1.set_ylabel("candidate coverage")
        ax1.set_xticks(x, labels)
        ax2 = ax1.twinx()
        ax2.plot(x, dee, marker="s", color="#F58518", label="DEE")
        ax2.set_ylabel("DEE")
        ax1.set_xlabel("sketch method / dim")
        fig.legend(loc="upper center", ncols=2, fontsize=8)
        plt.title("Sketch Dim Candidate Coverage")
        savefig("dim_candidate_coverage.png")


def summarize_experiments(inputs: Iterable[str | Path], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    resource_rows: list[dict] = []
    quality_rows: list[dict] = []
    final_rows: list[dict] = []
    level_rows_all: list[dict] = []
    failure_rows: list[dict] = []

    run_dirs = discover_run_dirs(inputs)
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        diagnostics_paths = sorted(run_dir.glob("level_*/diagnostics.json"))
        experiment_block = str(metadata.get("experiment_block", ""))
        unique_run_key = str(metadata.get("unique_run_key", metadata.get("run_name", run_dir.name)))
        scalar_metadata = {
            key: value
            for key, value in metadata.items()
            if not isinstance(value, (dict, list))
        }
        base = {
            "run_name": metadata.get("run_name", run_dir.name),
            "status": metadata.get("status", "success" if diagnostics_paths else "unknown"),
            "dataset": metadata.get("dataset", ""),
            "variant": metadata.get("variant", ""),
            "experiment_block": experiment_block,
            "unique_run_key": unique_run_key,
            "run_dir": str(run_dir),
            "failure_reason": metadata.get("failure_reason", ""),
        }
        base.update({key: value for key, value in scalar_metadata.items() if key not in base})
        if diagnostics_paths:
            level_rows: list[dict] = []
            for diagnostics_path in diagnostics_paths:
                row = {**base, **diagnostics_row(run_dir, diagnostics_path), "row_type": "level"}
                level_rows.append(row)
                level_rows_all.append(row)
                all_rows.append(row)
                resource_rows.append(
                    {
                        "run_name": base["run_name"],
                        "row_type": "level",
                        "level": row.get("level", ""),
                        "disk_usage_bytes": disk_usage_bytes(run_dir),
                        "runtime_total": sum(
                            float(row.get(f"runtime_by_stage.{key}", 0) or 0)
                            for key in (
                                "sketch",
                                "candidates",
                                "scoring",
                                "matching_and_aggregation",
                                "spectral_diagnostics",
                            )
                        ),
                    }
                )
                quality_rows.append(_quality_row(base, row, row_type="level"))
            final_row = _final_cumulative_row(base, level_rows, metadata)
            task_eval_path = run_dir / "task_eval.json"
            if task_eval_path.exists():
                task_payload = read_json(task_eval_path)
                final_row.update(flatten_mapping({"task": task_payload}))
                _with_task_aliases(final_row)
            final_rows.append(final_row)
            all_rows.append(final_row)
            quality_rows.append(_quality_row(base, final_row, row_type="final"))
        else:
            all_rows.append({**base, "row_type": "run"})
        if base["status"] == "failed":
            failure_rows.append(base)

    for final_row in final_rows:
        final_row["run_count_unique"] = int(len(final_rows))
    target_rows = _target_check_rows(final_rows)

    write_csv(output / "all_runs.csv", all_rows)
    write_csv(output / "all_levels.csv", level_rows_all)
    write_csv(output / "final_summary.csv", final_rows)
    write_csv(output / "run_final_summary.csv", final_rows)
    run_resource_rows = _run_resource_rows(final_rows)
    write_csv(output / "resource_summary.csv", run_resource_rows)
    write_csv(output / "resource_summary_runlevel.csv", run_resource_rows)
    write_csv(output / "resource_summary_levels.csv", resource_rows)
    write_csv(output / "quality_summary.csv", quality_rows)
    write_csv(output / "score_term_scale.csv", _score_term_scale_rows(final_rows))
    write_csv(output / "baseline_summary.csv", _baseline_summary_rows(final_rows))
    write_csv(output / "candidate_source_pareto.csv", _candidate_source_pareto_rows(final_rows))
    write_csv(output / "task_summary.csv", _task_summary_rows(final_rows))
    write_csv(output / "target_check.csv", target_rows)
    write_csv(output / "compare_by_variant.csv", _compare_rows(final_rows, ["dataset", "variant"]))
    write_csv(
        output / "compare_by_dataset_variant.csv",
        _compare_rows(final_rows, ["dataset", "variant"]),
    )
    write_csv(output / "compare_by_source.csv", _compare_rows(final_rows, ["dataset", "candidate_source"]))
    write_csv(
        output / "compare_by_dim.csv",
        _compare_rows(final_rows, ["dataset", "sketch_method", "sketch_dim"]),
    )
    paper_mean_std_rows = _mean_std_rows(final_rows, ["variant"])
    paper_dataset_variant_rows = _mean_std_rows(final_rows, ["dataset", "variant"])
    write_csv(output / "paper_table_mean_std.csv", paper_mean_std_rows)
    write_csv(output / "paper_table_dataset_variant.csv", paper_dataset_variant_rows)
    write_csv(output / "failures.csv", failure_rows)
    _maybe_write_figures(output, final_rows, target_rows)
    core_rows = _core_report_rows(final_rows)
    report_rows = final_rows[:20]
    gpu_marked_runs = sum(1 for row in final_rows if _compute_device_mark([row]) == "GPU")
    report = [
        "# Experiment Summary",
        "",
        f"Unique runs: {len(final_rows)}",
        f"GPU-marked runs: {gpu_marked_runs}",
        f"CPU-only runs: {len(final_rows) - gpu_marked_runs}",
        f"Level rows: {sum(int(row.get('level_row_count', 0) or 0) for row in final_rows)}",
        f"Rows: {len(all_rows)}",
        f"Failures: {len(failure_rows)}",
        "",
        "## Core Results",
        "",
        markdown_table(
            core_rows,
            [
                "variant",
                "final ratio",
                "DEE \u2193",
                "FSE-unweighted \u2193",
                "REE-max \u2193",
                "SIPE \u2193",
                "macro-F1 \u2191",
                "runtime \u2193",
                "peak RAM",
            ],
        ),
        "",
        "## Next4 Method Notes",
        "",
        "HeSF-LVC is the main method: heterogeneous fused low-pass sketch + type-compatible small-cluster local variation coarsening + convolution-aware scoring.",
        "Primary spectral metrics are cumulative metrics; final-level metrics are diagnostics only.",
        "meta-path is optional / disabled in the main method unless a future run proves changed matches and metric gains.",
        "Non-uniform relation weighting is optional; uniform relation fusion is the default claim.",
        "mutual_best is retained as an ablation baseline, not the default kernel.",
        "0.25 aggressive mode is a stress setting, not high-quality spectral coarsening.",
        "GPU/system claims require measured GPU paths; CPU-only runs must not be described as GPU validated.",
        "",
        "## Completed Runs",
        "",
        markdown_table(
            report_rows,
            [
                "run_name",
                "status",
                "dataset",
                "variant",
                "target_ratio",
                "final_cumulative_ratio",
                "target_abs_error",
                "target_hit",
                "cumulative_dee",
                "cumulative_fwe_weighted",
                "cumulative_fse_unweighted",
                "cumulative_ree_max",
                "cumulative_sipe",
                "task_projected_macro_f1",
                "task_refined_macro_f1",
                "task_coarse_train_macro_f1",
                "task_primary_macro_f1",
                "task_primary_metric",
                "task_primary_metric_name",
                "spectral_baseline_computed_count",
                "cumulative_spectral_baseline_computed_count",
                "spectral_exact_eigenvalue_sanity_status",
                "spectral_exact_eigenvalue_sanity_mode",
                "runtime_total_run",
                "peak_rss_gb",
                "peak_cpu_memory_gb",
                "peak_vram_allocated_gb",
                "peak_vram_reserved_gb",
                "peak_gpu_memory_allocated_gb",
                "peak_gpu_memory_reserved_gb",
                "compute_device",
                "cuda_available",
                "cpu_only",
                "matched_units",
                "node_reduction",
                "cluster_count",
                "cluster_size_histogram",
                "score_contribution_share",
                "failure_reason",
            ],
        ),
        "",
        "## Failed Runs",
        "",
        markdown_table(failure_rows[:20], ["run_name", "status", "dataset", "failure_reason"]),
        "",
        "## Correctness Invariants",
        "",
        "Invariant fields are preserved in `all_runs.csv` when emitted by each runner.",
        "",
        "## Compression Ratios",
        "",
        "See `quality_summary.csv`.",
        "",
        "## Candidate Source Distribution",
        "",
        "Flattened diagnostics fields are preserved in `all_runs.csv`.",
        "",
        "## Runtime Breakdown",
        "",
        "Standard stage fields map to `runtime_by_stage.sketch`, `runtime_by_stage.candidates`, `runtime_by_stage.scoring`, split `runtime_by_stage.matching` / `runtime_by_stage.aggregation`, combined `runtime_by_stage.matching_and_aggregation`, `runtime_by_stage.cumulative_diagnostics`, task train/refine, and baseline totals.",
        "",
        "## Memory And Disk Footprint",
        "",
        "See `resource_summary.csv` for run-level resource fields and `resource_summary_levels.csv` for per-level artifact disk usage.",
        "",
        "## Figures",
        "",
        "Generated figures are written under `figures/` when matplotlib is available.",
        "",
        "## Spectral Diagnostics",
        "",
        "Sketch-based diagnostics are optional and are recorded when present in run artifacts.",
        "",
        "final-level baseline diagnostics compare each method against the current level only.",
        "cumulative baseline diagnostics compare the composed assignment from the original graph to the final coarse graph.",
        "",
        "## Bottleneck Analysis",
        "",
        "Compare resource and runtime summaries across presets before scaling full graph runs.",
        "",
        "## Recommended Next Engineering Fixes",
        "",
        "- Promote any failing run in `failures.csv` to a focused regression test.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize experiment run directories.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--inputs", nargs="+", type=Path, dest="input_flags", help="Run roots to summarize; accepted for plan command compatibility.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = [*args.inputs, *(args.input_flags or [])]
    if not inputs:
        parser.error("at least one input root is required via positional inputs or --inputs")
    summarize_experiments(inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
