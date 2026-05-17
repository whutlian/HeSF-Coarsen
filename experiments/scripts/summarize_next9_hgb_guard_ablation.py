from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter


VARIANTS = [
    "P_baseline",
    "P_spectral_guard",
    "P_source_aware_auto",
    "P_spectral_guard_plus_source_aware_auto",
    "S_baseline",
    "S_spectral_guard",
    "S_source_aware_auto",
    "S_spectral_guard_plus_source_aware_auto",
    "flatten-sum",
    "H6-no-spec",
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


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return default


def _variant_from_paper_method(method: str) -> str | None:
    return {
        "HeSF-LVC-P": "P_baseline",
        "HeSF-LVC-S": "S_baseline",
        "flatten-sum": "flatten-sum",
        "H6-no-spec": "H6-no-spec",
    }.get(method)


def _variant_from_sourceaware(row: Mapping[str, Any]) -> str | None:
    lambda_spec = _as_float(row.get("lambda_spec", row.get("config.scoring.lambda_spec")), None)
    if lambda_spec == 0.25:
        return "P_source_aware_auto"
    if lambda_spec == 0.5:
        return "S_source_aware_auto"
    return None


def _actual_variant(row: Mapping[str, Any]) -> str:
    variant = str(row.get("variant", "") or "")
    if variant in VARIANTS:
        return variant
    method = str(row.get("method", "") or "")
    return {
        "HeSF-LVC-P": "P_baseline",
        "HeSF-LVC-S": "S_baseline",
        "flatten-sum": "flatten-sum",
        "H6-no-spec": "H6-no-spec",
    }.get(method, variant)


def _guard_flag(row: Mapping[str, Any], suffix: str) -> bool:
    return _truthy(row.get(f"spectral_guard.{suffix}")) or _truthy(row.get(f"source_aware_guard.{suffix}"))


def _guard_count(row: Mapping[str, Any], suffix: str) -> float:
    return (_as_float(row.get(f"spectral_guard.{suffix}"), 0.0) or 0.0) + (
        _as_float(row.get(f"source_aware_guard.{suffix}"), 0.0) or 0.0
    )


def _source_value(row: Mapping[str, Any], stem: str, source: str) -> str:
    return _fmt(row.get(f"source_aware_guard.{stem}.{source}", ""))


def _actual_rows(actual_summary: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = _read_csv(actual_summary / "run_final_summary.csv") or _read_csv(actual_summary / "final_summary.csv")
    main: list[dict[str, Any]] = []
    trigger: list[dict[str, Any]] = []
    source_distribution: list[dict[str, Any]] = []
    for row in source_rows:
        variant = _actual_variant(row)
        if variant not in VARIANTS:
            continue
        spectral_reason = str(row.get("spectral_guard.trigger_reason", "") or "")
        source_reason = str(row.get("source_aware_guard.trigger_reason", "") or "")
        guard_triggered = _guard_flag(row, "guard_triggered")
        guard_enabled = _guard_flag(row, "guard_enabled")
        rejected = _guard_count(row, "rejected_by_spec_count")
        rejected_share = max(
            _as_float(row.get("spectral_guard.rejected_by_spec_share"), 0.0) or 0.0,
            _as_float(row.get("source_aware_guard.rejected_by_spec_share"), 0.0) or 0.0,
        )
        source_before_onehop = _as_float(
            row.get("source_aware_guard.source_selected_share_before.onehop"),
            None,
        )
        source_after_onehop = _as_float(
            row.get("source_aware_guard.source_selected_share_after.onehop"),
            None,
        )
        onehop_high_delta = _as_float(
            row.get("source_aware_guard.onehop_high_delta_selected_share"),
            None,
        )
        if onehop_high_delta is None:
            onehop_high_delta = source_after_onehop
        main.append(
            {
                "variant": variant,
                "method": row.get("method", "HeSF-LVC-P" if variant.startswith("P_") else "HeSF-LVC-S"),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "target_hit": row.get("target_hit", ""),
                "DEE": _first(row, ("DEE", "cumulative_dee", "final_DEE")),
                "REEmax": _first(row, ("REEmax", "cumulative_ree_max", "final_REE_max")),
                "SIPE": _first(row, ("SIPE", "cumulative_sipe", "final_SIPE")),
                "projected_macro_f1": _first(row, ("projected_macro_f1", "task_projected_macro_f1")),
                "refined_macro_f1@5": _first(row, ("refined_macro_f1@5", "task_refined_macro_f1@5")),
                "best_macro_f1": _first(row, ("best_macro_f1", "task_best_refined_macro_f1")),
                "refine_auc_macro_f1": _first(row, ("refine_auc_macro_f1", "task_refine_auc_macro_f1")),
                "guard_enabled": guard_enabled,
                "guard_triggered": guard_triggered,
                "trigger_reason": "; ".join(part for part in (spectral_reason, source_reason) if part),
                "source_selected_share_before": _fmt(source_before_onehop),
                "source_selected_share_after": _fmt(source_after_onehop),
                "source_avg_delta_spec_before": _source_value(row, "source_avg_delta_spec_before", "onehop"),
                "source_avg_delta_spec_after": _source_value(row, "source_avg_delta_spec_after", "onehop"),
                "rejected_by_spec_count": int(rejected),
                "rejected_by_spec_share": _fmt(rejected_share),
                "fallback_used_count": int(_guard_count(row, "fallback_used_count")),
                "target_pressure_accept_count": int(_guard_count(row, "target_pressure_accept_count")),
                "cluster_size_hist": _first(
                    row,
                    ("source_aware_guard.cluster_size_hist", "cluster_size_histogram", "cluster_size_hist"),
                ),
                "onehop_high_delta_selected_share": _fmt(onehop_high_delta),
                "guard_source": "actual_next10_guard_ablation",
                "run_status": "available",
            }
        )
        trigger.append(
            {
                "variant": variant,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "guard_enabled": guard_enabled,
                "guard_triggered": guard_triggered,
                "activation_rate": _fmt(1.0 if guard_triggered else 0.0),
                "trigger_reason": "; ".join(part for part in (spectral_reason, source_reason) if part),
                "source_selected_share_before": _fmt(source_before_onehop),
                "source_selected_share_after": _fmt(source_after_onehop),
                "source_avg_delta_spec_before": _source_value(row, "source_avg_delta_spec_before", "onehop"),
                "source_avg_delta_spec_after": _source_value(row, "source_avg_delta_spec_after", "onehop"),
                "onehop_high_delta_selected_share": _fmt(onehop_high_delta),
                "rejected_by_spec_count": int(rejected),
                "rejected_by_spec_share": _fmt(rejected_share),
                "fallback_used_count": int(_guard_count(row, "fallback_used_count")),
                "target_pressure_accept_count": int(_guard_count(row, "target_pressure_accept_count")),
                "cluster_size_hist": _first(
                    row,
                    ("source_aware_guard.cluster_size_hist", "cluster_size_histogram", "cluster_size_hist"),
                ),
                "target_hit": row.get("target_hit", ""),
            }
        )
        for src in ("bucket", "onehop", "capped_twohop", "fallback"):
            source_distribution.append(
                {
                    "variant": variant,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "source": src,
                    "source_selected_share_before": _source_value(row, "source_selected_share_before", src),
                    "source_selected_share_after": _source_value(row, "source_selected_share_after", src),
                    "source_avg_delta_spec_before": _source_value(row, "source_avg_delta_spec_before", src),
                    "source_avg_delta_spec_after": _source_value(row, "source_avg_delta_spec_after", src),
                }
            )
    return main, trigger, source_distribution


def _paper_rows(hgb_summary: Path) -> list[dict[str, Any]]:
    out = []
    for row in _read_csv(hgb_summary / "final_main_table_by_seed.csv"):
        variant = _variant_from_paper_method(str(row.get("method", "")))
        if variant is None:
            continue
        out.append(
            {
                "variant": variant,
                "method": row.get("method", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "target_hit": row.get("target_hit", ""),
                "DEE": row.get("DEE", ""),
                "REEmax": row.get("REEmax", ""),
                "SIPE": row.get("SIPE", ""),
                "projected_macro_f1": row.get("projected_macro_f1", ""),
                "refined_macro_f1@5": row.get("refined_macro_f1@5", ""),
                "best_macro_f1": row.get("best_macro_f1", ""),
                "refine_auc_macro_f1": row.get("refine_auc_macro_f1", ""),
                "guard_source": "baseline_or_negative_control",
                "run_status": "available",
            }
        )
    return out


def _sourceaware_rows(sourceaware_summary: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main = []
    trigger = []
    source = []
    for row in _read_csv(sourceaware_summary / "run_final_summary.csv"):
        variant = _variant_from_sourceaware(row)
        if variant is None:
            continue
        before = _as_float(row.get("source_policy_filter.pairs_before"), None)
        after = _as_float(row.get("source_policy_filter.pairs_after"), None)
        rejected = _as_float(row.get("source_policy_filter.onehop_rejected_by_spec"), 0.0) or 0.0
        main.append(
            {
                "variant": variant,
                "method": "HeSF-LVC-P" if variant.startswith("P_") else "HeSF-LVC-S",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "target_hit": row.get("target_hit", ""),
                "DEE": row.get("cumulative_dee", row.get("final_DEE", "")),
                "REEmax": row.get("cumulative_ree_max", row.get("final_REE_max", "")),
                "SIPE": row.get("cumulative_sipe", row.get("final_SIPE", "")),
                "projected_macro_f1": row.get("task_projected_macro_f1", ""),
                "refined_macro_f1@5": row.get("task_refined_macro_f1@5", ""),
                "best_macro_f1": row.get("task_best_refined_macro_f1", ""),
                "refine_auc_macro_f1": row.get("task_refine_auc_macro_f1", ""),
                "guard_source": "legacy_source_policy_bucket_q95",
                "run_status": "available_legacy_sourceaware",
            }
        )
        trigger.append(
            {
                "variant": variant,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "guard_enabled": True,
                "guard_triggered": rejected > 0,
                "trigger_reason": "legacy onehop > bucket q95" if rejected > 0 else "legacy policy inactive",
                "rejected_by_spec_count": int(rejected),
                "rejected_by_spec_share": _fmt(rejected / max(before or 1.0, 1.0)),
                "pairs_before": _fmt(before),
                "pairs_after": _fmt(after),
                "target_hit": row.get("target_hit", ""),
            }
        )
        for src in ("bucket", "onehop", "capped_twohop", "fallback"):
            selected = _as_float(row.get(f"selected_merges_by_source.{src}"), 0.0) or 0.0
            avg_spec = _as_float(row.get(f"selected_source_avg_delta_spec.{src}"), None)
            source.append(
                {
                    "variant": variant,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "source": src,
                    "selected_merges": int(selected),
                    "source_avg_delta_spec_after": _fmt(avg_spec),
                }
            )
    return main, trigger, source


def _placeholder_guard_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    seeds = sorted({str(row.get("seed", "")) for row in rows if row.get("seed")})
    out = []
    target_failures = []
    for variant in (
        "P_spectral_guard",
        "P_spectral_guard_plus_source_aware_auto",
        "S_spectral_guard",
        "S_spectral_guard_plus_source_aware_auto",
    ):
        for dataset in datasets:
            for seed in seeds:
                out.append(
                    {
                        "variant": variant,
                        "method": "HeSF-LVC-P" if variant.startswith("P_") else "HeSF-LVC-S",
                        "dataset": dataset,
                        "seed": seed,
                        "target_hit": "",
                        "DEE": "",
                        "REEmax": "",
                        "SIPE": "",
                        "projected_macro_f1": "",
                        "refined_macro_f1@5": "",
                        "best_macro_f1": "",
                        "refine_auc_macro_f1": "",
                        "guard_source": "implemented_not_full_local_rerun",
                        "run_status": "not_run",
                    }
                )
                target_failures.append(
                    {
                        "variant": variant,
                        "dataset": dataset,
                        "seed": seed,
                        "target_hit": "",
                        "reason": "full guard ablation not rerun in legacy summary source",
                    }
                )
    return out, target_failures


def _delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row.get("variant"), row.get("dataset"), row.get("seed")): row for row in rows}
    out = []
    pairs = [
        ("P_source_aware_auto", "P_baseline"),
        ("S_source_aware_auto", "S_baseline"),
        ("P_spectral_guard", "P_baseline"),
        ("S_spectral_guard", "S_baseline"),
        ("P_spectral_guard_plus_source_aware_auto", "P_baseline"),
        ("S_spectral_guard_plus_source_aware_auto", "S_baseline"),
    ]
    for variant, baseline in pairs:
        for key, row in by_key.items():
            if key[0] != variant:
                continue
            base = by_key.get((baseline, key[1], key[2]))
            if base is None:
                continue
            for metric in ("DEE", "REEmax", "SIPE", "best_macro_f1", "refined_macro_f1@5"):
                value = _as_float(row.get(metric), None)
                base_value = _as_float(base.get(metric), None)
                out.append(
                    {
                        "variant": variant,
                        "baseline": baseline,
                        "dataset": key[1],
                        "seed": key[2],
                        "metric": metric,
                        "delta_vs_baseline": _fmt(value - base_value)
                        if value is not None and base_value is not None
                        else "",
                        "best_macro_f1_drop_vs_baseline": _fmt(base_value - value)
                        if metric == "best_macro_f1" and value is not None and base_value is not None
                        else "",
                        "refined_macro_f1@5_drop_vs_baseline": _fmt(base_value - value)
                        if metric == "refined_macro_f1@5" and value is not None and base_value is not None
                        else "",
                    }
                )
    return out


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("variant", "")), str(row.get("dataset", "")))].append(row)
    out = []
    for (variant, dataset), group in sorted(groups.items()):
        item = {"variant": variant, "dataset": dataset, "run_count": len(group)}
        for metric in ("DEE", "REEmax", "SIPE", "best_macro_f1", "refined_macro_f1@5"):
            values = [value for value in (_as_float(row.get(metric), None) for row in group) if value is not None]
            item[f"{metric}_mean"] = _fmt(mean(values)) if values else ""
        out.append(item)
    return out


def summarize_next9_hgb_guard_ablation(
    *,
    hgb_summary: str | Path | None = None,
    sourceaware_summary: str | Path | None = None,
    actual_summary: str | Path | None = None,
    output: str | Path,
    command_lines: Sequence[str] = (),
) -> dict[str, Any]:
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    target_failures: list[dict[str, Any]] = []
    if actual_summary is not None:
        rows, trigger_rows, source_distribution = _actual_rows(Path(actual_summary))
    else:
        if hgb_summary is None:
            raise ValueError("hgb_summary is required unless actual_summary is provided")
        rows = _paper_rows(Path(hgb_summary))
        trigger_rows = []
        source_distribution = []
        if sourceaware_summary is not None:
            source_rows, trigger_rows, source_distribution = _sourceaware_rows(Path(sourceaware_summary))
            rows.extend(source_rows)
        placeholders, target_failures = _placeholder_guard_rows(rows)
        rows.extend(placeholders)
    delta_rows = _delta_rows(rows)
    aggregate = _aggregate(rows)
    if actual_summary is not None:
        for row in rows:
            if str(row.get("target_hit", "")).lower() not in {"true", "1", "yes"}:
                target_failures.append(
                    {
                        "variant": row.get("variant", ""),
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "target_hit": row.get("target_hit", ""),
                        "reason": "target_hit was not true in actual guard ablation row",
                    }
                )

    write_csv(output / "guard_ablation_main_table.csv", rows)
    write_csv(output / "guard_trigger_diagnostics.csv", trigger_rows)
    write_csv(output / "guard_source_distribution.csv", source_distribution)
    write_csv(output / "guard_target_hit_failures.csv", target_failures)
    write_csv(output / "guard_delta_vs_baseline.csv", delta_rows)

    _plot_scatter(aggregate, "DEE_mean", "best_macro_f1_mean", output / "figures" / "guard_dee_vs_best_macro_f1.png")
    _plot_scatter(source_distribution, "selected_merges", "source_avg_delta_spec_after", output / "figures" / "guard_selected_source_share.png")
    _plot_scatter(trigger_rows, "rejected_by_spec_count", "rejected_by_spec_share", output / "figures" / "guard_rejection_counts.png")

    lines = [
        "# Next9 Guard Ablation Summary",
        "",
        "Spectral guard and auto source-aware guard code paths are implemented and unit-tested.",
        (
            "This summary is based on actual guard ablation rows."
            if actual_summary is not None
            else "The available local legacy evidence includes P/S baselines and the prior source-aware bucket-q95 policy; missing spectral-guard and combined full HGB reruns are marked `not_run` rather than inferred."
        ),
        "",
        markdown_table(aggregate, ["variant", "dataset", "run_count", "DEE_mean", "REEmax_mean", "SIPE_mean", "best_macro_f1_mean"]),
        "",
        "Acceptance status: source-aware filtering remains optional/appendix until a full auto-trigger ablation is rerun; spectral guard is not promoted by this summary alone.",
    ]
    if command_lines:
        lines.extend(["", "## Commands", *[f"- `{line}`" for line in command_lines]])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "main_rows": rows,
        "trigger_rows": trigger_rows,
        "source_distribution": source_distribution,
        "target_failures": target_failures,
        "delta_rows": delta_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgb-summary", "--input", required=True)
    parser.add_argument("--sourceaware-summary", default="outputs/exp_next8_hgb_lvc_sourceaware_5seed_20260517_summary")
    parser.add_argument("--actual-summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_hgb_guard_ablation(
        hgb_summary=args.hgb_summary,
        sourceaware_summary=args.sourceaware_summary,
        actual_summary=args.actual_summary,
        output=args.output,
        command_lines=args.command_lines,
    )


if __name__ == "__main__":
    main()
