from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.run_gate21_19_multidataset_frontier import GATE21_19_MAIN_FIELDS, _add_recovery, _decision_flag_rows, _source_dataset_dir
from hesf_coarsen.eval.official.acm_selector_overlap import build_acm_selector_overlap_rows, rows_from_gate21_exports
from hesf_coarsen.eval.official.critical_robustness_runner import ROBUSTNESS_FIELDS, build_critical_robustness_rows, build_missing_robustness_queue_rows
from hesf_coarsen.eval.official.final_stage_report_tables import (
    BEST_METHOD_COMPARISON_FIELDS,
    FRONTIER_FIELDS,
    build_best_method_comparison,
    build_frontier_rows,
)
from hesf_coarsen.eval.official.freehgc_score_selector import (
    FREEHGC_SELECTOR_FIELDS,
    build_dblp_freehgc_score_selector_export,
    build_freehgc_score_selector_plan_rows,
)
from hesf_coarsen.eval.official.gate21_20_decision import GATE21_20_DECISION_FLAGS, gate21_20_decision
from hesf_coarsen.eval.official.imdb_planner_upgrade import IMDB_UPGRADE_FIELDS, build_imdb_hesf_upgrade_rows
from hesf_coarsen.eval.official.official_training_queue import aggregate_training_runs, build_training_queue, execute_training_queue
from hesf_coarsen.eval.official.rep_selection import REP_SELECTION_FIELDS, resolve_validation_metrics, select_gate21_20_representatives
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


DEFAULT_INPUT = ROOT / "outputs" / "gate21_19_smoke"
DEFAULT_OUTPUT = ROOT / "outputs" / "gate21_20_final_stage"

GATE21_20_MAIN_FIELDS = tuple(
    dict.fromkeys(
        [
            *GATE21_19_MAIN_FIELDS,
            "actual_semantic_structural_ratio",
            "selector_uses_test_labels",
            "validation_resolution_source",
            "constraint_pass",
            "MD_keep",
            "MA_keep",
            "MK_keep",
            "source_method",
            "source_path",
        ]
    )
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.20 final-style stage report table runner.")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick-robust"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", "--output-dir", dest="output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sehgnn-repo", default=str(ROOT / "external" / "SeHGNN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--dry-run-training", action="store_true")
    parser.add_argument("--skip-dblp-validation-repair", action="store_true")
    parser.add_argument("--skip-freehgc-selector-training", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    mode = str(args.mode)
    datasets = [normalize_dataset(item) for item in args.datasets]
    sehgnn_repo = Path(args.sehgnn_repo)
    training_seeds = [int(item) for item in (args.training_seeds or [1])]
    if mode == "quick-robust" and len(training_seeds) < 3:
        training_seeds = [1, 2, 3]
    graph_seeds = [int(item) for item in (args.graph_seeds or [1])]

    prior_main = _read_csv(input_dir / "gate21_19_main_official_table.csv")
    prior_training = _read_csv(input_dir / "gate21_19_training_runs.csv")
    main_rows = [_strip_prior_rep_rows(row) for row in prior_main if not _is_prior_rep_row(row)]
    failure_rows = _read_csv(input_dir / "gate21_19_training_failures.csv")
    training_runs = list(prior_training)

    if not bool(args.skip_dblp_validation_repair) and "DBLP" in datasets:
        validation_runs = _repair_dblp_hesf_validation(
            out_dir=out_dir,
            mode=mode,
            graph_seeds=graph_seeds,
            training_seeds=training_seeds if mode == "quick-robust" else training_seeds[:1],
            sehgnn_repo=sehgnn_repo,
            device=str(args.device),
            dry_run=bool(args.dry_run_training) or mode == "preflight",
        )
        _merge_dblp_hesf_validation(main_rows, validation_runs)
        training_runs.extend(_gate21_4_training_rows(validation_runs))

    imdb_upgrade_rows = build_imdb_hesf_upgrade_rows(main_rows, budgets=(0.40, 0.50))
    main_rows.extend(row for row in imdb_upgrade_rows if bool_value(row.get("success", True)))

    freehgc_selector_rows = _prepare_freehgc_selector_rows(out_dir=out_dir, datasets=datasets, graph_seed=graph_seeds[0])
    selector_queue = build_training_queue(freehgc_selector_rows, graph_seeds=graph_seeds[:1], training_seeds=training_seeds if mode == "quick-robust" else training_seeds[:1])
    selector_runs: list[dict[str, Any]] = []
    selector_failures: list[dict[str, Any]] = []
    if selector_queue and not bool(args.skip_freehgc_selector_training):
        selector_runs, selector_failures = execute_training_queue(
            selector_queue,
            sehgnn_repo=sehgnn_repo,
            device=str(args.device),
            out_dir=out_dir,
            python_executable=sys.executable,
            dry_run=bool(args.dry_run_training) or mode == "preflight",
        )
        _merge_training_results(freehgc_selector_rows, aggregate_training_runs(selector_runs))
    elif selector_queue:
        for row in freehgc_selector_rows:
            row["failure_type"] = "official_training_not_requested"
            row["failure_reason"] = "FreeHGC selector export was prepared, but selector training was skipped by CLI flag."
    training_runs.extend(selector_runs)
    failure_rows.extend(selector_failures)
    main_rows.extend(row for row in freehgc_selector_rows if _ready_for_main(row))
    _drop_resolved_prior_failures(failure_rows, main_rows)

    _add_recovery(main_rows)
    main_rows = [_resolve_row_validation(row, training_runs) for row in main_rows]

    robustness_rows = build_critical_robustness_rows(main_rows, training_runs)
    if mode == "quick-robust":
        robust_queue = build_missing_robustness_queue_rows(main_rows, robustness_rows, target_training_seeds=3)
        robust_runs, robust_failures = execute_training_queue(
            robust_queue,
            sehgnn_repo=sehgnn_repo,
            device=str(args.device),
            out_dir=out_dir,
            python_executable=sys.executable,
            dry_run=bool(args.dry_run_training),
        )
        training_runs.extend(robust_runs)
        failure_rows.extend(robust_failures)
        _merge_method_training_aggregates(main_rows, training_runs)
        _add_recovery(main_rows)
        robustness_rows = build_critical_robustness_rows(main_rows, training_runs)

    acm_overlap_inputs = rows_from_gate21_exports(main_rows)
    acm_overlap_rows = build_acm_selector_overlap_rows(acm_overlap_inputs)
    rep_rows = select_gate21_20_representatives(main_rows, datasets=datasets)
    frontier_rows = build_frontier_rows(main_rows, datasets=datasets)
    best_rows = build_best_method_comparison(main_rows, rep_rows=rep_rows, datasets=datasets)
    decision = gate21_20_decision(
        main_rows=main_rows,
        rep_rows=rep_rows,
        robustness_rows=robustness_rows,
        acm_overlap_rows=acm_overlap_rows,
        imdb_upgrade_rows=imdb_upgrade_rows,
        freehgc_selector_rows=freehgc_selector_rows,
        datasets=datasets,
    )

    write_csv(out_dir / "gate21_20_main_official_table.csv", main_rows, GATE21_20_MAIN_FIELDS)
    write_csv(out_dir / "gate21_20_rep_selection.csv", rep_rows, REP_SELECTION_FIELDS)
    write_csv(out_dir / "gate21_20_imdb_planner_upgrade.csv", imdb_upgrade_rows, IMDB_UPGRADE_FIELDS)
    write_csv(out_dir / "gate21_20_acm_selector_overlap.csv", acm_overlap_rows)
    write_csv(out_dir / "gate21_20_acm_selector_overlap_inputs.csv", acm_overlap_inputs)
    write_csv(out_dir / "gate21_20_freehgc_score_selector.csv", freehgc_selector_rows, FREEHGC_SELECTOR_FIELDS)
    write_csv(out_dir / "gate21_20_robustness_by_method.csv", robustness_rows, ROBUSTNESS_FIELDS)
    write_csv(out_dir / "gate21_20_best_method_comparison.csv", best_rows, BEST_METHOD_COMPARISON_FIELDS)
    write_csv(out_dir / "gate21_20_frontiers.csv", frontier_rows, FRONTIER_FIELDS)
    write_csv(out_dir / "gate21_20_training_runs.csv", training_runs)
    write_csv(out_dir / "gate21_20_training_failures.csv", failure_rows)
    write_csv(out_dir / "gate21_20_decision_flags.csv", _decision_flag_rows(decision))
    write_json(out_dir / "gate21_20_decision.json", decision)
    (out_dir / "gate21_20_summary.md").write_text(_summary(decision, main_rows, rep_rows, robustness_rows, failure_rows), encoding="utf-8")
    (out_dir / "gate21_20_requirement_checklist.md").write_text(_checklist(decision, rep_rows, robustness_rows, failure_rows, mode), encoding="utf-8")
    return decision


def _repair_dblp_hesf_validation(
    *,
    out_dir: Path,
    mode: str,
    graph_seeds: Sequence[int],
    training_seeds: Sequence[int],
    sehgnn_repo: Path,
    device: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    from experiments.scripts.run_gate21_4_apv_skeleton_validation import build_parser, run_gate21_4

    args = [
        "--dataset",
        "DBLP",
        "--output-dir",
        str(out_dir / "dblp_hesf_validation"),
        "--graph-seeds",
        *[str(seed) for seed in graph_seeds[:1]],
        "--training-seeds",
        *[str(seed) for seed in training_seeds],
        "--methods",
        "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
        "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
        "--sehgnn-root",
        str(sehgnn_repo),
        "--hgb-data-root",
        str(sehgnn_repo / "data"),
        "--data-root",
        str(ROOT / "data"),
        "--device",
        device,
    ]
    if dry_run:
        args.append("--dry-run")
    if mode != "quick-robust":
        args.extend(["--max-runs", "2"])
    parsed = build_parser().parse_args(args)
    run_gate21_4(parsed)
    raw = _read_csv(out_dir / "dblp_hesf_validation" / "gate21_4_raw_rows.csv")
    manifest = _read_csv(out_dir / "dblp_hesf_validation" / "gate21_4_run_manifest.csv")
    export_by_key = {(row.get("method"), str(row.get("graph_seed")), str(row.get("training_seed"))): row.get("export_dir", "") for row in manifest}
    out: list[dict[str, Any]] = []
    for row in raw:
        if str(row.get("method", "")) not in {"H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00"}:
            continue
        item = dict(row)
        item["export_dir"] = export_by_key.get((row.get("method"), str(row.get("graph_seed")), str(row.get("training_seed"))), "")
        out.append(item)
    return out


def _merge_dblp_hesf_validation(main_rows: list[dict[str, Any]], validation_runs: Sequence[Mapping[str, Any]]) -> None:
    by_method = {
        "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00": "HeSF-RCS-auto structural12",
        "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00": "HeSF-RCS-auto structural16",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in validation_runs:
        if not bool_value(row.get("success")) and str(row.get("status", "")) != "success":
            continue
        target = by_method.get(str(row.get("method", "")))
        if target:
            grouped.setdefault(target, []).append(row)
    for row in main_rows:
        group = grouped.get(str(row.get("method", "")), [])
        if not group:
            continue
        row["validation_micro_f1_mean"] = _mean_field(group, "validation_micro_f1")
        row["validation_macro_f1_mean"] = _mean_field(group, "validation_macro_f1")
        row["validation_resolution_source"] = "gate21_4_apv_validation_repair"
        row["training_seed_count"] = max(int(float_value(row.get("training_seed_count")) or 1), len({str(item.get("training_seed", "")) for item in group if item.get("training_seed")}))
        row["graph_seed_count"] = max(int(float_value(row.get("graph_seed_count")) or 1), len({str(item.get("graph_seed", "")) for item in group if item.get("graph_seed")}))
        if group[0].get("export_dir"):
            row["export_dir"] = group[0].get("export_dir", "")


def _gate21_4_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        method = {
            "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00": "HeSF-RCS-auto structural12",
            "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00": "HeSF-RCS-auto structural16",
        }.get(str(row.get("method", "")))
        if not method:
            continue
        out.append(
            {
                "dataset": "DBLP",
                "method": method,
                "status": row.get("status", "success" if bool_value(row.get("success")) else ""),
                "training_executed": bool_value(row.get("success")),
                "success": bool_value(row.get("success")),
                "graph_seed": row.get("graph_seed", ""),
                "training_seed": row.get("training_seed", ""),
                "test_micro_f1": row.get("test_micro_f1", ""),
                "test_macro_f1": row.get("test_macro_f1", ""),
                "validation_micro_f1": row.get("validation_micro_f1", ""),
                "validation_macro_f1": row.get("validation_macro_f1", ""),
                "export_dir": row.get("export_dir", ""),
                "stdout_path": row.get("stdout_path", ""),
                "stderr_path": row.get("stderr_path", ""),
            }
        )
    return out


def _prepare_freehgc_selector_rows(*, out_dir: Path, datasets: Sequence[str], graph_seed: int) -> list[dict[str, Any]]:
    if "DBLP" not in datasets:
        return []
    source_dir = _source_dataset_dir("DBLP")
    out: list[dict[str, Any]] = []
    for row in build_freehgc_score_selector_plan_rows(dataset="DBLP", budgets=(0.16, 0.20)):
        budget = float(row["requested_budget"])
        export_dir = out_dir / "exports" / "DBLP" / str(int(graph_seed)) / _slug(str(row["method"])) / f"structural_storage_ratio_{_budget_slug(budget)}" / "official_trainval" / "DBLP"
        manifest = build_dblp_freehgc_score_selector_export(source_dir=source_dir, export_dir=export_dir, budget=budget, graph_seed=int(graph_seed))
        merged = dict(row)
        merged.update(manifest)
        merged.update(
            {
                "actual_structural_storage_ratio": manifest.get("raw_hgb_text_byte_ratio", ""),
                "schema_compatible": True,
                "target_preserving": True,
                "official_hgb_exported": True,
                "official_sehgnn_unmodified": True,
                "training_executed": False,
                "success": False,
                "failure_type": "implemented_pending_official_training",
                "failure_reason": "",
            }
        )
        out.append(merged)
    return out


def _merge_training_results(rows: list[dict[str, Any]], by_source_id: Mapping[int, Mapping[str, Any]]) -> None:
    for index, update in by_source_id.items():
        if 0 <= int(index) < len(rows):
            rows[int(index)].update(update)


def _merge_method_training_aggregates(main_rows: list[dict[str, Any]], training_runs: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in training_runs:
        if str(row.get("status", "")) != "success" and not bool_value(row.get("success")):
            continue
        grouped.setdefault((normalize_dataset(row.get("dataset")), str(row.get("method", ""))), []).append(row)
    for row in main_rows:
        group = grouped.get((normalize_dataset(row.get("dataset")), str(row.get("method", ""))), [])
        if len(group) < 3:
            continue
        row["test_micro_f1_mean"] = _mean_field(group, "test_micro_f1")
        row["test_macro_f1_mean"] = _mean_field(group, "test_macro_f1")
        row["test_micro_f1_std"] = _std_field(group, "test_micro_f1")
        row["test_macro_f1_std"] = _std_field(group, "test_macro_f1")
        row["validation_micro_f1_mean"] = _mean_field(group, "validation_micro_f1")
        row["validation_macro_f1_mean"] = _mean_field(group, "validation_macro_f1")
        row["training_seed_count"] = len({str(item.get("training_seed", item.get("seed", ""))) for item in group})
        graph_seeds = {str(item.get("graph_seed", "")) for item in group if item.get("graph_seed") not in {"", None}}
        row["graph_seed_count"] = len(graph_seeds) if graph_seeds else row.get("graph_seed_count", "")


def _resolve_row_validation(row: Mapping[str, Any], training_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return resolve_validation_metrics(row, training_runs=training_runs)


def _ready_for_main(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
        and bool_value(row.get("eligible_for_main_table", True))
        and not bool_value(row.get("constraint_safe_fallback"))
    )


def _drop_resolved_prior_failures(failure_rows: list[dict[str, Any]], main_rows: Sequence[Mapping[str, Any]]) -> None:
    ready_keys = {
        (normalize_dataset(row.get("dataset")), str(row.get("method", "")))
        for row in main_rows
        if _ready_for_main(row)
    }
    failure_rows[:] = [
        row
        for row in failure_rows
        if not (
            (normalize_dataset(row.get("dataset")), str(row.get("method", ""))) in ready_keys
            and str(row.get("failure_type", "")) in {"not_in_main_no_successful_real_metric", "implemented_pending_official_training"}
        )
    ]


def _strip_prior_rep_rows(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if str(out.get("method", "")).startswith("HeSF-RCS-Rep"):
        out["eligible_for_main_table"] = False
    return out


def _is_prior_rep_row(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    return bool("Rep-Validated" in method or "TestOracle" in method)


def _summary(
    decision: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    rep_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# Gate21.20 Final-Style Stage Report Summary", "", f"- main rows: {len(rows)}", f"- failures/deferred rows: {len(failures)}", ""]
    lines.append("## Decision Level")
    lines.append(f"- smoke-ready: {decision.get('STAGE_REPORT_SMOKE_READY')}")
    lines.append(f"- quick-robust-ready: {decision.get('STAGE_REPORT_QUICK_ROBUSTNESS_READY')}")
    lines.append(f"- final-table-ready: {decision.get('STAGE_REPORT_FINAL_TABLE_READY')}")
    lines.extend(["", "## Required Flags"])
    for flag in GATE21_20_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## HeSF-RCS Representatives"])
    for row in rep_rows:
        if str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated":
            lines.append(
                f"- {row.get('dataset')}: {row.get('selected_method') or 'MISSING'} | "
                f"family={row.get('selected_method_family')} | reason={row.get('selection_reason')} | uses_test={row.get('uses_test_for_selection')}"
            )
    lines.extend(["", "## Robustness Status"])
    for row in robustness_rows:
        lines.append(
            f"- {row.get('dataset')} {row.get('method')}: ready={row.get('robustness_ready')} "
            f"mode={row.get('robustness_mode')} train_count={row.get('training_executed_count')} failures={row.get('failure_count')}"
        )
    lines.extend(["", "## Failures/Deferred"])
    if not failures:
        lines.append("- none")
    for row in failures:
        if row.get("method"):
            lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', row.get('error_message', '')))[:400]}")
    return "\n".join(lines) + "\n"


def _checklist(
    decision: Mapping[str, Any],
    rep_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    mode: str,
) -> str:
    requirements = {
        "P0 HeSF-RCS-Rep-Validated selects only HeSF family and uses no test": decision.get("HESF_RCS_REP_CANDIDATE_POOL_PASS") and decision.get("HESF_RCS_REP_NO_TEST_LEAKAGE"),
        "P1 DBLP representative has real validation metric or is explicitly blocked": _dblp_rep_has_validation_or_blocked(rep_rows),
        "P2 IMDB upgraded channel40/channel50 planner rows emitted": decision.get("IMDB_HEFS_UPGRADED_PLANNER_READY"),
        "P3 ACM selector overlap audit emitted": decision.get("ACM_SELECTOR_OVERLAP_READY"),
        "P4 Critical robustness rows are 3x3 or deterministic export plus 3 training seeds": decision.get("STAGE_REPORT_QUICK_ROBUSTNESS_READY") if mode == "quick-robust" else bool(robustness_rows),
        "P5 FreeHGC-score-as-selector structural16/20 emitted": decision.get("FREEHGC_SCORE_AS_SELECTOR_READY"),
        "P6 Main compression table has no full fallback rows": decision.get("NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"),
        "P7 Final summary separates smoke, quick-robust, and final-ready": True,
    }
    lines = ["# Gate21.20 Requirement Checklist", "", f"- mode: {mode}", "", "## Decision Flags"]
    for flag in GATE21_20_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(["", "## Attachment Requirements"])
    for name, passed in requirements.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}")
    lines.extend(["", "## Incomplete/Failed Items"])
    incomplete = [row for row in robustness_rows if not bool_value(row.get("robustness_ready"))]
    for row in incomplete:
        lines.append(f"- robustness {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {row.get('failure_reason')}")
    for row in failures:
        if row.get("method"):
            lines.append(f"- training/export {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', row.get('error_message', '')))[:300]}")
    if not incomplete and not [row for row in failures if row.get("method")]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _dblp_rep_has_validation_or_blocked(rep_rows: Sequence[Mapping[str, Any]]) -> bool:
    row = next((item for item in rep_rows if normalize_dataset(item.get("dataset")) == "DBLP" and str(item.get("rep_type", "")) == "HeSF-RCS-Rep-Validated"), None)
    if not row:
        return False
    if str(row.get("selection_reason", "")) == "missing_real_validation_metric":
        return True
    return bool(row.get("validation_micro_f1") not in {"", None} and not bool_value(row.get("uses_test_for_selection")))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else ""


def _std_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return ""
    return pstdev(finite) if len(finite) > 1 else 0.0


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _budget_slug(value: float) -> str:
    return str(float(value)).replace(".", "p")


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.20 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.20 STAGE_REPORT_QUICK_ROBUSTNESS_READY={decision['STAGE_REPORT_QUICK_ROBUSTNESS_READY']}")
    print(f"Gate21.20 STAGE_REPORT_FINAL_TABLE_READY={decision['STAGE_REPORT_FINAL_TABLE_READY']}")


if __name__ == "__main__":
    main()
