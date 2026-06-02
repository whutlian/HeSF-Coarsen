from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.external_repo_manager import audit_required_external_repos
from hesf_coarsen.eval.official.external_tp_5x5_runner import build_gate21_15_external_tp_rows
from hesf_coarsen.eval.official.freehgc_score_tp_adapter import build_freehgc_score_tp_rows
from hesf_coarsen.eval.official.freehgc_standard_runner import build_gate21_15_freehgc_standard_rows
from hesf_coarsen.eval.official.gcond_standard_runner import build_gcond_standard_rows
from hesf_coarsen.eval.official.hgcond_standard_runner import build_hgcond_standard_rows
from hesf_coarsen.eval.official.relation_structural_baselines import build_relation_structural_baseline_rows
from hesf_coarsen.eval.official.stage_report_protocol import (
    DATASETS,
    EXTERNAL_TP_BASELINES,
    FULL_METHODS,
    INTERNAL_BASELINES,
    STRUCTURAL_BUDGETS,
    SUPPORT_NODE_BUDGETS,
    bool_value,
    failure_main_row,
    finite_metric,
    float_value,
    normalize_dataset,
    select_hesf_rcs_representatives,
    success_main_row,
)
from hesf_coarsen.eval.official.stage_report_summarizer import write_gate21_15_artifacts


ROOT = Path(__file__).resolve().parents[2]
GATE21_0 = ROOT / "outputs" / "gate21_0_sehgnn_native_export"
GATE21_14 = ROOT / "outputs" / "gate21_14_full_execution_push"
GATE21_14_H6 = ROOT / "outputs" / "gate21_14_cross_h6_training"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Gate21.15 DBLP/ACM/IMDB stage-report benchmark tables.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--budgets", nargs="+", type=float, default=list(STRUCTURAL_BUDGETS))
    parser.add_argument("--support-node-budgets", nargs="+", type=float, default=list(SUPPORT_NODE_BUDGETS))
    parser.add_argument("--mode", choices=("quick", "paper", "dry-run"), default="quick")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--run-internal-baselines", action="store_true")
    parser.add_argument("--run-structural-baselines", action="store_true")
    parser.add_argument("--run-external-tp", action="store_true")
    parser.add_argument("--run-standard-condensation", action="store_true")
    parser.add_argument("--clone-missing-baselines", action="store_true")
    parser.add_argument("--external-repos-dir", default="external_repos")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="outputs/gate21_15_stage_report")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in args.datasets)
    budgets = tuple(float(item) for item in args.budgets)
    support_node_budgets = tuple(float(item) for item in args.support_node_budgets)
    run_full, run_internal, run_structural, run_external_tp, run_standard = _resolve_run_switches(args)

    repo_rows = audit_required_external_repos(args.external_repos_dir, clone_missing=bool(args.clone_missing_baselines))
    structural_rows = (
        build_relation_structural_baseline_rows(datasets=datasets, budgets=budgets, mode=args.mode)
        if run_structural or args.mode == "dry-run"
        else []
    )
    external_tp_rows = []
    if run_external_tp or args.mode == "dry-run":
        external_methods = tuple(method for method in EXTERNAL_TP_BASELINES if method != "FreeHGC-score-TP")
        external_tp_rows.extend(
            build_gate21_15_external_tp_rows(
                datasets=list(datasets),
                methods=external_methods,
                support_node_budgets=list(support_node_budgets),
                structural_budgets=[budget for budget in budgets if budget in {0.30, 0.20, 0.16}],
                mode=args.mode if args.mode != "dry-run" else "quick",
            )
        )
        external_tp_rows.extend(
            build_freehgc_score_tp_rows(
                datasets=datasets,
                support_node_budgets=support_node_budgets,
                structural_budgets=[budget for budget in budgets if budget in {0.30, 0.20, 0.16}],
                repo_audit_rows=repo_rows,
            )
        )
    freehgc_rows = build_gate21_15_freehgc_standard_rows(datasets=datasets, repo_audit_rows=repo_rows) if run_standard or args.mode == "dry-run" else []
    hgcond_gcond_rows = []
    if run_standard or args.mode == "dry-run":
        hgcond_gcond_rows.extend(build_hgcond_standard_rows(datasets=datasets, repo_audit_rows=repo_rows))
        hgcond_gcond_rows.extend(build_gcond_standard_rows(datasets=datasets, repo_audit_rows=repo_rows))

    if args.mode == "dry-run":
        main_rows = _dry_run_main_rows(datasets=datasets, budgets=budgets, support_node_budgets=support_node_budgets)
    else:
        evidence = _load_evidence()
        main_rows = []
        if run_full:
            main_rows.extend(_full_rows(datasets=datasets, evidence=evidence))
        if run_internal:
            main_rows.extend(_internal_rows(datasets=datasets, evidence=evidence))
        if run_structural:
            main_rows.extend(_structural_main_rows(structural_rows))
        if run_external_tp:
            main_rows.extend(_external_tp_main_rows(external_tp_rows))
        main_rows.extend(_hesf_auto_rows(datasets=datasets, budgets=budgets, evidence=evidence))

    rep_rows = select_hesf_rcs_representatives(main_rows, datasets=datasets)
    main_rows.extend(_rep_main_rows(rep_rows=rep_rows, main_rows=main_rows, datasets=datasets))

    return write_gate21_15_artifacts(
        output_dir=args.output_dir,
        main_rows=main_rows,
        rep_rows=rep_rows,
        structural_rows=structural_rows,
        external_tp_rows=external_tp_rows,
        external_repo_rows=repo_rows,
        freehgc_standard_rows=freehgc_rows,
        hgcond_gcond_rows=hgcond_gcond_rows,
        datasets=datasets,
    )


def _resolve_run_switches(args: argparse.Namespace) -> tuple[bool, bool, bool, bool, bool]:
    selected = [args.run_full, args.run_internal_baselines, args.run_structural_baselines, args.run_external_tp, args.run_standard_condensation]
    if not any(selected):
        return True, True, True, True, True
    return (
        bool(args.run_full),
        bool(args.run_internal_baselines),
        bool(args.run_structural_baselines),
        bool(args.run_external_tp),
        bool(args.run_standard_condensation),
    )


def _dry_run_main_rows(*, datasets: tuple[str, ...], budgets: tuple[float, ...], support_node_budgets: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in FULL_METHODS:
            rows.append(_planned_failure(dataset, method, "full_fidelity_baseline", "dry_run_planned", "Dry-run plan row; no training executed."))
        for method in INTERNAL_BASELINES:
            rows.append(
                _planned_failure(
                    dataset,
                    method,
                    "internal_historical_baseline",
                    "support_node_ratio",
                    0.30,
                    "dry_run_planned",
                    "Dry-run support-node baseline row; no training executed.",
                )
            )
        for method in ("Random-edge-relwise", "Degree-edge-relwise", "Proportional-relation-budget"):
            for budget in budgets:
                rows.append(_planned_failure(dataset, method, "relation_structural_baseline", "structural_storage_ratio", budget, "dry_run_planned", "Dry-run structural baseline row; no training executed."))
        for method in EXTERNAL_TP_BASELINES:
            for support in support_node_budgets:
                rows.append(_planned_failure(dataset, method, "external_tp_baseline", "support_node_ratio", support, "dry_run_planned", "Dry-run external TP row; no training executed."))
            for budget in [item for item in budgets if item in {0.30, 0.20, 0.16}]:
                rows.append(_planned_failure(dataset, method, "external_tp_baseline", "structural_storage_ratio", budget, "dry_run_planned", "Dry-run external TP row; no training executed."))
        for budget in budgets:
            rows.append(_planned_failure(dataset, _hesf_auto_name(budget), "schema_preserving_rcs", "structural_storage_ratio", budget, "dry_run_planned", "Dry-run HeSF-RCS-auto row; no training executed."))
        rows.append(_planned_failure(dataset, "HeSF-RCS-Rep", "schema_preserving_rcs", "", "", "dry_run_planned", "Dry-run representative row; validation selection not executed."))
    return rows


def _planned_failure(
    dataset: str,
    method: str,
    method_family: str,
    requested_budget_type: object = "",
    requested_budget: object = "",
    failure_type: str = "not_executed",
    failure_reason: str = "",
) -> dict[str, Any]:
    return failure_main_row(
        dataset=dataset,
        method=method,
        method_family=method_family,
        requested_budget_type=requested_budget_type,
        requested_budget=requested_budget,
        failure_type=failure_type,
        failure_reason=failure_reason,
        official_hgb_exported=False,
    )


def _load_evidence() -> dict[str, Any]:
    native_runs = _read_csv(GATE21_0 / "native" / "native_metrics.csv")
    export_runs = _read_csv(GATE21_0 / "fidelity" / "gate21_0_export_full_metrics.csv")
    compressed_runs = _read_csv(GATE21_0 / "compressed" / "gate21_0_compressed_metrics.csv")
    cross_h6_runs = _read_csv(GATE21_14_H6 / "compressed" / "gate21_0_compressed_metrics.csv")
    official_main = _read_csv(GATE21_14 / "gate21_14_official_main_by_method.csv")
    selectors = _read_csv(GATE21_14 / "gate21_14_budgeted_selector_by_method.csv")
    return {
        "native": _aggregate_native(native_runs),
        "export": _aggregate_export(export_runs),
        "compressed": _aggregate_compressed([*compressed_runs, *cross_h6_runs]),
        "compressed_failures": _compressed_failures([*compressed_runs, *cross_h6_runs]),
        "official_main": official_main,
        "selectors": selectors,
    }


def _full_rows(*, datasets: tuple[str, ...], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        native = evidence["native"].get(dataset)
        if native:
            rows.append(
                success_main_row(
                    dataset=dataset,
                    method="Full-native-SeHGNN",
                    method_family="full_fidelity_baseline",
                    test_micro_f1_mean=native["test_micro_f1_mean"],
                    test_micro_f1_std=native["test_micro_f1_std"],
                    test_macro_f1_mean=native["test_macro_f1_mean"],
                    test_macro_f1_std=native["test_macro_f1_std"],
                    graph_seed_count=1,
                    training_seed_count=native["training_seed_count"],
                    recovery_vs_native_full_micro=1.0,
                    recovery_vs_native_full_macro=1.0,
                    full_minus_micro=0.0,
                    full_minus_macro=0.0,
                    source_gate="gate21_0",
                    source_path=str(GATE21_0 / "native" / "native_metrics.csv"),
                )
            )
        else:
            rows.append(_planned_failure(dataset, "Full-native-SeHGNN", "full_fidelity_baseline", failure_type="missing_native_metric", failure_reason="No native SeHGNN metric found."))
        export = evidence["export"].get(dataset)
        if export:
            rows.append(
                success_main_row(
                    dataset=dataset,
                    method="Export-full-SeHGNN",
                    method_family="full_fidelity_baseline",
                    test_micro_f1_mean=export["test_micro_f1_mean"],
                    test_micro_f1_std=export["test_micro_f1_std"],
                    test_macro_f1_mean=export["test_macro_f1_mean"],
                    test_macro_f1_std=export["test_macro_f1_std"],
                    graph_seed_count=1,
                    training_seed_count=export["training_seed_count"],
                    recovery_vs_native_full_micro=1.0,
                    recovery_vs_native_full_macro=1.0,
                    full_minus_micro=0.0,
                    full_minus_macro=0.0,
                    source_gate="gate21_0",
                    source_path=str(GATE21_0 / "fidelity" / "gate21_0_export_full_metrics.csv"),
                )
            )
        else:
            rows.append(_planned_failure(dataset, "Export-full-SeHGNN", "full_fidelity_baseline", failure_type="missing_export_metric", failure_reason="No export-full SeHGNN metric found."))
    return rows


def _internal_rows(*, datasets: tuple[str, ...], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in INTERNAL_BASELINES:
            metric = evidence["compressed"].get((dataset, method))
            if metric:
                rows.append(
                    success_main_row(
                        dataset=dataset,
                        method=method,
                        method_family="internal_historical_baseline",
                        requested_budget_type="support_node_ratio",
                        requested_budget=0.30,
                        actual_structural_storage_ratio=metric["actual_structural_storage_ratio"],
                        support_node_ratio=metric["support_node_ratio"],
                        support_edge_ratio=metric["support_edge_ratio"],
                        raw_hgb_text_byte_ratio=metric["actual_structural_storage_ratio"],
                        graph_seed_count=1,
                        training_seed_count=metric["training_seed_count"],
                        test_micro_f1_mean=metric["test_micro_f1_mean"],
                        test_micro_f1_std=metric["test_micro_f1_std"],
                        test_macro_f1_mean=metric["test_macro_f1_mean"],
                        test_macro_f1_std=metric["test_macro_f1_std"],
                        validation_micro_f1_mean=metric["validation_micro_f1_mean"],
                        validation_macro_f1_mean=metric["validation_macro_f1_mean"],
                        recovery_vs_native_full_micro=metric["recovery_vs_native_full_micro"],
                        recovery_vs_native_full_macro=metric["recovery_vs_native_full_macro"],
                        source_gate=metric["source_gate"],
                        source_path=metric["source_path"],
                    )
                )
            else:
                failure = evidence["compressed_failures"].get((dataset, method), {})
                rows.append(
                    failure_main_row(
                        dataset=dataset,
                        method=method,
                        method_family="internal_historical_baseline",
                        requested_budget_type="support_node_ratio",
                        requested_budget=0.30,
                        actual_structural_storage_ratio=failure.get("total_storage_ratio_vs_full_graph", ""),
                        support_node_ratio=failure.get("support_node_ratio", 0.30),
                        support_edge_ratio=failure.get("support_edge_ratio", ""),
                        failure_type=failure.get("status", "missing_task_metric"),
                        failure_reason=failure.get("error_message", f"{method} has no official SeHGNN task metric for {dataset}."),
                        official_hgb_exported=bool(failure),
                        source_gate=failure.get("source_gate", ""),
                    )
                )
    return rows


def _structural_main_rows(structural_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        failure_main_row(
            dataset=str(row.get("dataset", "")),
            method=str(row.get("method", "")),
            method_family="relation_structural_baseline",
            requested_budget_type=row.get("requested_budget_type", ""),
            requested_budget=row.get("requested_budget", ""),
            failure_type=str(row.get("failure_type", "not_executed")),
            failure_reason=str(row.get("failure_reason", "")),
            official_hgb_exported=bool_value(row.get("official_hgb_exported")),
        )
        for row in structural_rows
    ]


def _external_tp_main_rows(external_tp_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        failure_main_row(
            dataset=str(row.get("dataset", "")),
            method=str(row.get("method", "")),
            method_family="external_tp_baseline",
            requested_budget_type=row.get("requested_budget_type", row.get("budget_type", "")),
            requested_budget=row.get("requested_budget", row.get("budget_value", "")),
            support_node_ratio=row.get("support_node_ratio", ""),
            failure_type=str(row.get("failure_type", "not_executed")),
            failure_reason=str(row.get("failure_reason", row.get("failure_message", ""))),
            official_hgb_exported=bool_value(row.get("official_hgb_exported")),
            repo_url=str(row.get("repo_url", "")),
        )
        for row in external_tp_rows
    ]


def _hesf_auto_rows(*, datasets: tuple[str, ...], budgets: tuple[float, ...], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    official_by_method = {(row.get("dataset", ""), row.get("method", "")): row for row in evidence["official_main"]}
    selectors = [row for row in evidence["selectors"] if str(row.get("requested_structural_budget", ""))]
    for dataset in datasets:
        for budget in budgets:
            method = _hesf_auto_name(budget)
            if dataset == "DBLP":
                selector = _selector_for_budget(selectors, budget)
                linked_method = selector.get("linked_official_task_method") or selector.get("selected_canonical_method")
                linked = official_by_method.get(("DBLP", linked_method))
                if selector and linked:
                    rows.append(
                        success_main_row(
                            dataset=dataset,
                            method=method,
                            method_family="schema_preserving_rcs",
                            requested_budget_type="structural_storage_ratio",
                            requested_budget=budget,
                            actual_structural_storage_ratio=selector.get("actual_structural_storage_ratio", linked.get("actual_structural_storage_ratio", linked.get("structural_storage_ratio", ""))),
                            support_node_ratio=linked.get("support_node_ratio", ""),
                            support_edge_ratio=linked.get("support_edge_ratio", linked.get("actual_support_edge_ratio", "")),
                            raw_hgb_text_byte_ratio=linked.get("raw_hgb_text_byte_ratio", ""),
                            graph_seed_count=linked.get("graph_seed_count", 1),
                            training_seed_count=linked.get("training_seed_count", linked.get("success_count", 5)),
                            test_micro_f1_mean=linked.get("test_micro_f1_mean", linked.get("test_micro_mean", "")),
                            test_micro_f1_std=linked.get("test_micro_f1_std", linked.get("test_micro_std", "")),
                            test_macro_f1_mean=linked.get("test_macro_f1_mean", linked.get("test_macro_mean", "")),
                            test_macro_f1_std=linked.get("test_macro_f1_std", linked.get("test_macro_std", "")),
                            validation_micro_f1_mean=linked.get("validation_micro_f1_mean", ""),
                            validation_macro_f1_mean=linked.get("validation_macro_f1_mean", ""),
                            recovery_vs_native_full_micro=linked.get("recovery_micro_mean", ""),
                            recovery_vs_native_full_macro=linked.get("recovery_macro_mean", ""),
                            full_minus_micro=linked.get("full_minus_micro", ""),
                            full_minus_macro=linked.get("full_minus_macro", ""),
                            source_gate="gate21_14",
                            source_path=str(GATE21_14 / "gate21_14_budgeted_selector_by_method.csv"),
                            selected_edge_hash=selector.get("selected_edge_hash", ""),
                            planner_config_hash=selector.get("selection_config_hash", selector.get("selected_plan_hash", "")),
                        )
                    )
                    continue
            rows.append(
                failure_main_row(
                    dataset=dataset,
                    method=method,
                    method_family="schema_preserving_rcs",
                    requested_budget_type="structural_storage_ratio",
                    requested_budget=budget,
                    failure_type="missing_official_task_metric",
                    failure_reason=f"No official-unmodified HeSF-RCS-auto task metric with validation fields exists for {dataset} at structural budget {budget:.2f}.",
                    official_hgb_exported=False,
                )
            )
    return rows


def _rep_main_rows(*, rep_rows: list[Mapping[str, Any]], main_rows: list[Mapping[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main_by_method = {(row.get("dataset", ""), row.get("method", "")): dict(row) for row in main_rows}
    for dataset in datasets:
        selected = next((row for row in rep_rows if normalize_dataset(row.get("dataset")) == dataset and bool_value(row.get("selected_as_rep"))), None)
        if selected:
            source = main_by_method.get((dataset, selected.get("candidate_method", "")), {})
            rep = dict(source)
            rep["method"] = "HeSF-RCS-Rep"
            rep["method_family"] = "schema_preserving_rcs"
            rep["source_method"] = selected.get("candidate_method", "")
            rows.append(rep)
        else:
            rows.append(
                failure_main_row(
                    dataset=dataset,
                    method="HeSF-RCS-Rep",
                    method_family="schema_preserving_rcs",
                    failure_type="validation_metrics_missing",
                    failure_reason="No HeSF-RCS-auto representative can be selected without validation metrics; test metrics were not used.",
                    official_hgb_exported=False,
                )
            )
    return rows


def _hesf_auto_name(budget: float) -> str:
    return f"HeSF-RCS-auto structural{int(round(float(budget) * 100)):02d}"


def _selector_for_budget(selectors: list[dict[str, str]], budget: float) -> dict[str, str]:
    for row in selectors:
        value = float_value(row.get("requested_structural_budget"))
        if value is not None and abs(value - float(budget)) < 1e-9:
            return row
    return {}


def _aggregate_native(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dataset = normalize_dataset(row.get("dataset"))
        if finite_metric(row.get("test_micro_f1")) and finite_metric(row.get("test_macro_f1")):
            grouped.setdefault(dataset, []).append(row)
    return {
        dataset: {
            "training_seed_count": len(group),
            "test_micro_f1_mean": _mean(group, "test_micro_f1"),
            "test_micro_f1_std": _std(group, "test_micro_f1"),
            "test_macro_f1_mean": _mean(group, "test_macro_f1"),
            "test_macro_f1_std": _std(group, "test_macro_f1"),
        }
        for dataset, group in grouped.items()
    }


def _aggregate_export(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dataset = normalize_dataset(row.get("dataset"))
        if finite_metric(row.get("test_micro_f1")) and finite_metric(row.get("test_macro_f1")):
            grouped.setdefault(dataset, []).append(row)
    return {
        dataset: {
            "training_seed_count": len(group),
            "test_micro_f1_mean": _mean(group, "test_micro_f1"),
            "test_micro_f1_std": _std(group, "test_micro_f1"),
            "test_macro_f1_mean": _mean(group, "test_macro_f1"),
            "test_macro_f1_std": _std(group, "test_macro_f1"),
        }
        for dataset, group in grouped.items()
    }


def _aggregate_compressed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("status", "success")) != "success":
            continue
        if not (finite_metric(row.get("test_micro_f1")) and finite_metric(row.get("test_macro_f1"))):
            continue
        key = (normalize_dataset(row.get("dataset")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        source = "gate21_14_cross_h6_training" if key[0] in {"ACM", "IMDB"} and key[1] == "H6-node30" else "gate21_0"
        source_path = GATE21_14_H6 / "compressed" / "gate21_0_compressed_metrics.csv" if source.startswith("gate21_14") else GATE21_0 / "compressed" / "gate21_0_compressed_metrics.csv"
        out[key] = {
            "training_seed_count": len(group),
            "support_node_ratio": group[0].get("support_node_ratio", ""),
            "support_edge_ratio": group[0].get("support_edge_ratio", ""),
            "actual_structural_storage_ratio": group[0].get("total_storage_ratio_vs_full_graph", ""),
            "validation_micro_f1_mean": _mean(group, "validation_micro_f1"),
            "validation_macro_f1_mean": _mean(group, "validation_macro_f1"),
            "test_micro_f1_mean": _mean(group, "test_micro_f1"),
            "test_micro_f1_std": _std(group, "test_micro_f1"),
            "test_macro_f1_mean": _mean(group, "test_macro_f1"),
            "test_macro_f1_std": _std(group, "test_macro_f1"),
            "recovery_vs_native_full_micro": _mean(group, "recovery_vs_native_full_micro"),
            "recovery_vs_native_full_macro": _mean(group, "recovery_vs_native_full_macro"),
            "source_gate": source,
            "source_path": str(source_path),
        }
    return out


def _compressed_failures(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    failures: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if str(row.get("status", "success")) == "success":
            continue
        key = (normalize_dataset(row.get("dataset")), str(row.get("method", "")))
        if key not in failures:
            out = dict(row)
            out["source_gate"] = "gate21_14_cross_h6_training"
            failures[key] = out
    return failures


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else "NaN"


def _std(rows: Iterable[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return "NaN"
    return pstdev(finite) if len(finite) > 1 else 0.0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    decision = run(args)
    print(f"Gate21.15 STAGE_REPORT_TABLE_READY={decision['STAGE_REPORT_TABLE_READY']}")


if __name__ == "__main__":
    main()
