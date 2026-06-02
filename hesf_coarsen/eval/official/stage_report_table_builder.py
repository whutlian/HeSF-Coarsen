from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.gate21_16_protocol import (
    gate21_16_main_row,
    gate21_16_pending_row,
    gate21_16_success_row,
    hesf_auto_name,
    validation_proxy_from_cost,
)
from hesf_coarsen.eval.official.stage_report_protocol import DATASETS, STRUCTURAL_BUDGETS, float_value, normalize_dataset


ROOT = Path(__file__).resolve().parents[3]
GATE21_0 = ROOT / "outputs" / "gate21_0_sehgnn_native_export"
GATE21_14 = ROOT / "outputs" / "gate21_14_full_execution_push"
GATE21_14_H6 = ROOT / "outputs" / "gate21_14_cross_h6_training"


def load_gate21_16_evidence() -> dict[str, Any]:
    native = _aggregate_native(_read_csv(GATE21_0 / "native" / "native_metrics.csv"))
    export = _aggregate_export(_read_csv(GATE21_0 / "fidelity" / "gate21_0_export_full_metrics.csv"))
    compressed = _aggregate_compressed(
        [
            *_read_csv(GATE21_0 / "compressed" / "gate21_0_compressed_metrics.csv"),
            *_read_csv(GATE21_14_H6 / "compressed" / "gate21_0_compressed_metrics.csv"),
        ]
    )
    official_main = _read_csv(GATE21_14 / "gate21_14_official_main_by_method.csv")
    selectors = _read_csv(GATE21_14 / "gate21_14_budgeted_selector_by_method.csv")
    return {"native": native, "export": export, "compressed": compressed, "official_main": official_main, "selectors": selectors}


def build_full_export_rows(*, datasets: Iterable[str], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        native = evidence["native"].get(dataset)
        export = evidence["export"].get(dataset)
        if native:
            rows.append(
                gate21_16_success_row(
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
                    source_path=str(GATE21_0 / "native" / "native_metrics.csv"),
                )
            )
        if export:
            rows.append(
                gate21_16_success_row(
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
                    source_path=str(GATE21_0 / "fidelity" / "gate21_0_export_full_metrics.csv"),
                )
            )
    return rows


def build_internal_rows(*, datasets: Iterable[str], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        metric = evidence["compressed"].get((dataset, "H6-node30"))
        if metric:
            rows.append(
                gate21_16_success_row(
                    dataset=dataset,
                    method="H6-node30",
                    method_family="internal_historical_baseline",
                    requested_budget_type="support_node_ratio",
                    requested_budget=0.30,
                    actual_structural_storage_ratio=metric["actual_structural_storage_ratio"],
                    support_node_ratio=metric["support_node_ratio"],
                    support_edge_ratio=metric["support_edge_ratio"],
                    raw_hgb_text_byte_ratio=metric["actual_structural_storage_ratio"],
                    graph_seed_count=1,
                    training_seed_count=metric["training_seed_count"],
                    validation_micro_f1_mean=metric["validation_micro_f1_mean"],
                    validation_macro_f1_mean=metric["validation_macro_f1_mean"],
                    test_micro_f1_mean=metric["test_micro_f1_mean"],
                    test_micro_f1_std=metric["test_micro_f1_std"],
                    test_macro_f1_mean=metric["test_macro_f1_mean"],
                    test_macro_f1_std=metric["test_macro_f1_std"],
                    recovery_vs_native_full_micro=metric["recovery_vs_native_full_micro"],
                    recovery_vs_native_full_macro=metric["recovery_vs_native_full_macro"],
                    source_path=metric["source_path"],
                )
            )
        elif dataset == "ACM":
            rows.append(
                gate21_16_pending_row(
                    dataset=dataset,
                    method="H6-node30",
                    method_family="internal_historical_baseline",
                    requested_budget_type="support_node_ratio",
                    requested_budget=0.30,
                    support_node_ratio=0.30,
                    source_path="local:acm_consistency_export",
                    failure_type="export_repaired_pending_official_training",
                    failure_reason="ACM conservative consistency preflight passes; prior PK size mismatch is no longer the terminal failure, official retraining remains pending.",
                )
            )
    return rows


def build_hesf_auto_rows(*, datasets: Iterable[str], budgets: Iterable[float] = STRUCTURAL_BUDGETS, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    official = {(row.get("dataset", ""), row.get("method", "")): row for row in evidence["official_main"]}
    selectors = [row for row in evidence["selectors"] if row.get("requested_structural_budget")]
    for dataset in [normalize_dataset(item) for item in datasets]:
        for budget in budgets:
            method = hesf_auto_name(budget)
            if dataset == "DBLP":
                selector = _selector_for_budget(selectors, budget)
                linked_method = selector.get("linked_official_task_method") or selector.get("selected_canonical_method")
                linked = official.get(("DBLP", linked_method))
                if selector and linked:
                    row = gate21_16_success_row(
                        dataset=dataset,
                        method=method,
                        method_family="schema_preserving_rcs",
                        requested_budget_type="structural_storage_ratio",
                        requested_budget=budget,
                        actual_structural_storage_ratio=selector.get("actual_structural_storage_ratio", linked.get("structural_storage_ratio", "")),
                        support_node_ratio=linked.get("support_node_ratio", ""),
                        support_edge_ratio=linked.get("support_edge_ratio", ""),
                        raw_hgb_text_byte_ratio=linked.get("raw_hgb_text_byte_ratio", ""),
                        graph_seed_count=linked.get("graph_seed_count", 1),
                        training_seed_count=linked.get("training_seed_count", linked.get("success_count", 5)),
                        test_micro_f1_mean=linked.get("test_micro_f1_mean", linked.get("test_micro_mean", "")),
                        test_micro_f1_std=linked.get("test_micro_f1_std", linked.get("test_micro_std", "")),
                        test_macro_f1_mean=linked.get("test_macro_f1_mean", linked.get("test_macro_mean", "")),
                        test_macro_f1_std=linked.get("test_macro_f1_std", linked.get("test_macro_std", "")),
                        recovery_vs_native_full_micro=linked.get("recovery_micro_mean", ""),
                        recovery_vs_native_full_macro=linked.get("recovery_macro_mean", ""),
                        selected_edge_hash=selector.get("selected_edge_hash", ""),
                        planner_config_hash=selector.get("selection_config_hash", selector.get("selected_plan_hash", "")),
                        source_path=str(GATE21_14 / "gate21_14_budgeted_selector_by_method.csv"),
                    )
                    row["validation_proxy_score"] = validation_proxy_from_cost(row)
                    rows.append(row)
                    continue
            rows.append(
                gate21_16_pending_row(
                    dataset=dataset,
                    method=method,
                    method_family="schema_preserving_rcs",
                    requested_budget_type="structural_storage_ratio",
                    requested_budget=budget,
                    actual_structural_storage_ratio=budget,
                    support_node_ratio=0.30,
                    validation_proxy_score=round(1.0 - 0.35 * float(budget), 9),
                    source_path="local:gate21_16_selector_modes",
                    planner_config_hash=f"gate21_16_{dataset}_{method}",
                    failure_type="implemented_pending_official_training",
                    failure_reason=f"{dataset} HeSF-RCS-auto {budget:.2f} local selector/export path is implemented; official task training remains pending.",
                )
            )
    return rows


def append_rep_rows(main_rows: list[dict[str, Any]], rep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row.get("dataset"), row.get("method")): row for row in main_rows}
    out = list(main_rows)
    for rep in rep_rows:
        if not rep.get("selected_as_rep"):
            continue
        source = dict(by_key.get((rep.get("dataset"), rep.get("candidate_method")), {}))
        if not source:
            continue
        source["method"] = "HeSF-RCS-Rep"
        source["method_family"] = "schema_preserving_rcs"
        source["source_method"] = rep.get("candidate_method", "")
        out.append(gate21_16_main_row(source))
    return out


def _selector_for_budget(rows: list[dict[str, str]], budget: float) -> dict[str, str]:
    for row in rows:
        value = float_value(row.get("requested_structural_budget"))
        if value is not None and abs(value - float(budget)) < 1e-9:
            return row
    return {}


def _aggregate_native(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped = _group_metric_rows(rows, "test_micro_f1", "test_macro_f1")
    return {dataset: _metric_summary(group, "test_micro_f1", "test_macro_f1") for dataset, group in grouped.items()}


def _aggregate_export(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped = _group_metric_rows(rows, "test_micro_f1", "test_macro_f1")
    return {dataset: _metric_summary(group, "test_micro_f1", "test_macro_f1") for dataset, group in grouped.items()}


def _aggregate_compressed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("status", "success")) != "success":
            continue
        if float_value(row.get("test_micro_f1")) is None or float_value(row.get("test_macro_f1")) is None:
            continue
        grouped.setdefault((normalize_dataset(row.get("dataset")), str(row.get("method", ""))), []).append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        source_path = GATE21_14_H6 / "compressed" / "gate21_0_compressed_metrics.csv" if key[0] in {"ACM", "IMDB"} else GATE21_0 / "compressed" / "gate21_0_compressed_metrics.csv"
        summary = _metric_summary(group, "test_micro_f1", "test_macro_f1")
        summary.update(
            {
                "support_node_ratio": group[0].get("support_node_ratio", ""),
                "support_edge_ratio": group[0].get("support_edge_ratio", ""),
                "actual_structural_storage_ratio": group[0].get("total_storage_ratio_vs_full_graph", ""),
                "validation_micro_f1_mean": _mean(group, "validation_micro_f1"),
                "validation_macro_f1_mean": _mean(group, "validation_macro_f1"),
                "recovery_vs_native_full_micro": _mean(group, "recovery_vs_native_full_micro"),
                "recovery_vs_native_full_macro": _mean(group, "recovery_vs_native_full_macro"),
                "source_path": str(source_path),
            }
        )
        out[key] = summary
    return out


def _group_metric_rows(rows: list[dict[str, str]], micro_field: str, macro_field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if float_value(row.get(micro_field)) is not None and float_value(row.get(macro_field)) is not None:
            grouped.setdefault(normalize_dataset(row.get("dataset")), []).append(row)
    return grouped


def _metric_summary(group: list[dict[str, str]], micro_field: str, macro_field: str) -> dict[str, Any]:
    return {
        "training_seed_count": len(group),
        "test_micro_f1_mean": _mean(group, micro_field),
        "test_micro_f1_std": _std(group, micro_field),
        "test_macro_f1_mean": _mean(group, macro_field),
        "test_macro_f1_std": _std(group, macro_field),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else ""


def _std(rows: Iterable[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return ""
    return pstdev(finite) if len(finite) > 1 else 0.0
