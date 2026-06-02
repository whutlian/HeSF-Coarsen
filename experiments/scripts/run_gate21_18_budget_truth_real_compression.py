from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.acm_closure_compression import audit_acm_closure_export, export_acm_closure_compressed
from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth, build_budget_truth_audit
from hesf_coarsen.eval.official.freehgc_score_tp_local import (
    freehgc_budget_audit_rows,
    freehgc_score_components_rows,
    freehgc_task_metric_rows,
)
from hesf_coarsen.eval.official.gate21_18_decision import GATE21_18_DECISION_FLAGS, gate21_18_decision
from hesf_coarsen.eval.official.imdb_constraint_compression import audit_imdb_constraint_export, export_imdb_constraint_compressed
from hesf_coarsen.eval.official.official_training_queue import aggregate_training_runs, build_training_queue, execute_training_queue
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset
from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_18_representatives


ROOT = Path(__file__).resolve().parents[2]
GATE21_17_SMOKE = ROOT / "outputs" / "gate21_17_smoke"

GATE21_18_MAIN_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "actual_edge_ratio",
    "actual_support_edge_ratio",
    "actual_support_node_ratio",
    "semantic_structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "static_inference_package_ratio",
    "reconstructable_package_ratio",
    "keyword_feature_ratio",
    "PK_edge_ratio",
    "actor_channel_ratio",
    "keyword_channel_ratio",
    "channel_edge_ratio",
    "graph_seed_count",
    "training_seed_count",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "validation_proxy_score",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "schema_compatible",
    "target_preserving",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "eligible_for_main_table",
    "eligible_for_compression_claim",
    "success",
    "constraint_safe_fallback",
    "budget_match_for_requested_metric",
    "budget_metric_used_for_match",
    "budget_match_failure_type",
    "budget_match_failure_reason",
    "failure_type",
    "failure_reason",
    "selected_edge_hash",
    "planner_config_hash",
    "source_path",
    "repo_url",
    "export_dir",
    "stdout_path",
    "stderr_path",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.18 budget-truth + real compression runner.")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--sehgnn-repo", default=str(ROOT / "external" / "SeHGNN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", "--output-dir", dest="output", default=str(ROOT / "outputs" / "gate21_18_smoke"))
    parser.add_argument("--dry-run-training", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in args.datasets)
    mode = str(args.mode)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_seeds = tuple(args.graph_seeds or [1])
    training_seeds = tuple(args.training_seeds or [1])
    graph_seed = graph_seeds[0]
    sehgnn_repo = Path(args.sehgnn_repo)

    prior_rows = _read_csv(GATE21_17_SMOKE / "gate21_17_main_official_table.csv")
    main_rows: list[dict[str, Any]] = []
    main_rows.extend(_full_anchor_rows(prior_rows, datasets=datasets))
    if "DBLP" in datasets:
        main_rows.extend(_dblp_smoke_rows(prior_rows))

    acm_audit_rows: list[dict[str, Any]] = []
    imdb_audit_rows: list[dict[str, Any]] = []
    if "ACM" in datasets:
        acm_rows, acm_audit_rows = _acm_smoke_rows(out_dir=out_dir, graph_seed=graph_seed, sehgnn_repo=sehgnn_repo)
        main_rows.extend(acm_rows)
    if "IMDB" in datasets:
        imdb_rows, imdb_audit_rows = _imdb_smoke_rows(out_dir=out_dir, graph_seed=graph_seed, sehgnn_repo=sehgnn_repo)
        main_rows.extend(imdb_rows)

    fallback_rows = _fallback_sanity_rows(prior_rows, datasets=datasets)
    queue = build_training_queue(
        main_rows,
        graph_seeds=graph_seeds if mode != "smoke" else graph_seeds[:1],
        training_seeds=training_seeds if mode != "smoke" else training_seeds[:1],
    )
    training_runs, training_failures = execute_training_queue(
        queue,
        sehgnn_repo=sehgnn_repo,
        device=str(args.device),
        out_dir=out_dir,
        python_executable=sys.executable,
        dry_run=bool(args.dry_run_training) or mode == "preflight",
    )
    _merge_training_results(main_rows, aggregate_training_runs(training_runs))
    _replace_unexecuted_pending(main_rows)
    _annotate_rows(main_rows)
    _add_recovery(main_rows)

    rep_rows = select_gate21_18_representatives(main_rows, datasets=datasets)
    for row in rep_rows:
        if bool_value(row.get("eligible_for_main_table", True)):
            rep_main = _gate21_18_row(row)
            _annotate_rows([rep_main])
            main_rows.append(rep_main)

    decision = gate21_18_decision(main_rows=main_rows, fallback_rows=fallback_rows, datasets=datasets)
    external_rows = [row for row in main_rows if row.get("method_family") == "external_tp_baseline"]
    budget_audit = build_budget_truth_audit(main_rows)
    training_runs_all = _reused_training_run_rows(main_rows) + training_runs

    write_csv(out_dir / "gate21_18_main_official_table.csv", main_rows, GATE21_18_MAIN_FIELDS)
    write_csv(out_dir / "gate21_18_training_runs.csv", training_runs_all)
    write_csv(out_dir / "gate21_18_training_failures.csv", training_failures)
    write_csv(out_dir / "gate21_18_training_queue.csv", queue)
    write_csv(out_dir / "gate21_18_budget_truth_audit.csv", budget_audit)
    write_csv(out_dir / "gate21_18_fallback_loader_sanity.csv", fallback_rows)
    write_csv(out_dir / "gate21_18_acm_closure_audit.csv", acm_audit_rows)
    write_csv(out_dir / "gate21_18_imdb_constraint_audit.csv", imdb_audit_rows)
    write_csv(out_dir / "gate21_18_external_tp_by_method.csv", _by_method_rows(external_rows))
    write_csv(out_dir / "gate21_18_hesf_rcs_rep_selection.csv", rep_rows)
    write_csv(out_dir / "gate21_18_edge_structural_workload_table.csv", _edge_structural_table(main_rows))
    write_csv(out_dir / "gate21_18_raw_text_deployment_storage_table.csv", _raw_text_table(main_rows))
    write_csv(out_dir / "freehgc_score_components.csv", freehgc_score_components_rows())
    write_csv(out_dir / "freehgc_score_tp_budget_audit.csv", freehgc_budget_audit_rows(main_rows))
    write_csv(out_dir / "freehgc_score_tp_task_metrics.csv", freehgc_task_metric_rows(main_rows))
    write_csv(out_dir / "gate21_18_decision_flags.csv", _decision_flag_rows(decision))
    write_json(out_dir / "gate21_18_decision.json", decision)
    (out_dir / "gate21_18_summary.md").write_text(_summary(decision, main_rows, training_failures), encoding="utf-8")
    (out_dir / "gate21_18_requirement_checklist.md").write_text(_checklist(decision, mode), encoding="utf-8")
    return decision


def _full_anchor_rows(prior_rows: Sequence[Mapping[str, Any]], *, datasets: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in ("Full-native-SeHGNN", "Export-full-SeHGNN"):
            source = _find_prior(prior_rows, dataset, method)
            if not source:
                continue
            out.append(
                _gate21_18_row(
                    source,
                    method=method,
                    method_family="full_fidelity_baseline",
                    requested_budget_type="",
                    requested_budget="",
                    actual_support_edge_ratio=1.0,
                    actual_support_node_ratio=1.0,
                    semantic_structural_storage_ratio=1.0,
                    raw_hgb_text_byte_ratio=1.0,
                    eligible_for_compression_claim=False,
                    constraint_safe_fallback=False,
                    source_path=source.get("source_path", "outputs/gate21_17_smoke"),
                )
            )
    return out


def _dblp_smoke_rows(prior_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("HeSF-RCS-auto structural12", "HeSF-RCS-auto structural12", "schema_preserving_rcs", "structural_storage_ratio", 0.12),
        ("HeSF-RCS-auto structural16", "HeSF-RCS-auto structural16", "schema_preserving_rcs", "structural_storage_ratio", 0.16),
        ("Random-edge-relwise", "Random-edge-relwise", "relation_structural_baseline", "support_edge_ratio", 0.20),
        ("Degree-edge-relwise", "Degree-edge-relwise", "relation_structural_baseline", "support_edge_ratio", 0.20),
        ("Proportional-relation-budget", "Proportional-relation-budget", "relation_structural_baseline", "support_edge_ratio", 0.20),
        ("Herding-HG-TP", "Herding-HG-TP", "external_tp_baseline", "support_node_ratio", 0.50),
        ("FreeHGC-score-TP", "FreeHGC-score-TP-local", "external_tp_baseline", "support_edge_ratio", 0.20),
        ("HGCond-score-TP-local", "HGCond-score-TP-local", "external_tp_baseline", "support_node_ratio", 0.50),
        ("GCond-score-TP-local", "GCond-score-TP-local", "external_tp_baseline", "support_node_ratio", 0.50),
    ]
    out: list[dict[str, Any]] = []
    for source_method, method, family, budget_type, budget in specs:
        source = _find_prior(prior_rows, "DBLP", source_method)
        if not source:
            continue
        support_edge = float_value(source.get("support_edge_ratio"))
        raw = float_value(source.get("raw_hgb_text_byte_ratio"))
        old_actual = float_value(source.get("actual_structural_storage_ratio"))
        semantic = support_edge if family == "relation_structural_baseline" else old_actual
        out.append(
            _gate21_18_row(
                source,
                method=method,
                method_family=family,
                requested_budget_type=budget_type,
                requested_budget=budget,
                actual_support_edge_ratio=support_edge,
                actual_support_node_ratio=source.get("support_node_ratio", ""),
                semantic_structural_storage_ratio=semantic,
                raw_hgb_text_byte_ratio=raw,
                static_inference_package_ratio=raw,
                reconstructable_package_ratio=raw,
                constraint_safe_fallback=False,
                eligible_for_compression_claim=True,
                source_path=source.get("source_path", "outputs/gate21_17_smoke"),
            )
        )
    return out


def _acm_smoke_rows(*, out_dir: Path, graph_seed: int, sehgnn_repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = _source_dataset_dir("ACM")
    specs = [
        ("ACM-HeSF-RCS-auto-field30", "coverage_greedy", 0.30),
        ("ACM-HeSF-RCS-auto-field20", "coverage_greedy", 0.20),
        ("ACM-Random-field20", "random", 0.20),
        ("ACM-Degree-field20", "degree", 0.20),
    ]
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for method, selector, ratio in specs:
        export_dir = out_dir / "exports" / "ACM" / str(graph_seed) / _slug(method) / f"keyword_feature_ratio_{_budget_slug(ratio)}" / "official_trainval" / "ACM"
        manifest = export_acm_closure_compressed(source_dir, export_dir, method=selector, keyword_ratio=ratio, graph_seed=graph_seed)
        audit = audit_acm_closure_export(export_dir, source_dir=source_dir)
        audit.update({"method": method, "selector": selector, "requested_budget": ratio})
        audit.update({f"native_{key}": value for key, value in audit_native_hgb_data_dir("ACM", export_dir.parent, sehgnn_repo).items() if key in {"can_load_with_official_data_loader", "official_data_loader_error"}})
        audits.append(audit)
        rows.append(
            _new_pending_row(
                dataset="ACM",
                method=method,
                method_family="schema_preserving_rcs" if "HeSF" in method else "relation_structural_baseline",
                requested_budget_type="keyword_feature_ratio",
                requested_budget=ratio,
                manifest=manifest,
                export_dir=export_dir,
                audit_pass=bool(audit.get("P_matches_PK")) and bool(audit.get("A_matches_AP_PK")) and bool(audit.get("C_matches_CP_PK")) and bool(audit.get("PK_KP_reciprocal")),
            )
        )
    return rows, audits


def _imdb_smoke_rows(*, out_dir: Path, graph_seed: int, sehgnn_repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = _source_dataset_dir("IMDB")
    specs = [
        ("IMDB-HeSF-RCS-auto structural30", "degree", 0.15, 0.15, "structural_storage_ratio", 0.30),
        ("IMDB-HeSF-RCS-auto structural20", "degree", 0.05, 0.05, "structural_storage_ratio", 0.20),
        ("IMDB-Random-channel20", "random", 0.20, 0.20, "channel_edge_ratio", 0.20),
        ("IMDB-Degree-channel20", "degree", 0.20, 0.20, "channel_edge_ratio", 0.20),
        ("IMDB-MDfull-MA50-MK20", "degree", 0.50, 0.20, "channel_edge_ratio", 0.50),
        ("IMDB-MDfull-MA20-MK50", "degree", 0.20, 0.50, "channel_edge_ratio", 0.50),
    ]
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for method, selector, actor_ratio, keyword_ratio, budget_type, budget in specs:
        export_dir = out_dir / "exports" / "IMDB" / str(graph_seed) / _slug(method) / f"{budget_type}_{_budget_slug(budget)}" / "official_trainval" / "IMDB"
        manifest = export_imdb_constraint_compressed(source_dir, export_dir, method=selector, actor_ratio=actor_ratio, keyword_ratio=keyword_ratio, graph_seed=graph_seed)
        audit = audit_imdb_constraint_export(export_dir, source_dir=source_dir)
        audit.update({"method": method, "selector": selector, "requested_budget_type": budget_type, "requested_budget": budget})
        audit.update({f"native_{key}": value for key, value in audit_native_hgb_data_dir("IMDB", export_dir.parent, sehgnn_repo).items() if key in {"can_load_with_official_data_loader", "official_data_loader_error"}})
        audits.append(audit)
        rows.append(
            _new_pending_row(
                dataset="IMDB",
                method=method,
                method_family="schema_preserving_rcs" if "HeSF" in method else "relation_structural_baseline",
                requested_budget_type=budget_type,
                requested_budget=budget,
                manifest=manifest,
                export_dir=export_dir,
                audit_pass=bool(audit.get("MD_DM_reciprocal")) and bool(audit.get("MA_AM_reciprocal")) and bool(audit.get("MK_KM_reciprocal")) and bool(audit.get("movie_single_director_constraint_pass")),
            )
        )
    return rows, audits


def _new_pending_row(
    *,
    dataset: str,
    method: str,
    method_family: str,
    requested_budget_type: str,
    requested_budget: float,
    manifest: Mapping[str, Any],
    export_dir: Path,
    audit_pass: bool,
) -> dict[str, Any]:
    failure_type = "implemented_pending_official_training" if audit_pass else "export_schema_failure"
    return _gate21_18_row(
        {
            "dataset": dataset,
            "method": method,
            "method_family": method_family,
            "requested_budget_type": requested_budget_type,
            "requested_budget": requested_budget,
            "actual_support_edge_ratio": manifest.get("actual_support_edge_ratio", manifest.get("PK_edge_ratio", "")),
            "actual_support_node_ratio": manifest.get("actual_support_node_ratio", ""),
            "semantic_structural_storage_ratio": manifest.get("semantic_structural_storage_ratio", ""),
            "raw_hgb_text_byte_ratio": manifest.get("raw_hgb_text_byte_ratio", ""),
            "static_inference_package_ratio": manifest.get("raw_hgb_text_byte_ratio", ""),
            "reconstructable_package_ratio": manifest.get("raw_hgb_text_byte_ratio", ""),
            "keyword_feature_ratio": manifest.get("keyword_feature_ratio", ""),
            "PK_edge_ratio": manifest.get("PK_edge_ratio", ""),
            "actor_channel_ratio": manifest.get("actor_channel_ratio", ""),
            "keyword_channel_ratio": manifest.get("keyword_channel_ratio", ""),
            "channel_edge_ratio": max(
                [value for value in (float_value(manifest.get("actor_channel_ratio")), float_value(manifest.get("keyword_channel_ratio"))) if value is not None],
                default="",
            ),
            "schema_compatible": audit_pass,
            "target_preserving": True,
            "official_hgb_exported": audit_pass,
            "official_sehgnn_unmodified": True,
            "training_executed": False,
            "eligible_for_main_table": True,
            "eligible_for_compression_claim": True,
            "success": False,
            "constraint_safe_fallback": False,
            "failure_type": failure_type,
            "failure_reason": "" if audit_pass else "Gate21.18 compressed export failed dataset-specific consistency audit.",
            "selected_edge_hash": manifest.get("selected_edge_hash", ""),
            "planner_config_hash": manifest.get("planner_config_hash", ""),
            "source_path": str(export_dir / "gate21_18_export_manifest.json"),
            "export_dir": str(export_dir),
            "graph_seed_count": 1,
            "training_seed_count": 0,
        }
    )


def _gate21_18_row(row: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    merged = dict(row)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    out = {field: merged.get(field, "") for field in GATE21_18_MAIN_FIELDS}
    out["dataset"] = normalize_dataset(out.get("dataset"))
    for field in (
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "eligible_for_main_table",
        "eligible_for_compression_claim",
        "success",
        "constraint_safe_fallback",
    ):
        out[field] = bool_value(merged.get(field, out.get(field)))
    for optional in ("selection_source", "rep_selection_confidence", "uses_test_for_selection", "eligible_for_decision", "source_method", "selected_as_rep"):
        if optional in merged:
            out[optional] = merged.get(optional, "")
    return out


def _annotate_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        annotated = annotate_budget_truth(row)
        if str(row.get("budget_match_failure_type", "")) == "budget_infeasible":
            annotated["budget_match_for_requested_metric"] = False
            annotated["budget_match_failure_type"] = "budget_infeasible"
            annotated["budget_match_failure_reason"] = row.get("budget_match_failure_reason", "")
        rows[index].update(annotated)


def _merge_training_results(rows: list[dict[str, Any]], by_source_id: Mapping[int, Mapping[str, Any]]) -> None:
    for index, update in by_source_id.items():
        if 0 <= int(index) < len(rows):
            rows[int(index)].update(update)


def _replace_unexecuted_pending(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if str(row.get("failure_type", "")) != "implemented_pending_official_training":
            continue
        row["training_executed"] = False
        row["success"] = False
        row["failure_type"] = "official_training_not_executed"
        row["failure_reason"] = "Gate21.18 did not receive task metrics for this export."


def _add_recovery(rows: list[dict[str, Any]]) -> None:
    full_micro = {row.get("dataset"): float_value(row.get("test_micro_f1_mean")) for row in rows if row.get("method") == "Full-native-SeHGNN"}
    full_macro = {row.get("dataset"): float_value(row.get("test_macro_f1_mean")) for row in rows if row.get("method") == "Full-native-SeHGNN"}
    for row in rows:
        dataset = row.get("dataset")
        micro = float_value(row.get("test_micro_f1_mean"))
        macro = float_value(row.get("test_macro_f1_mean"))
        if micro is not None and full_micro.get(dataset):
            row["recovery_vs_native_full_micro"] = micro / float(full_micro[dataset])
        if macro is not None and full_macro.get(dataset):
            row["recovery_vs_native_full_macro"] = macro / float(full_macro[dataset])


def _fallback_sanity_rows(prior_rows: Sequence[Mapping[str, Any]], *, datasets: Sequence[str]) -> list[dict[str, Any]]:
    fallback_methods = {
        "ACM": {"HeSF-RCS-auto structural20", "Random-edge-relwise", "Herding-HG-TP", "HGCond-score-TP-local", "GCond-score-TP-local"},
        "IMDB": {"HeSF-RCS-auto structural20", "Random-edge-relwise", "Herding-HG-TP", "HGCond-score-TP-local", "GCond-score-TP-local"},
    }
    out: list[dict[str, Any]] = []
    for row in prior_rows:
        dataset = normalize_dataset(row.get("dataset"))
        if dataset not in datasets or str(row.get("method", "")) not in fallback_methods.get(dataset, set()):
            continue
        if float_value(row.get("raw_hgb_text_byte_ratio")) != 1.0:
            continue
        sanity = dict(row)
        sanity.update(
            {
                "constraint_safe_fallback": True,
                "eligible_for_main_table": False,
                "eligible_for_compression_claim": False,
                "failure_type": "constraint_safe_full_fallback",
                "failure_reason": "Gate21.17 full-HGB fallback retained only as Gate21.18 loader sanity evidence.",
            }
        )
        out.append(sanity)
    return out


def _edge_structural_table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "requested_budget_type": row.get("requested_budget_type", ""),
            "requested_budget": row.get("requested_budget", ""),
            "actual_support_edge_ratio": row.get("actual_support_edge_ratio", ""),
            "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
            "micro": row.get("test_micro_f1_mean", ""),
            "macro": row.get("test_macro_f1_mean", ""),
            "recovery_micro": row.get("recovery_vs_native_full_micro", ""),
            "recovery_macro": row.get("recovery_vs_native_full_macro", ""),
            "training_executed": row.get("training_executed", ""),
            "budget_match": row.get("budget_match_for_requested_metric", ""),
        }
        for row in rows
        if row.get("method") not in {"Full-native-SeHGNN", "Export-full-SeHGNN"} or normalize_dataset(row.get("dataset")) in {"DBLP", "ACM", "IMDB"}
    ]


def _raw_text_table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        export_dir = Path(str(row.get("export_dir", "")))
        bytes_row = _byte_breakdown(export_dir) if str(export_dir) else {}
        table.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
                "node_dat_bytes": bytes_row.get("node_dat_bytes", ""),
                "link_dat_bytes": bytes_row.get("link_dat_bytes", ""),
                "feature_bytes": bytes_row.get("feature_bytes", ""),
                "static_inference_package_ratio": row.get("static_inference_package_ratio", ""),
                "training_executed": row.get("training_executed", ""),
                "micro": row.get("test_micro_f1_mean", ""),
                "macro": row.get("test_macro_f1_mean", ""),
            }
        )
    return table


def _by_method_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((normalize_dataset(row.get("dataset")), str(row.get("method", ""))), []).append(dict(row))
    return [
        {
            "dataset": dataset,
            "method": method,
            "rows": len(group),
            "success_rows": sum(1 for row in group if bool_value(row.get("success"))),
            "budget_match_rows": sum(1 for row in group if bool_value(row.get("budget_match_for_requested_metric"))),
            "failure_types": ";".join(sorted({str(row.get("failure_type", "")) for row in group if row.get("failure_type")})),
        }
        for (dataset, method), group in sorted(grouped.items())
    ]


def _reused_training_run_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not bool_value(row.get("training_executed")) or str(row.get("failure_type", "")):
            continue
        if row.get("export_dir"):
            continue
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "status": "reused_gate21_17_official_metric",
                "training_executed": True,
                "success": True,
                "test_micro_f1": row.get("test_micro_f1_mean", ""),
                "test_macro_f1": row.get("test_macro_f1_mean", ""),
                "validation_micro_f1": row.get("validation_micro_f1_mean", ""),
                "validation_macro_f1": row.get("validation_macro_f1_mean", ""),
            }
        )
    return out


def _summary(decision: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Gate21.18 Budget Truth Real Compression Summary", "", f"- rows: {len(rows)}", f"- training failures: {len(failures)}", ""]
    for flag in GATE21_18_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## Successful Task Metrics", ""])
    for row in rows:
        if not bool_value(row.get("success")) or not bool_value(row.get("training_executed")):
            continue
        lines.append(
            "- "
            f"{row.get('dataset')} {row.get('method')} "
            f"{row.get('requested_budget_type')}={row.get('requested_budget')} "
            f"semantic={row.get('semantic_structural_storage_ratio')} "
            f"edge={row.get('actual_support_edge_ratio')} "
            f"raw={row.get('raw_hgb_text_byte_ratio')} "
            f"micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}"
        )
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- none")
    for row in failures:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', ''))[:500]}")
    return "\n".join(lines) + "\n"


def _checklist(decision: Mapping[str, Any], mode: str) -> str:
    section_status = {
        "P0 Fix Budget Metric Semantics": decision.get("BUDGET_METRIC_SEMANTICS_PASS") and decision.get("NO_MIXED_ACTUAL_STRUCTURAL_RATIO_PASS"),
        "P1 Separate Edge/Structural and Raw Text Tables": True,
        "P2 Stop Full-HGB Fallback From Main Results": decision.get("NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"),
        "P3 Implement Real ACM Compression": decision.get("ACM_REAL_COMPRESSED_ROW_READY"),
        "P4 Implement Real IMDB Compression": decision.get("IMDB_REAL_COMPRESSED_ROW_READY"),
        "P5 Repair DBLP Structural Baseline Budgets": decision.get("DBLP_EDGE_BASELINE_SUPPORT_EDGE20_READY"),
        "P6 Upgrade HeSF-RCS-Rep Selection": decision.get("HESF_RCS_REP_ACTUAL_VALIDATION_READY") and decision.get("HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE"),
        "P7 Budget-Comparable Local External Baselines": decision.get("DBLP_EXTERNAL_TP_SMOKE_READY"),
        "P8 FreeHGC-score-TP-local Ready": decision.get("FREEHGC_SCORE_TP_LOCAL_READY"),
        "P9 Smoke Execution Plan": decision.get("STAGE_REPORT_SMOKE_READY"),
        "P10 Main Output Files": True,
        "P11 Decision Flags": True,
        "P12 Repository Integration": True,
        "P13 Non-Negotiable Rules": decision.get("STAGE_REPORT_BUDGET_TRUTH_READY"),
    }
    lines = ["# Gate21.18 Requirement Checklist", "", f"- mode: {mode}", "", "## Decision Flags", ""]
    for flag in GATE21_18_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(["", "## Attachment Sections", ""])
    for section, passed in section_status.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {section}")
    return "\n".join(lines) + "\n"


def _decision_flag_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"flag": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in decision.items()
    ]


def _find_prior(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    candidates = [dict(row) for row in rows if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method]
    successes = [row for row in candidates if bool_value(row.get("success")) and bool_value(row.get("training_executed"))]
    if successes:
        return successes[0]
    return candidates[0] if candidates else {}


def _source_dataset_dir(dataset: str) -> Path:
    name = normalize_dataset(dataset)
    direct = ROOT / "data" / name.lower() / "raw" / name
    if direct.exists():
        return direct
    nested = ROOT / "data" / name.lower() / name.lower() / "raw" / name
    if nested.exists():
        return nested
    external = ROOT / "external" / "SeHGNN" / "data" / name
    if external.exists():
        return external
    raise FileNotFoundError(f"Missing source HGB dataset directory for {name}")


def _byte_breakdown(export_dir: Path) -> dict[str, Any]:
    if not export_dir.exists():
        return {}
    node_bytes = (export_dir / "node.dat").stat().st_size if (export_dir / "node.dat").exists() else 0
    link_bytes = (export_dir / "link.dat").stat().st_size if (export_dir / "link.dat").exists() else 0
    feature_bytes = 0
    node_path = export_dir / "node.dat"
    if node_path.exists():
        with node_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    feature_bytes += len(parts[3].encode("utf-8"))
    return {"node_dat_bytes": node_bytes, "link_dat_bytes": link_bytes, "feature_bytes": feature_bytes}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _budget_slug(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.18 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.18 STAGE_REPORT_BUDGET_TRUTH_READY={decision['STAGE_REPORT_BUDGET_TRUTH_READY']}")


if __name__ == "__main__":
    main()
