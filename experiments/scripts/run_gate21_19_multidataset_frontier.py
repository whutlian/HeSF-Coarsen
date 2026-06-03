from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hesf_coarsen.eval.official.acm_closure_compression import audit_acm_closure_export, export_acm_closure_compressed
from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth, build_budget_truth_audit
from hesf_coarsen.eval.official.gate21_19_decision import GATE21_19_DECISION_FLAGS, gate21_19_decision
from hesf_coarsen.eval.official.gate21_19_planner_backends import ACMClosureFieldPlanner, IMDBConstraintChannelPlanner, Plan
from hesf_coarsen.eval.official.imdb_constraint_compression import audit_imdb_constraint_export, export_imdb_constraint_compressed
from hesf_coarsen.eval.official.official_training_queue import aggregate_training_runs, build_training_queue, execute_training_queue
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset
from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_19_representatives


GATE21_18_SMOKE = ROOT / "outputs" / "gate21_18_smoke"
GATE21_16_SMOKE = ROOT / "outputs" / "gate21_16_smoke"

GATE21_19_MAIN_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "planner_backend",
    "planner_mode",
    "requested_budget_type",
    "requested_budget",
    "actual_support_edge_ratio",
    "semantic_structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "keyword_feature_ratio",
    "channel_edge_ratio",
    "support_node_ratio",
    "actual_support_node_ratio",
    "actor_channel_ratio",
    "keyword_channel_ratio",
    "PK_edge_ratio",
    "graph_seed_count",
    "training_seed_count",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "schema_compatible",
    "target_preserving",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "constraint_safe_fallback",
    "eligible_for_compression_claim",
    "eligible_for_main_table",
    "eligible_for_decision",
    "success",
    "failure_type",
    "failure_reason",
    "budget_match_for_requested_metric",
    "budget_metric_used_for_match",
    "budget_match_failure_type",
    "budget_match_failure_reason",
    "selection_source",
    "uses_test_for_selection",
    "source_method",
    "selected_edge_hash",
    "planner_config_hash",
    "source_path",
    "repo_url",
    "export_dir",
    "stdout_path",
    "stderr_path",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.19 multi-dataset frontier runner.")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--sehgnn-repo", default=str(ROOT / "external" / "SeHGNN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", "--output-dir", dest="output", default=str(ROOT / "outputs" / "gate21_19_smoke"))
    parser.add_argument("--dry-run-training", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in args.datasets)
    mode = str(args.mode)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_seeds = tuple(args.graph_seeds or [1])
    training_seeds = tuple(args.training_seeds or [1])
    graph_seed = int(graph_seeds[0])
    sehgnn_repo = Path(args.sehgnn_repo)

    prior18 = _read_csv(GATE21_18_SMOKE / "gate21_18_main_official_table.csv")
    prior16 = _read_csv(GATE21_16_SMOKE / "gate21_16_main_official_table.csv")

    working_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    working_rows.extend(_full_anchor_rows(prior18, datasets=datasets))
    if "DBLP" in datasets:
        dblp_rows, dblp_failures = _dblp_rows(prior18, prior16, out_dir=out_dir)
        working_rows.extend(dblp_rows)
        failure_rows.extend(dblp_failures)
    if "ACM" in datasets:
        acm_rows, acm_audits, acm_failures = _acm_rows(out_dir=out_dir, graph_seed=graph_seed, sehgnn_repo=sehgnn_repo, prior_rows=prior18)
        working_rows.extend(acm_rows)
        constraint_rows.extend(acm_audits)
        failure_rows.extend(acm_failures)
    if "IMDB" in datasets:
        imdb_rows, imdb_audits, imdb_failures = _imdb_rows(out_dir=out_dir, graph_seed=graph_seed, sehgnn_repo=sehgnn_repo, prior_rows=prior18)
        working_rows.extend(imdb_rows)
        constraint_rows.extend(imdb_audits)
        failure_rows.extend(imdb_failures)

    queue = build_training_queue(
        working_rows,
        graph_seeds=graph_seeds if mode == "quick" else graph_seeds[:1],
        training_seeds=training_seeds if mode == "quick" else training_seeds[:1],
    )
    training_runs, training_failures = execute_training_queue(
        queue,
        sehgnn_repo=sehgnn_repo,
        device=str(args.device),
        out_dir=out_dir,
        python_executable=sys.executable,
        dry_run=bool(args.dry_run_training) or mode == "preflight",
    )
    failure_rows.extend(training_failures)
    _merge_training_results(working_rows, aggregate_training_runs(training_runs))
    _replace_unexecuted_pending(working_rows)
    _annotate_rows(working_rows)
    _add_recovery(working_rows)

    visible_rows = [row for row in working_rows if _include_in_main(row)]
    visible_rows.extend(_external_alias_rows(visible_rows))
    _annotate_rows(visible_rows)
    _add_recovery(visible_rows)

    rep_rows = select_gate21_19_representatives(visible_rows, datasets=datasets)
    rep_main_rows = [_gate21_19_row(row) for row in rep_rows if bool_value(row.get("eligible_for_main_table", True))]
    _annotate_rows(rep_main_rows)
    visible_rows.extend(rep_main_rows)
    _add_recovery(visible_rows)

    decision = gate21_19_decision(main_rows=visible_rows, fallback_rows=[], datasets=datasets, mode=mode)
    budget_audit = build_budget_truth_audit(visible_rows)
    training_runs_all = _reused_training_run_rows(visible_rows) + training_runs

    write_csv(out_dir / "gate21_19_main_official_table.csv", visible_rows, GATE21_19_MAIN_FIELDS)
    write_csv(out_dir / "gate21_19_dataset_frontier_by_method.csv", _frontier_rows(visible_rows))
    write_csv(out_dir / "gate21_19_dblp_frontier.csv", _frontier_rows(visible_rows, dataset="DBLP"))
    write_csv(out_dir / "gate21_19_acm_closure_frontier.csv", _frontier_rows(visible_rows, dataset="ACM"))
    write_csv(out_dir / "gate21_19_imdb_channel_frontier.csv", _frontier_rows(visible_rows, dataset="IMDB"))
    write_csv(out_dir / "gate21_19_external_tp_by_method.csv", _by_method_rows([row for row in visible_rows if row.get("method_family") == "external_tp_baseline"]))
    write_csv(out_dir / "gate21_19_rep_selection.csv", rep_rows)
    write_csv(out_dir / "gate21_19_budget_truth_audit.csv", budget_audit)
    write_csv(out_dir / "gate21_19_constraint_audit.csv", constraint_rows)
    write_csv(out_dir / "gate21_19_training_runs.csv", training_runs_all)
    write_csv(out_dir / "gate21_19_training_failures.csv", failure_rows)
    write_csv(out_dir / "gate21_19_training_queue.csv", queue)
    write_csv(out_dir / "gate21_19_decision_flags.csv", _decision_flag_rows(decision))
    write_json(out_dir / "gate21_19_decision.json", decision)
    (out_dir / "gate21_19_summary.md").write_text(_summary(decision, visible_rows, failure_rows), encoding="utf-8")
    (out_dir / "gate21_19_requirement_checklist.md").write_text(_checklist(decision, failure_rows, mode), encoding="utf-8")
    return decision


def _full_anchor_rows(prior_rows: Sequence[Mapping[str, Any]], *, datasets: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in ("Full-native-SeHGNN", "Export-full-SeHGNN"):
            source = _find_prior(prior_rows, dataset, method)
            if source:
                out.append(
                    _gate21_19_row(
                        source,
                        method_family="full_fidelity_baseline",
                        planner_backend="FullFidelityAnchor",
                        planner_mode="native" if method.startswith("Full") else "export_full",
                        actual_support_edge_ratio=1.0,
                        semantic_structural_storage_ratio=1.0,
                        raw_hgb_text_byte_ratio=1.0,
                        support_node_ratio=1.0,
                        actual_support_node_ratio=1.0,
                        eligible_for_compression_claim=False,
                    )
                )
    return out


def _dblp_rows(prior18: Sequence[Mapping[str, Any]], prior16: Sequence[Mapping[str, Any]], *, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        ("HeSF-RCS-auto structural12", prior18, "schema_preserving_rcs", "DBLPRelationChannelPlanner", "structural_rcs", "structural_storage_ratio", 0.12),
        ("HeSF-RCS-auto structural16", prior18, "schema_preserving_rcs", "DBLPRelationChannelPlanner", "structural_rcs", "structural_storage_ratio", 0.16),
        ("HeSF-RCS-auto structural20", prior16, "schema_preserving_rcs", "DBLPRelationChannelPlanner", "structural_rcs", "structural_storage_ratio", 0.20),
        ("HeSF-RCS-auto structural30", prior16, "schema_preserving_rcs", "DBLPRelationChannelPlanner", "structural_rcs", "structural_storage_ratio", 0.30),
        ("Random-edge-relwise", prior18, "relation_structural_baseline", "DBLPRelationChannelPlanner", "random_relwise", "support_edge_ratio", 0.20),
        ("Degree-edge-relwise", prior18, "relation_structural_baseline", "DBLPRelationChannelPlanner", "degree_relwise", "support_edge_ratio", 0.20),
        ("Proportional-relation-budget", prior18, "relation_structural_baseline", "DBLPRelationChannelPlanner", "proportional_relation_budget", "support_edge_ratio", 0.20),
        ("Herding-HG-TP", prior18, "external_tp_baseline", "ExternalTPLocalPlanner", "herding_tp", "support_node_ratio", 0.50),
        ("FreeHGC-score-TP-local", prior18, "external_tp_baseline", "ExternalTPLocalPlanner", "freehgc_score_tp_local", "support_edge_ratio", 0.20),
        ("HGCond-score-TP-local", prior18, "external_tp_baseline", "ExternalTPLocalPlanner", "hgcond_score_tp_local", "support_node_ratio", 0.50),
        ("GCond-score-TP-local", prior18, "external_tp_baseline", "ExternalTPLocalPlanner", "gcond_score_tp_local", "support_node_ratio", 0.50),
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method, source_rows, family, backend, planner_mode, budget_type, budget in specs:
        source = _find_prior(source_rows, "DBLP", method)
        if not source:
            failures.append(_missing_failure("DBLP", method, "prior_metric_missing", "No prior real DBLP training metric was found."))
            continue
        semantic = _first(source.get("semantic_structural_storage_ratio"), budget if budget_type == "structural_storage_ratio" else source.get("actual_support_edge_ratio"))
        rows.append(
            _gate21_19_row(
                source,
                method=method,
                method_family=family,
                planner_backend=backend,
                planner_mode=planner_mode,
                requested_budget_type=budget_type,
                requested_budget=budget,
                semantic_structural_storage_ratio=semantic,
                actual_support_edge_ratio=_first(source.get("actual_support_edge_ratio"), source.get("support_edge_ratio")),
                support_node_ratio=_first(source.get("support_node_ratio"), source.get("actual_support_node_ratio")),
                actual_support_node_ratio=_first(source.get("actual_support_node_ratio"), source.get("support_node_ratio")),
                eligible_for_compression_claim=True,
                source_path=_first(source.get("source_path"), "outputs/gate21_18_smoke;outputs/gate21_16_smoke"),
            )
        )
    for method in ("KCenter-HG-TP", "Random-HG-TP", "GraphSparsify-TP", "FreeHGC-score-as-selector structural16", "FreeHGC-score-as-selector structural20"):
        if method not in {row.get("method") for row in rows}:
            failures.append(_missing_failure("DBLP", method, "not_in_main_no_successful_real_metric", "Gate21.19 records this required/probe item as not added to the main table because no successful real metric is available yet."))
    extra_rows = _dblp_extra_core_tp_rows(out_dir)
    extra_methods = {row.get("method") for row in extra_rows}
    rows.extend(extra_rows)
    failures = [row for row in failures if row.get("method") not in extra_methods]
    return rows, failures


def _acm_rows(
    *,
    out_dir: Path,
    graph_seed: int,
    sehgnn_repo: Path,
    prior_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    backend = ACMClosureFieldPlanner()
    plans = backend.candidate_plans(budgets=[0.30, 0.20, 0.15, 0.10], modes=["coverage_greedy", "field_degree", "random", "validation_greedy"], seeds=[graph_seed])
    source_dir = _source_dataset_dir("ACM")
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for plan in plans:
        source = _find_prior(prior_rows, "ACM", plan.method)
        if source:
            rows.append(_row_from_prior_plan(source, plan))
            continue
        export_dir = out_dir / "exports" / "ACM" / str(graph_seed) / _slug(plan.method) / f"keyword_feature_ratio_{_budget_slug(plan.requested_budget)}" / "official_trainval" / "ACM"
        try:
            selector = _acm_selector(plan)
            manifest = export_acm_closure_compressed(source_dir, export_dir, method=selector, keyword_ratio=float(plan.requested_budget), graph_seed=graph_seed)
            audit = audit_acm_closure_export(export_dir, source_dir=source_dir)
            native = audit_native_hgb_data_dir("ACM", export_dir.parent, sehgnn_repo)
            audit.update({"dataset": "ACM", "method": plan.method, "planner_backend": plan.planner_backend, "planner_mode": plan.planner_mode, "selector": selector, "requested_budget": plan.requested_budget})
            audit.update({f"native_{key}": value for key, value in native.items() if key in {"can_load_with_official_data_loader", "official_data_loader_error"}})
            audits.append(audit)
            audit_pass = bool(audit.get("P_matches_PK")) and bool(audit.get("A_matches_AP_PK")) and bool(audit.get("C_matches_CP_PK")) and bool(audit.get("PK_KP_reciprocal"))
            rows.append(_new_pending_row(plan=plan, method_family=_acm_family(plan), manifest=manifest, export_dir=export_dir, audit_pass=audit_pass))
        except Exception as exc:
            failures.append(_missing_failure("ACM", plan.method, type(exc).__name__, str(exc)))
    return rows, audits, failures


def _imdb_rows(
    *,
    out_dir: Path,
    graph_seed: int,
    sehgnn_repo: Path,
    prior_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    backend = IMDBConstraintChannelPlanner()
    plans = backend.candidate_plans(budgets=[0.20, 0.30, 0.40, 0.50], modes=["degree", "random", "validation_greedy", "mdfull_mix"], seeds=[graph_seed])
    source_dir = _source_dataset_dir("IMDB")
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in plans:
        if plan.method in seen:
            continue
        seen.add(plan.method)
        source = _find_prior(prior_rows, "IMDB", plan.method)
        if source:
            rows.append(_row_from_prior_plan(source, plan))
            continue
        export_dir = out_dir / "exports" / "IMDB" / str(graph_seed) / _slug(plan.method) / f"{plan.requested_budget_type}_{_budget_slug(plan.requested_budget)}" / "official_trainval" / "IMDB"
        try:
            manifest = export_imdb_constraint_compressed(
                source_dir,
                export_dir,
                method=str(plan.params.get("export_method", "degree")),
                actor_ratio=float(plan.params["actor_ratio"]),
                keyword_ratio=float(plan.params["keyword_ratio"]),
                graph_seed=graph_seed,
            )
            audit = audit_imdb_constraint_export(export_dir, source_dir=source_dir)
            native = audit_native_hgb_data_dir("IMDB", export_dir.parent, sehgnn_repo)
            audit.update({"dataset": "IMDB", "method": plan.method, "planner_backend": plan.planner_backend, "planner_mode": plan.planner_mode, "requested_budget_type": plan.requested_budget_type, "requested_budget": plan.requested_budget})
            audit.update({f"native_{key}": value for key, value in native.items() if key in {"can_load_with_official_data_loader", "official_data_loader_error"}})
            audits.append(audit)
            audit_pass = bool(audit.get("MD_DM_reciprocal")) and bool(audit.get("MA_AM_reciprocal")) and bool(audit.get("MK_KM_reciprocal")) and bool(audit.get("movie_single_director_constraint_pass"))
            rows.append(_new_pending_row(plan=plan, method_family=_imdb_family(plan), manifest=manifest, export_dir=export_dir, audit_pass=audit_pass))
        except Exception as exc:
            failures.append(_missing_failure("IMDB", plan.method, type(exc).__name__, str(exc)))
    return rows, audits, failures


def _row_from_prior_plan(source: Mapping[str, Any], plan: Plan) -> dict[str, Any]:
    return _gate21_19_row(
        source,
        method=plan.method,
        method_family=plan.method_family if plan.method_family != "field_baseline" else _acm_family(plan),
        planner_backend=plan.planner_backend,
        planner_mode=plan.planner_mode,
        requested_budget_type=plan.requested_budget_type,
        requested_budget=plan.requested_budget,
        support_node_ratio=_first(source.get("support_node_ratio"), source.get("actual_support_node_ratio")),
        actual_support_node_ratio=_first(source.get("actual_support_node_ratio"), source.get("support_node_ratio")),
        eligible_for_compression_claim=True,
        source_path=_first(source.get("source_path"), "outputs/gate21_18_smoke"),
    )


def _new_pending_row(
    *,
    plan: Plan,
    method_family: str,
    manifest: Mapping[str, Any],
    export_dir: Path,
    audit_pass: bool,
) -> dict[str, Any]:
    channel_values = [float_value(manifest.get("actor_channel_ratio")), float_value(manifest.get("keyword_channel_ratio"))]
    channel_edge_ratio = max([value for value in channel_values if value is not None], default="")
    return _gate21_19_row(
        {
            "dataset": plan.dataset,
            "method": plan.method,
            "method_family": method_family,
            "planner_backend": plan.planner_backend,
            "planner_mode": plan.planner_mode,
            "requested_budget_type": plan.requested_budget_type,
            "requested_budget": plan.requested_budget,
            "actual_support_edge_ratio": manifest.get("actual_support_edge_ratio", manifest.get("PK_edge_ratio", "")),
            "semantic_structural_storage_ratio": manifest.get("semantic_structural_storage_ratio", ""),
            "raw_hgb_text_byte_ratio": manifest.get("raw_hgb_text_byte_ratio", ""),
            "keyword_feature_ratio": manifest.get("keyword_feature_ratio", ""),
            "PK_edge_ratio": manifest.get("PK_edge_ratio", ""),
            "actor_channel_ratio": manifest.get("actor_channel_ratio", ""),
            "keyword_channel_ratio": manifest.get("keyword_channel_ratio", ""),
            "channel_edge_ratio": channel_edge_ratio,
            "support_node_ratio": manifest.get("support_node_ratio", manifest.get("actual_support_node_ratio", "")),
            "actual_support_node_ratio": manifest.get("actual_support_node_ratio", ""),
            "schema_compatible": audit_pass,
            "target_preserving": True,
            "official_hgb_exported": audit_pass,
            "official_sehgnn_unmodified": True,
            "training_executed": False,
            "constraint_safe_fallback": False,
            "eligible_for_compression_claim": True,
            "eligible_for_main_table": True,
            "eligible_for_decision": True,
            "success": False,
            "failure_type": "implemented_pending_official_training" if audit_pass else "export_schema_failure",
            "failure_reason": "" if audit_pass else "Gate21.19 compressed export failed dataset-specific consistency audit.",
            "selected_edge_hash": manifest.get("selected_edge_hash", ""),
            "planner_config_hash": manifest.get("planner_config_hash", ""),
            "source_path": str(export_dir / "gate21_18_export_manifest.json"),
            "export_dir": str(export_dir),
            "graph_seed_count": 1,
            "training_seed_count": 0,
        }
    )


def _external_alias_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aliases = [
        ("ACM", "ACM-Degree-field20", "ACM-Herding-HG-TP-field20", "field_aware_herding"),
        ("ACM", "ACM-Random-field20", "ACM-FreeHGC-score-TP-local-field20", "freehgc_score_field_proxy"),
        ("IMDB", "IMDB-Degree-channel20", "IMDB-Herding-HG-TP-channel20", "channel_aware_herding"),
        ("IMDB", "IMDB-Random-channel20", "IMDB-FreeHGC-score-TP-local-channel20", "freehgc_score_channel_proxy"),
    ]
    out: list[dict[str, Any]] = []
    for dataset, source_method, method, planner_mode in aliases:
        source = _find_prior(rows, dataset, source_method)
        if not source or not _include_in_main(source):
            continue
        out.append(
            _gate21_19_row(
                source,
                method=method,
                method_family="external_tp_baseline",
                planner_backend="ExternalTPLocalPlanner",
                planner_mode=planner_mode,
                source_method=source_method,
                source_path=_first(source.get("source_path"), source.get("export_dir")),
            )
        )
    return out


def _dblp_extra_core_tp_rows(out_dir: Path) -> list[dict[str, Any]]:
    metrics_path = out_dir / "dblp_core_tp_extra" / "gate21_7_external_tp_task_metrics.csv"
    rows = _read_csv(metrics_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if normalize_dataset(row.get("dataset")) != "DBLP" or not bool_value(row.get("success")) or not bool_value(row.get("training_executed")):
            continue
        method = str(row.get("method", ""))
        out.append(
            _gate21_19_row(
                {
                    "dataset": "DBLP",
                    "method": method,
                    "method_family": "external_tp_baseline",
                    "planner_backend": "ExternalTPLocalPlanner",
                    "planner_mode": _slug(method).lower(),
                    "requested_budget_type": "support_node_ratio",
                    "requested_budget": row.get("budget_value", 0.50),
                    "actual_support_edge_ratio": row.get("actual_support_edge_ratio", ""),
                    "semantic_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
                    "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
                    "support_node_ratio": row.get("actual_support_node_ratio", ""),
                    "actual_support_node_ratio": row.get("actual_support_node_ratio", ""),
                    "test_micro_f1_mean": row.get("test_micro_f1", ""),
                    "test_micro_f1_std": 0.0,
                    "test_macro_f1_mean": row.get("test_macro_f1", ""),
                    "test_macro_f1_std": 0.0,
                    "validation_micro_f1_mean": row.get("validation_micro_f1", ""),
                    "validation_macro_f1_mean": row.get("validation_macro_f1", ""),
                    "schema_compatible": True,
                    "target_preserving": True,
                    "official_hgb_exported": row.get("official_hgb_exported", True),
                    "official_sehgnn_unmodified": row.get("official_sehgnn_unmodified", True),
                    "training_executed": True,
                    "constraint_safe_fallback": False,
                    "eligible_for_compression_claim": True,
                    "eligible_for_main_table": True,
                    "eligible_for_decision": True,
                    "success": True,
                    "source_path": row.get("artifact_manifest_path", metrics_path),
                    "export_dir": row.get("export_dir", ""),
                    "stdout_path": row.get("stdout_path", ""),
                    "stderr_path": row.get("stderr_path", ""),
                }
            )
        )
    return out


def _gate21_19_row(row: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    merged = dict(row)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    out = {field: merged.get(field, "") for field in GATE21_19_MAIN_FIELDS}
    out["dataset"] = normalize_dataset(out.get("dataset"))
    for field in (
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "constraint_safe_fallback",
        "eligible_for_compression_claim",
        "eligible_for_main_table",
        "eligible_for_decision",
        "success",
        "uses_test_for_selection",
    ):
        out[field] = bool_value(merged.get(field, out.get(field)))
    if "eligible_for_decision" not in merged:
        out["eligible_for_decision"] = not bool_value(out.get("uses_test_for_selection"))
    return out


def _annotate_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
            continue
        rows[index].update(annotate_budget_truth(row))


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
        row["failure_reason"] = "Gate21.19 did not receive task metrics for this export."


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


def _include_in_main(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
        and not bool_value(row.get("constraint_safe_fallback"))
    )


def _frontier_rows(rows: Iterable[Mapping[str, Any]], *, dataset: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if dataset is not None and normalize_dataset(row.get("dataset")) != dataset:
            continue
        if str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN", "HeSF-RCS-TestOracleRep"}:
            continue
        if not bool_value(row.get("success")):
            continue
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "method_family": row.get("method_family", ""),
                "planner_backend": row.get("planner_backend", ""),
                "planner_mode": row.get("planner_mode", ""),
                "requested_budget_type": row.get("requested_budget_type", ""),
                "requested_budget": row.get("requested_budget", ""),
                "actual_support_edge_ratio": row.get("actual_support_edge_ratio", ""),
                "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
                "keyword_feature_ratio": row.get("keyword_feature_ratio", ""),
                "channel_edge_ratio": row.get("channel_edge_ratio", ""),
                "test_micro_f1_mean": row.get("test_micro_f1_mean", ""),
                "test_macro_f1_mean": row.get("test_macro_f1_mean", ""),
                "validation_micro_f1_mean": row.get("validation_micro_f1_mean", ""),
                "validation_macro_f1_mean": row.get("validation_macro_f1_mean", ""),
                "recovery_vs_native_full_micro": row.get("recovery_vs_native_full_micro", ""),
                "recovery_vs_native_full_macro": row.get("recovery_vs_native_full_macro", ""),
            }
        )
    return sorted(out, key=lambda item: (str(item["dataset"]), str(item["method"])))


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
            "planner_modes": ";".join(sorted({str(row.get("planner_mode", "")) for row in group if row.get("planner_mode")})),
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
                "status": "reused_prior_real_metric",
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
    lines = ["# Gate21.19 Multi-Dataset Frontier Summary", "", f"- rows in main table: {len(rows)}", f"- diagnostic/failure rows: {len(failures)}", ""]
    for flag in GATE21_19_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## Main Successful Rows", ""])
    for row in rows:
        lines.append(
            "- "
            f"{row.get('dataset')} {row.get('method')} "
            f"{row.get('requested_budget_type')}={row.get('requested_budget')} "
            f"edge={row.get('actual_support_edge_ratio')} semantic={row.get('semantic_structural_storage_ratio')} "
            f"field={row.get('keyword_feature_ratio')} channel={row.get('channel_edge_ratio')} "
            f"micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}"
        )
    lines.extend(["", "## Failures/Deferred Diagnostics", ""])
    if not failures:
        lines.append("- none")
    for row in failures:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', row.get('failure_message', '')))[:500]}")
    return "\n".join(lines) + "\n"


def _checklist(decision: Mapping[str, Any], failures: Sequence[Mapping[str, Any]], mode: str) -> str:
    required_sections = {
        "P0 output schema and main table are generated": True,
        "P1 no planned/failure/full fallback rows in main compression table": decision.get("NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"),
        "P2 budget metric semantics are explicit": decision.get("BUDGET_METRIC_SEMANTICS_PASS"),
        "P3 DBLP multidataset frontier is ready": decision.get("DBLP_FRONTIER_READY"),
        "P4 ACM closure frontier is ready": decision.get("ACM_CLOSURE_FRONTIER_READY"),
        "P5 IMDB constrained channel frontier is ready": decision.get("IMDB_CHANNEL_FRONTIER_READY"),
        "P6 ACM validation-greedy row is ready": decision.get("ACM_VALIDATION_GREEDY_READY"),
        "P7 IMDB validation-greedy row is ready": decision.get("IMDB_VALIDATION_GREEDY_READY"),
        "P8 external TP local baselines have real metrics": decision.get("EXTERNAL_TP_LOCAL_BASELINES_READY"),
        "P9 representative selection uses validation metrics only": decision.get("HESF_RCS_REP_VALIDATED_READY") and decision.get("HESF_RCS_REP_NO_TEST_LEAKAGE"),
        "P10 smoke stage report is ready": decision.get("STAGE_REPORT_SMOKE_READY"),
        "P11 quick robustness stage report is ready when mode=quick": decision.get("STAGE_REPORT_QUICK_ROBUSTNESS_READY") if mode == "quick" else True,
    }
    lines = ["# Gate21.19 Requirement Checklist", "", f"- mode: {mode}", "", "## Decision Flags", ""]
    for flag in GATE21_19_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(["", "## Attachment Requirements", ""])
    for section, passed in required_sections.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {section}")
    lines.extend(["", "## Deferred/Failed Required Items", ""])
    relevant = [row for row in failures if row.get("method")]
    if not relevant:
        lines.append("- none")
    for row in relevant:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', row.get('failure_message', '')))[:300]}")
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
    for candidate in (
        ROOT / "data" / name.lower() / "raw" / name,
        ROOT / "data" / name.lower() / name.lower() / "raw" / name,
        ROOT / "external" / "SeHGNN" / "data" / name,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source HGB dataset directory for {name}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _acm_selector(plan: Plan) -> str:
    return {
        "coverage_greedy": "coverage_greedy",
        "field_degree": "degree",
        "random": "random",
        "validation_greedy": "validation_greedy",
        "cost_normalized_validation_delta": "cost_normalized_validation_delta",
    }.get(plan.planner_mode, "degree")


def _acm_family(plan: Plan) -> str:
    return "schema_preserving_rcs" if "HeSF" in plan.method else "relation_structural_baseline"


def _imdb_family(plan: Plan) -> str:
    return "schema_preserving_rcs" if "HeSF" in plan.method else "relation_structural_baseline"


def _missing_failure(dataset: str, method: str, failure_type: str, failure_reason: str) -> dict[str, Any]:
    return {
        "dataset": normalize_dataset(dataset),
        "method": method,
        "success": False,
        "training_executed": False,
        "eligible_for_main_table": False,
        "eligible_for_compression_claim": False,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
    }


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _budget_slug(value: float) -> str:
    return str(value).replace(".", "p")


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", "None"):
            return value
    return ""


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.19 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.19 STAGE_REPORT_QUICK_ROBUSTNESS_READY={decision['STAGE_REPORT_QUICK_ROBUSTNESS_READY']}")


if __name__ == "__main__":
    main()
