from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.run_gate21_19_multidataset_frontier import _add_recovery
from hesf_coarsen.eval.official.compact_table_builder import (
    GATE21_22_COMPACT_FIELDS,
    build_gate21_22_compact_table,
    compact_table_markdown,
)
from hesf_coarsen.eval.official.condensation_score_tp_proxy import (
    CONDENSATION_SCORE_FIELDS,
    build_gate21_22_condensation_proxy_rows,
    mark_training_eligible,
    split_condensation_rows,
)
from hesf_coarsen.eval.official.external_repo_manager import GATE21_22_EXTERNAL_REPO_AUDIT_FIELDS, audit_gate21_22_required_external_repos
from hesf_coarsen.eval.official.freehgc_standard_protocol_runner import FREEHGC_STANDARD_PROTOCOL_FIELDS, build_freehgc_standard_protocol_rows
from hesf_coarsen.eval.official.gate21_22_decision import GATE21_22_DECISION_FLAGS, decision_flag_rows, gate21_22_decision
from hesf_coarsen.eval.official.official_training_queue import aggregate_training_runs, build_training_queue, execute_training_queue
from hesf_coarsen.eval.official.pareto_frontier import GATE21_21_FRONTIER_FIELDS, build_gate21_21_frontier_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


DEFAULT_SOURCE = ROOT / "results" / "gate21_21_final_rep_external_baselines_quick"
DEFAULT_OUT = ROOT / "outputs" / "gate21_22_external_baseline_completion"
BASELINES = ("FreeHGC", "HGCond", "GCond", "GCondenser")
REQUIRED_OUTPUTS = (
    "gate21_22_external_repo_audit.csv",
    "gate21_22_freehgc_standard_protocol.csv",
    "gate21_22_condensation_score_tp_results.csv",
    "gate21_22_condensation_score_selector_results.csv",
    "gate21_22_official_training_runs.csv",
    "gate21_22_training_failures.csv",
    "gate21_22_final_compact_table.csv",
    "gate21_22_final_compact_table.md",
    "gate21_22_best_method_comparison.csv",
    "gate21_22_frontiers.csv",
    "gate21_22_decision_flags.csv",
    "gate21_22_decision.json",
    "gate21_22_external_baseline_status.json",
    "gate21_22_summary.md",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.22 external condensation-score baseline completion.")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--mode", choices=("smoke", "quick", "preflight"), default="smoke")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--run-official-sehgnn", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sehgnn-repo", default=str(ROOT / "external" / "SeHGNN"))
    parser.add_argument("--external-repos-dir", default=str(ROOT / "external_repos"))
    parser.add_argument("--source-gate21-21-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", "--out", dest="output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--no-clone-external-repos", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in args.datasets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_gate21_21_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Gate21.21 source directory not found: {source_dir}")

    source_rows = [_normalize_main_row(row) for row in _read_csv(source_dir / "gate21_21_main_official_table.csv")]
    if not source_rows:
        raise FileNotFoundError(f"No Gate21.21 main rows found under {source_dir}")

    base_main_rows = [_normalize_main_row(row) for row in source_rows if not _is_prior_score_row(row)]
    repo_rows = audit_gate21_22_required_external_repos(Path(args.external_repos_dir), clone_missing=not bool(args.no_clone_external_repos))
    standard_rows = build_freehgc_standard_protocol_rows(repo_rows, datasets=datasets)
    proxy_rows = build_gate21_22_condensation_proxy_rows(source_rows, datasets=datasets, baselines=tuple(args.baselines))

    queue = build_training_queue(proxy_rows, graph_seeds=tuple(args.graph_seeds), training_seeds=tuple(args.training_seeds))
    training_runs: list[dict[str, Any]] = []
    training_failures: list[dict[str, Any]] = []
    if queue:
        training_runs, training_failures = _execute_queue_with_export_cache(
            queue,
            sehgnn_repo=Path(args.sehgnn_repo),
            device=str(args.device),
            out_dir=out_dir,
            dry_run=(not bool(args.run_official_sehgnn)) or str(args.mode) == "preflight",
        )
        _merge_training_results(proxy_rows, aggregate_training_runs(training_runs))
    proxy_rows = mark_training_eligible(_annotate_proxy_counts(proxy_rows, graph_seeds=args.graph_seeds))

    tp_rows, selector_rows = split_condensation_rows(proxy_rows)
    main_rows = [_normalize_main_row(row) for row in base_main_rows + [row for row in proxy_rows if bool_value(row.get("eligible_for_official_main_table"))]]
    _add_recovery(main_rows)
    _mirror_recovery_to_proxy(proxy_rows, main_rows)
    tp_rows, selector_rows = split_condensation_rows(proxy_rows)

    compact_rows = build_gate21_22_compact_table(main_rows, datasets=datasets)
    frontier_rows = build_gate21_21_frontier_rows(main_rows, datasets=datasets)
    decision = gate21_22_decision(
        main_rows=main_rows,
        compact_rows=compact_rows,
        repo_rows=repo_rows,
        standard_rows=standard_rows,
        condensation_tp_rows=tp_rows,
        condensation_selector_rows=selector_rows,
        datasets=datasets,
    )
    status = _external_status(decision, repo_rows, proxy_rows, standard_rows, training_runs, training_failures, args)

    write_csv(out_dir / "gate21_22_main_official_table.csv", main_rows)
    write_csv(out_dir / "gate21_22_external_repo_audit.csv", repo_rows, GATE21_22_EXTERNAL_REPO_AUDIT_FIELDS)
    write_csv(out_dir / "gate21_22_freehgc_standard_protocol.csv", standard_rows, FREEHGC_STANDARD_PROTOCOL_FIELDS)
    write_csv(out_dir / "gate21_22_condensation_score_tp_results.csv", tp_rows, CONDENSATION_SCORE_FIELDS)
    write_csv(out_dir / "gate21_22_condensation_score_selector_results.csv", selector_rows, CONDENSATION_SCORE_FIELDS)
    write_csv(out_dir / "gate21_22_official_training_runs.csv", training_runs)
    write_csv(out_dir / "gate21_22_training_failures.csv", training_failures)
    write_csv(out_dir / "gate21_22_training_queue.csv", queue)
    write_csv(out_dir / "gate21_22_final_compact_table.csv", compact_rows, GATE21_22_COMPACT_FIELDS)
    (out_dir / "gate21_22_final_compact_table.md").write_text(compact_table_markdown(compact_rows), encoding="utf-8")
    write_csv(out_dir / "gate21_22_best_method_comparison.csv", compact_rows, GATE21_22_COMPACT_FIELDS)
    write_csv(out_dir / "gate21_22_frontiers.csv", frontier_rows, GATE21_21_FRONTIER_FIELDS)
    write_csv(out_dir / "gate21_22_decision_flags.csv", decision_flag_rows(decision))
    write_json(out_dir / "gate21_22_decision.json", decision)
    write_json(out_dir / "gate21_22_external_baseline_status.json", status)
    (out_dir / "gate21_22_summary.md").write_text(_summary(decision, status, compact_rows, training_failures, args), encoding="utf-8")
    (out_dir / "gate21_22_requirement_checklist.md").write_text(_checklist(decision, out_dir, status, args), encoding="utf-8")
    return decision


def _execute_queue_with_export_cache(
    queue: Sequence[Mapping[str, Any]],
    *,
    sehgnn_repo: Path,
    device: str,
    out_dir: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in queue:
        grouped[_queue_cache_key(item)].append(dict(item))
    unique_queue = [items[0] for _, items in sorted(grouped.items(), key=lambda pair: pair[0])]
    unique_runs, _ = execute_training_queue(
        unique_queue,
        sehgnn_repo=sehgnn_repo,
        device=device,
        out_dir=out_dir,
        python_executable=sys.executable,
        dry_run=dry_run,
    )
    runs_by_key = {_queue_cache_key(row): row for row in unique_runs}
    expanded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key, aliases in sorted(grouped.items(), key=lambda pair: pair[0]):
        run = runs_by_key.get(key, {})
        for alias in aliases:
            row = dict(run)
            row.update(
                {
                    "dataset": alias.get("dataset", row.get("dataset", "")),
                    "method": alias.get("method", row.get("method", "")),
                    "requested_budget_type": alias.get("requested_budget_type", row.get("requested_budget_type", "")),
                    "requested_budget": alias.get("requested_budget", row.get("requested_budget", "")),
                    "actual_structural_storage_ratio": alias.get("actual_structural_storage_ratio", row.get("actual_structural_storage_ratio", "")),
                    "export_dir": alias.get("export_dir", row.get("export_dir", "")),
                    "selected_edge_hash": alias.get("selected_edge_hash", row.get("selected_edge_hash", "")),
                    "planner_config_hash": alias.get("planner_config_hash", row.get("planner_config_hash", "")),
                    "graph_seed": alias.get("graph_seed", row.get("graph_seed", "")),
                    "training_seed": alias.get("training_seed", row.get("training_seed", "")),
                    "source_row_id": alias.get("source_row_id", row.get("source_row_id", "")),
                    "cache_reused_from_method": run.get("method", ""),
                }
            )
            expanded.append(row)
            if not bool_value(row.get("success")):
                failures.append(row)
    return expanded, failures


def _queue_cache_key(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_dataset(item.get("dataset")),
        str(Path(str(item.get("export_dir", "")))),
        str(item.get("graph_seed", "")),
        str(item.get("training_seed", "")),
    )


def _normalize_main_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    dataset = normalize_dataset(out.get("dataset"))
    out["dataset"] = dataset
    method = str(out.get("method", ""))
    semantic = _first_value(out, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio", "actual_structural_storage_ratio", "channel_edge_ratio", "keyword_feature_ratio")
    support_edge = _first_value(out, "actual_support_edge_ratio", "support_edge_ratio", "channel_edge_ratio")
    support_node = _first_value(out, "actual_support_node_ratio", "support_node_ratio", "total_node_ratio")
    raw_ratio = _first_value(out, "raw_hgb_text_byte_ratio", "hgb_raw_file_byte_ratio", "official_text_hgb_byte_ratio")
    if method in {"Full-native-SeHGNN", "Export-full-SeHGNN"}:
        out.setdefault("requested_budget_type", "full_graph")
        out.setdefault("requested_budget", 1.0)
        semantic = semantic or 1.0
        support_edge = support_edge or 1.0
        support_node = support_node or 1.0
        raw_ratio = raw_ratio or 1.0
    out["semantic_structural_storage_ratio"] = semantic if semantic not in {"", None} else _first_value(out, "requested_budget", 1.0)
    out["actual_semantic_structural_ratio"] = out["semantic_structural_storage_ratio"]
    out["actual_support_edge_ratio"] = support_edge if support_edge not in {"", None, "induced_schema_preserving"} else 1.0
    out["support_edge_ratio"] = out["actual_support_edge_ratio"]
    out["actual_support_node_ratio"] = support_node if support_node not in {"", None} and float_value(support_node) is not None else 1.0
    out["support_node_ratio"] = out["actual_support_node_ratio"]
    out["raw_hgb_text_byte_ratio"] = raw_ratio if raw_ratio not in {"", None} else out["semantic_structural_storage_ratio"]
    out["static_inference_package_ratio"] = _first_value(out, "static_inference_package_ratio", "preprocessed_cache_byte_ratio", "raw_hgb_text_byte_ratio")
    out["reconstructable_package_ratio"] = _first_value(out, "reconstructable_package_ratio", "transform_recipe_package_ratio", "raw_hgb_text_byte_ratio")
    out["eligible_for_official_main_table"] = bool_value(out.get("eligible_for_official_main_table", out.get("eligible_for_main_table", True)))
    out["eligible_for_main_table"] = bool_value(out.get("eligible_for_main_table", out.get("eligible_for_official_main_table", True)))
    out["eligible_for_main_decision"] = bool_value(out.get("eligible_for_main_decision", out.get("eligible_for_main_table", True)))
    out["full_fallback"] = bool_value(out.get("full_fallback", out.get("constraint_safe_fallback", False)))
    out["constraint_safe_fallback"] = bool_value(out.get("constraint_safe_fallback", False))
    out["uses_weighted_superedges"] = bool_value(out.get("uses_weighted_superedges", False))
    out["uses_synthetic_target_nodes"] = bool_value(out.get("uses_synthetic_target_nodes", False))
    out["success"] = bool_value(out.get("success", True))
    out["training_executed"] = bool_value(out.get("training_executed", True))
    out["schema_compatible"] = bool_value(out.get("schema_compatible", True))
    out["target_preserving"] = bool_value(out.get("target_preserving", True))
    out["official_hgb_exported"] = bool_value(out.get("official_hgb_exported", True))
    out["official_sehgnn_unmodified"] = bool_value(out.get("official_sehgnn_unmodified", True))
    out["uses_test_for_selection"] = bool_value(out.get("uses_test_for_selection", False))
    out["selector_uses_test_labels"] = bool_value(out.get("selector_uses_test_labels", False))
    return out


def _annotate_proxy_counts(rows: Iterable[Mapping[str, Any]], *, graph_seeds: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if bool_value(item.get("training_executed")):
            item["graph_seed_count"] = len(set(int(seed) for seed in graph_seeds)) or item.get("graph_seed_count", 1)
        out.append(item)
    return out


def _mirror_recovery_to_proxy(proxy_rows: list[dict[str, Any]], main_rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {(normalize_dataset(row.get("dataset")), str(row.get("method", ""))): row for row in main_rows}
    for row in proxy_rows:
        source = by_key.get((normalize_dataset(row.get("dataset")), str(row.get("method", ""))))
        if source:
            row["recovery_micro"] = source.get("recovery_vs_native_full_micro", "")
            row["recovery_macro"] = source.get("recovery_vs_native_full_macro", "")


def _merge_training_results(rows: list[dict[str, Any]], by_source_id: Mapping[int, Mapping[str, Any]]) -> None:
    for index, update in by_source_id.items():
        if 0 <= int(index) < len(rows):
            rows[int(index)].update(update)


def _is_prior_score_row(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    return method.startswith(("FreeHGC-score", "HGCond-score", "GCond-score", "GCondenser-score")) or family in {"condensation_score_tp_proxy", "condensation_score_as_selector"}


def _external_status(
    decision: Mapping[str, Any],
    repo_rows: Sequence[Mapping[str, Any]],
    proxy_rows: Sequence[Mapping[str, Any]],
    standard_rows: Sequence[Mapping[str, Any]],
    training_runs: Sequence[Mapping[str, Any]],
    training_failures: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "datasets": [normalize_dataset(item) for item in args.datasets],
        "baselines": list(args.baselines),
        "run_official_sehgnn": bool(args.run_official_sehgnn),
        "training_seed_count": len(args.training_seeds),
        "graph_seed_count": len(args.graph_seeds),
        "repo_audit_rows": len(repo_rows),
        "standard_protocol_rows": len(standard_rows),
        "proxy_rows": len(proxy_rows),
        "proxy_success_rows": sum(1 for row in proxy_rows if bool_value(row.get("success"))),
        "official_training_run_rows": len(training_runs),
        "official_training_failure_rows": len(training_failures),
        "ready_by_dataset_and_baseline": decision.get("READY_BY_DATASET_AND_BASELINE", {}),
        "paper_final_table_ready": decision.get("PAPER_FINAL_TABLE_READY", False),
        "blockers": decision.get("PAPER_FINAL_TABLE_BLOCKERS", []),
    }


def _summary(
    decision: Mapping[str, Any],
    status: Mapping[str, Any],
    compact_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Gate21.22 External Baseline Completion Summary",
        "",
        f"- mode: {args.mode}",
        f"- datasets: {', '.join(normalize_dataset(item) for item in args.datasets)}",
        f"- baselines: {', '.join(args.baselines)}",
        f"- official SeHGNN requested: {bool(args.run_official_sehgnn)}",
        f"- official training rows: {status.get('official_training_run_rows')}",
        f"- official training failures: {status.get('official_training_failure_rows')}",
        f"- compact rows: {len(compact_rows)}",
        "",
        "## Decision Flags",
    ]
    for flag in GATE21_22_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## Compact Table Rows"])
    for row in compact_rows:
        lines.append(
            "- "
            f"{row.get('dataset')} | {row.get('row_category')} | {row.get('method') or 'MISSING'} | "
            f"micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')} | "
            f"eligible={row.get('eligible_for_main_decision')}"
        )
    lines.extend(["", "## Failures"])
    if not failures:
        lines.append("- none")
    for row in failures:
        lines.append(f"- {row.get('dataset')} {row.get('method')}: {row.get('failure_type')} | {str(row.get('failure_reason', row.get('error_message', '')))[:400]}")
    blockers = decision.get("PAPER_FINAL_TABLE_BLOCKERS", [])
    lines.extend(["", "## Blockers"])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _checklist(decision: Mapping[str, Any], out_dir: Path, status: Mapping[str, Any], args: argparse.Namespace) -> str:
    requirements = {
        "P0 external repo audit generated for FreeHGC/HGCond/GCond/GCondenser": (out_dir / "gate21_22_external_repo_audit.csv").exists(),
        "P1 FreeHGC standard protocol is separated from official main table": bool(decision.get("FREEHGC_STANDARD_PROTOCOL_HANDLED")),
        "P2 DBLP condensation matrix includes FreeHGC/HGCond/GCond/GCondenser TP and selector rows": bool(decision.get("DBLP_CONDENSATION_SCORE_BASELINES_READY")),
        "P3 ACM condensation matrix includes TP-local-field20 and selector-field20 rows": bool(decision.get("ACM_CONDENSATION_SCORE_BASELINES_READY")),
        "P4 IMDB condensation matrix includes TP-local-channel50 and selector-channel50 rows": bool(decision.get("IMDB_CONDENSATION_SCORE_BASELINES_READY")),
        "P5 GCondenser proxy is implemented and has metrics": bool(decision.get("GCONDENSER_PROXY_READY")),
        "P6 compact table separates external TP and condensation-score categories": bool(decision.get("COMPACT_TABLE_EXTERNAL_CATEGORIES_SEPARATED")),
        "P7 compact table has nine required rows per dataset": bool(decision.get("STAGE_COMPACT_TABLE_READY")),
        "P8 paper-final external baselines are ready": bool(decision.get("PAPER_FINAL_EXTERNAL_BASELINES_READY")),
        "P9 all required output files are present": all((out_dir / name).exists() for name in REQUIRED_OUTPUTS),
        "P10 official SeHGNN was requested by CLI": bool(args.run_official_sehgnn),
    }
    lines = ["# Gate21.22 Requirement Checklist", "", f"- mode: {args.mode}", "", "## Decision Flags", ""]
    for flag in GATE21_22_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(["", "## Attachment Requirements", ""])
    for label, passed in requirements.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {label}")
    lines.extend(["", "## Required Outputs", ""])
    for name in REQUIRED_OUTPUTS:
        lines.append(f"- [{'PASS' if (out_dir / name).exists() else 'FAIL'}] {name}")
    lines.extend(["", "## Status", "", "```json", json.dumps(dict(status), indent=2, default=str), "```"])
    return "\n".join(lines) + "\n"


def _first_value(row: Mapping[str, Any], *fields_or_values: Any) -> Any:
    for field in fields_or_values:
        value = row.get(field, "") if isinstance(field, str) else field
        if value not in {"", None, "induced_schema_preserving"}:
            return value
    return ""


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decision = run(build_arg_parser().parse_args())
    print(f"Gate21.22 PAPER_FINAL_EXTERNAL_BASELINES_READY={decision['PAPER_FINAL_EXTERNAL_BASELINES_READY']}")
    print(f"Gate21.22 PAPER_FINAL_TABLE_READY={decision['PAPER_FINAL_TABLE_READY']}")


if __name__ == "__main__":
    main()
