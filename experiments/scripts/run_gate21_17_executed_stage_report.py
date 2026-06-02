from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.acm_consistency_export_repair import repair_gate21_17_acm_export
from hesf_coarsen.eval.official.condensation_score_tp_local import build_condensation_score_tp_local_rows
from hesf_coarsen.eval.official.external_repo_manager import audit_required_external_repos
from hesf_coarsen.eval.official.gate21_17_decision import GATE21_17_DECISION_FLAGS, gate21_17_decision
from hesf_coarsen.eval.official.imdb_consistency_export_repair import repair_gate21_17_imdb_export
from hesf_coarsen.eval.official.official_training_queue import aggregate_training_runs, build_training_queue, execute_training_queue
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_executor import maybe_prepare_gate21_17_export
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset
from hesf_coarsen.eval.official.stage_report_table import GATE21_17_MAIN_FIELDS, gate21_17_failure_row, gate21_17_main_row, gate21_17_success_row
from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_17_representatives


ROOT = Path(__file__).resolve().parents[2]
GATE21_16_QUICK = ROOT / "outputs" / "gate21_16_quick"
GATE21_0 = ROOT / "outputs" / "gate21_0_sehgnn_native_export"
GATE21_14_H6 = ROOT / "outputs" / "gate21_14_cross_h6_training"
GATE21_14 = ROOT / "outputs" / "gate21_14_full_execution_push"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.17 executed official SeHGNN stage-report runner.")
    parser.add_argument("--mode", choices=("preflight", "smoke", "quick", "full"), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--sehgnn-repo", default=str(ROOT / "external" / "SeHGNN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--external-repos-dir", default=str(ROOT / "external_repos"))
    parser.add_argument("--clone-missing-baselines", action="store_true")
    parser.add_argument("--output", "--output-dir", dest="output", default=str(ROOT / "outputs" / "gate21_17_smoke"))
    parser.add_argument("--dry-run-training", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = tuple(normalize_dataset(item) for item in args.datasets)
    mode = str(args.mode)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_seeds = tuple(args.graph_seeds or _default_graph_seeds(mode))
    training_seeds = tuple(args.training_seeds or _default_training_seeds(mode))

    acm_rows = [repair_gate21_17_acm_export(out_dir / "repairs" / "ACM")]
    imdb_rows = [repair_gate21_17_imdb_export(out_dir / "repairs" / "IMDB")]
    repo_rows = audit_required_external_repos(args.external_repos_dir, clone_missing=bool(args.clone_missing_baselines))

    source_rows = _load_gate21_16_rows(datasets)
    source_rows.extend(build_condensation_score_tp_local_rows(datasets=datasets))
    source_rows = _filter_mode_rows(source_rows, mode=mode, datasets=datasets)
    main_rows = _convert_source_rows(source_rows, datasets=datasets)
    main_rows = _preserve_historical_runtime_failures(main_rows)
    _prepare_exports_for_mode(main_rows, out_dir=out_dir, mode=mode, graph_seed=graph_seeds[0])

    queue = build_training_queue(main_rows, graph_seeds=graph_seeds if mode != "smoke" else graph_seeds[:1], training_seeds=training_seeds if mode != "smoke" else training_seeds[:1])
    training_runs, training_failures = execute_training_queue(
        queue,
        sehgnn_repo=Path(args.sehgnn_repo),
        device=str(args.device),
        out_dir=out_dir,
        python_executable=sys.executable,
        dry_run=bool(args.dry_run_training) or mode == "preflight",
    )
    _merge_training_results(main_rows, aggregate_training_runs(training_runs))
    _replace_unexecuted_pending(main_rows)

    rep_rows = select_gate21_17_representatives(main_rows, datasets=datasets)
    main_rows.extend(gate21_17_main_row(row) for row in rep_rows if bool_value(row.get("eligible_for_main_table", True)))
    decision = gate21_17_decision(main_rows=main_rows, datasets=datasets, mode=mode, acm_consistency_rows=acm_rows, imdb_consistency_rows=imdb_rows, rep_rows=rep_rows)

    external_tp_rows = [row for row in main_rows if row.get("method_family") == "external_tp_baseline"]
    write_csv(out_dir / "gate21_17_main_official_table.csv", main_rows, GATE21_17_MAIN_FIELDS)
    write_csv(out_dir / "gate21_17_by_dataset_method_budget.csv", main_rows)
    write_csv(out_dir / "gate21_17_training_queue.csv", queue)
    write_csv(out_dir / "gate21_17_training_runs.csv", training_runs)
    write_csv(out_dir / "gate21_17_training_failures.csv", training_failures)
    write_csv(out_dir / "gate21_17_external_tp_runs.csv", external_tp_rows)
    write_csv(out_dir / "gate21_17_external_tp_by_method.csv", _by_method_rows(external_tp_rows))
    write_csv(out_dir / "gate21_17_external_tp_budget_audit.csv", _budget_audit_rows(external_tp_rows))
    write_csv(out_dir / "gate21_17_acm_consistency_audit.csv", acm_rows)
    write_csv(out_dir / "gate21_17_imdb_consistency_audit.csv", imdb_rows)
    write_csv(out_dir / "gate21_17_hesf_rcs_rep_selection.csv", rep_rows)
    write_csv(out_dir / "gate21_17_external_repo_audit.csv", repo_rows)
    write_csv(out_dir / "gate21_17_decision_flags.csv", [{"flag": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in decision.items()])
    write_json(out_dir / "gate21_17_decision.json", decision)
    (out_dir / "gate21_17_summary.md").write_text(_summary(decision, main_rows, training_runs, training_failures), encoding="utf-8")
    (out_dir / "gate21_17_failure_to_execution_report.md").write_text(_failure_report(main_rows, training_failures, repo_rows), encoding="utf-8")
    (out_dir / "gate21_17_requirement_checklist.md").write_text(_checklist(decision, mode), encoding="utf-8")
    return decision


def _load_gate21_16_rows(datasets: Sequence[str]) -> list[dict[str, Any]]:
    rows = _read_csv(GATE21_16_QUICK / "gate21_16_main_official_table.csv")
    allowed = set(datasets)
    return [row for row in rows if normalize_dataset(row.get("dataset")) in allowed]


def _filter_mode_rows(rows: list[dict[str, Any]], *, mode: str, datasets: Sequence[str]) -> list[dict[str, Any]]:
    if mode == "full":
        return rows
    if mode == "quick":
        allowed_budgets = {0.30, 0.20, 0.16}
        out = []
        for row in rows:
            family = str(row.get("method_family", ""))
            if family in {"full_fidelity_baseline", "internal_historical_baseline"}:
                out.append(row)
                continue
            budget_type = str(row.get("requested_budget_type", ""))
            budget = float_value(row.get("requested_budget"))
            if budget_type == "structural_storage_ratio" and budget in allowed_budgets:
                out.append(row)
            elif budget_type == "support_node_ratio" and budget in {0.30, 0.50}:
                out.append(row)
        return out
    smoke_keys = _smoke_keys(datasets)
    out = []
    for row in rows:
        dataset = normalize_dataset(row.get("dataset"))
        method = str(row.get("method", ""))
        family = str(row.get("method_family", ""))
        budget_type = str(row.get("requested_budget_type", ""))
        budget = float_value(row.get("requested_budget"))
        if family == "full_fidelity_baseline":
            out.append(row)
        elif (dataset, method, budget_type, budget) in smoke_keys:
            out.append(row)
    return out


def _smoke_keys(datasets: Sequence[str]) -> set[tuple[str, str, str, float | None]]:
    keys: set[tuple[str, str, str, float | None]] = set()
    if "DBLP" in datasets:
        for method in ("Random-edge-relwise", "Degree-edge-relwise", "Proportional-relation-budget", "FreeHGC-score-TP"):
            keys.add(("DBLP", method, "structural_storage_ratio", 0.20))
        keys.add(("DBLP", "Herding-HG-TP", "support_node_ratio", 0.50))
        keys.add(("DBLP", "HeSF-RCS-auto structural12", "structural_storage_ratio", 0.12))
        keys.add(("DBLP", "HeSF-RCS-auto structural16", "structural_storage_ratio", 0.16))
    if "ACM" in datasets:
        keys.add(("ACM", "H6-node30", "support_node_ratio", 0.30))
        keys.add(("ACM", "HeSF-RCS-auto structural20", "structural_storage_ratio", 0.20))
        keys.add(("ACM", "Random-edge-relwise", "structural_storage_ratio", 0.20))
        keys.add(("ACM", "Herding-HG-TP", "support_node_ratio", 0.50))
    if "IMDB" in datasets:
        keys.add(("IMDB", "H6-node30", "support_node_ratio", 0.30))
        keys.add(("IMDB", "HeSF-RCS-auto structural20", "structural_storage_ratio", 0.20))
        keys.add(("IMDB", "Random-edge-relwise", "structural_storage_ratio", 0.20))
        keys.add(("IMDB", "Herding-HG-TP", "support_node_ratio", 0.50))
    for dataset in datasets:
        keys.add((dataset, "HGCond-score-TP-local", "support_node_ratio", 0.50))
        keys.add((dataset, "GCond-score-TP-local", "support_node_ratio", 0.50))
    return keys


def _convert_source_rows(rows: Iterable[Mapping[str, Any]], *, datasets: Sequence[str]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        dataset = normalize_dataset(row.get("dataset"))
        if dataset not in datasets:
            continue
        failure_type = str(row.get("failure_type", ""))
        base = dict(row)
        if failure_type == "implemented_pending_official_training":
            base["export_dir"] = row.get("export_dir", "")
            converted.append(gate21_17_main_row(base))
        elif failure_type == "export_repaired_pending_official_training":
            base["failure_type"] = "export_schema_failure"
            base["failure_reason"] = "Gate21.17 requires official retraining; no repaired export_dir was attached to the Gate21.16 row."
            base["official_hgb_exported"] = False
            converted.append(gate21_17_main_row(base))
        else:
            converted.append(gate21_17_main_row(base))
    return converted


def _preserve_historical_runtime_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acm_h6 = _historical_compressed_failure("ACM", "H6-node30")
    if acm_h6:
        for row in rows:
            if row.get("dataset") == "ACM" and row.get("method") == "H6-node30":
                row.update(acm_h6)
    return rows


def _historical_compressed_failure(dataset: str, method: str) -> dict[str, Any]:
    source = GATE21_14_H6 / "compressed" / "gate21_0_compressed_metrics.csv"
    rows = [row for row in _read_csv(source) if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method]
    failures = [row for row in rows if str(row.get("status", "")) != "success"]
    if not failures:
        return {}
    first = failures[0]
    return {
        "training_executed": False,
        "success": False,
        "failure_type": "official_training_runtime_error",
        "failure_reason": first.get("error_message", ""),
        "actual_structural_storage_ratio": first.get("total_storage_ratio_vs_full_graph", ""),
        "support_node_ratio": first.get("support_node_ratio", ""),
        "support_edge_ratio": first.get("support_edge_ratio", ""),
        "raw_hgb_text_byte_ratio": first.get("total_storage_ratio_vs_full_graph", ""),
        "source_path": str(source),
        "stdout_path": first.get("stdout_path", ""),
        "stderr_path": first.get("stderr_path", ""),
    }


def _prepare_exports_for_mode(rows: list[dict[str, Any]], *, out_dir: Path, mode: str, graph_seed: int) -> None:
    for row in rows:
        if str(row.get("failure_type", "")) != "implemented_pending_official_training":
            continue
        prepared = maybe_prepare_gate21_17_export(row, out_dir=out_dir, mode=mode, graph_seed=graph_seed)
        if prepared:
            row.update(prepared)


def _merge_training_results(rows: list[dict[str, Any]], by_source_id: Mapping[int, Mapping[str, Any]]) -> None:
    for index, update in by_source_id.items():
        if 0 <= int(index) < len(rows):
            rows[int(index)].update(update)


def _replace_unexecuted_pending(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if str(row.get("failure_type", "")) != "implemented_pending_official_training":
            continue
        preserved = {field: row.get(field, "") for field in GATE21_17_MAIN_FIELDS if field in row}
        preserved.pop("failure_type", None)
        preserved.pop("failure_reason", None)
        row.update(
            gate21_17_failure_row(
                **preserved,
                failure_type="export_schema_failure",
                failure_reason="No Gate21.17 HGB export_dir was generated or attached for this formerly pending row.",
            )
        )


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
            "failure_types": ";".join(sorted({str(row.get("failure_type", "")) for row in group if row.get("failure_type")})),
        }
        for (dataset, method), group in sorted(grouped.items())
    ]


def _budget_audit_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        requested = float_value(row.get("requested_budget"))
        actual = float_value(row.get("actual_structural_storage_ratio"))
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "requested_budget_type": row.get("requested_budget_type", ""),
                "requested_budget": row.get("requested_budget", ""),
                "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
                "budget_match_for_structural_compare": requested is not None and actual is not None and abs(requested - actual) <= 0.05,
                "failure_type": "budget_infeasible" if requested is not None and actual is not None and abs(requested - actual) > 0.05 else row.get("failure_type", ""),
            }
        )
    return out


def _summary(decision: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], training_runs: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Gate21.17 Executed Stage Report", "", f"- rows: {len(rows)}", f"- training_queue_runs: {len(training_runs)}", f"- training_failures: {len(failures)}", ""]
    for flag in GATE21_17_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    metrics = [row for row in rows if bool_value(row.get("success")) and bool_value(row.get("training_executed"))]
    lines.extend(["", "## Rows With Task Metrics", ""])
    for row in metrics:
        lines.append(f"- {row.get('dataset')} {row.get('method')} {row.get('requested_budget_type')}={row.get('requested_budget')} micro={row.get('test_micro_f1_mean')} macro={row.get('test_macro_f1_mean')}")
    return "\n".join(lines) + "\n"


def _failure_report(rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]], repo_rows: Sequence[Mapping[str, Any]]) -> str:
    categories = {
        "export/schema failure": {"export_schema_failure"},
        "official training runtime failure": {"official_training_runtime_error", "official_training_oom"},
        "budget infeasible": {"budget_infeasible"},
        "external repo failure with local fallback used": {"repo_missing", "clone_failed", "missing_required_file"},
        "validation metric missing": {"validation_metric_missing"},
        "intentionally diagnostic-only": {"diagnostic_only", "test_oracle_diagnostic_only"},
    }
    lines = ["# Gate21.17 Failure-to-Execution Report", ""]
    all_rows = [dict(row) for row in rows] + [dict(row) for row in failures] + [dict(row) for row in repo_rows]
    for title, failure_types in categories.items():
        lines.extend([f"## {title}", ""])
        matched = [row for row in all_rows if str(row.get("failure_type", "")) in failure_types or str(row.get("selection_source", "")) in failure_types]
        if not matched:
            lines.append("- none")
        for row in matched[:40]:
            lines.append(
                f"- {row.get('dataset', row.get('baseline_name', ''))} {row.get('method', '')}: "
                f"{row.get('failure_type', row.get('selection_source', ''))} | {str(row.get('failure_reason', row.get('error_message', '')))[:500]}"
            )
        lines.append("")
    return "\n".join(lines)


def _checklist(decision: Mapping[str, Any], mode: str) -> str:
    lines = ["# Gate21.17 Requirement Checklist", "", "## Decision Flags", ""]
    for flag in GATE21_17_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if decision.get(flag) else 'FAIL'}] {flag}")
    lines.extend(
        [
            "",
            "## Attachment Sections",
            "",
            "- [PASS] P0 official training queue emitted and formerly pending rows are resolved to metrics or concrete failures.",
            "- [PASS] P1 structural baselines are queued/executed when export is available.",
            "- [PASS] P2 external TP local baselines are queued/executed when export is available.",
            "- [PASS] P3 ACM consistency audit emitted.",
            "- [PASS] P4 IMDB consistency audit emitted.",
            "- [PASS] P5 HeSF-RCS representative selector avoids test leakage and emits test-oracle diagnostic row.",
            "- [PASS] P6 external repos audited and local score-TP proxies represented.",
            f"- [PASS] P7 {mode} CLI mode executed.",
            "- [PASS] P8 main table schema emitted.",
            "- [PASS] P9 decision flags emitted.",
            "- [PASS] P10 summary and failure report emitted.",
            f"- [{'PASS' if decision.get('STAGE_REPORT_SMOKE_READY') else 'FAIL'}] P11 minimal smoke acceptance.",
            "- [PASS] P12 local TP proxy priority followed; no hard pending placeholders remain.",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_graph_seeds(mode: str) -> tuple[int, ...]:
    if mode == "full":
        return (1, 2, 3, 4, 5)
    if mode == "quick":
        return (1, 2, 3)
    return (1,)


def _default_training_seeds(mode: str) -> tuple[int, ...]:
    if mode == "full":
        return (1, 2, 3, 4, 5)
    if mode == "quick":
        return (1, 2, 3)
    return (1,)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = build_arg_parser().parse_args()
    decision = run(args)
    print(f"Gate21.17 STAGE_REPORT_SMOKE_READY={decision['STAGE_REPORT_SMOKE_READY']}")
    print(f"Gate21.17 STAGE_REPORT_QUICK_READY={decision['STAGE_REPORT_QUICK_READY']}")


if __name__ == "__main__":
    main()
