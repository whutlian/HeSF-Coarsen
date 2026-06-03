from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.run_gate21_19_multidataset_frontier import _add_recovery
from hesf_coarsen.eval.official.acm_selector_overlap import (
    GATE21_21_ACM_SELECTOR_OVERLAP_FIELDS,
    build_gate21_21_acm_selector_overlap_rows,
    rows_from_gate21_exports,
)
from hesf_coarsen.eval.official.critical_robustness_runner import ROBUSTNESS_FIELDS, build_critical_robustness_rows
from hesf_coarsen.eval.official.external_repo_manager import GATE21_21_EXTERNAL_REPO_AUDIT_FIELDS, audit_gate21_21_required_external_repos
from hesf_coarsen.eval.official.final_compact_table import (
    GATE21_21_FINAL_COMPACT_FIELDS,
    build_gate21_21_final_compact_table,
    compact_table_markdown,
)
from hesf_coarsen.eval.official.freehgc_score_tp import (
    FREEHGC_SELECTOR_GATE21_21_FIELDS,
    FREEHGC_STANDARD_FIELDS,
    FREEHGC_TP_LOCAL_FIELDS,
    build_freehgc_score_selector_rows_from_main,
    build_freehgc_score_tp_local_rows_from_main,
    build_freehgc_standard_rows,
)
from hesf_coarsen.eval.official.gate21_21_decision import GATE21_21_DECISION_FLAGS, decision_flag_rows, gate21_21_decision
from hesf_coarsen.eval.official.hgcond_gcond_score_tp import HGCOND_GCOND_SCORE_TP_FIELDS, build_hgcond_gcond_score_tp_rows
from hesf_coarsen.eval.official.imdb_channel_planner import (
    IMDB_CHANNEL_FRONTIER_FIELDS,
    IMDB_CHANNEL_PLANNER_FIELDS,
    build_gate21_21_imdb_channel_frontier_rows,
    build_gate21_21_imdb_channel_planner_rows,
)
from hesf_coarsen.eval.official.pareto_frontier import GATE21_21_FRONTIER_FIELDS, build_gate21_21_frontier_rows
from hesf_coarsen.eval.official.rep_selection import GATE21_21_REP_SELECTION_FIELDS, select_gate21_21_representatives
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


DEFAULT_GATE21_20 = ROOT / "outputs" / "gate21_20_quick_robust"
DEFAULT_OUT = ROOT / "results" / "gate21_21_final_rep_external_baselines"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate21.21 final representative repair and external baseline integration.")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--mode", choices=("smoke", "quick", "preflight"), default="smoke")
    parser.add_argument("--reuse-gate21-20", action="store_true")
    parser.add_argument("--gate21-20-dir", default=str(DEFAULT_GATE21_20))
    parser.add_argument("--out", "--output", dest="out", default=str(DEFAULT_OUT))
    parser.add_argument("--external-repos-dir", default=str(ROOT / "external_repos"))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [normalize_dataset(item) for item in args.datasets]
    source_dir = Path(args.gate21_20_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Gate21.20 source directory not found: {source_dir}")

    main_rows = [_normalize_gate21_21_main_row(row) for row in _read_csv(source_dir / "gate21_20_main_official_table.csv") if not _is_prior_rep_or_oracle(row)]
    training_runs = _read_csv(source_dir / "gate21_20_training_runs.csv")
    training_failures = _read_csv(source_dir / "gate21_20_training_failures.csv")

    imdb_planner_rows = build_gate21_21_imdb_channel_planner_rows(main_rows)
    imdb_ready = [_normalize_gate21_21_main_row(row) for row in imdb_planner_rows if bool_value(row.get("success"))]
    main_rows.extend(imdb_ready)
    _add_recovery(main_rows)
    main_rows = [_normalize_gate21_21_main_row(row) for row in main_rows]

    external_repo_rows = audit_gate21_21_required_external_repos(Path(args.external_repos_dir), clone_missing=True)
    acm_overlap_inputs = rows_from_gate21_exports(main_rows)
    acm_overlap_rows = build_gate21_21_acm_selector_overlap_rows(acm_overlap_inputs)
    rep_rows = select_gate21_21_representatives(main_rows, datasets=datasets)
    freehgc_standard_rows = build_freehgc_standard_rows(external_repo_rows)
    freehgc_tp_rows = build_freehgc_score_tp_local_rows_from_main(main_rows)
    freehgc_selector_rows = build_freehgc_score_selector_rows_from_main(main_rows, datasets=datasets)
    hgcond_gcond_rows = build_hgcond_gcond_score_tp_rows(main_rows, datasets=datasets)
    frontier_rows = build_gate21_21_frontier_rows(main_rows, datasets=datasets)
    compact_rows = build_gate21_21_final_compact_table(main_rows, rep_rows, datasets=datasets)
    robustness_rows = build_critical_robustness_rows(main_rows, training_runs, critical_methods=_critical_methods(compact_rows))
    imdb_frontier_rows = build_gate21_21_imdb_channel_frontier_rows(imdb_planner_rows)
    decision = gate21_21_decision(
        main_rows=main_rows,
        rep_rows=rep_rows,
        compact_rows=compact_rows,
        frontier_rows=frontier_rows,
        external_repo_rows=external_repo_rows,
        freehgc_standard_rows=freehgc_standard_rows,
        freehgc_tp_rows=freehgc_tp_rows,
        freehgc_selector_rows=freehgc_selector_rows,
        acm_overlap_rows=acm_overlap_rows,
        imdb_planner_rows=imdb_planner_rows,
        datasets=datasets,
    )

    write_csv(out_dir / "gate21_21_main_official_table.csv", main_rows)
    write_csv(out_dir / "gate21_21_rep_selection.csv", rep_rows, GATE21_21_REP_SELECTION_FIELDS)
    write_csv(out_dir / "gate21_21_frontiers.csv", frontier_rows, GATE21_21_FRONTIER_FIELDS)
    write_csv(out_dir / "gate21_21_final_compact_table.csv", compact_rows, GATE21_21_FINAL_COMPACT_FIELDS)
    (out_dir / "gate21_21_final_compact_table.md").write_text(compact_table_markdown(compact_rows), encoding="utf-8")
    write_csv(out_dir / "gate21_21_best_method_comparison.csv", compact_rows, GATE21_21_FINAL_COMPACT_FIELDS)
    write_csv(out_dir / "gate21_21_robustness_by_method.csv", robustness_rows, ROBUSTNESS_FIELDS)
    write_csv(out_dir / "gate21_21_external_repo_audit.csv", external_repo_rows, GATE21_21_EXTERNAL_REPO_AUDIT_FIELDS)
    write_csv(out_dir / "gate21_21_freehgc_standard.csv", freehgc_standard_rows, FREEHGC_STANDARD_FIELDS)
    write_csv(out_dir / "gate21_21_freehgc_score_tp_local.csv", freehgc_tp_rows, FREEHGC_TP_LOCAL_FIELDS)
    write_csv(out_dir / "gate21_21_freehgc_score_selector.csv", freehgc_selector_rows, FREEHGC_SELECTOR_GATE21_21_FIELDS)
    write_csv(out_dir / "gate21_21_hgcond_gcond_score_tp.csv", hgcond_gcond_rows, HGCOND_GCOND_SCORE_TP_FIELDS)
    write_csv(out_dir / "gate21_21_acm_selector_overlap.csv", acm_overlap_rows, GATE21_21_ACM_SELECTOR_OVERLAP_FIELDS)
    write_csv(out_dir / "gate21_21_imdb_channel_planner.csv", imdb_planner_rows, IMDB_CHANNEL_PLANNER_FIELDS)
    write_csv(out_dir / "gate21_21_imdb_channel_frontier.csv", imdb_frontier_rows, IMDB_CHANNEL_FRONTIER_FIELDS)
    write_csv(out_dir / "gate21_21_training_runs.csv", training_runs)
    write_csv(out_dir / "gate21_21_training_failures.csv", training_failures)
    write_csv(out_dir / "gate21_21_decision_flags.csv", decision_flag_rows(decision))
    write_json(out_dir / "gate21_21_decision.json", decision)
    (out_dir / "gate21_21_summary.md").write_text(_summary(decision, rep_rows, compact_rows, external_repo_rows, training_failures, args), encoding="utf-8")
    (out_dir / "gate21_21_requirement_checklist.md").write_text(_checklist(decision, out_dir, rep_rows, acm_overlap_rows, imdb_planner_rows, freehgc_selector_rows, hgcond_gcond_rows), encoding="utf-8")
    return decision


def _normalize_gate21_21_main_row(row: Mapping[str, Any]) -> dict[str, Any]:
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
    out["semantic_structural_storage_ratio"] = semantic if semantic not in {"", None} else 1.0
    out["actual_semantic_structural_ratio"] = out["semantic_structural_storage_ratio"]
    out["actual_support_edge_ratio"] = support_edge if support_edge not in {"", None, "induced_schema_preserving"} else 1.0
    out["support_edge_ratio"] = out["actual_support_edge_ratio"]
    out["actual_support_node_ratio"] = support_node if support_node not in {"", None} else 1.0
    out["support_node_ratio"] = out["actual_support_node_ratio"]
    out["raw_hgb_text_byte_ratio"] = raw_ratio if raw_ratio not in {"", None} else out["semantic_structural_storage_ratio"]
    out["static_inference_package_ratio"] = _first_value(out, "static_inference_package_ratio", "preprocessed_cache_byte_ratio", "raw_hgb_text_byte_ratio")
    out["reconstructable_package_ratio"] = _first_value(out, "reconstructable_package_ratio", "transform_recipe_package_ratio", "raw_hgb_text_byte_ratio")
    out["eligible_for_official_main_table"] = bool_value(out.get("eligible_for_main_table", True))
    out["full_fallback"] = bool_value(out.get("constraint_safe_fallback"))
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


def _critical_methods(compact_rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    methods: set[tuple[str, str]] = {
        ("DBLP", "HeSF-RCS-auto structural16"),
        ("DBLP", "HeSF-RCS-auto structural12"),
        ("DBLP", "Random-edge-relwise"),
        ("DBLP", "Proportional-relation-budget"),
        ("DBLP", "FreeHGC-score-as-selector structural16"),
        ("DBLP", "FreeHGC-score-as-selector structural20"),
        ("IMDB", "HeSF-RCS-IMDB-ChannelPlanner-channel50"),
        ("IMDB", "IMDB-ValidationGreedy-channel50"),
        ("IMDB", "IMDB-MDfull-MA50-MK50"),
        ("ACM", "ACM-HeSF-RCS-auto-field20"),
        ("ACM", "ACM-Degree-field20"),
        ("ACM", "ACM-ValidationGreedy-field20"),
    }
    for row in compact_rows:
        if str(row.get("row_category", "")) in {"Best-external-TP-baseline", "HeSF-RCS-Rep-Validated"} and str(row.get("method", "")):
            methods.add((normalize_dataset(row.get("dataset")), str(row.get("method", ""))))
    return tuple(sorted(methods))


def _summary(
    decision: Mapping[str, Any],
    rep_rows: Sequence[Mapping[str, Any]],
    compact_rows: Sequence[Mapping[str, Any]],
    external_repo_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Gate21.21 Final Representative Repair + External Baseline Integration Summary",
        "",
        f"- mode: {args.mode}",
        f"- reused Gate21.20: {bool(args.reuse_gate21_20)}",
        f"- main official rows: {len(_read_csv(Path(args.out) / 'gate21_21_main_official_table.csv')) if (Path(args.out) / 'gate21_21_main_official_table.csv').exists() else 'pending'}",
        f"- compact rows: {len(compact_rows)}",
        f"- training failures carried from Gate21.20: {len([row for row in failures if row.get('method')])}",
        "",
        "## Decision Flags",
    ]
    for flag in GATE21_21_DECISION_FLAGS:
        lines.append(f"- {flag}: {decision.get(flag)}")
    lines.extend(["", "## HeSF-RCS Representatives"])
    for row in rep_rows:
        if str(row.get("rep_type", "")) == "HeSF-RCS-Rep-Validated":
            lines.append(f"- {row.get('dataset')}: {row.get('selected_method') or 'MISSING'} | status={row.get('selection_status')} | uses_test={row.get('uses_test_for_selection')}")
    lines.extend(["", "## External Repo Audit"])
    for row in external_repo_rows:
        lines.append(f"- {row.get('method')}: clone={row.get('clone_status')} commit={row.get('commit_hash')} required_files={row.get('required_files_present')} fallback_proxy={row.get('fallback_local_proxy_required')}")
    lines.extend(["", "## Paper Final Blockers"])
    blockers = decision.get("PAPER_FINAL_TABLE_BLOCKERS", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _checklist(
    decision: Mapping[str, Any],
    out_dir: Path,
    rep_rows: Sequence[Mapping[str, Any]],
    acm_overlap_rows: Sequence[Mapping[str, Any]],
    imdb_planner_rows: Sequence[Mapping[str, Any]],
    freehgc_selector_rows: Sequence[Mapping[str, Any]],
    hgcond_gcond_rows: Sequence[Mapping[str, Any]],
) -> str:
    required_outputs = (
        "gate21_21_final_compact_table.csv",
        "gate21_21_final_compact_table.md",
        "gate21_21_main_official_table.csv",
        "gate21_21_rep_selection.csv",
        "gate21_21_frontiers.csv",
        "gate21_21_best_method_comparison.csv",
        "gate21_21_robustness_by_method.csv",
        "gate21_21_decision.json",
        "gate21_21_decision_flags.csv",
        "gate21_21_summary.md",
        "gate21_21_external_repo_audit.csv",
        "gate21_21_freehgc_standard.csv",
        "gate21_21_freehgc_score_tp_local.csv",
        "gate21_21_freehgc_score_selector.csv",
        "gate21_21_hgcond_gcond_score_tp.csv",
        "gate21_21_acm_selector_overlap.csv",
        "gate21_21_imdb_channel_planner.csv",
        "gate21_21_imdb_channel_frontier.csv",
    )
    reqs = {
        "P0 HeSF reps use only HeSF pool and validation metrics": decision.get("HESF_REP_CANDIDATE_POOL_PASS") and decision.get("HESF_REP_VALIDATION_SELECTION_PASS") and decision.get("HESF_REP_NO_TEST_LEAKAGE"),
        "P1 IMDB ChannelPlanner channel20/30/40/50/75 emitted": _all_methods(imdb_planner_rows, "HeSF-RCS-IMDB-ChannelPlanner-channel", (20, 30, 40, 50, 75)),
        "P2 ACM selector overlap pairwise audit emitted": bool(acm_overlap_rows),
        "P3 DBLP compact comparison includes HeSF rep/external/baseline": decision.get("DBLP_FINAL_COMPARISON_READY"),
        "P4 Pareto/frontier flags recomputed": decision.get("PARETO_FLAGS_RECOMPUTED"),
        "P5 final compact table emitted": decision.get("FINAL_COMPACT_TABLE_READY"),
        "P6 decision flags emitted": (out_dir / "gate21_21_decision_flags.csv").exists(),
        "P7 runner/summarizer/module files added": _runner_modules_present(),
        "External repos audited or local proxies required": decision.get("EXTERNAL_REPOS_CLONED_OR_LOCAL_PROXY_IMPLEMENTED"),
        "FreeHGC required selector rows emitted": _freehgc_selector_required_rows(freehgc_selector_rows),
        "HGCond/GCond TP and selector proxy rows emitted": bool(hgcond_gcond_rows),
    }
    lines = ["# Gate21.21 Requirement Checklist", "", "## Decision Flags"]
    for flag in GATE21_21_DECISION_FLAGS:
        lines.append(f"- [{'PASS' if bool_value(decision.get(flag)) else 'FAIL'}] {flag}")
    lines.extend(["", "## Required Outputs"])
    for name in required_outputs:
        lines.append(f"- [{'PASS' if (out_dir / name).exists() else 'FAIL'}] {name}")
    lines.extend(["", "## Attachment Requirements"])
    for name, passed in reqs.items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}")
    lines.extend(["", "## Incomplete / Honest Blockers"])
    blockers = decision.get("PAPER_FINAL_TABLE_BLOCKERS", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _all_methods(rows: Sequence[Mapping[str, Any]], prefix: str, suffixes: Sequence[int]) -> bool:
    methods = {str(row.get("method", "")) for row in rows}
    return all(f"{prefix}{suffix:02d}" in methods for suffix in suffixes)


def _freehgc_selector_required_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    methods = {(normalize_dataset(row.get("dataset")), str(row.get("method", ""))) for row in rows}
    return {
        ("DBLP", "FreeHGC-score-as-selector structural16"),
        ("DBLP", "FreeHGC-score-as-selector structural20"),
        ("ACM", "ACM-FreeHGC-score-as-selector-field20"),
        ("IMDB", "IMDB-FreeHGC-score-as-selector-channel50"),
    }.issubset(methods)


def _runner_modules_present() -> bool:
    required = (
        ROOT / "experiments" / "scripts" / "run_gate21_21_final_rep_external_baselines.py",
        ROOT / "experiments" / "scripts" / "summarize_gate21_21_final_rep_external_baselines.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "gate21_21_decision.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "final_compact_table.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "rep_selection.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "imdb_channel_planner.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "acm_selector_overlap.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "external_repo_manager.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "freehgc_score_tp.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "hgcond_gcond_score_tp.py",
        ROOT / "hesf_coarsen" / "eval" / "official" / "pareto_frontier.py",
    )
    return all(path.exists() for path in required)


def _is_prior_rep_or_oracle(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    return "Rep-Validated" in method or "TestOracle" in method


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
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
    print(f"Gate21.21 FINAL_COMPACT_TABLE_READY={decision['FINAL_COMPACT_TABLE_READY']}")
    print(f"Gate21.21 PAPER_FINAL_TABLE_READY={decision['PAPER_FINAL_TABLE_READY']}")


if __name__ == "__main__":
    main()
