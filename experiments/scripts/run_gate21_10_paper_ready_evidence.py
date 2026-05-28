from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_10_common import (
    COMPRESSION_FIELDS,
    DEFAULT_GATE21_9_ROOT,
    DEFAULT_OUTPUT_ROOT,
    NO_LEAKAGE_FIELDS,
    add_required_gate21_10_fields,
    bool_value,
    ensure_layout,
    float_value,
    mean_field,
    normalize_metric_fields,
    parse_bool_arg,
    read_csv,
    read_json,
    write_summary_csv,
    write_summary_json,
)
from hesf_coarsen.eval.official.adapter_package_manifest import aggregate_adapter_by_method_gate21_10
from hesf_coarsen.eval.official.budgeted_channel_planner import deterministic_selection_proof, plan_budgeted_channels
from hesf_coarsen.eval.official.external_tp_task_runner import budget_match_status, external_tp_by_method
from hesf_coarsen.eval.official.feature_ablation_task_runner import (
    GATE21_10_FEATURE_TRANSFORMS,
    GATE21_10_LABEL_GRAPH_SETTINGS,
    GATE21_10_METHODS,
)
from hesf_coarsen.eval.official.freehgc_standard_runner import freehgc_standard_ratios
from hesf_coarsen.eval.official.freehgc_tp_export_adapter import build_gate21_10_freehgc_tp_audit_rows
from hesf_coarsen.eval.official.runner_utils import git_commit_hash
from hesf_coarsen.eval.official.storage_denominator_audit import storage_denominator_audit
from hesf_coarsen.eval.official.system_workload_cost import normalize_system_workload_row


EXTERNAL_TP_METHODS = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
    "FreeHGC-TP-selection",
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = ensure_layout(Path(args.output_root))
    stages = _selected_stages(args.stage)
    graph_seeds = _seed_list(args.graph_seeds, quick=bool(args.quick))
    training_seeds = _seed_list(args.training_seeds, quick=bool(args.quick))
    seeds = _seed_list(args.seeds, quick=bool(args.quick))
    manifest = {
        "gate": "21.10",
        "dataset": str(args.dataset).upper(),
        "output_root": str(Path(args.output_root)),
        "gate21_9_root": str(Path(args.gate21_9_root)),
        "freehgc_root": str(Path(args.freehgc_root)),
        "freehgc_zip": str(Path(args.freehgc_zip)),
        "stages": sorted(stages),
        "graph_seeds": graph_seeds,
        "training_seeds": training_seeds,
        "seeds": seeds,
        "structural_budgets": [float(item) for item in args.structural_budgets],
        "support_node_budgets": [float(item) for item in args.support_node_budgets],
        "condensation_ratios": [float(item) for item in args.condensation_ratios],
        "device": str(args.device),
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "reuse_gate21_9_anchors": bool(args.reuse_gate21_9_anchors),
        "hesf_commit": git_commit_hash(Path.cwd()) or "",
    }
    write_summary_json(paths["audits"] / "gate21_10_run_manifest.json", manifest)
    _write_readmes(paths)

    if not args.dry_run:
        if "official_main" in stages:
            _write_official_main(paths, args)
        if "auto_selector" in stages:
            _write_auto_selector(paths, args)
        if "external_tp" in stages:
            _write_external_tp(paths, args, graph_seeds, training_seeds)
        if "freehgc_standard" in stages or "freehgc_tp" in stages:
            _write_freehgc(paths, args, seeds)
        if "metapath_cache" in stages:
            _write_metapath_cache(paths, args)
        if "feature_ablation" in stages:
            _write_feature_ablation(paths, args)
        if "adapter" in stages:
            _write_adapter(paths, args)
        if "storage_system" in stages:
            _write_storage_system(paths, args)
        if "cross_dataset" in stages:
            _write_cross_dataset(paths, args)
        if "audits" in stages:
            _write_audits(paths, args)
    else:
        _write_dry_run_manifest(paths, stages)

    from experiments.scripts.summarize_gate21_10_paper_ready_evidence import summarize

    decision = summarize(
        input_root=Path(args.output_root),
        output_root=paths["summary"],
        gate21_9_root=Path(args.gate21_9_root),
        fail_on_missing_required=bool(args.fail_on_missing_required),
    )
    return {
        "output_root": str(Path(args.output_root)),
        "summary_root": str(paths["summary"]),
        "paper_ready_status": decision.get("paper_ready_status"),
        "blocking_issues": decision.get("blocking_issues", []),
    }


def _write_official_main(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_9_root) / "gate21_9_main_table_official.csv")
    rows = [_official_row(row, args) for row in source]
    if not rows:
        rows = [
            add_required_gate21_10_fields(
                {
                    "dataset": str(args.dataset).upper(),
                    "method": "missing-gate21-9-official-anchor",
                    "training_executed": False,
                    "success": False,
                    "failure_type": "missing_gate21_9_main_table_official",
                    "failure_message": "Gate21.10 could not find Gate21.9 official anchors to reuse.",
                    "source_gate": "gate21_10",
                },
                table="official_main",
            )
        ]
    write_summary_csv(paths["official_main"] / "gate21_10_official_main_by_method.csv", rows)


def _official_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = normalize_metric_fields(dict(row))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["source_gate"] = out.get("source_gate", "gate21_9")
    out["protocol"] = "schema_preserving_tp"
    out["structural_storage_ratio"] = out.get("structural_storage_ratio", out.get("actual_structural_storage_ratio", ""))
    out["support_node_ratio"] = out.get("support_node_ratio", out.get("actual_support_node_ratio", ""))
    out["support_edge_ratio"] = out.get("support_edge_ratio", out.get("actual_support_edge_ratio", ""))
    out["official_text_hgb_byte_ratio"] = out.get("official_text_hgb_byte_ratio", out.get("raw_hgb_text_byte_ratio", ""))
    out = add_required_gate21_10_fields(out, table="official_main")
    method = str(out.get("method", ""))
    if "APV12" in method or "APV16" in method:
        proof = deterministic_selection_proof(
            {
                "dataset": out.get("dataset"),
                "method": out.get("method"),
                "canonical_method": out.get("canonical_method"),
                "structural_storage_ratio": out.get("structural_storage_ratio"),
                "support_node_ratio": out.get("support_node_ratio"),
                "support_edge_ratio": out.get("support_edge_ratio"),
                "selection_rule_name": out.get("selection_rule_name") or "deterministic_relation_keep_plan_from_gate21_9",
            },
            repeat_count=3,
        )
        out.update({key: json.dumps(value) if isinstance(value, list) else value for key, value in proof.items()})
    return out


def _write_auto_selector(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    official = read_csv(paths["official_main"] / "gate21_10_official_main_by_method.csv")
    planner = plan_budgeted_channels(str(args.dataset), [float(item) for item in args.structural_budgets])
    rows: list[dict[str, Any]] = []
    for row in planner["plan_rows"]:
        out = dict(row)
        out["method"] = out.get("method_name", "HeSF-RCS-auto-budgeted")
        out["protocol"] = "schema_preserving_tp"
        out["structural_storage_ratio"] = out.get("actual_structural_storage_ratio", "")
        out["official_text_hgb_byte_ratio"] = _match_metric(official, out, "official_text_hgb_byte_ratio")
        out["raw_hgb_text_byte_ratio"] = _match_metric(official, out, "raw_hgb_text_byte_ratio")
        out["support_node_ratio"] = _match_metric(official, out, "support_node_ratio")
        out["support_edge_ratio"] = _match_metric(official, out, "support_edge_ratio")
        out["training_executed"] = bool(out.get("test_micro_f1"))
        out["success"] = bool(out.get("test_micro_f1"))
        out["source_gate"] = "gate21_10_budgeted_channel_planner"
        out.update(deterministic_selection_proof(out, repeat_count=3))
        out = add_required_gate21_10_fields(normalize_metric_fields(out), table="auto_selector")
        rows.append(out)
    utility_rows = [add_required_gate21_10_fields(row, table="auto_selector") for row in planner["utility_rows"]]
    write_summary_csv(paths["auto_selector"] / "gate21_10_auto_selector_by_method.csv", rows)
    write_summary_csv(paths["auto_selector"] / "gate21_10_channel_utility.csv", utility_rows)
    write_summary_csv(paths["auto_selector"] / "gate21_10_budgeted_channel_plans.csv", rows)


def _write_external_tp(paths: Mapping[str, Path], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> None:
    source_rows = read_csv(Path(args.gate21_9_root) / "external_tp_5x5" / "gate21_9_external_tp_task_rows.csv")
    allowed_graph = {str(seed) for seed in graph_seeds}
    allowed_training = {str(seed) for seed in training_seeds}
    rows = [
        _external_tp_row(row, args)
        for row in source_rows
        if (not allowed_graph or str(row.get("graph_seed", "")) in allowed_graph)
        and (not allowed_training or str(row.get("training_seed", "")) in allowed_training)
    ]
    rows.extend(_missing_external_tp_rows(rows, args, graph_seeds, training_seeds))
    budget_rows = [_external_budget_row(row) for row in rows]
    by_method = external_tp_by_method(rows, required_methods=EXTERNAL_TP_METHODS)
    write_summary_csv(paths["external_tp"] / "gate21_10_external_tp_task_rows.csv", rows)
    write_summary_csv(paths["external_tp"] / "gate21_10_external_tp_by_method.csv", by_method)
    write_summary_csv(paths["external_tp"] / "gate21_10_external_tp_budget_audit.csv", budget_rows)


def _external_tp_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = normalize_metric_fields(dict(row))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = out.get("method", out.get("baseline_name", ""))
    out["baseline_name"] = out.get("method")
    out["protocol"] = "schema_preserving_tp"
    if out.get("budget_type") == "structural_budget":
        out["budget_type"] = "structural_storage_ratio"
    out["requested_budget"] = out.get("requested_budget", out.get("budget_value", ""))
    out["budget_value"] = out.get("budget_value", out.get("requested_budget", ""))
    out["structural_storage_ratio"] = out.get("actual_structural_storage_ratio", out.get("structural_storage_ratio", ""))
    out["support_node_ratio"] = out.get("actual_support_node_ratio", out.get("support_node_ratio", ""))
    out["support_edge_ratio"] = out.get("actual_support_edge_ratio", out.get("support_edge_ratio", ""))
    out.update(budget_match_status(out))
    if not bool_value(out.get("training_executed")) and not out.get("failure_type"):
        out["failure_type"] = "not_executed_missing_gate21_10_task_metric"
        out["failure_message"] = "No successful official SeHGNN task metric exists for this Gate21.10 external TP cell."
    out["source_gate"] = out.get("source_gate", "gate21_9_external_tp")
    out = add_required_gate21_10_fields(out, table="external_tp")
    return out


def _missing_external_tp_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, graph_seeds: Sequence[int], training_seeds: Sequence[int]) -> list[dict[str, Any]]:
    existing = {
        (
            str(row.get("method")),
            str(row.get("budget_type")),
            str(row.get("requested_budget", row.get("budget_value", ""))),
            str(row.get("graph_seed")),
            str(row.get("training_seed")),
        )
        for row in rows
    }
    missing: list[dict[str, Any]] = []
    budgets = [("structural_storage_ratio", value) for value in args.structural_budgets] + [("support_node_ratio", value) for value in args.support_node_budgets]
    for method in EXTERNAL_TP_METHODS:
        for budget_type, budget in budgets:
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    key = (method, budget_type, str(float(budget)), str(graph_seed), str(training_seed))
                    if key in existing:
                        continue
                    row = {
                        "dataset": str(args.dataset).upper(),
                        "method": method,
                        "baseline_name": method,
                        "protocol": "schema_preserving_tp",
                        "budget_type": budget_type,
                        "budget_value": float(budget),
                        "requested_budget": float(budget),
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "training_executed": False,
                        "success": False,
                        "failure_type": "missing_gate21_10_external_tp_task_metric",
                        "failure_message": "Required Gate21.10 external TP cell is present as an explicit failure row; no local task metric was available.",
                        "budget_match_status": "not_evaluated",
                        "budget_match_pass": False,
                        "source_gate": "gate21_10_grid_completion",
                    }
                    missing.append(add_required_gate21_10_fields(row, table="external_tp"))
    return missing


def _external_budget_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "budget_type": row.get("budget_type", ""),
        "requested_budget": row.get("requested_budget", row.get("budget_value", "")),
        "actual_support_node_ratio": row.get("actual_support_node_ratio", row.get("support_node_ratio", "")),
        "actual_support_edge_ratio": row.get("actual_support_edge_ratio", row.get("support_edge_ratio", "")),
        "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", row.get("structural_storage_ratio", "")),
        "budget_tolerance": 0.02,
        "budget_match_status": row.get("budget_match_status", budget_match_status(row)),
        "budget_match_pass": row.get("budget_match_pass", False),
        "failure_type": row.get("failure_type", ""),
        "failure_message": row.get("failure_message", ""),
    }


def _write_freehgc(paths: Mapping[str, Path], args: argparse.Namespace, seeds: Sequence[int]) -> None:
    env_row = _freehgc_env_audit(args)
    write_summary_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_env_audit.csv", [env_row])
    write_summary_json(paths["freehgc_standard"] / "gate21_10_freehgc_standard_env_audit.json", env_row)

    source = read_csv(Path(args.gate21_9_root) / "freehgc_protocols" / "gate21_9_freehgc_standard_task_rows.csv")
    rows = [_freehgc_standard_row(row, env_row, args) for row in source]
    rows.extend(_missing_freehgc_standard_rows(rows, env_row, args, seeds))
    by_method = _freehgc_standard_by_method(rows)
    write_summary_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_task_rows.csv", rows)
    write_summary_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_by_method.csv", by_method)

    tp_rows = [add_required_gate21_10_fields(row, table="freehgc_tp") for row in build_gate21_10_freehgc_tp_audit_rows(dataset=str(args.dataset).upper())]
    for row in tp_rows:
        row["method"] = row.get("freehgc_variant", "FreeHGC-TP")
        row["official_hgb_exported"] = row.get("official_hgb_export_possible", False)
        row["official_sehgnn_unmodified"] = row.get("official_sehgnn_loader_accepts", False)
        row["failure_type"] = "hard_incompatibility"
        row["failure_message"] = row.get("minimal_blocking_artifact", "")
    by_tp = _freehgc_tp_by_method(tp_rows)
    write_summary_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_adapter_audit.csv", tp_rows)
    write_summary_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_by_method.csv", by_tp)
    write_summary_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_selection_task_rows.csv", [row for row in tp_rows if "selection" in str(row.get("method", "")).lower()])
    write_summary_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_selection_by_method.csv", [row for row in by_tp if "selection" in str(row.get("method", "")).lower()])


def _freehgc_env_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.freehgc_root)
    zip_path = Path(args.freehgc_zip)
    zip_members: list[str] = []
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                zip_members = archive.namelist()
        except zipfile.BadZipFile:
            zip_members = []
    required = {
        "HGB/train_hgb.py": (root / "HGB" / "train_hgb.py").exists() or any(name.replace("\\", "/").endswith("HGB/train_hgb.py") for name in zip_members),
        "HGB/model_hgb.py": (root / "HGB" / "model_hgb.py").exists() or any(name.replace("\\", "/").endswith("HGB/model_hgb.py") for name in zip_members),
    }
    required_present = all(required.values())
    return {
        "dataset": str(args.dataset).upper(),
        "freehgc_repo_url": "https://github.com/GooLiang/FreeHGC",
        "freehgc_root": str(root),
        "freehgc_root_exists": root.exists(),
        "freehgc_zip": str(zip_path),
        "freehgc_zip_exists": zip_path.exists(),
        "freehgc_zip_member_count": len(zip_members),
        "freehgc_zip_top_level": _zip_top_level(zip_members),
        "upstream_config_verified": required_present,
        "split_matches_hgb_official": False,
        "required_files_present": required_present,
        "required_files_json": json.dumps(required, sort_keys=True),
        "standard_condensation_supported": required_present,
        "hard_failure_reason": "" if required_present else "freehgc_required_files_missing",
        "source_gate": "gate21_10_freehgc_zip_audit",
    }


def _freehgc_standard_row(row: Mapping[str, Any], env_row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = normalize_metric_fields(dict(row))
    ratio = out.get("ratio", out.get("support_node_ratio", out.get("reduction_rate", "")))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["method"] = out.get("method", f"FreeHGC-standard-ratio{ratio}")
    out["ratio"] = ratio
    out["seed"] = out.get("seed", out.get("training_seed", ""))
    out["protocol"] = "standard_condensation"
    out["official_sehgnn_unmodified"] = False
    out["official_hgb_exported"] = False
    out["uses_patched_loader"] = bool_value(out.get("uses_patched_loader", True))
    out["uses_patched_model"] = bool_value(out.get("uses_patched_model", False))
    out["success"] = bool_value(out.get("success")) and bool_value(env_row.get("required_files_present"))
    out["training_executed"] = bool_value(out.get("training_executed")) and out["success"]
    out["upstream_config_verified"] = env_row.get("upstream_config_verified", False)
    out["split_matches_hgb_official"] = env_row.get("split_matches_hgb_official", False)
    out["required_files_present"] = env_row.get("required_files_present", False)
    out["failure_type"] = "" if out["success"] else out.get("failure_type", "freehgc_standard_not_ready")
    out["failure_message"] = "" if out["success"] else out.get("failure_message", env_row.get("hard_failure_reason", "FreeHGC standard run is not verified in this local environment."))
    out = add_required_gate21_10_fields(out, table="freehgc_standard")
    out["eligible_for_standard_condensation_table"] = True
    out["eligible_for_official_main_table"] = False
    return out


def _missing_freehgc_standard_rows(rows: Sequence[Mapping[str, Any]], env_row: Mapping[str, Any], args: argparse.Namespace, seeds: Sequence[int]) -> list[dict[str, Any]]:
    existing = {(str(row.get("ratio", "")), str(row.get("seed", ""))) for row in rows}
    missing: list[dict[str, Any]] = []
    for ratio in [float(item) for item in args.condensation_ratios]:
        for seed in seeds:
            key = (str(ratio), str(seed))
            if key in existing:
                continue
            row = {
                "dataset": str(args.dataset).upper(),
                "method": f"FreeHGC-standard-ratio{ratio:.3f}",
                "ratio": ratio,
                "seed": int(seed),
                "protocol": "standard_condensation",
                "training_executed": False,
                "success": False,
                "upstream_config_verified": env_row.get("upstream_config_verified", False),
                "split_matches_hgb_official": env_row.get("split_matches_hgb_official", False),
                "required_files_present": env_row.get("required_files_present", False),
                "failure_type": "missing_freehgc_standard_5seed_metric",
                "failure_message": "Required Gate21.10 FreeHGC standard ratio/seed cell has no verified local metric.",
                "source_gate": "gate21_10_grid_completion",
            }
            missing.append(add_required_gate21_10_fields(row, table="freehgc_standard"))
    return missing


def _freehgc_standard_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("ratio", "")), []).append(row)
    return [
        {
            "method": f"FreeHGC-standard-ratio{ratio}",
            "ratio": ratio,
            "row_count": len(group),
            "success_count": len([row for row in group if bool_value(row.get("success"))]),
            "seed_count": len({str(row.get("seed")) for row in group if bool_value(row.get("success"))}),
            "test_micro_f1_mean": mean_field(group, "test_micro_f1"),
            "test_macro_f1_mean": mean_field(group, "test_macro_f1"),
            "eligible_for_standard_condensation_table": True,
            "eligible_for_official_main_table": False,
        }
        for ratio, group in sorted(grouped.items())
    ]


def _freehgc_tp_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", row.get("freehgc_variant", "")),
            "training_executed": row.get("training_executed", False),
            "official_hgb_exported": row.get("official_hgb_exported", False),
            "hard_incompatibility": row.get("hard_incompatibility", False),
            "hard_reason": row.get("hard_reason", ""),
            "eligible_for_official_main_table": False,
            "eligible_for_tp_workload_table": False,
            "failure_type": row.get("failure_type", ""),
            "failure_message": row.get("failure_message", ""),
        }
        for row in rows
    ]


def _write_metapath_cache(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    metapath = [add_required_gate21_10_fields(row, table="audits") for row in read_csv(Path(args.gate21_9_root) / "metapath_cache_dump" / "gate21_9_metapath_tensor_audit.csv")]
    cache = []
    for row in read_csv(Path(args.gate21_9_root) / "metapath_cache_dump" / "gate21_9_cache_hash_assertions.csv"):
        out = dict(row)
        cache_hash = out.get("cache_hash", out.get("cache_file_hash", ""))
        out["cache_hash_non_empty"] = bool(cache_hash) and str(cache_hash) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        cache.append(add_required_gate21_10_fields(out, table="audits"))
    write_summary_csv(paths["metapath_cache"] / "gate21_10_metapath_tensor_audit.csv", metapath)
    write_summary_csv(paths["metapath_cache"] / "gate21_10_cache_hash_audit.csv", cache)


def _write_feature_ablation(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_9_root) / "feature_ablation_tasks" / "gate21_9_feature_ablation_task_rows.csv")
    rows = [_feature_ablation_row(row, args) for row in source]
    rows.extend(_missing_feature_ablation_rows(rows, args))
    by_method = _feature_ablation_by_method(rows)
    write_summary_csv(paths["feature_ablation"] / "gate21_10_feature_ablation_task_rows.csv", rows)
    write_summary_csv(paths["feature_ablation"] / "gate21_10_feature_ablation_by_method.csv", by_method)


def _feature_ablation_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = normalize_metric_fields(dict(row))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["feature_transform"] = out.get("feature_transform", out.get("feature_setting", ""))
    out["label_graph_setting"] = out.get("label_graph_setting", "default")
    out["source_gate"] = out.get("source_gate", "gate21_9")
    out = add_required_gate21_10_fields(out, table="audits")
    out["eligible_for_decision"] = bool_value(out.get("training_executed")) and bool_value(out.get("success"))
    return out


def _missing_feature_ablation_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    existing = {(str(row.get("method")), str(row.get("feature_transform")), str(row.get("label_graph_setting"))) for row in rows}
    missing: list[dict[str, Any]] = []
    for method in GATE21_10_METHODS:
        for transform in GATE21_10_FEATURE_TRANSFORMS:
            for setting in GATE21_10_LABEL_GRAPH_SETTINGS:
                if (method, transform, setting) in existing:
                    continue
                row = {
                    "dataset": str(args.dataset).upper(),
                    "method": method,
                    "feature_transform": transform,
                    "label_graph_setting": setting,
                    "training_executed": False,
                    "success": False,
                    "failure_type": "missing_feature_ablation_task_metric",
                    "failure_message": "Gate21.10 required mechanism cell has no official task metric yet.",
                    "source_gate": "gate21_10_grid_completion",
                }
                missing.append(add_required_gate21_10_fields(row, table="audits"))
    return missing


def _feature_ablation_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    return [
        {
            "method": method,
            "row_count": len(group),
            "success_count": len([row for row in group if bool_value(row.get("success")) and bool_value(row.get("training_executed"))]),
            "test_micro_f1_mean": mean_field(group, "test_micro_f1"),
            "test_macro_f1_mean": mean_field(group, "test_macro_f1"),
        }
        for method, group in sorted(grouped.items())
    ]


def _write_adapter(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_9_root) / "adapter_package_v4" / "gate21_9_adapter_task_rows.csv")
    rows = [_adapter_row(row, args) for row in source]
    by_method = aggregate_adapter_by_method_gate21_10(rows)
    audit = [_adapter_audit_row(row) for row in rows]
    write_summary_csv(paths["adapter"] / "gate21_10_adapter_task_rows.csv", rows)
    write_summary_csv(paths["adapter"] / "gate21_10_adapter_by_method.csv", by_method)
    write_summary_csv(paths["adapter"] / "gate21_10_adapter_package_audit.csv", audit)


def _adapter_row(row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = normalize_metric_fields(dict(row))
    out["dataset"] = out.get("dataset", str(args.dataset).upper())
    out["base_method"] = out.get("base_method", out.get("base_graph_method", ""))
    out["adapter_method"] = out.get("adapter_method", out.get("feature_adapter", out.get("adapter_name", "")))
    out["uses_feature_adapter"] = True
    out["official_sehgnn_unmodified"] = False
    out["eligible_for_official_main_table"] = False
    out["eligible_for_adapter_table"] = True
    out["source_gate"] = out.get("source_gate", "gate21_9")
    out = add_required_gate21_10_fields(out, table="adapter")
    out["eligible_for_official_main_table"] = False
    return out


def _adapter_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "base_method": row.get("base_method", ""),
        "adapter_method": row.get("adapter_method", ""),
        "success": row.get("success", ""),
        "static_inference_package_ratio": row.get("static_inference_package_ratio", ""),
        "transform_recipe_package_ratio": row.get("transform_recipe_package_ratio", ""),
        "reconstructable_package_ratio": row.get("reconstructable_package_ratio", ""),
        "pca_reproducible_package_complete": row.get("pca_reproducible_package_complete", ""),
        "eligible_for_adapter_table": row.get("eligible_for_adapter_table", ""),
        "eligible_for_official_main_table": row.get("eligible_for_official_main_table", ""),
        "failure_type": row.get("failure_type", ""),
        "failure_message": row.get("failure_message", row.get("failed_reason", "")),
    }


def _write_storage_system(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_9_root) / "storage_system_costs" / "gate21_9_storage_system_by_method.csv")
    artifact_rows = [add_required_gate21_10_fields(row, table="audits") for row in source]
    denom_rows = [storage_denominator_audit(row) for row in source]
    workload_rows = [normalize_system_workload_row(_storage_workload_row(row)) for row in source]
    write_summary_csv(paths["storage_system"] / "gate21_10_storage_system_by_artifact.csv", artifact_rows)
    write_summary_csv(paths["storage_system"] / "gate21_10_storage_denominator_audit.csv", denom_rows)
    write_summary_csv(paths["storage_system"] / "gate21_10_system_workload_cost.csv", workload_rows)


def _storage_workload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["method"] = out.get("method", out.get("artifact_name", ""))
    out["load_time_seconds"] = out.get("load_time_seconds", out.get("load_wall_time_seconds", ""))
    out["task_micro_f1"] = out.get("task_micro_f1", out.get("test_micro_f1", ""))
    out["task_macro_f1"] = out.get("task_macro_f1", out.get("test_macro_f1", ""))
    out["peak_cpu_rss_mb"] = out.get("peak_cpu_rss_mb", out.get("peak_cpu_memory_mb", ""))
    out["preprocessed_cache_bytes"] = out.get("preprocessed_cache_bytes", "")
    out["training_executed"] = bool_value(out.get("training_executed"))
    return out


def _write_cross_dataset(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    source = read_csv(Path(args.gate21_9_root) / "cross_dataset_auto_channel" / "gate21_9_cross_dataset_task_rows.csv")
    rows = [add_required_gate21_10_fields(normalize_metric_fields(row), table="official_main") for row in source]
    write_summary_csv(paths["cross_dataset"] / "gate21_10_cross_dataset_task_rows.csv", rows)
    write_summary_csv(paths["cross_dataset"] / "gate21_10_cross_dataset_by_method.csv", _cross_dataset_by_method(rows))


def _cross_dataset_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    return [
        {
            "dataset": dataset,
            "method": method,
            "row_count": len(group),
            "success_count": len([row for row in group if bool_value(row.get("training_executed")) and bool_value(row.get("success", True))]),
            "test_micro_f1_mean": mean_field(group, "test_micro_f1"),
            "test_macro_f1_mean": mean_field(group, "test_macro_f1"),
        }
        for (dataset, method), group in sorted(grouped.items())
    ]


def _write_audits(paths: Mapping[str, Path], args: argparse.Namespace) -> None:
    coverage_rows = []
    for row in read_csv(Path(args.gate21_9_root) / "audits" / "gate21_9_coverage_v4.csv"):
        out = dict(row)
        out.setdefault("reachability_assertions_pass", out.get("coverage_pass", out.get("assertion_pass", "")))
        out.setdefault("per_class_venue_coverage_json", "")
        out.setdefault("semantic_distributional_coverage_pass", False)
        coverage_rows.append(add_required_gate21_10_fields(out, table="audits"))
    write_summary_csv(paths["audits"] / "gate21_10_coverage_semantic.csv", coverage_rows)


def _write_dry_run_manifest(paths: Mapping[str, Path], stages: set[str]) -> None:
    rows = [{"stage": stage, "would_write_outputs": True, "dry_run": True} for stage in sorted(stages)]
    write_summary_csv(paths["audits"] / "gate21_10_dry_run_stage_manifest.csv", rows)


def _selected_stages(stage_arg: str) -> set[str]:
    aliases = {
        "official-main": "official_main",
        "official_main": "official_main",
        "auto-selector": "auto_selector",
        "auto_selector": "auto_selector",
        "external-tp": "external_tp",
        "external_tp": "external_tp",
        "freehgc": "freehgc_standard,freehgc_tp",
        "freehgc-standard": "freehgc_standard",
        "freehgc-tp": "freehgc_tp",
        "metapath-cache": "metapath_cache",
        "feature-ablation": "feature_ablation",
        "storage-cost": "storage_system",
        "storage-system": "storage_system",
        "cross-dataset": "cross_dataset",
    }
    all_stages = {
        "official_main",
        "auto_selector",
        "external_tp",
        "freehgc_standard",
        "freehgc_tp",
        "metapath_cache",
        "feature_ablation",
        "adapter",
        "storage_system",
        "cross_dataset",
        "audits",
    }
    raw = str(stage_arg).strip()
    if raw == "all":
        return set(all_stages)
    selected: set[str] = set()
    for part in raw.split(","):
        key = part.strip()
        expanded = aliases.get(key, key)
        for item in expanded.split(","):
            if item:
                selected.add(item)
    return selected


def _seed_list(values: Sequence[int], *, quick: bool) -> list[int]:
    seeds = [int(value) for value in values]
    return seeds[:1] if quick and seeds else seeds


def _write_readmes(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if name == "root":
            continue
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text(f"# Gate21.10 {name}\n\nEvidence generated by `run_gate21_10_paper_ready_evidence.py`.\n", encoding="utf-8")


def _match_metric(official: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], field: str) -> Any:
    budget = float_value(plan.get("budget_target"))
    token = "APV12" if budget is not None and budget <= 0.125 else "APV16"
    for row in official:
        if token in str(row.get("method", "")):
            return row.get(field, "")
    return ""


def _zip_top_level(members: Sequence[str]) -> str:
    tops = sorted({name.replace("\\", "/").split("/", 1)[0] for name in members if name})
    return ";".join(tops[:5])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate21.10 paper-ready evidence assembly.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--stage", default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate21-9-root", type=Path, default=DEFAULT_GATE21_9_ROOT)
    parser.add_argument("--freehgc-root", type=Path, default=Path("external/FreeHGC"))
    parser.add_argument("--freehgc-zip", type=Path, default=Path("FreeHGC-main (1).zip"))
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--structural-budgets", nargs="+", type=float, default=[0.12, 0.16, 0.20, 0.30])
    parser.add_argument("--support-node-budgets", nargs="+", type=float, default=[0.30, 0.50])
    parser.add_argument("--condensation-ratios", nargs="+", type=float, default=list(freehgc_standard_ratios()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--dry-run", nargs="?", const=True, default=False, type=parse_bool_arg)
    parser.add_argument("--reuse-gate21-9-anchors", nargs="?", const=True, default=True, type=parse_bool_arg)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
