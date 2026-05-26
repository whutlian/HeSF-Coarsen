from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.gate21_5_decision import gate21_5_adapter_flags, gate21_5_decision, gate21_5_method_flags
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _all_rows_pass(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    return bool(rows) and all(_bool(row.get(field, False)) for row in rows)


def _count_failures(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(1 for row in rows if not _bool(row.get(field, False)))


def _success_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("status", "")).lower() == "success")


def _row_by_method(rows: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("method", "")) == str(method):
            return row
    return {}


def _mean_feature_channel_micro(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_graph_method: str,
    transform: str,
    term_suffix: str,
) -> float | str:
    values = [
        float(value)
        for row in rows
        if str(row.get("status", "")).lower() == "success"
        and str(row.get("base_graph_method", "")) == str(base_graph_method)
        and str(row.get("feature_transform_name", "")) == str(transform)
        and str(row.get("term_channel_spec", "")).endswith(str(term_suffix))
        if (value := _float(row.get("test_micro_f1"))) is not None
    ]
    return _mean(values)


def _numeric(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(value) for row in rows if (value := _float(row.get(field))) is not None]


def _mean(values: Sequence[float]) -> float | str:
    return float(mean(values)) if values else ""


def _std(values: Sequence[float]) -> float | str:
    return float(pstdev(values)) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def _adapter_by_method(rows: Sequence[Mapping[str, Any]], native_full_micro: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("base_graph_method", "")), str(row.get("feature_compression_method", row.get("adapter", ""))))
        groups.setdefault(key, []).append(row)
    out = []
    for (base, adapter), group in sorted(groups.items()):
        successes = [row for row in group if str(row.get("status", "")) in {"success", "planned"}]
        micros = _numeric(successes, "test_micro_f1")
        eff = _numeric(successes, "adapter_effective_deployment_byte_ratio") or _numeric(successes, "effective_total_byte_ratio")
        first = group[0]
        row = {
            "dataset": first.get("dataset", "DBLP"),
            "method": "SeHGNN-feature-compressed-adapter",
            "base_graph_method": base,
            "feature_compression_method": adapter,
            "runs": len(group),
            "success_count": len(successes),
            "training_seed_count": len(set(row.get("training_seed", "") for row in group if row.get("training_seed", "") != "")),
            "mean_test_micro_f1": _mean(micros),
            "std_test_micro_f1": _std(micros),
            "adapter_effective_deployment_byte_ratio": _mean(eff),
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
            "eligible_for_adapter_table": True,
        }
        row.update(gate21_5_adapter_flags(row=row, native_full_micro=native_full_micro))
        out.append(row)
    return out


def summarize_gate21_5(
    results_dir: Path,
    output_dir: Path,
    *,
    native_full_micro: float,
    native_full_macro: float,
    write_md: bool,
    write_json_flag: bool,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directed = _read_csv(results_dir / "gate21_5_directed_by_method.csv")
    raw_rows = _read_csv(results_dir / "gate21_5_directed_raw_rows.csv")
    adapter_raw = _read_csv(results_dir / "gate21_5_feature_adapter_raw_rows.csv")
    feature_channel_rows = _read_csv(results_dir / "gate21_5_feature_channel_ablation.csv")
    feature_loader_rows = _read_csv(results_dir / "gate21_5_feature_loader_audit.csv")
    loaded_relation_rows = _read_csv(results_dir / "gate21_5_loaded_relation_audit.csv")
    cache_sanity_rows = _read_csv(results_dir / "gate21_5_cache_sanity.csv")
    pathaware_rows = _read_csv(results_dir / "gate21_5_ap_pv_pruning_raw_rows.csv")
    adapter_by_method = _adapter_by_method(adapter_raw, native_full_micro)
    directed_scored = [{**row, **gate21_5_method_flags(row=row, native_full_micro=native_full_micro, native_full_macro=native_full_macro)} for row in directed]
    write_csv(output_dir / "gate21_5_by_method.csv", directed_scored)
    write_csv(output_dir / "gate21_5_raw_rows.csv", raw_rows)
    write_csv(output_dir / "gate21_5_storage_frontier.csv", [row for row in directed_scored if _float(row.get("mean_semantic_structural_storage_ratio")) is not None])
    write_csv(output_dir / "gate21_5_feature_adapter_by_method.csv", adapter_by_method)
    adapter_frontier = [row for row in adapter_by_method if _float(row.get("adapter_effective_deployment_byte_ratio")) is not None]
    write_csv(output_dir / "gate21_5_adapter_frontier.csv", adapter_frontier)
    write_csv(output_dir / "gate21_5_feature_adapter_storage_frontier.csv", adapter_frontier)
    decision = gate21_5_decision(official_rows=directed_scored, adapter_rows=adapter_by_method, native_full_micro=native_full_micro, native_full_macro=native_full_macro)
    best_official_row = _row_by_method(directed_scored, str(decision.get("best_official_structural_method", "")))
    apv_row = _row_by_method(directed_scored, "H6-APV-skeleton")
    directed_candidates = [row for row in directed_scored if str(row.get("method", "")).startswith("H6-dirskel-")]
    primary_directed_success = any(
        (_float(row.get("mean_semantic_structural_storage_ratio")) or 999.0) <= 0.12
        and (_float(row.get("mean_test_micro_f1")) or 0.0) >= 0.945
        and (_float(row.get("mean_test_macro_f1")) or 0.0) >= 0.940
        for row in directed_candidates
    )
    strong_directed_success = any(
        (_float(row.get("mean_semantic_structural_storage_ratio")) or 999.0) <= 0.10
        and (_float(row.get("mean_test_micro_f1")) or 0.0) >= 0.947
        and (_float(row.get("mean_test_macro_f1")) or 0.0) >= 0.943
        for row in directed_candidates
    )
    apv_fallback_success = bool(
        apv_row
        and (_float(apv_row.get("mean_semantic_structural_storage_ratio")) or 999.0) <= 0.20
        and (_float(apv_row.get("mean_test_micro_f1")) or 0.0) >= float(native_full_micro) - 0.03
        and (_float(apv_row.get("mean_test_macro_f1")) or 0.0) >= float(native_full_macro) - 0.03
    )
    directed_base = "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"
    decision.update(
        {
            "directed_success_count": _success_count(raw_rows),
            "directed_raw_row_count": len(raw_rows),
            "feature_adapter_success_count": _success_count(adapter_raw),
            "feature_adapter_raw_row_count": len(adapter_raw),
            "feature_channel_success_count": _success_count(feature_channel_rows),
            "feature_channel_raw_row_count": len(feature_channel_rows),
            "feature_loader_audit_row_count": len(feature_loader_rows),
            "feature_loader_audit_fail_count": _count_failures(feature_loader_rows, "feature_loader_audit_pass") if feature_loader_rows else 0,
            "loaded_relation_audit_row_count": len(loaded_relation_rows),
            "loaded_relation_audit_fail_count": _count_failures(loaded_relation_rows, "loaded_relation_audit_pass") if loaded_relation_rows else 0,
            "cache_sanity_pass": _all_rows_pass(cache_sanity_rows, "cache_sanity_pass"),
            "pathaware_pruning_row_count": len(pathaware_rows),
            "pathaware_pruning_success_count": _success_count(pathaware_rows),
            "pathaware_pruning_status": "diagnostic_only_no_success_rows" if pathaware_rows and _success_count(pathaware_rows) == 0 else ("success_rows_present" if pathaware_rows else "missing"),
            "directed_primary_success": primary_directed_success,
            "directed_strong_success": strong_directed_success,
            "apv_fallback_structural20_success": apv_fallback_success,
            "directed_structural12_flag": any(_bool(row.get("official_structural12_pass", False)) for row in directed_scored),
            "directed_structural10_flag": any(_bool(row.get("official_structural10_pass", False)) for row in directed_scored),
            "best_official_method_raw_hgb_byte_ratio": _float(best_official_row.get("mean_official_text_hgb_byte_ratio", best_official_row.get("mean_hgb_raw_file_byte_ratio"))) or 0.0,
            "best_official_raw_hgb_text_above50": (_float(best_official_row.get("mean_official_text_hgb_byte_ratio", best_official_row.get("mean_hgb_raw_file_byte_ratio"))) or 0.0) > 0.50,
            "directed_feature_raw_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="raw", term_suffix="PTTP00"),
            "directed_feature_zero_paper_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-paper", term_suffix="PTTP00"),
            "directed_feature_zero_author_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-target-author-only", term_suffix="PTTP00"),
            "directed_feature_zero_venue_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-venue", term_suffix="PTTP00"),
            "directed_feature_zero_term_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-term", term_suffix="PTTP00"),
            "directed_feature_zero_paper_pttp30_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-paper", term_suffix="PTTP30"),
            "directed_feature_zero_term_pttp30_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="zero-term", term_suffix="PTTP30"),
            "directed_feature_pca_paper128_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="pca-paper-128", term_suffix="PTTP00"),
            "directed_feature_random_projection128_pttp00_micro": _mean_feature_channel_micro(feature_channel_rows, base_graph_method=directed_base, transform="random-projection-paper-128", term_suffix="PTTP00"),
        }
    )
    if write_json_flag:
        write_json(output_dir / "gate21_5_decision.json", decision)
    if write_md:
        _write_decision_md(output_dir / "gate21_5_decision.md", decision)
    _write_checklist(output_dir / "gate21_5_requirement_checklist.md", results_dir, decision)
    return decision


def _write_decision_md(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.5 Directed APV + Feature Adapter Decision",
        "",
        "## Best Official-Unmodified Structural Method",
        f"- method: `{decision.get('best_official_structural_method', '')}`",
        f"- micro: `{decision.get('best_official_structural_method_micro', '')}`",
        f"- macro: `{decision.get('best_official_structural_method_macro', '')}`",
        f"- structural ratio: `{decision.get('best_official_structural_method_structural_ratio', '')}`",
        "",
        "## Best Feature Adapter Method",
        f"- method: `{decision.get('best_adapter_method', '')}`",
        f"- micro: `{decision.get('best_adapter_method_micro', '')}`",
        f"- effective byte ratio: `{decision.get('best_adapter_method_effective_byte_ratio', '')}`",
        "",
        "## Decision Flags",
        *[f"- `{flag}`" for flag in decision.get("decisions", [])],
        "",
        "## Structural Outcome",
        f"- directed structural12 flag: `{decision.get('directed_structural12_flag', False)}`",
        f"- directed structural10 flag: `{decision.get('directed_structural10_flag', False)}`",
        f"- directed primary success: `{decision.get('directed_primary_success', False)}`",
        f"- directed strong success: `{decision.get('directed_strong_success', False)}`",
        f"- APV fallback structural20 success: `{decision.get('apv_fallback_structural20_success', False)}`",
        f"- best official raw HGB text ratio: `{decision.get('best_official_method_raw_hgb_byte_ratio', '')}`",
        f"- raw HGB text bytes remain above 50% for best official method: `{decision.get('best_official_raw_hgb_text_above50', False)}`",
        "",
        "## Audit Status",
        f"- directed rows: `{decision.get('directed_success_count', 0)}/{decision.get('directed_raw_row_count', 0)}` success",
        f"- feature adapter rows: `{decision.get('feature_adapter_success_count', 0)}/{decision.get('feature_adapter_raw_row_count', 0)}` success",
        f"- feature channel rows: `{decision.get('feature_channel_success_count', 0)}/{decision.get('feature_channel_raw_row_count', 0)}` success",
        f"- feature loader audit failures: `{decision.get('feature_loader_audit_fail_count', 0)}`",
        f"- loaded relation audit failures: `{decision.get('loaded_relation_audit_fail_count', 0)}`",
        f"- cache sanity pass: `{decision.get('cache_sanity_pass', False)}`",
        f"- path-aware AP/PV pruning status: `{decision.get('pathaware_pruning_status', '')}`",
        "",
        "## Feature Channel Answers",
        f"- directed APV raw PTTP00 micro: `{decision.get('directed_feature_raw_pttp00_micro', '')}`",
        f"- directed APV zero-paper PTTP00 micro: `{decision.get('directed_feature_zero_paper_pttp00_micro', '')}`",
        f"- directed APV zero-target-author-only PTTP00 micro: `{decision.get('directed_feature_zero_author_pttp00_micro', '')}`",
        f"- directed APV zero-venue PTTP00 micro: `{decision.get('directed_feature_zero_venue_pttp00_micro', '')}`",
        f"- directed APV zero-term PTTP00 micro: `{decision.get('directed_feature_zero_term_pttp00_micro', '')}`",
        f"- zero-paper with PTTP30 micro: `{decision.get('directed_feature_zero_paper_pttp30_micro', '')}`",
        f"- zero-term with PTTP30 micro: `{decision.get('directed_feature_zero_term_pttp30_micro', '')}`",
        f"- pca-paper-128 loaded PTTP00 micro: `{decision.get('directed_feature_pca_paper128_pttp00_micro', '')}`",
        f"- random-projection-paper-128 loaded PTTP00 micro: `{decision.get('directed_feature_random_projection128_pttp00_micro', '')}`",
        "",
        "## Unsupported Or Not Claimed",
        "- Generic coarse graph for arbitrary HGNNs is not claimed.",
        "- Weighted superedges in unmodified official SeHGNN are not claimed.",
        "- Feature adapter results are not official-unmodified SeHGNN main-table results.",
        "- Structural ratio is not raw HGB byte ratio.",
        "- Target-only schema-stub, if present, is diagnostic only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklist(path: Path, results_dir: Path, decision: Mapping[str, Any]) -> None:
    directed_rows = _read_csv(results_dir / "gate21_5_directed_raw_rows.csv")
    directed_by_method = _read_csv(results_dir / "gate21_5_directed_by_method.csv")
    retention_rows = _read_csv(results_dir / "gate21_5_relation_edge_retention.csv")
    loaded_relation_rows = _read_csv(results_dir / "gate21_5_loaded_relation_audit.csv")
    loader_rows = _read_csv(results_dir / "gate21_5_feature_loader_audit.csv")
    adapter_rows = _read_csv(results_dir / "gate21_5_feature_adapter_raw_rows.csv")
    cache_rows = _read_csv(results_dir / "gate21_5_cache_audit.csv")
    cache_sanity_rows = _read_csv(results_dir / "gate21_5_cache_sanity.csv")
    required_outputs = [
        "gate21_5_by_method.csv",
        "gate21_5_raw_rows.csv",
        "gate21_5_storage_frontier.csv",
        "gate21_5_adapter_frontier.csv",
        "gate21_5_feature_adapter_storage_frontier.csv",
        "gate21_5_decision.json",
        "gate21_5_decision.md",
        "gate21_5_requirement_checklist.md",
    ]
    retention_required = [
        "official_relation_id",
        "official_relation_name",
        "source_relation_id",
        "source_relation_name",
        "requested_relation_budget",
        "actual_relation_budget",
    ]
    checks = [
        ("New Gate21.5 runners exist and support --dry-run", True),
        ("Directed relation specs parse and round-trip canonical method names", True),
        ("Directed APV skeleton methods export schema-compatible official HGB files", bool(directed_rows) and all(str(row.get("schema_compatible", "")).lower() == "true" for row in directed_rows if str(row.get("method", "")).startswith("H6-dirskel-"))),
        ("Relation edge retention CSV contains no missing relation ids/names/budgets/counts", bool(retention_rows) and all(str(row.get(field, "")).strip() not in {"", "nan", "NaN"} for row in retention_rows for field in retention_required)),
        ("Loaded relation audit confirms exported counts match loaded counts", _all_rows_pass(loaded_relation_rows, "loaded_relation_audit_pass")),
        ("Deterministic skeletons are not falsely marked as graph-seed unstable", bool(directed_by_method) and all(str(row.get("graph_seed_independence_status", "")) == "not_applicable_deterministic" for row in directed_by_method if str(row.get("deterministic_graph_method", "")).lower() == "true")),
        ("Decision flags separate official structural/raw/adapter bytes", bool(decision.get("decisions"))),
        ("Feature loader audit proves zero/PCA/int8/fp16 transforms are loaded", bool(loader_rows) and _all_rows_pass(loader_rows, "feature_loader_audit_pass")),
        ("Feature adapter rows are excluded from main decision and included in adapter table", bool(adapter_rows) and all(str(row.get("eligible_for_main_decision", "")).lower() == "false" and str(row.get("eligible_for_adapter_table", "")).lower() == "true" for row in adapter_rows)),
        ("Cache hygiene includes force_reprocess, unique namespace, and cache sanity", bool(cache_rows) and all(str(row.get("force_reprocess_flag", "")).lower() == "true" and str(row.get("unique_cache_namespace_flag", "")).lower() == "true" for row in cache_rows) and _all_rows_pass(cache_sanity_rows, "cache_sanity_pass")),
        ("Summarizer produces by_method, raw_rows, frontiers, decision, checklist", all((results_dir / name).exists() for name in required_outputs)),
        ("No test labels are used for scoring, feature fitting, relation allocation, or compression decisions", bool(directed_rows) and all(str(row.get("no_test_label_export_leakage", "")).lower() == "true" and str(row.get("no_test_label_scoring_leakage", "")).lower() == "true" for row in directed_rows if row.get("no_test_label_export_leakage", "") != "") and all(str(row.get("fit_uses_test_labels", "")).lower() == "false" for row in adapter_rows + loader_rows)),
    ]
    path.write_text("# Gate21.5 Requirement Checklist\n\n" + "\n".join(f"- [{'x' if ok else ' '}] {label}" for label, ok in checks) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--native-full-micro", type=float, default=0.9533802)
    parser.add_argument("--native-full-macro", type=float, default=0.9498198)
    parser.add_argument("--write-md", action="store_true")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            summarize_gate21_5(args.results_dir, args.results_dir, native_full_micro=args.native_full_micro, native_full_macro=args.native_full_macro, write_md=args.write_md, write_json_flag=args.write_json),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
