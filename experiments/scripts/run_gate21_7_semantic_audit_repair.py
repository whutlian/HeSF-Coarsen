from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import GATE21_6_SOURCE, add_gate21_7_common_args, ensure_layout, read_csv
from hesf_coarsen.eval.official.coverage_diagnostics_v2 import (
    compute_hgb_coverage_diagnostics_v2,
    coverage_sanity_assertion_rows,
    write_gate21_7_coverage_outputs,
)
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_metapath_introspection_patch import compare_cache_hashes_for_perturbation


DEFAULT_METHODS = [
    "export-full-SeHGNN",
    "H6-node30",
    "H6-APV-skeleton",
    "HeSF-RCS-APV12",
    "HeSF-RCS-APV16",
    "APV12-PTTP10",
    "APV12-PV75",
]
METHOD_ALIASES = {
    "HeSF-RCS-APV12": "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
    "HeSF-RCS-APV16": "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
    "APV12-PTTP10": "H6-dirskel-AP100-PA00-PV100-VP00-PTTP10",
    "APV12-PV75": "H6-dirskel-AP100-PA00-PV75-VP00-PTTP00",
}


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["semantic_audit"]
    source = Path(args.gate21_6_dir)
    cache = read_csv(Path(args.gate21_5_dir) / "gate21_5_cache_audit.csv")
    retention = read_csv(Path(args.gate21_5_dir) / "gate21_5_relation_edge_retention.csv")
    metapath_source = read_csv(source / "gate21_6_metapath_cache_audit.csv")
    coverage_rows: list[dict[str, Any]] = []
    assertion_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method in args.methods:
        source_method = METHOD_ALIASES.get(str(method), str(method))
        row = _first_cache_row(cache, source_method)
        if not row:
            failures.append({"method": method, "failure_type": "missing_export_dir", "failure_message": "No Gate21.5 HGB export found for semantic audit."})
            continue
        export_dir = Path(row.get("export_dir", ""))
        if not export_dir.exists():
            failures.append({"method": method, "failure_type": "missing_export_dir", "failure_message": str(export_dir)})
            continue
        try:
            cov = compute_hgb_coverage_diagnostics_v2(export_dir, dataset=args.dataset, method=str(method), graph_seed=int(float(row.get("graph_seed") or 1)))
            coverage_rows.append(cov)
            rel_rows = [_retention_for(ret, source_method) for ret in retention if ret.get("method") == source_method and ret.get("training_seed") == "1"]
            assertions = coverage_sanity_assertion_rows(cov, relation_retention_rows=rel_rows)
            for assertion in assertions:
                assertion.update({"method": method, "dataset": str(args.dataset).upper()})
            assertion_rows.extend(assertions)
        except Exception as exc:
            failures.append({"method": method, "failure_type": "coverage_runtime_failure", "failure_message": str(exc)})
    write_gate21_7_coverage_outputs(out, coverage_rows=coverage_rows, assertion_rows=assertion_rows)
    write_csv(out / "gate21_7_coverage_failure_log.csv", failures)

    metapath_rows = [_metapath_gate21_7_row(row) for row in metapath_source]
    cache_hash_rows = _cache_hash_rows(cache)
    write_csv(out / "gate21_7_metapath_cache_audit.csv", metapath_rows)
    write_csv(out / "gate21_7_cache_hash_audit.csv", cache_hash_rows)
    write_csv(out / "gate21_7_cache_sanity.csv", cache_hash_rows)
    write_csv(
        out / "gate21_7_metapath_introspection_failure_log.csv",
        [row for row in metapath_rows if not _bool(row.get("real_tensor_dumped"))],
    )
    write_json(out / "gate21_7_semantic_audit_plan.json", {"methods": list(args.methods), "source": str(source)})
    return {"coverage_rows": len(coverage_rows), "assertion_rows": len(assertion_rows), "metapath_rows": len(metapath_rows)}


def _first_cache_row(rows: Sequence[Mapping[str, str]], method: str) -> Mapping[str, str] | None:
    return next((row for row in rows if row.get("method") == method and row.get("training_seed") == "1"), None)


def _retention_for(row: Mapping[str, str], method: str) -> dict[str, Any]:
    return {
        **dict(row),
        "relation_name": row.get("official_relation_name", ""),
        "retained_edges": row.get("retained_edge_count", row.get("actual_relation_budget", "")),
    }


def _metapath_gate21_7_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        **dict(row),
        "real_tensor_dumped": bool(str(row.get("metapath_key", "")).strip() or str(row.get("feature_tensor_hash", "")).strip()),
        "tensor_key_dumped": bool(str(row.get("metapath_key", "")).strip()),
        "fallback_loaded_relation_audit_used": row.get("fallback_loaded_relation_audit_used", True),
        "failure_type": "" if row.get("metapath_key") else "official_sehgnn_intermediate_tensors_not_exposed",
    }


def _cache_hash_rows(cache_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    pttp00 = next((row for row in cache_rows if row.get("method") == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00" and row.get("training_seed") == "1"), None)
    pttp10 = next((row for row in cache_rows if row.get("method") == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP10" and row.get("training_seed") == "1"), None)
    if not pttp00 or not pttp10:
        return []
    left = {**pttp00, "cache_hash": pttp00.get("cache_hash_after", ""), "fallback_loaded_relation_audit_used": True}
    right = {**pttp10, "cache_hash": pttp10.get("cache_hash_after", ""), "fallback_loaded_relation_audit_used": True}
    return [compare_cache_hashes_for_perturbation(left, right, comparison_name="APV12_vs_APV12_PTTP10")]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--gate21-6-dir", type=Path, default=GATE21_6_SOURCE)
    parser.add_argument("--gate21-5-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
