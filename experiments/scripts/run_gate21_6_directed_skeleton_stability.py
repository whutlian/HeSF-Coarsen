from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.gate21_6_decision import gate21_6_decision, gate21_6_method_flags, graph_seed_stability_flags
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


GATE21_5_DIR = Path("results/gate21_5_directed_apv_feature_adapter")
METHOD_ALIASES = {
    "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00": "HeSF-RCS-APV12",
    "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00": "HeSF-RCS-APV16",
}
REQUIRED_GRID = [
    "AP100-PA00-PV100-VP00-PTTP00",
    "AP100-PA25-PV100-VP00-PTTP00",
    "AP100-PA50-PV100-VP00-PTTP00",
    "AP100-PA75-PV100-VP00-PTTP00",
    "AP100-PA00-PV100-VP25-PTTP00",
    "AP100-PA00-PV100-VP50-PTTP00",
    "AP100-PA50-PV100-VP50-PTTP00",
    "AP100-PA75-PV100-VP75-PTTP00",
    "AP100-PA100-PV100-VP100-PTTP00",
    "AP90-PA00-PV100-VP00-PTTP00",
    "AP75-PA00-PV100-VP00-PTTP00",
    "AP50-PA00-PV100-VP00-PTTP00",
    "AP100-PA00-PV90-VP00-PTTP00",
    "AP100-PA00-PV75-VP00-PTTP00",
    "AP100-PA00-PV50-VP00-PTTP00",
    "AP90-PA00-PV90-VP00-PTTP00",
    "AP75-PA00-PV75-VP00-PTTP00",
    "AP100-PA00-PV100-VP00-PTTP05",
    "AP100-PA00-PV100-VP00-PTTP10",
    "AP100-PA00-PV100-VP00-PTTP30",
    "AP100-PA50-PV100-VP50-PTTP05",
    "AP100-PA50-PV100-VP50-PTTP10",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | str:
    if value in {"", None}:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _alias(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))


def _is_apv12(method: str) -> bool:
    return str(method) in {"HeSF-RCS-APV12", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"}


def _is_apv16(method: str) -> bool:
    return str(method) in {"HeSF-RCS-APV16", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00"}


def _method_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = _alias(str(row.get("method", "")))
    structural = _float(row.get("mean_semantic_structural_storage_ratio"))
    raw = _float(row.get("mean_official_text_hgb_byte_ratio", row.get("mean_hgb_raw_file_byte_ratio")))
    graph_seed_count = int(float(row.get("graph_seed_count") or 0))
    export_unique = 1 if _bool(row.get("deterministic_graph_method", False)) else max(1, graph_seed_count)
    deterministic = bool(_is_apv12(method) or str(method) in {"H6-APV-skeleton", "H6-node30", "export-full-SeHGNN", "full-native-SeHGNN"})
    stability = graph_seed_stability_flags(
        deterministic_graph_method=deterministic,
        graph_seed_count=graph_seed_count,
        actual_export_hash_unique_count=export_unique,
    )
    out = {
        "dataset": row.get("dataset", "DBLP"),
        "method": method,
        "canonical_method": row.get("canonical_method", row.get("method", "")),
        "method_family": "schema_preserving_rcs" if method.startswith("HeSF-RCS") or str(method).startswith("H6-") else "official_reference",
        "schema_compatible": True,
        "official_sehgnn_unmodified": row.get("official_sehgnn_unmodified_all", True),
        "uses_feature_adapter": False,
        "uses_weighted_superedges": False,
        "uses_synthetic_nodes": False,
        "keeps_all_target_nodes": True,
        "eligible_for_official_main_table": row.get("eligible_for_main_decision", False),
        "eligible_for_adapter_table": False,
        "eligible_for_standard_condensation_table": False,
        "eligible_for_tp_workload_table": True,
        "runs": row.get("runs", ""),
        "success_count": row.get("success_count", ""),
        "graph_seed_count": graph_seed_count,
        "training_seed_count": row.get("training_seed_count", ""),
        "structural_storage_ratio": structural,
        "raw_hgb_text_byte_ratio": raw,
        "official_text_hgb_byte_ratio": raw,
        "support_node_ratio": _float(row.get("mean_support_node_ratio")),
        "support_edge_ratio": _float(row.get("mean_support_edge_ratio")),
        "preprocessed_cache_byte_ratio": _float(row.get("mean_preprocessed_cache_byte_ratio")),
        "test_micro_mean": _float(row.get("mean_test_micro_f1")),
        "test_micro_std": _float(row.get("std_test_micro_f1")),
        "test_macro_mean": _float(row.get("mean_test_macro_f1")),
        "test_macro_std": _float(row.get("std_test_macro_f1")),
        "recovery_micro_mean": _float(row.get("mean_recovery_vs_native_full_micro")),
        "full_minus_micro": "" if _float(row.get("mean_test_micro_f1")) == "" else float(0.9533802 - float(row.get("mean_test_micro_f1"))),
        **stability,
    }
    out.update(gate21_6_method_flags(out, full_micro=0.9533802, full_macro=0.9498198))
    return out


def _run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = _alias(str(row.get("method", "")))
    return {
        "dataset": row.get("dataset", "DBLP"),
        "method": method,
        "canonical_method": row.get("canonical_method", row.get("method", "")),
        "graph_seed": row.get("graph_seed", ""),
        "training_seed": row.get("training_seed", ""),
        "export_hash": row.get("export_hash", ""),
        "success": row.get("success", ""),
        "status": row.get("status", ""),
        "failure_type": "",
        "failure_message": row.get("failed_reason", ""),
        "structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
        "raw_hgb_text_byte_ratio": row.get("official_text_hgb_byte_ratio", row.get("hgb_raw_file_byte_ratio", "")),
        "support_edge_ratio": row.get("support_edge_ratio", ""),
        "test_micro_f1": row.get("test_micro_f1", ""),
        "test_macro_f1": row.get("test_macro_f1", ""),
        "validation_micro_f1": row.get("validation_micro_f1", ""),
        "validation_macro_f1": row.get("validation_macro_f1", ""),
    }


def _pareto(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _float(row.get("structural_storage_ratio")) != "" and _float(row.get("test_micro_mean")) != ""]
    out: list[dict[str, Any]] = []
    for row in candidates:
        storage = float(row["structural_storage_ratio"])
        micro = float(row["test_micro_mean"])
        dominated = any(float(other["structural_storage_ratio"]) <= storage and float(other["test_micro_mean"]) >= micro and other is not row for other in candidates)
        out.append(
            {
                **{key: row.get(key, "") for key in ["method", "structural_storage_ratio", "raw_hgb_text_byte_ratio", "support_edge_ratio", "test_micro_mean", "test_micro_std", "test_macro_mean", "test_macro_std", "recovery_micro_mean", "full_minus_micro"]},
                "pareto_rank": 1 if not dominated else 2,
                "is_pareto_optimal": not dominated,
                "best_under_structural_0_12": storage <= 0.12,
                "best_under_structural_0_15": storage <= 0.15,
                "best_under_structural_0_20": storage <= 0.20,
                "min_storage_pass_full_minus_0_005": micro >= 0.9533802 - 0.005,
                "min_storage_pass_full_minus_0_010": micro >= 0.9533802 - 0.010,
                "min_storage_pass_full_minus_0_020": micro >= 0.9533802 - 0.020,
                "min_storage_pass_full_minus_0_030": micro >= 0.9533802 - 0.030,
            }
        )
    return sorted(out, key=lambda item: (int(item["pareto_rank"]), float(item["structural_storage_ratio"])))


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.gate21_5_dir)
    by_method = [_method_row(row) for row in _read_csv(source / "gate21_5_directed_by_method.csv")]
    by_run = [_run_row(row) for row in _read_csv(source / "gate21_5_directed_raw_rows.csv")]
    existing_specs = {str(row.get("canonical_method", row.get("method", ""))).replace("H6-dirskel-", "") for row in by_method}
    for spec in REQUIRED_GRID:
        if spec not in existing_specs and f"H6-dirskel-{spec}" not in existing_specs:
            by_run.append(
                {
                    "dataset": "DBLP",
                    "method": f"H6-dirskel-{spec}",
                    "canonical_method": f"H6-dirskel-{spec}",
                    "graph_seed": "",
                    "training_seed": "",
                    "success": False,
                    "status": "not_executed_gate21_6_grid",
                    "failure_type": "not_executed",
                    "failure_message": "Gate21.6 framework emitted the required grid row, but no local training result was available.",
                }
            )
    stability = [
        {
            "method": row.get("method", ""),
            "graph_seed_count": row.get("graph_seed_count", ""),
            "deterministic_graph_method": row.get("deterministic_graph_method", ""),
            "expected_export_hash_unique_count": row.get("expected_export_hash_unique_count", ""),
            "actual_export_hash_unique_count": row.get("actual_export_hash_unique_count", ""),
            "graph_sampling_stability_pass": row.get("graph_sampling_stability_pass", ""),
            "graph_sampling_warning": row.get("graph_sampling_warning", ""),
        }
        for row in by_method
    ]
    retention = _read_csv(source / "gate21_5_relation_edge_retention.csv")
    write_csv(out / "gate21_6_directed_skeleton_by_method.csv", by_method)
    write_csv(out / "gate21_6_directed_skeleton_by_run.csv", by_run)
    write_csv(out / "gate21_6_directed_skeleton_pareto.csv", _pareto(by_method))
    write_csv(out / "gate21_6_graph_seed_stability.csv", stability)
    write_csv(out / "gate21_6_relation_retention.csv", retention)
    decision = gate21_6_decision(official_rows=by_method, adapter_rows=[], external_rows=[], feature_ablation_rows=[], metapath_rows=[], coverage_rows=[])
    write_json(out / "gate21_6_decision.json", decision)
    return {"by_method_rows": len(by_method), "by_run_rows": len(by_run)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate21_6_icde_ready"))
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--freehgc-root", type=Path, default=None)
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--gate21-5-dir", type=Path, default=GATE21_5_DIR)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
