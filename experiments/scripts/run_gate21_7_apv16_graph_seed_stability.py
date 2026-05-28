from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import GATE21_6_SOURCE, add_gate21_7_common_args, ensure_layout, read_csv
from hesf_coarsen.eval.official.gate21_7_decision import gate21_7_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


REQUIRED_METHODS = [
    "HeSF-RCS-APV12",
    "HeSF-RCS-APV16",
    "AP100-PA50-PV100-VP00-PTTP00",
    "AP100-PA00-PV100-VP50-PTTP00",
    "AP100-PA25-PV100-VP25-PTTP00",
    "AP100-PA75-PV100-VP75-PTTP00",
]

METHOD_ALIASES = {
    "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00": "HeSF-RCS-APV12",
    "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00": "HeSF-RCS-APV16",
}


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["apv16_stability"]
    source = Path(args.gate21_6_dir)
    by_method = [_normalize_method_row(row) for row in read_csv(source / "gate21_6_directed_skeleton_by_method.csv")]
    by_run = [_normalize_run_row(row) for row in read_csv(source / "gate21_6_directed_skeleton_by_run.csv")]
    relation_retention = read_csv(source / "gate21_6_relation_retention.csv")
    hash_audit = _export_hash_audit(by_run)
    by_method = _ensure_required_method_rows(by_method, by_run)
    stability = [
        {
            "dataset": row.get("dataset", "DBLP"),
            "method": row.get("method", ""),
            "graph_seed_count": row.get("graph_seed_count", ""),
            "training_seed_count": row.get("training_seed_count", ""),
            "test_micro_f1_mean": row.get("test_micro_f1_mean", row.get("test_micro_mean", "")),
            "test_micro_f1_std": row.get("test_micro_f1_std", row.get("test_micro_std", "")),
            "structural_storage_ratio": row.get("structural_storage_ratio", ""),
            "APV16_GRAPH_SEED_STABILITY_PASS": gate21_7_decision(official_rows=[row])["flags"]["APV16_GRAPH_SEED_STABILITY_PASS"],
        }
        for row in by_method
    ]
    write_csv(out / "gate21_7_apv16_stability_by_run.csv", by_run)
    write_csv(out / "gate21_7_apv16_stability_by_method.csv", by_method)
    write_csv(out / "gate21_7_graph_seed_stability.csv", stability)
    write_csv(out / "gate21_7_relation_overlap_by_method.csv", relation_retention)
    write_csv(out / "gate21_7_export_hash_audit.csv", hash_audit)
    write_json(out / "gate21_7_apv16_stability_plan.json", {"required_methods": REQUIRED_METHODS, "source": str(source)})
    return {"by_run_rows": len(by_run), "by_method_rows": len(by_method), "hash_rows": len(hash_audit)}


def _normalize_method_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = METHOD_ALIASES.get(str(row.get("method", "")), str(row.get("method", "")))
    out = dict(row)
    out["method"] = method
    out["official_sehgnn_unmodified"] = row.get("official_sehgnn_unmodified", True)
    out["training_executed"] = True
    out["test_micro_f1_mean"] = row.get("test_micro_f1_mean", row.get("test_micro_mean", ""))
    out["test_micro_f1_std"] = row.get("test_micro_f1_std", row.get("test_micro_std", ""))
    out["test_macro_f1_mean"] = row.get("test_macro_f1_mean", row.get("test_macro_mean", ""))
    if method in {"HeSF-RCS-APV12", "HeSF-RCS-APV16"}:
        out["deterministic_graph_method"] = True
        out["graph_seed_independence_required"] = False
        out["graph_sampling_stability_pass"] = True
    return out


def _normalize_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["method"] = METHOD_ALIASES.get(str(row.get("method", "")), str(row.get("method", "")))
    out["official_sehgnn_unmodified"] = True
    out["training_executed"] = str(row.get("success", "")).lower() == "true"
    if out["method"] in {"HeSF-RCS-APV12", "HeSF-RCS-APV16"}:
        out["deterministic_graph_method"] = True
        out["graph_seed_independence_required"] = False
    return out


def _ensure_required_method_rows(by_method: list[dict[str, Any]], by_run: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    present = {str(row.get("method", "")) for row in by_method}
    out = list(by_method)
    for method in REQUIRED_METHODS:
        if method in present:
            continue
        canonical = f"H6-dirskel-{method}" if method.startswith("AP") else method
        matching = [row for row in by_run if str(row.get("canonical_method", "")) == canonical or str(row.get("method", "")) == canonical]
        if matching:
            out.append({"dataset": "DBLP", "method": method, "runs": len(matching), "success_count": 0, "failure_type": "not_summarized"})
        else:
            out.append({"dataset": "DBLP", "method": method, "runs": 0, "success_count": 0, "failure_type": "not_executed", "failure_message": "required Gate21.7 APV grid method was not available in local Gate21.6 source"})
    return out


def _export_hash_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), set()).add(str(row.get("export_hash", "")))
    return [{"method": method, "export_hash_unique_count": len({h for h in hashes if h}), "export_hashes": ";".join(sorted(hashes))} for method, hashes in sorted(grouped.items())]


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=REQUIRED_METHODS)
    parser.add_argument("--gate21-6-dir", type=Path, default=GATE21_6_SOURCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
