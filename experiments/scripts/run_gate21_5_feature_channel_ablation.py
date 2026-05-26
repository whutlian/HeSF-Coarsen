from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.run_gate21_4_cache_feature_validation import build_parser as build_gate21_4_feature_parser
from experiments.scripts.run_gate21_4_cache_feature_validation import run_gate21_4_cache_feature
from experiments.scripts.run_gate21_5_feature_adapter_validation import _node_features_from_hgb
from experiments.scripts.summarize_gate21_5_directed_apv import summarize_gate21_5
from hesf_coarsen.eval.official.feature_loader_audit import FEATURE_LOADER_AUDIT_FIELDS, feature_loader_audit_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


FEATURE_CHANNEL_FIELDS = [
    "dataset",
    "method",
    "base_graph_method",
    "canonical_base_graph_method",
    "graph_seed",
    "training_seed",
    "relation_channel_spec",
    "term_channel_spec",
    "feature_transform_name",
    "feature_transform_family",
    "node_types_modified",
    "feature_dim_by_type_after_loader",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "effective_total_byte_ratio",
    "support_edge_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "eligible_for_adapter_table",
    "cache_hygiene_pass",
    "feature_loader_audit_pass",
    "fit_uses_labels",
    "fit_uses_test_labels",
    "success",
    "status",
    "failed_reason",
]


BASE_GRAPH_MAP = {
    "APV": ("H6-APV-skeleton", "H6-relgrid-APPA100-PVVP100-PTTP00"),
    "dirskel-best": ("H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"),
}


DEFAULT_TRANSFORMS = "raw,zero-author,zero-paper,zero-venue,zero-term,zero-target-author-only,zero-support-author-only,zero-all-support-features,author-only-features,paper-only-features,venue-only-features,term-only-features,type-constant-support-features,all-type-constant-features,pca-all-types-128,pca-paper-128,pca-paper-64,random-projection-paper-128"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _items(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _method_safe(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(method))


def _base_graph(token: str) -> tuple[str, str]:
    return BASE_GRAPH_MAP.get(str(token), (str(token), str(token)))


def _transform_family(name: str) -> str:
    if name == "raw":
        return "raw"
    if name.startswith("zero-"):
        return "zero"
    if name.endswith("-only-features"):
        return "type_only"
    if "constant" in name:
        return "type_constant"
    if name.startswith("pca"):
        return "pca"
    if name.startswith("random_projection") or name.startswith("random-projection"):
        return "random_projection"
    return "feature_ablation"


def _dry_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_token in _items(args.base_graphs):
        base, canonical = _base_graph(base_token)
        for transform in _items(args.feature_transforms):
            for term in args.term_channel_specs:
                for seed in args.training_seeds:
                    rows.append(
                        {
                            "dataset": str(args.dataset).upper(),
                            "method": "feature_channel_ablation",
                            "base_graph_method": base,
                            "canonical_base_graph_method": canonical,
                            "graph_seed": 1,
                            "training_seed": int(seed),
                            "relation_channel_spec": "",
                            "term_channel_spec": str(term),
                            "feature_transform_name": str(transform),
                            "feature_transform_family": _transform_family(str(transform)),
                            "official_sehgnn_unmodified": False,
                            "eligible_for_main_decision": False,
                            "eligible_for_adapter_table": True,
                            "success": False,
                            "status": "planned",
                            "failed_reason": "",
                        }
                    )
    return rows


def _loader_rows_from_manifest(internal: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _read_csv(internal / "gate21_4_run_manifest.csv"):
        if item.get("run_group") != "feature_channel":
            continue
        loaded = _node_features_from_hgb(Path(item.get("export_dir", "")))
        transform = item.get("paper_feature_transform", "")
        rows.extend(
            feature_loader_audit_rows(
                dataset=item.get("dataset", "DBLP"),
                method="feature_channel_ablation",
                canonical_method=item.get("canonical_method", item.get("method", "")),
                graph_seed=int(float(item.get("graph_seed") or 1)),
                training_seed=int(float(item.get("training_seed") or 1)),
                feature_transform_name=transform,
                before_features=loaded,
                after_features=loaded,
                loaded_features=loaded,
                feature_transform_family=_transform_family(transform),
                node_types_modified="",
                fit_uses_labels=False,
                fit_uses_test_labels=False,
            )
        )
    return rows


def _copy_rows(out: Path, internals: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for internal in internals:
        for row in _read_csv(internal / "gate21_4_feature_channel_ablation.csv"):
            rows.append(
                {
                    "dataset": row.get("dataset", "DBLP"),
                    "method": "feature_channel_ablation",
                    "base_graph_method": row.get("base_graph_method", ""),
                    "canonical_base_graph_method": row.get("canonical_base_graph_method", ""),
                    "graph_seed": row.get("graph_seed", 1),
                    "training_seed": row.get("training_seed", ""),
                    "relation_channel_spec": row.get("term_channel_spec", ""),
                    "term_channel_spec": row.get("term_channel_spec", ""),
                    "feature_transform_name": row.get("paper_feature_transform", ""),
                    "feature_transform_family": _transform_family(row.get("paper_feature_transform", "")),
                    "node_types_modified": "",
                    "feature_dim_by_type_after_loader": "",
                    "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
                    "hgb_raw_file_byte_ratio": row.get("hgb_raw_file_byte_ratio", ""),
                    "effective_total_byte_ratio": row.get("hgb_raw_file_byte_ratio", ""),
                    "support_edge_ratio": row.get("support_edge_ratio", ""),
                    "test_micro_f1": row.get("test_micro_f1", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "validation_micro_f1": row.get("validation_micro_f1", ""),
                    "validation_macro_f1": row.get("validation_macro_f1", ""),
                    "official_sehgnn_unmodified": False,
                    "eligible_for_main_decision": False,
                    "eligible_for_adapter_table": True,
                    "cache_hygiene_pass": row.get("cache_hygiene_pass", ""),
                    "feature_loader_audit_pass": "",
                    "fit_uses_labels": row.get("fit_uses_labels", False),
                    "fit_uses_test_labels": row.get("fit_uses_test_labels", False),
                    "success": row.get("success", ""),
                    "status": row.get("status", ""),
                    "failed_reason": row.get("failed_reason", ""),
                }
            )
    write_csv(out / "gate21_5_feature_channel_ablation.csv", rows, fieldnames=FEATURE_CHANNEL_FIELDS)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if args.force and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        rows = _dry_rows(args)
        write_csv(out / "gate21_5_feature_channel_ablation.csv", rows, fieldnames=FEATURE_CHANNEL_FIELDS)
        write_json(out / "gate21_5_feature_channel_plan.json", {"dataset": str(args.dataset).upper(), "rows": len(rows), "dry_run": True})
    else:
        internals: list[Path] = []
        for base_token in _items(args.base_graphs):
            base = _base_graph(base_token)[0]
            internal = out / "diagnostics" / "fc" / _method_safe(base)
            gate_args = build_gate21_4_feature_parser().parse_args(
                [
                    "--dataset",
                    str(args.dataset),
                    "--output-dir",
                    str(internal),
                    "--base-graph",
                    base,
                    "--graph-seeds",
                    "1",
                    "--training-seeds",
                    *[str(seed) for seed in args.training_seeds],
                    "--feature-transforms",
                    *[str(item) for item in _items(args.feature_transforms)],
                    "--term-channel-specs",
                    *[str(item) for item in args.term_channel_specs],
                    "--feature-compression-methods",
                    "raw_features_adapter_control",
                    "--force",
                    "--force-reprocess",
                    "--sehgnn-root",
                    str(args.sehgnn_root),
                    "--hgb-data-root",
                    str(args.hgb_data_root),
                    "--data-root",
                    str(args.data_root),
                    "--device",
                    str(args.device),
                ]
            )
            gate_args.feature_compression_methods = []
            run_gate21_4_cache_feature(gate_args)
            internals.append(internal)
        rows = _copy_rows(out, internals)
        loader_rows: list[dict[str, Any]] = []
        for internal in internals:
            loader_rows.extend(_loader_rows_from_manifest(internal))
        write_csv(out / "gate21_5_feature_channel_loader_audit.csv", loader_rows, fieldnames=FEATURE_LOADER_AUDIT_FIELDS)
        write_csv(out / "gate21_5_feature_loader_audit.csv", loader_rows, fieldnames=FEATURE_LOADER_AUDIT_FIELDS)
        write_json(out / "gate21_5_feature_channel_plan.json", {"dataset": str(args.dataset).upper(), "rows": len(rows), "dry_run": False})
    summarize_gate21_5(out, out, native_full_micro=0.9533802, native_full_macro=0.9498198, write_md=True, write_json_flag=True)
    return {"dry_run": bool(args.dry_run), "feature_channel_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-graphs", default="APV,dirskel-best")
    parser.add_argument("--custom-base-graphs", default="")
    parser.add_argument("--feature-transforms", default=DEFAULT_TRANSFORMS)
    parser.add_argument("--term-channel-specs", nargs="+", default=["PTTP00", "PTTP30", "PTTP100"])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.training_seeds = list(args.training_seeds[:1])
        args.term_channel_specs = list(args.term_channel_specs[:1])
        args.feature_transforms = ",".join(_items(args.feature_transforms)[:3])
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
