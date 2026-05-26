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

from experiments.scripts.summarize_gate21_5_directed_apv import summarize_gate21_5
from experiments.scripts.run_gate21_4_cache_feature_validation import build_parser as build_gate21_4_feature_parser
from experiments.scripts.run_gate21_4_cache_feature_validation import run_gate21_4_cache_feature
from hesf_coarsen.eval.official.feature_loader_audit import FEATURE_LOADER_AUDIT_FIELDS, feature_loader_audit_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


ADAPTER_FIELDS = [
    "dataset",
    "method",
    "method_family",
    "base_graph_method",
    "canonical_base_graph_method",
    "graph_seed",
    "training_seed",
    "feature_compression_method",
    "uses_feature_adapter",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "eligible_for_adapter_table",
    "official_text_hgb_byte_ratio",
    "hgb_raw_file_byte_ratio",
    "adapter_binary_feature_byte_ratio",
    "adapter_effective_deployment_byte_ratio",
    "effective_total_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "feature_storage_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "feature_loader_audit_pass",
    "fit_uses_labels",
    "fit_uses_test_labels",
    "fit_uses_validation_labels",
    "fit_uses_test_features",
    "success",
    "status",
    "failed_reason",
]

ADAPTER_MAP = {
    "raw": "raw_features_adapter_control",
    "fp16": "fp16_node_features",
    "int8": "int8_per_feature",
    "pca128": "pca_svd_dim128",
    "pca64": "pca_svd_dim64",
    "rp128": "random_projection_dim128",
    "rp64": "random_projection_dim64",
    "pca256": "pca_svd_dim256",
    "int4": "int4_per_feature_diagnostic",
}

BASE_GRAPH_MAP = {
    "APV": ("H6-APV-skeleton", "H6-relgrid-APPA100-PVVP100-PTTP00"),
    "dirskel-best": ("H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"),
}


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


def _adapter_name(token: str) -> str:
    return ADAPTER_MAP.get(str(token), str(token))


def _base_graph(token: str) -> tuple[str, str]:
    return BASE_GRAPH_MAP.get(str(token), (str(token), str(token)))


def _rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_token in _items(args.base_graphs):
        base, canonical = _base_graph(base_token)
        for adapter_token in _items(args.adapters):
            adapter = _adapter_name(adapter_token)
            for seed in args.training_seeds:
                rows.append(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": "SeHGNN-feature-compressed-adapter",
                        "method_family": "feature_compressed_adapter",
                        "base_graph_method": base,
                        "canonical_base_graph_method": canonical,
                        "graph_seed": 1,
                        "training_seed": int(seed),
                        "feature_compression_method": adapter,
                        "uses_feature_adapter": True,
                        "official_sehgnn_unmodified": False,
                        "eligible_for_main_decision": False,
                        "eligible_for_adapter_table": True,
                        "official_text_hgb_byte_ratio": "",
                        "hgb_raw_file_byte_ratio": "",
                        "adapter_binary_feature_byte_ratio": "",
                        "adapter_effective_deployment_byte_ratio": "",
                        "effective_total_byte_ratio": "",
                        "preprocessed_cache_byte_ratio": "",
                        "feature_storage_ratio": "",
                        "feature_loader_audit_pass": "",
                        "fit_uses_labels": False,
                        "fit_uses_test_labels": False,
                        "fit_uses_validation_labels": False,
                        "fit_uses_test_features": str(args.fit_split) == "all-nodes",
                        "success": False,
                        "status": "planned" if args.dry_run else "pending",
                        "failed_reason": "",
                    }
                )
    return rows


def _source_dtype(transform: str) -> str:
    if str(transform) in {"fp16-paper", "fp16_node_features"}:
        return "fp16"
    if str(transform) in {"int8-paper", "int8_per_feature"}:
        return "int8"
    return "fp32"


def _node_features_from_hgb(export_dir: Path) -> dict[int, Any]:
    import numpy as np

    by_type: dict[int, list[list[float]]] = {}
    node_path = Path(export_dir) / "node.dat"
    if not node_path.exists():
        return {}
    with node_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            type_id = int(parts[2])
            values = [] if len(parts) < 4 or parts[3] == "" else [float(item) for item in parts[3].split(",") if item != ""]
            by_type.setdefault(type_id, []).append(values)
    out = {}
    for type_id, rows in by_type.items():
        width = max((len(row) for row in rows), default=0)
        arr = np.zeros((len(rows), width), dtype=np.float32)
        for idx, row in enumerate(rows):
            if row:
                arr[idx, : len(row)] = np.asarray(row, dtype=np.float32)
        out[int(type_id)] = arr
    return out


def _feature_loader_rows_from_manifest(internal: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _read_csv(internal / "gate21_4_run_manifest.csv"):
        if item.get("run_group") != "adapter":
            continue
        export_dir = Path(item.get("export_dir", ""))
        loaded = _node_features_from_hgb(export_dir)
        transform = item.get("paper_feature_transform", "") or item.get("feature_compression_method", "")
        rows.extend(
            feature_loader_audit_rows(
                dataset=item.get("dataset", "DBLP"),
                method="SeHGNN-feature-compressed-adapter",
                canonical_method=item.get("canonical_method", item.get("method", "")),
                graph_seed=int(float(item.get("graph_seed") or 1)),
                training_seed=int(float(item.get("training_seed") or 1)),
                feature_transform_name=transform,
                before_features=loaded,
                after_features=loaded,
                loaded_features=loaded,
                fit_uses_labels=False,
                fit_uses_test_labels=False,
                loader_uses_sidecar_flag=transform not in {"raw", "raw-paper", "raw_features_adapter_control"},
                loader_uses_text_node_dat_flag=True,
                feature_transform_family="feature_compressed_adapter",
                node_types_modified="paper",
                source_storage_dtype=_source_dtype(transform),
                model_input_dtype="fp32",
            )
        )
    return rows


def _copy_gate21_4_adapter_rows(out: Path, internals: Sequence[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for internal in internals:
        for row in _read_csv(internal / "gate21_4_feature_cache_compression_results.csv"):
            rows.append(
                {
                    "dataset": row.get("dataset", str(args.dataset).upper()),
                    "method": "SeHGNN-feature-compressed-adapter",
                    "method_family": "feature_compressed_adapter",
                    "base_graph_method": row.get("base_graph_method", ""),
                    "canonical_base_graph_method": row.get("canonical_base_graph_method", ""),
                    "graph_seed": row.get("graph_seed", 1),
                    "training_seed": row.get("training_seed", ""),
                    "feature_compression_method": row.get("feature_compression_method", ""),
                    "uses_feature_adapter": True,
                    "official_sehgnn_unmodified": False,
                    "eligible_for_main_decision": False,
                    "eligible_for_adapter_table": True,
                    "official_text_hgb_byte_ratio": row.get("raw_hgb_byte_ratio", row.get("hgb_raw_file_byte_ratio", "")),
                    "hgb_raw_file_byte_ratio": row.get("raw_hgb_byte_ratio", row.get("hgb_raw_file_byte_ratio", "")),
                    "adapter_binary_feature_byte_ratio": row.get("binary_feature_sidecar_byte_ratio", ""),
                    "adapter_effective_deployment_byte_ratio": row.get("effective_total_byte_ratio", ""),
                    "effective_total_byte_ratio": row.get("effective_total_byte_ratio", ""),
                    "preprocessed_cache_byte_ratio": row.get("preprocessed_cache_byte_ratio", ""),
                    "feature_storage_ratio": row.get("feature_storage_ratio", ""),
                    "test_micro_f1": row.get("test_micro_f1", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "validation_micro_f1": row.get("validation_micro_f1", ""),
                    "validation_macro_f1": row.get("validation_macro_f1", ""),
                    "feature_loader_audit_pass": "",
                    "fit_uses_labels": False,
                    "fit_uses_test_labels": False,
                    "fit_uses_validation_labels": False,
                    "fit_uses_test_features": str(args.fit_split) == "all-nodes",
                    "success": row.get("success", ""),
                    "status": row.get("status", ""),
                    "failed_reason": row.get("failed_reason", ""),
                    "sidecar_feature_bytes": row.get("sidecar_feature_bytes", ""),
                    "sidecar_metadata_bytes": row.get("sidecar_metadata_bytes", ""),
                    "node_dat_bytes": row.get("node_dat_bytes", ""),
                    "link_dat_bytes": row.get("link_dat_bytes", ""),
                    "label_dat_bytes": row.get("label_dat_bytes", ""),
                    "label_test_dat_bytes": row.get("label_test_dat_bytes", ""),
                    "info_dat_bytes": row.get("info_dat_bytes", ""),
                    "export_total_bytes": row.get("export_total_bytes", ""),
                    "native_full_total_bytes": row.get("native_full_total_bytes", ""),
                    "preprocessed_cache_bytes": row.get("preprocessed_cache_bytes", ""),
                    "train_time_seconds": row.get("train_time_seconds", ""),
                    "peak_memory_mb": row.get("peak_memory_mb", ""),
                }
            )
    write_csv(out / "gate21_5_feature_adapter_raw_rows.csv", rows, fieldnames=[*ADAPTER_FIELDS, "sidecar_feature_bytes", "sidecar_metadata_bytes", "node_dat_bytes", "link_dat_bytes", "label_dat_bytes", "label_test_dat_bytes", "info_dat_bytes", "export_total_bytes", "native_full_total_bytes", "preprocessed_cache_bytes", "train_time_seconds", "peak_memory_mb"])
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if args.force and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        rows = _rows(args)
        write_csv(out / "gate21_5_feature_adapter_raw_rows.csv", rows, fieldnames=ADAPTER_FIELDS)
        write_json(out / "gate21_5_feature_adapter_plan.json", {"dataset": str(args.dataset).upper(), "rows": len(rows), "dry_run": True, "fit_split": str(args.fit_split)})
    else:
        internals: list[Path] = []
        for base_token in _items(args.base_graphs):
            base = _base_graph(base_token)[0]
            internal = out / "diagnostics" / "fa" / _method_safe(base)
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
                    "raw",
                    "--feature-compression-methods",
                    *[_adapter_name(item) for item in _items(args.adapters)],
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
            gate_args.feature_transforms = []
            run_gate21_4_cache_feature(gate_args)
            internals.append(internal)
        rows = _copy_gate21_4_adapter_rows(out, internals, args)
        loader_rows: list[dict[str, Any]] = []
        for internal in internals:
            loader_rows.extend(_feature_loader_rows_from_manifest(internal))
        write_csv(out / "gate21_5_feature_adapter_loader_audit.csv", loader_rows, fieldnames=FEATURE_LOADER_AUDIT_FIELDS)
        write_csv(out / "gate21_5_feature_loader_audit.csv", loader_rows, fieldnames=FEATURE_LOADER_AUDIT_FIELDS)
        write_json(out / "gate21_5_feature_adapter_plan.json", {"dataset": str(args.dataset).upper(), "rows": len(rows), "dry_run": False, "fit_split": str(args.fit_split)})
    summarize_gate21_5(out, out, native_full_micro=0.9533802, native_full_macro=0.9498198, write_md=True, write_json_flag=True)
    return {"dry_run": bool(args.dry_run), "adapter_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-graphs", default="APV,dirskel-best")
    parser.add_argument("--custom-base-graphs", default="")
    parser.add_argument("--adapters", default="raw,fp16,int8,pca128,pca64,rp128,rp64")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fit-split", choices=["all-nodes", "trainval-only"], default="all-nodes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.training_seeds = list(args.training_seeds[:3])
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
