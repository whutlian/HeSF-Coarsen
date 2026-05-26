from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate13_task_first_common import load_hgb_graph
from experiments.scripts.run_gate21_1_sehgnn_schema_edge_budget import _compressed_labels_and_splits, _float, _native_labels, _storage_fields
from experiments.scripts.run_gate21_3_relation_channel import (
    MethodSpec,
    _prune_graph,
    _run_id,
    _support_graph,
)
from experiments.scripts.summarize_gate21_4_cache_feature_validation import summarize_gate21_4_cache_feature
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.cache_hygiene import CacheNamespace, collect_cache_audit_before_after, compute_export_file_list_hash, file_hashes_for_export, prepare_unique_cache_dir
from hesf_coarsen.eval.official.edge_pruning_baselines import semantic_storage_ratio
from hesf_coarsen.eval.official.feature_cache_storage import adapter_storage_row
from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command, run_native_command
from hesf_coarsen.eval.official.storage_audit import audit_hgb_directory


FEATURE_CHANNEL_FIELDS = [
    "dataset",
    "method",
    "base_graph_method",
    "canonical_base_graph_method",
    "graph_seed",
    "training_seed",
    "term_channel_spec",
    "paper_feature_transform",
    "paper_feature_dim",
    "fit_uses_labels",
    "fit_uses_test_labels",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "support_edge_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "cache_hygiene_pass",
    "success",
    "status",
    "failed_reason",
]

FEATURE_CACHE_FIELDS_21_4 = [
    "dataset",
    "method",
    "base_graph_method",
    "canonical_base_graph_method",
    "graph_seed",
    "training_seed",
    "feature_compression_method",
    "feature_dtype",
    "feature_dim",
    "feature_storage_ratio",
    "raw_hgb_byte_ratio",
    "effective_total_byte_ratio",
    "binary_feature_sidecar_byte_ratio",
    "sidecar_feature_bytes",
    "sidecar_metadata_bytes",
    "node_dat_bytes",
    "link_dat_bytes",
    "label_dat_bytes",
    "label_test_dat_bytes",
    "info_dat_bytes",
    "export_total_bytes",
    "native_full_total_bytes",
    "preprocessed_cache_byte_ratio",
    "preprocessed_cache_bytes",
    "train_time_seconds",
    "train_time_ratio",
    "peak_memory_mb",
    "peak_memory_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_vs_uncompressed_apv_micro",
    "recovery_vs_native_full_micro",
    "official_sehgnn_unmodified",
    "eligible_for_main_decision",
    "adapter_family",
    "cache_hygiene_pass",
    "success",
    "status",
    "failed_reason",
]

FEATURE_TRANSFORM_AUDIT_FIELDS = [
    "dataset",
    "graph_seed",
    "training_seed",
    "transform_name",
    "term_channel_spec",
    "input_shape",
    "output_shape",
    "feature_dtype",
    "feature_dim",
    "sidecar_metadata_bytes",
    "metadata_keys",
    "fit_uses_labels",
    "fit_uses_test_labels",
]

CACHE_AUDIT_FIELDS = [
    "dataset",
    "method",
    "graph_seed",
    "training_seed",
    "export_dir",
    "export_file_list_hash",
    "node_dat_hash",
    "link_dat_hash",
    "label_dat_hash",
    "label_test_dat_hash",
    "info_dat_hash",
    "preprocess_cache_dir",
    "cache_dir_exists_before_run",
    "cache_dir_deleted_before_run",
    "force_reprocess_flag",
    "unique_cache_namespace_flag",
    "cache_files_count_before",
    "cache_files_count_after",
    "cache_hash_before",
    "cache_hash_after",
    "cache_reused_flag",
    "cache_hygiene_pass",
    "notes",
]

RUN_MANIFEST_FIELDS = [
    "run_id",
    "dataset",
    "method",
    "canonical_method",
    "run_group",
    "graph_seed",
    "training_seed",
    "relation_channel_spec",
    "relation_direction_spec",
    "paper_feature_transform",
    "feature_compression_method",
    "sehgnn_command_json",
    "export_dir",
    "cache_dir",
    "output_dir",
    "status",
]

STORAGE_AUDIT_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "semantic_structural_storage_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "total_node_ratio",
    "total_edge_ratio",
    "hgb_raw_file_byte_ratio",
    "effective_total_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "node_dat_bytes",
    "link_dat_bytes",
    "label_dat_bytes",
    "label_test_dat_bytes",
    "info_dat_bytes",
    "metadata_bytes",
    "feature_sidecar_bytes",
    "feature_sidecar_metadata_bytes",
    "preprocessed_cache_bytes",
    "export_total_bytes",
    "native_full_total_bytes",
    "structural_storage50_pass",
    "structural_storage40_pass",
    "structural_storage30_pass",
    "structural_storage20_pass",
    "raw_hgb_byte50_pass",
    "raw_hgb_byte30_pass",
    "effective_byte50_pass",
    "effective_byte30_pass",
    "cache_byte50_pass",
    "cache_byte30_pass",
    "official_sehgnn_unmodified",
    "adapter_family",
    "eligible_for_main_decision",
]


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _term_spec(token: str) -> str:
    token = str(token).upper()
    if token.startswith("APPA"):
        return token
    if token.startswith("PTTP"):
        return f"APPA100-PVVP100-{token}"
    raise ValueError(f"unsupported term channel spec: {token}")


def _compression_to_transform(method: str) -> tuple[str, str, int | str, float]:
    mapping = {
        "raw_features_adapter_control": ("raw", "fp32", "", 1.0),
        "fp16_node_features": ("fp16-paper", "fp16", "", 0.5),
        "int8_per_feature": ("int8-paper", "int8", "", 0.25),
        "pca_svd_dim256": ("pca-paper-256", "fp32", 256, 256 / 4231),
        "pca_svd_dim128": ("pca-paper-128", "fp32", 128, 128 / 4231),
        "pca_svd_dim64": ("pca-paper-64", "fp32", 64, 64 / 4231),
        "random_projection_dim128": ("random_projection_dim128", "fp32", 128, 128 / 4231),
    }
    if method not in mapping:
        raise ValueError(f"unsupported feature compression method: {method}")
    return mapping[method]


def _empty_outputs(out: Path) -> None:
    write_csv(out / "gate21_4_feature_channel_ablation.csv", [], fieldnames=FEATURE_CHANNEL_FIELDS)
    write_csv(out / "gate21_4_feature_cache_compression_results.csv", [], fieldnames=FEATURE_CACHE_FIELDS_21_4)
    write_csv(out / "gate21_4_feature_transform_audit.csv", [], fieldnames=FEATURE_TRANSFORM_AUDIT_FIELDS)
    write_csv(out / "gate21_4_cache_audit.csv", [], fieldnames=CACHE_AUDIT_FIELDS)
    write_csv(out / "gate21_4_storage_audit.csv", [], fieldnames=STORAGE_AUDIT_FIELDS)
    write_csv(out / "gate21_4_run_manifest.csv", [], fieldnames=RUN_MANIFEST_FIELDS)


def _write_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    feature_rows = [
        {
            "dataset": str(args.dataset).upper(),
            "method": "feature_channel_ablation",
            "base_graph_method": str(args.base_graph),
            "canonical_base_graph_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "term_channel_spec": str(term),
            "paper_feature_transform": str(transform),
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
            "success": False,
            "status": "planned",
        }
        for graph_seed in args.graph_seeds
        for training_seed in args.training_seeds
        for transform in args.feature_transforms
        for term in args.term_channel_specs
    ]
    adapter_rows = [
        {
            "dataset": str(args.dataset).upper(),
            "method": "SeHGNN-feature-compressed-adapter",
            "base_graph_method": str(args.base_graph),
            "canonical_base_graph_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "feature_compression_method": str(method),
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
            "adapter_family": "feature_cache_compression",
            "success": False,
            "status": "planned",
        }
        for graph_seed in args.graph_seeds
        for training_seed in args.training_seeds
        for method in args.feature_compression_methods
    ]
    manifest_rows = [
        {
            "run_id": _run_id(str(args.dataset).upper(), f"feature_channel-{transform}-{term}", int(graph_seed), int(training_seed)),
            "dataset": str(args.dataset).upper(),
            "method": "feature_channel_ablation",
            "canonical_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "run_group": "feature_channel_ablation",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "relation_channel_spec": _term_spec(str(term)),
            "relation_direction_spec": "",
            "paper_feature_transform": str(transform),
            "feature_compression_method": "",
            "sehgnn_command_json": "",
            "export_dir": str(out / "exports" / str(args.dataset).upper() / f"graph_seed_{graph_seed}"),
            "cache_dir": str(out / "cache"),
            "output_dir": str(out / "runs" / str(args.dataset).upper() / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}"),
            "status": "planned",
        }
        for graph_seed in args.graph_seeds
        for training_seed in args.training_seeds
        for transform in args.feature_transforms
        for term in args.term_channel_specs
    ]
    manifest_rows.extend(
        {
            "run_id": _run_id(str(args.dataset).upper(), f"adapter-{method}", int(graph_seed), int(training_seed)),
            "dataset": str(args.dataset).upper(),
            "method": "SeHGNN-feature-compressed-adapter",
            "canonical_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "run_group": "feature_cache_adapter",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "relation_channel_spec": "APPA100-PVVP100-PTTP00",
            "relation_direction_spec": "",
            "paper_feature_transform": _compression_to_transform(str(method))[0],
            "feature_compression_method": str(method),
            "sehgnn_command_json": "",
            "export_dir": str(out / "exports" / str(args.dataset).upper() / f"graph_seed_{graph_seed}"),
            "cache_dir": str(out / "cache"),
            "output_dir": str(out / "runs" / str(args.dataset).upper() / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}"),
            "status": "planned",
        }
        for graph_seed in args.graph_seeds
        for training_seed in args.training_seeds
        for method in args.feature_compression_methods
    )
    write_csv(out / "gate21_4_feature_channel_ablation.csv", feature_rows, fieldnames=FEATURE_CHANNEL_FIELDS)
    write_csv(out / "gate21_4_feature_cache_compression_results.csv", adapter_rows, fieldnames=FEATURE_CACHE_FIELDS_21_4)
    write_csv(out / "gate21_4_run_manifest.csv", manifest_rows, fieldnames=RUN_MANIFEST_FIELDS)
    write_json(out / "gate21_4_plan.json", {"dataset": str(args.dataset).upper(), "dry_run": True, "feature_rows": len(feature_rows), "adapter_rows": len(adapter_rows)})
    summary = summarize_gate21_4_cache_feature(out, out)
    return {"dry_run": True, "feature_rows": len(feature_rows), "adapter_rows": len(adapter_rows), "summary": summary}


def run_gate21_4_cache_feature(args: argparse.Namespace) -> dict[str, Any]:
    if _bool_arg(args.dry_run) or _bool_arg(args.plan_only):
        return _write_dry_run(args)
    out = Path(args.output_dir)
    if _bool_arg(args.force) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out)
    dataset = str(args.dataset).upper()
    original = load_hgb_graph(Path(args.data_root), dataset)
    target_type = int(infer_target_node_type(original))
    labels_native, trainval_native, test_native = _native_labels(Path(args.hgb_data_root), dataset, original.num_nodes)
    feature_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    transform_audit_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    write_json(
        out / "gate21_4_plan.json",
        {
            "dataset": dataset,
            "dry_run": False,
            "base_graph": str(args.base_graph),
            "graph_seeds": [int(seed) for seed in args.graph_seeds],
            "training_seeds": [int(seed) for seed in args.training_seeds],
            "feature_transforms": [str(item) for item in args.feature_transforms],
            "term_channel_specs": [str(item) for item in args.term_channel_specs],
            "feature_compression_methods": [str(item) for item in args.feature_compression_methods],
        },
    )
    for graph_seed in args.graph_seeds:
        base_graph, assignment = _support_graph(original, "H6", graph_seed=int(graph_seed), candidate_k=int(args.candidate_k))
        for training_seed in args.training_seeds:
            for transform in args.feature_transforms:
                for term in args.term_channel_specs:
                    row, audit_rows, cache_row, storage_row, manifest_row = _run_one(
                        args=args,
                        original=original,
                        base_graph=base_graph,
                        assignment=assignment,
                        labels_native=labels_native,
                        trainval_native=trainval_native,
                        test_native=test_native,
                        target_type=target_type,
                        graph_seed=int(graph_seed),
                        training_seed=int(training_seed),
                        relation_spec=_term_spec(term),
                        transform_name=str(transform),
                        result_kind="feature_channel",
                    )
                    feature_rows.append(row)
                    transform_audit_rows.extend(audit_rows)
                    cache_rows.extend(cache_row)
                    storage_rows.extend(storage_row)
                    manifest_rows.extend(manifest_row)
            for compression_method in args.feature_compression_methods:
                transform, dtype, dim, storage_ratio = _compression_to_transform(compression_method)
                row, audit_rows, cache_row, storage_row, manifest_row = _run_one(
                    args=args,
                    original=original,
                    base_graph=base_graph,
                    assignment=assignment,
                    labels_native=labels_native,
                    trainval_native=trainval_native,
                    test_native=test_native,
                    target_type=target_type,
                    graph_seed=int(graph_seed),
                    training_seed=int(training_seed),
                    relation_spec="APPA100-PVVP100-PTTP00",
                    transform_name=transform,
                    result_kind="adapter",
                    feature_compression_method=str(compression_method),
                    feature_dtype=str(dtype),
                    feature_dim=dim,
                    feature_storage_ratio=float(storage_ratio),
                )
                adapter_rows.append(row)
                transform_audit_rows.extend(audit_rows)
                cache_rows.extend(cache_row)
                storage_rows.extend(storage_row)
                manifest_rows.extend(manifest_row)
    write_csv(out / "gate21_4_feature_channel_ablation.csv", feature_rows, fieldnames=FEATURE_CHANNEL_FIELDS)
    write_csv(out / "gate21_4_feature_cache_compression_results.csv", adapter_rows, fieldnames=FEATURE_CACHE_FIELDS_21_4)
    write_csv(out / "gate21_4_feature_transform_audit.csv", transform_audit_rows, fieldnames=FEATURE_TRANSFORM_AUDIT_FIELDS)
    write_csv(out / "gate21_4_cache_audit.csv", cache_rows, fieldnames=CACHE_AUDIT_FIELDS)
    write_csv(out / "gate21_4_storage_audit.csv", storage_rows, fieldnames=STORAGE_AUDIT_FIELDS)
    write_csv(out / "gate21_4_run_manifest.csv", manifest_rows, fieldnames=RUN_MANIFEST_FIELDS)
    summary = summarize_gate21_4_cache_feature(out, out)
    return {"feature_rows": len(feature_rows), "adapter_rows": len(adapter_rows), "summary": summary}


def _run_one(
    *,
    args: argparse.Namespace,
    original,
    base_graph,
    assignment,
    labels_native,
    trainval_native,
    test_native,
    target_type: int,
    graph_seed: int,
    training_seed: int,
    relation_spec: str,
    transform_name: str,
    result_kind: str,
    feature_compression_method: str = "",
    feature_dtype: str = "",
    feature_dim: int | str = "",
    feature_storage_ratio: float | str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = str(args.dataset).upper()
    out = Path(args.output_dir)
    method_name = f"{result_kind}-{_safe(transform_name)}-{_safe(relation_spec)}"
    try:
        spec = MethodSpec(method_name, "gate21_4_cache_feature", "feature_cache_adapter_probe", "relation_channel_grid", "random_edge_within_relation", relation_spec, eligible_for_main_decision=False)
        graph, _candidate, _retained, _requested, _min_active, _score, _coverage = _prune_graph(
            graph=base_graph,
            original=original,
            dataset=dataset,
            spec=spec,
            graph_seed=int(graph_seed),
            target_type=int(target_type),
            train_idx=trainval_native,
            labels=labels_native,
            min_edges_per_relation=int(args.min_edges_per_relation),
        )
        graph = _transform_paper_features(graph, transform_name, seed=int(graph_seed))
        transform_audit = getattr(graph, "_gate21_4_transform_audit", {})
        labels, trainval, test = _compressed_labels_and_splits(
            original=original,
            graph=graph,
            assignment=assignment,
            target_type=int(target_type),
            labels_native=labels_native,
            trainval_native=trainval_native,
            test_native=test_native,
        )
        manifest = export_graph_to_sehgnn_hgb(
            graph=graph,
            dataset_name=dataset,
            target_type="A",
            output_dir=out / "exports" / dataset / f"graph_seed_{graph_seed}",
            split_mode="official_trainval",
            train_idx=trainval,
            val_idx=np.array([], dtype=np.int64),
            test_idx=test,
            labels=labels,
            method_name=method_name,
            seed=int(graph_seed),
        )
        export_dir = Path(manifest["export_dir"])
        export_hash = str(manifest.get("file_list_hash") or compute_export_file_list_hash(export_dir))
        namespace = CacheNamespace(dataset, method_name, int(graph_seed), int(training_seed), export_hash, out / "cache")
        before = prepare_unique_cache_dir(namespace, force_reprocess=_bool_arg(args.force_reprocess))
        storage_base = _storage_fields(original, graph, int(target_type), Path(args.hgb_data_root) / dataset, export_dir)
        storage = audit_hgb_directory(
            dataset=dataset,
            method=method_name,
            seed=int(graph_seed),
            export_dir=export_dir,
            native_full_dir=Path(args.hgb_data_root) / dataset,
            semantic_structural_storage_ratio=float(semantic_storage_ratio(graph, original)),
            support_node_ratio=_float(storage_base.get("support_node_ratio")),
            support_edge_ratio=_float(storage_base.get("support_edge_ratio")),
            total_node_ratio=_float(storage_base.get("total_node_ratio")),
            total_edge_ratio=_float(storage_base.get("total_edge_ratio")),
            structural_budget=None,
            raw_byte_budget=0.50,
            cache_dir=namespace.cache_dir,
        ).to_row(method_family="feature_cache_adapter_probe")
        command = build_official_hgb_command(
            dataset=dataset,
            seed=int(training_seed),
            repo_dir=Path(args.sehgnn_root),
            data_root=export_dir.parent,
            device=str(args.device),
            python_executable=sys.executable,
        )
        run_dir = out / "runs" / dataset / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}" / _safe(method_name)
        if _bool_arg(args.skip_official_training):
            run_row = {"status": "skipped", "error_message": "skip_official_training"}
        else:
            run_row = run_native_command(command, stdout_path=run_dir / "stdout.log", stderr_path=run_dir / "stderr.log")
        cache_after = collect_cache_audit_before_after(namespace.cache_dir, before)
        cache_row = {"dataset": dataset, "method": method_name, "graph_seed": int(graph_seed), "training_seed": int(training_seed), "export_dir": str(export_dir), **file_hashes_for_export(export_dir), **cache_after}
        common = {
            "dataset": dataset,
            "base_graph_method": "H6-APV-skeleton",
            "canonical_base_graph_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "semantic_structural_storage_ratio": storage.get("semantic_structural_storage_ratio", ""),
            "hgb_raw_file_byte_ratio": storage.get("hgb_raw_file_byte_ratio", ""),
            "support_edge_ratio": storage.get("support_edge_ratio", ""),
            "test_micro_f1": run_row.get("test_micro_f1", ""),
            "test_macro_f1": run_row.get("test_macro_f1", ""),
            "validation_micro_f1": run_row.get("validation_micro_f1", ""),
            "validation_macro_f1": run_row.get("validation_macro_f1", ""),
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
            "cache_hygiene_pass": cache_after["cache_hygiene_pass"],
            "success": run_row.get("status") == "success",
            "status": run_row.get("status", ""),
            "failed_reason": "" if run_row.get("status") == "success" else run_row.get("error_message", ""),
        }
        audit_row = {"dataset": dataset, "graph_seed": int(graph_seed), "training_seed": int(training_seed), "term_channel_spec": relation_spec, **transform_audit}
        manifest_row = _manifest_row(
            dataset=dataset,
            method=method_name,
            run_group=result_kind,
            graph_seed=int(graph_seed),
            training_seed=int(training_seed),
            relation_spec=relation_spec,
            transform_name=transform_name,
            feature_compression_method=feature_compression_method,
            command=command,
            export_dir=export_dir,
            cache_dir=namespace.cache_dir,
            run_dir=run_dir,
            status=run_row.get("status", ""),
        )
        if result_kind == "adapter":
            node_bytes = int(float(storage.get("node_dat_bytes") or 0))
            native_total = int(float(storage.get("native_full_total_bytes") or 1))
            sidecar_bytes = int(node_bytes * float(feature_storage_ratio or 1.0))
            adapter = adapter_storage_row(
                dataset=dataset,
                method="SeHGNN-feature-compressed-adapter",
                base_graph_method="H6-APV-skeleton",
                graph_seed=int(graph_seed),
                training_seed=int(training_seed),
                native_full_total_bytes=native_total,
                export_total_bytes=int(float(storage.get("export_total_bytes") or 0)),
                node_dat_bytes=node_bytes,
                link_dat_bytes=int(float(storage.get("link_dat_bytes") or 0)),
                label_dat_bytes=int(float(storage.get("label_dat_bytes") or 0)),
                label_test_dat_bytes=int(float(storage.get("label_test_dat_bytes") or 0)),
                info_dat_bytes=int(float(storage.get("info_dat_bytes") or 0)),
                sidecar_feature_bytes=sidecar_bytes,
                sidecar_metadata_bytes=int(transform_audit.get("sidecar_metadata_bytes") or 0),
            )
            adapter.update(
                {
                    **common,
                    "method": "SeHGNN-feature-compressed-adapter",
                    "feature_compression_method": feature_compression_method,
                    "feature_dtype": feature_dtype,
                    "feature_dim": feature_dim,
                    "feature_storage_ratio": feature_storage_ratio,
                    "preprocessed_cache_byte_ratio": storage.get("preprocessed_cache_byte_ratio", ""),
                    "train_time_seconds": run_row.get("train_time_sec", ""),
                    "peak_memory_mb": run_row.get("peak_memory_mb", ""),
                    "recovery_vs_uncompressed_apv_micro": "",
                    "recovery_vs_native_full_micro": _recovery(run_row.get("test_micro_f1"), 0.9533802),
                }
            )
            storage_row = _storage_audit_row(storage, adapter, int(graph_seed), int(training_seed), "feature_cache_compression")
            return (
                {field: adapter.get(field, "") for field in FEATURE_CACHE_FIELDS_21_4},
                [audit_row],
                [{field: cache_row.get(field, "") for field in CACHE_AUDIT_FIELDS}],
                [storage_row],
                [manifest_row],
            )
        row = {
            **common,
            "method": "feature_channel_ablation",
            "term_channel_spec": relation_spec.replace("APPA100-PVVP100-", ""),
            "paper_feature_transform": transform_name,
            "paper_feature_dim": transform_audit.get("feature_dim", ""),
            "fit_uses_labels": transform_audit.get("fit_uses_labels", False),
            "fit_uses_test_labels": transform_audit.get("fit_uses_test_labels", False),
        }
        storage_row = _storage_audit_row(storage, row, int(graph_seed), int(training_seed), "feature_channel_ablation")
        return (
            {field: row.get(field, "") for field in FEATURE_CHANNEL_FIELDS},
            [audit_row],
            [{field: cache_row.get(field, "") for field in CACHE_AUDIT_FIELDS}],
            [storage_row],
            [manifest_row],
        )
    except Exception as exc:
        trace_path = Path(args.output_dir) / "tracebacks" / f"{_run_id(dataset, method_name, graph_seed, training_seed)}.txt"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        failed = {
            "dataset": dataset,
            "method": "SeHGNN-feature-compressed-adapter" if result_kind == "adapter" else "feature_channel_ablation",
            "base_graph_method": "H6-APV-skeleton",
            "canonical_base_graph_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "success": False,
            "status": "failed_runtime",
            "failed_reason": f"{type(exc).__name__}: {exc}; traceback={trace_path}",
            "official_sehgnn_unmodified": False,
            "eligible_for_main_decision": False,
        }
        if result_kind == "adapter":
            failed["feature_compression_method"] = feature_compression_method
            return ({field: failed.get(field, "") for field in FEATURE_CACHE_FIELDS_21_4}, [], [], [], [])
        failed["paper_feature_transform"] = transform_name
        failed["term_channel_spec"] = relation_spec
        return ({field: failed.get(field, "") for field in FEATURE_CHANNEL_FIELDS}, [], [], [], [])


def _manifest_row(
    *,
    dataset: str,
    method: str,
    run_group: str,
    graph_seed: int,
    training_seed: int,
    relation_spec: str,
    transform_name: str,
    feature_compression_method: str,
    command,
    export_dir: Path,
    cache_dir: Path,
    run_dir: Path,
    status: str,
) -> dict[str, Any]:
    command_json = json.dumps(
        {"command": list(command.command), "cwd": str(command.cwd), "dataset": command.dataset, "seed": int(command.seed)},
        sort_keys=True,
    )
    return {
        "run_id": _run_id(dataset, method, int(graph_seed), int(training_seed)),
        "dataset": dataset,
        "method": method,
        "canonical_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
        "run_group": run_group,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "relation_channel_spec": relation_spec,
        "relation_direction_spec": "",
        "paper_feature_transform": transform_name,
        "feature_compression_method": feature_compression_method,
        "sehgnn_command_json": command_json,
        "export_dir": str(export_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(run_dir),
        "status": status,
    }


def _storage_audit_row(storage: Mapping[str, Any], row: Mapping[str, Any], graph_seed: int, training_seed: int, adapter_family: str) -> dict[str, Any]:
    semantic = _float(storage.get("semantic_structural_storage_ratio"))
    raw = _float(storage.get("hgb_raw_file_byte_ratio"))
    cache = _float(storage.get("preprocessed_cache_byte_ratio"))
    effective = _float(row.get("effective_total_byte_ratio")) or raw
    out = {
        "dataset": storage.get("dataset", ""),
        "method": row.get("method", ""),
        "canonical_method": "H6-relgrid-APPA100-PVVP100-PTTP00",
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "semantic_structural_storage_ratio": storage.get("semantic_structural_storage_ratio", ""),
        "support_node_ratio": storage.get("support_node_ratio", ""),
        "support_edge_ratio": storage.get("support_edge_ratio", ""),
        "total_node_ratio": storage.get("total_node_ratio", ""),
        "total_edge_ratio": storage.get("total_edge_ratio", ""),
        "hgb_raw_file_byte_ratio": storage.get("hgb_raw_file_byte_ratio", ""),
        "effective_total_byte_ratio": effective if effective is not None else "",
        "preprocessed_cache_byte_ratio": storage.get("preprocessed_cache_byte_ratio", ""),
        "node_dat_bytes": storage.get("node_dat_bytes", ""),
        "link_dat_bytes": storage.get("link_dat_bytes", ""),
        "label_dat_bytes": storage.get("label_dat_bytes", ""),
        "label_test_dat_bytes": storage.get("label_test_dat_bytes", ""),
        "info_dat_bytes": storage.get("info_dat_bytes", ""),
        "metadata_bytes": storage.get("metadata_sidecar_bytes", ""),
        "feature_sidecar_bytes": row.get("sidecar_feature_bytes", 0),
        "feature_sidecar_metadata_bytes": row.get("sidecar_metadata_bytes", 0),
        "preprocessed_cache_bytes": storage.get("export_preprocessed_cache_bytes", ""),
        "export_total_bytes": storage.get("export_total_bytes", ""),
        "native_full_total_bytes": storage.get("native_full_total_bytes", ""),
        "structural_storage50_pass": _le(semantic, 0.50),
        "structural_storage40_pass": _le(semantic, 0.40),
        "structural_storage30_pass": _le(semantic, 0.30),
        "structural_storage20_pass": _le(semantic, 0.20),
        "raw_hgb_byte50_pass": _le(raw, 0.50),
        "raw_hgb_byte30_pass": _le(raw, 0.30),
        "effective_byte50_pass": _le(effective, 0.50),
        "effective_byte30_pass": _le(effective, 0.30),
        "cache_byte50_pass": _le(cache, 0.50),
        "cache_byte30_pass": _le(cache, 0.30),
        "official_sehgnn_unmodified": False,
        "adapter_family": adapter_family,
        "eligible_for_main_decision": False,
    }
    return {field: out.get(field, "") for field in STORAGE_AUDIT_FIELDS}


def _le(value: float | None, threshold: float) -> bool | str:
    return "" if value is None else bool(float(value) <= float(threshold))


def _transform_paper_features(graph, transform_name: str, *, seed: int):
    features = {} if graph.features is None else {int(k): v.copy() for k, v in graph.features.items()}
    audit: dict[str, Any] = {"transform_name": transform_name, "fit_uses_labels": False, "fit_uses_test_labels": False}
    if 1 in features:
        transformed, audit = transform_feature_matrix(features[1], transform_name, seed=int(seed))
        features[1] = transformed
    from hesf_coarsen.io.schema import HeteroGraph

    out = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=graph.relations,
        relation_specs=graph.relation_specs,
        features=features,
        labels=None if graph.labels is None else np.asarray(graph.labels).copy(),
    )
    object.__setattr__(out, "_gate21_4_transform_audit", audit)
    return out


def _recovery(value: Any, baseline: float) -> float | str:
    parsed = _float(value)
    return "" if parsed is None else float(parsed / baseline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-graph", default="APV-skeleton")
    parser.add_argument("--graph-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--feature-transforms", nargs="+", default=["raw", "zero-paper", "pca-paper-128", "pca-paper-64", "int8-paper", "fp16-paper"])
    parser.add_argument("--term-channel-specs", nargs="+", default=["PTTP00", "PTTP30", "PTTP100"])
    parser.add_argument("--feature-compression-methods", nargs="+", default=["raw_features_adapter_control", "fp16_node_features", "int8_per_feature", "pca_svd_dim256", "pca_svd_dim128", "pca_svd_dim64", "random_projection_dim128"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unique-cache-namespace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-official-training", action="store_true")
    parser.add_argument("--sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--hgb-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--min-edges-per-relation", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_gate21_4_cache_feature(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
