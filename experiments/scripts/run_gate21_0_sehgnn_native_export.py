from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.summarize_gate21_0_sehgnn_native_export import summarize_gate21_0
from experiments.scripts.gate13_task_first_common import load_hgb_graph
from experiments.scripts.run_gate21_open_sota_bridge import _build_method_graph
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb, require_native_reproduction_pass
from hesf_coarsen.eval.official.sehgnn_native_runner import NATIVE_METRIC_FIELDS, build_official_hgb_command, run_native_command, run_native_stage
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


def _run_native(args: argparse.Namespace) -> dict[str, object]:
    return run_native_stage(
        repo_dir=Path(args.sehgnn_repo),
        data_root=Path(args.sehgnn_data_root),
        datasets=[str(dataset).upper() for dataset in args.datasets],
        seeds=[int(seed) for seed in args.seeds],
        device=str(args.device),
        out_dir=Path(args.out_dir),
        python_executable=sys.executable,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _native_labels(data_root: Path, dataset: str, node_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = Path(data_root) / dataset
    entries: list[tuple[int, list[int], str]] = []
    for filename, split in (("label.dat", "trainval"), ("label.dat.test", "test")):
        with (dataset_dir / filename).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    entries.append((int(parts[0]), [int(value) for value in parts[3].split(",") if value != ""], split))
    num_classes = max((max(labels) for _node, labels, _split in entries if labels), default=-1) + 1
    is_multi = any(len(labels) > 1 for _node, labels, _split in entries)
    if is_multi:
        labels_arr = np.zeros((int(node_count), int(num_classes)), dtype=np.int64)
        for node, labels, _split in entries:
            for label in labels:
                labels_arr[int(node), int(label)] = 1
    else:
        labels_arr = np.full(int(node_count), -1, dtype=np.int64)
        for node, labels, _split in entries:
            if labels:
                labels_arr[int(node)] = int(labels[0])
    trainval = np.asarray([node for node, _labels, split in entries if split == "trainval"], dtype=np.int64)
    test = np.asarray([node for node, _labels, split in entries if split == "test"], dtype=np.int64)
    return labels_arr, trainval, test


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_key_count(stdout_path: str) -> tuple[str, str]:
    path = Path(stdout_path)
    if not path.exists():
        return "", ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    import re

    feat = re.search(r"feature keys \(num=(\d+)\)", text)
    label = re.search(r"Involved label keys dict_keys\(\[(.*?)\]\)", text, re.DOTALL)
    if label is None:
        label = re.search(r"Involved label keys (.*?)\n", text)
    label_count = ""
    if label is not None:
        label_count = str(label.group(1).count("'") // 2) if "'" in label.group(1) else ""
    return feat.group(1) if feat else "", label_count


def _edge_count(graph: HeteroGraph) -> int:
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _dir_size(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return int(sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file()))


def _storage_ratio_fields(
    original: HeteroGraph,
    compressed: HeteroGraph,
    *,
    target_type: int,
    export_file_bytes: int,
    native_full_file_bytes: int,
    method: str,
) -> dict[str, Any]:
    original_support = int(np.sum(original.node_type != int(target_type)))
    compressed_support = int(np.sum(compressed.node_type != int(target_type)))
    original_edges = _edge_count(original)
    compressed_edges = _edge_count(compressed)
    original_storage = max(int(original.num_nodes) + int(original_edges), 1)
    compressed_storage = int(compressed.num_nodes) + int(compressed_edges)
    return {
        "method": str(method),
        "support_node_ratio": float(compressed_support / max(original_support, 1)),
        "support_edge_ratio": float(compressed_edges / max(original_edges, 1)),
        "total_node_ratio": float(compressed.num_nodes / max(original.num_nodes, 1)),
        "total_edge_ratio": float(compressed_edges / max(original_edges, 1)),
        "total_storage_ratio_vs_full_graph": float(compressed_storage / original_storage),
        "feature_storage_ratio": "",
        "label_storage_ratio": "",
        "edge_storage_ratio": float(compressed_edges / max(original_edges, 1)),
        "export_file_bytes": int(export_file_bytes),
        "native_full_file_bytes": int(native_full_file_bytes),
    }


def _compressed_method_label(method: str) -> str:
    if method == "target-only":
        return "target-only"
    if method == "typedhash":
        return "TypedHash-node30"
    return f"{method}-node30"


def _target_local_ids(graph: HeteroGraph, target_type: int, global_ids: np.ndarray) -> np.ndarray:
    target_nodes = nodes_of_type(graph, int(target_type))
    lookup = {int(node): int(pos) for pos, node in enumerate(target_nodes.tolist())}
    return np.asarray([lookup[int(node)] for node in np.asarray(global_ids, dtype=np.int64).reshape(-1).tolist()], dtype=np.int64)


def _run_export_full(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    current = summarize_gate21_0(out_dir)
    require_native_reproduction_pass(current)
    native_rows = _read_csv(out_dir / "native" / "native_metrics.csv")
    native_by_key = {(row["dataset"], str(row["seed"])): row for row in native_rows}
    export_rows: list[dict[str, Any]] = []
    export_audits: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    stdout_dir = out_dir / "fidelity" / "export_full_raw_stdout"
    stderr_dir = out_dir / "fidelity" / "export_full_raw_stderr"
    export_root = out_dir / "export_full_hgb"
    for dataset in [str(value).upper() for value in args.datasets]:
        graph = load_hgb_graph(Path(args.data_root), dataset)
        labels, trainval, test = _native_labels(Path(args.sehgnn_data_root), dataset, graph.num_nodes)
        manifest = export_graph_to_sehgnn_hgb(
            graph=graph,
            dataset_name=dataset,
            target_type={"DBLP": "A", "ACM": "P", "IMDB": "M"}[dataset],
            output_dir=export_root,
            split_mode="official_trainval",
            train_idx=trainval,
            val_idx=np.array([], dtype=np.int64),
            test_idx=test,
            labels=labels,
            method_name="full",
            seed=0,
        )
        audit = audit_native_hgb_data_dir(dataset, Path(manifest["export_dir"]).parent, Path(args.sehgnn_repo))
        export_audits.append({**manifest, **audit})
        if not (manifest["mapping_bijective"] and manifest["relation_order_matches_official"] and audit["can_load_with_official_data_loader"]):
            continue
        for seed in [int(seed) for seed in args.seeds]:
            command = build_official_hgb_command(
                dataset=dataset,
                seed=seed,
                repo_dir=Path(args.sehgnn_repo),
                data_root=Path(manifest["export_dir"]).parent,
                device=str(args.device),
                python_executable=sys.executable,
            )
            row = run_native_command(
                command,
                stdout_path=stdout_dir / f"{dataset}_{seed}.log",
                stderr_path=stderr_dir / f"{dataset}_{seed}.stderr",
            )
            export_rows.append(row)
            native = native_by_key.get((dataset, str(seed)), {})
            native_micro = _float(native.get("test_micro_f1"))
            native_macro = _float(native.get("test_macro_f1"))
            export_micro = _float(row.get("test_micro_f1"))
            export_macro = _float(row.get("test_macro_f1"))
            native_acc = _float(native.get("test_accuracy_if_single_label"))
            export_acc = _float(row.get("test_accuracy_if_single_label"))
            metric_gap = None if native_micro is None or export_micro is None else abs(native_micro - export_micro)
            macro_gap = None if native_macro is None or export_macro is None else abs(native_macro - export_macro)
            acc_gap = None if native_acc is None or export_acc is None else abs(native_acc - export_acc)
            fidelity_pass = bool(row.get("status") == "success" and metric_gap is not None and metric_gap <= 0.02 and (macro_gap is None or macro_gap <= 0.02))
            fidelity_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "native_official_micro_f1": "" if native_micro is None else native_micro,
                    "native_official_macro_f1": "" if native_macro is None else native_macro,
                    "native_official_accuracy_if_single_label": "" if native_acc is None else native_acc,
                    "export_full_micro_f1": "" if export_micro is None else export_micro,
                    "export_full_macro_f1": "" if export_macro is None else export_macro,
                    "export_full_accuracy_if_single_label": "" if export_acc is None else export_acc,
                    "micro_gap_native_minus_export": "" if native_micro is None or export_micro is None else native_micro - export_micro,
                    "macro_gap_native_minus_export": "" if native_macro is None or export_macro is None else native_macro - export_macro,
                    "accuracy_gap_native_minus_export": "" if native_acc is None or export_acc is None else native_acc - export_acc,
                    "fidelity_pass": fidelity_pass,
                }
            )
            native_feat_count, native_label_count = _feature_key_count(str(native.get("stdout_path", "")))
            export_feat_count, export_label_count = _feature_key_count(str(row.get("stdout_path", "")))
            feature_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "native_feature_key_count": native_feat_count,
                    "export_feature_key_count": export_feat_count,
                    "feature_key_overlap_count": min(int(native_feat_count or 0), int(export_feat_count or 0)) if native_feat_count and export_feat_count else "",
                    "feature_key_overlap_ratio": 1.0 if native_feat_count and native_feat_count == export_feat_count else "",
                    "native_label_feature_key_count": native_label_count,
                    "export_label_feature_key_count": export_label_count,
                    "num_hops": {"DBLP": 2, "ACM": 4, "IMDB": 4}[dataset],
                    "num_label_hops": 4,
                    "uses_label_feats": True,
                    "uses_official_preprocess": True,
                    "feature_shape_summary_native": "",
                    "feature_shape_summary_export": "",
                }
            )
    write_csv(out_dir / "export" / "gate21_0_hgb_export_audit.csv", export_audits)
    write_csv(out_dir / "fidelity" / "gate21_0_export_full_metrics.csv", export_rows, fieldnames=NATIVE_METRIC_FIELDS)
    write_csv(out_dir / "fidelity" / "gate21_0_sehgnn_full_fidelity.csv", fidelity_rows)
    write_csv(out_dir / "fidelity" / "gate21_0_sehgnn_feature_audit.csv", feature_rows)
    return {"export_rows": len(export_rows), "fidelity_rows": len(fidelity_rows)}


def _run_compressed(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    current = summarize_gate21_0(out_dir)
    if not bool(current.get("compressed_eval_allowed")):
        raise RuntimeError(f"compressed stage is blocked until export-full fidelity passes; current decision={current['decision']}")
    native_rows = _read_csv(out_dir / "native" / "native_metrics.csv")
    export_full_rows = _read_csv(out_dir / "fidelity" / "gate21_0_export_full_metrics.csv")
    native_by_key = {(row["dataset"], str(row["seed"])): row for row in native_rows}
    export_by_key = {(row["dataset"], str(row["seed"])): row for row in export_full_rows}
    metrics_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    stdout_dir = out_dir / "compressed" / "compressed_raw_stdout"
    stderr_dir = out_dir / "compressed" / "compressed_raw_stderr"
    export_root = out_dir / "compressed_hgb"
    for dataset in [str(value).upper() for value in args.datasets]:
        original = load_hgb_graph(Path(args.data_root), dataset)
        target_type = infer_target_node_type(original)
        labels_native, trainval_native, test_native = _native_labels(Path(args.sehgnn_data_root), dataset, original.num_nodes)
        for method in [str(value) for value in args.compressed_methods]:
            graph, assignment, _diag = _build_method_graph(
                original,
                method=method,
                ratio=float(args.support_ratio),
                seed=int(args.seeds[0]),
                candidate_k=int(args.candidate_k),
                target_type=int(target_type),
            )
            target_nodes = nodes_of_type(original, int(target_type))
            compressed_target_globals = np.asarray(assignment[target_nodes], dtype=np.int64)
            compressed_target_local = _target_local_ids(graph, int(target_type), compressed_target_globals)
            original_target_local = {int(node): int(pos) for pos, node in enumerate(target_nodes.tolist())}
            compressed_labels = np.zeros((graph.num_nodes, labels_native.shape[1]), dtype=np.int64) if labels_native.ndim == 2 else np.full(graph.num_nodes, -1, dtype=np.int64)
            for original_global, compressed_global in zip(target_nodes.tolist(), compressed_target_globals.tolist()):
                if labels_native.ndim == 2:
                    compressed_labels[int(compressed_global)] = labels_native[int(original_global)]
                else:
                    compressed_labels[int(compressed_global)] = int(labels_native[int(original_global)])
            trainval_compressed = compressed_target_local[[original_target_local[int(node)] for node in trainval_native.tolist()]]
            test_compressed = compressed_target_local[[original_target_local[int(node)] for node in test_native.tolist()]]
            method_label = _compressed_method_label(method)
            manifest = export_graph_to_sehgnn_hgb(
                graph=graph,
                dataset_name=dataset,
                target_type={"DBLP": "A", "ACM": "P", "IMDB": "M"}[dataset],
                output_dir=export_root,
                split_mode="official_trainval",
                train_idx=trainval_compressed,
                val_idx=np.array([], dtype=np.int64),
                test_idx=test_compressed,
                labels=compressed_labels,
                method_name=method_label,
                seed=int(args.seeds[0]),
            )
            native_full_dir = Path(args.sehgnn_data_root) / dataset
            storage = _storage_ratio_fields(
                original,
                graph,
                target_type=int(target_type),
                export_file_bytes=_dir_size(Path(manifest["export_dir"])),
                native_full_file_bytes=_dir_size(native_full_dir),
                method=method_label,
            )
            storage_rows.append({"dataset": dataset, "seed": int(args.seeds[0]), "support_node_ratio": float(args.support_ratio), **storage})
            for seed in [int(seed) for seed in args.seeds]:
                command = build_official_hgb_command(
                    dataset=dataset,
                    seed=seed,
                    repo_dir=Path(args.sehgnn_repo),
                    data_root=Path(manifest["export_dir"]).parent,
                    device=str(args.device),
                    python_executable=sys.executable,
                )
                row = run_native_command(
                    command,
                    stdout_path=stdout_dir / f"{dataset}_{method_label}_{seed}.log",
                    stderr_path=stderr_dir / f"{dataset}_{method_label}_{seed}.stderr",
                )
                native = native_by_key.get((dataset, str(seed)), {})
                export_full = export_by_key.get((dataset, str(seed)), {})
                native_micro = _float(native.get("test_micro_f1"))
                native_macro = _float(native.get("test_macro_f1"))
                export_micro = _float(export_full.get("test_micro_f1"))
                export_macro = _float(export_full.get("test_macro_f1"))
                test_micro = _float(row.get("test_micro_f1"))
                test_macro = _float(row.get("test_macro_f1"))
                metrics_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method_label,
                        "support_node_ratio": float(args.support_ratio),
                        "support_edge_ratio": storage["support_edge_ratio"],
                        "total_storage_ratio_vs_full_graph": storage["total_storage_ratio_vs_full_graph"],
                        "validation_micro_f1": row.get("validation_micro_f1", ""),
                        "validation_macro_f1": row.get("validation_macro_f1", ""),
                        "test_micro_f1": row.get("test_micro_f1", ""),
                        "test_macro_f1": row.get("test_macro_f1", ""),
                        "test_accuracy_if_single_label": row.get("test_accuracy_if_single_label", ""),
                        "recovery_vs_native_full_micro": "" if native_micro in {None, 0.0} or test_micro is None else test_micro / native_micro,
                        "recovery_vs_native_full_macro": "" if native_macro in {None, 0.0} or test_macro is None else test_macro / native_macro,
                        "recovery_vs_export_full_micro": "" if export_micro in {None, 0.0} or test_micro is None else test_micro / export_micro,
                        "recovery_vs_export_full_macro": "" if export_macro in {None, 0.0} or test_macro is None else test_macro / export_macro,
                        "status": row.get("status", ""),
                        "error_message": row.get("error_message", ""),
                        "stdout_path": row.get("stdout_path", ""),
                        "stderr_path": row.get("stderr_path", ""),
                    }
                )
    write_csv(out_dir / "compressed" / "gate21_0_compressed_storage_audit.csv", storage_rows)
    write_csv(out_dir / "compressed" / "gate21_0_compressed_metrics.csv", metrics_rows)
    return {"compressed_rows": len(metrics_rows), "storage_rows": len(storage_rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["native", "export-full", "compressed"], required=True)
    parser.add_argument("--sehgnn-repo", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--sehgnn-data-root", type=Path, default=Path("external/SeHGNN/data"))
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gate21_0_sehgnn_native_export"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--compressed-methods", nargs="+", default=["H6", "flatten", "typedhash", "target-only"])
    parser.add_argument("--support-ratio", type=float, default=0.30)
    parser.add_argument("--candidate-k", type=int, default=16)
    args = parser.parse_args(argv)

    if args.stage == "native":
        run_result = _run_native(args)
    elif args.stage == "export-full":
        run_result = _run_export_full(args)
    else:
        run_result = _run_compressed(args)
    summary = summarize_gate21_0(Path(args.out_dir))
    print(json.dumps({"run": run_result, "summary": summary}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
