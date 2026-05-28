from __future__ import annotations

import hashlib
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir
from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command, run_native_command


CORE_REAL_TASK_METHODS = ("Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "GraphSparsify-TP")


def ensure_external_tp_task_metrics(
    *,
    dataset: str,
    methods: Sequence[str],
    source_data_root: str | Path,
    sehgnn_repo: str | Path,
    output_dir: str | Path,
    support_node_ratio: float,
    graph_seed: int,
    training_seed: int,
    device: str,
    task_epochs: int = 200,
    force_reprocess: bool = False,
) -> Path:
    """Build minimal real TP artifacts and run official SeHGNN metrics for core baselines."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "gate21_7_external_tp_task_metrics.csv"
    if metrics_path.exists() and not force_reprocess:
        return metrics_path

    source_dataset_dir = Path(source_data_root) / str(dataset).upper()
    rows: list[dict[str, Any]] = []
    for method in CORE_REAL_TASK_METHODS:
        if method not in set(methods):
            continue
        artifact_root = out / "task_artifacts" / _safe(method) / f"support_{_ratio_token(support_node_ratio)}" / f"graph_seed_{int(graph_seed)}"
        try:
            export = build_tp_hgb_artifact(
                dataset=str(dataset).upper(),
                source_dataset_dir=source_dataset_dir,
                output_data_root=artifact_root,
                method=method,
                support_node_ratio=float(support_node_ratio),
                graph_seed=int(graph_seed),
            )
            command = build_official_hgb_command(
                dataset=str(dataset).upper(),
                seed=int(training_seed),
                repo_dir=Path(sehgnn_repo),
                data_root=artifact_root,
                device=str(device),
                python_executable=sys.executable,
            )
            command = command.__class__(
                command=_with_epoch(command.command, int(task_epochs)),
                cwd=command.cwd,
                dataset=command.dataset,
                seed=command.seed,
            )
            raw_dir = out / "task_raw"
            task = run_native_command(
                command,
                stdout_path=raw_dir / "stdout" / f"{_safe(method)}_g{int(graph_seed)}_t{int(training_seed)}.log",
                stderr_path=raw_dir / "stderr" / f"{_safe(method)}_g{int(graph_seed)}_t{int(training_seed)}.stderr",
            )
            success = str(task.get("status", "")).lower() == "success"
            rows.append(
                {
                    "dataset": str(dataset).upper(),
                    "method": method,
                    "baseline_name": method,
                    "graph_seed": int(graph_seed),
                    "training_seed": int(training_seed),
                    "budget_type": "support_node_ratio",
                    "budget_value": float(support_node_ratio),
                    "official_hgb_exported": True,
                    "official_sehgnn_unmodified": True,
                    "training_executed": bool(success),
                    "eligible_for_tp_main_comparison": bool(success),
                    "success": bool(success),
                    "success_count": 1 if success else 0,
                    "test_micro_f1": task.get("test_micro_f1", ""),
                    "test_macro_f1": task.get("test_macro_f1", ""),
                    "validation_micro_f1": task.get("validation_micro_f1", ""),
                    "validation_macro_f1": task.get("validation_macro_f1", ""),
                    "best_epoch": task.get("best_epoch", ""),
                    "compress_wall_time_seconds": export.get("compress_wall_time_seconds", ""),
                    "export_wall_time_seconds": export.get("export_wall_time_seconds", ""),
                    "preprocess_wall_time_seconds": "",
                    "train_wall_time_seconds": task.get("train_time_sec", ""),
                    "peak_cpu_rss_mb": "",
                    "peak_gpu_memory_mb": task.get("peak_memory_mb", ""),
                    "actual_support_node_ratio": export.get("actual_support_node_ratio", ""),
                    "actual_support_edge_ratio": export.get("actual_support_edge_ratio", ""),
                    "actual_structural_storage_ratio": export.get("actual_structural_storage_ratio", ""),
                    "raw_hgb_text_byte_ratio": export.get("raw_hgb_text_byte_ratio", ""),
                    "artifact_manifest_path": export.get("artifact_manifest_path", ""),
                    "export_dir": export.get("export_dir", ""),
                    "failure_type": "" if success else task.get("status", "failed_runtime"),
                    "failure_message": "" if success else task.get("error_message", ""),
                    "stdout_path": task.get("stdout_path", ""),
                    "stderr_path": task.get("stderr_path", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": str(dataset).upper(),
                    "method": method,
                    "baseline_name": method,
                    "graph_seed": int(graph_seed),
                    "training_seed": int(training_seed),
                    "budget_type": "support_node_ratio",
                    "budget_value": float(support_node_ratio),
                    "official_hgb_exported": False,
                    "official_sehgnn_unmodified": True,
                    "training_executed": False,
                    "eligible_for_tp_main_comparison": False,
                    "success": False,
                    "success_count": 0,
                    "test_micro_f1": "",
                    "test_macro_f1": "",
                    "failure_type": "external_tp_task_exception",
                    "failure_message": str(exc),
                }
            )
    write_csv(metrics_path, rows)
    return metrics_path


def build_tp_hgb_artifact(
    *,
    dataset: str,
    source_dataset_dir: str | Path,
    output_data_root: str | Path,
    method: str,
    support_node_ratio: float,
    graph_seed: int,
) -> dict[str, Any]:
    dataset_name = str(dataset).upper()
    if dataset_name != "DBLP":
        raise ValueError("Gate21.7 real TP HGB artifact runner currently supports DBLP")
    start = time.perf_counter()
    source = Path(source_dataset_dir)
    output_data_root = Path(output_data_root)
    export_dir = output_data_root / dataset_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    nodes, by_type = _read_nodes(source / "node.dat")
    edges = _read_edges(source / "link.dat")
    relation_order = _relation_order(edges)
    degree = _degree_scores(edges, len(nodes))
    selected_old_ids = _select_nodes(
        method=method,
        by_type=by_type,
        degree=degree,
        support_node_ratio=float(support_node_ratio),
        seed=int(graph_seed),
    )
    if method == "GraphSparsify-TP":
        selected_old_ids = sorted(nodes)
    selected = set(int(node) for node in selected_old_ids)
    candidate_edges = [edge for edge in edges if edge[0] in selected and edge[1] in selected]
    if method == "GraphSparsify-TP":
        candidate_edges = _sample_edges_by_relation(candidate_edges, edge_ratio=float(support_node_ratio), seed=int(graph_seed), relation_order=relation_order)
    candidate_edges = _ensure_target_incidence(edges, candidate_edges, selected, target_nodes=by_type.get(0, []))
    active_old_ids = _active_nodes_for_official_loader(candidate_edges, target_nodes=by_type.get(0, []))
    old_to_new = _write_nodes(export_dir / "node.dat", nodes, active_old_ids)
    export_wall_start = time.perf_counter()
    kept_edges = _write_link_edges(export_dir / "link.dat", candidate_edges, old_to_new=old_to_new, relation_order=relation_order)
    for name in ("label.dat", "label.dat.test", "info.dat", "meta.dat"):
        if (source / name).exists():
            if name.startswith("label"):
                _write_labels(source / name, export_dir / name, old_to_new)
            else:
                shutil.copy2(source / name, export_dir / name)
    export_wall = time.perf_counter() - export_wall_start

    audit = audit_native_hgb_data_dir(dataset_name, output_data_root, None)
    if not bool(audit.get("node_dat_exists")) or not bool(audit.get("link_dat_exists")):
        raise RuntimeError("exported TP artifact is missing node.dat or link.dat")
    relation_counts = _relation_counts(kept_edges)
    if len(relation_counts) < 6:
        raise RuntimeError(f"exported TP artifact lost required DBLP relation types: {relation_counts}")
    native_bytes = _dir_bytes(source)
    export_bytes = _dir_bytes(export_dir)
    target_count = len(by_type.get(0, []))
    native_support = max(len(nodes) - target_count, 1)
    selected_support = max(len(active_old_ids) - target_count, 0)
    native_edges = max(len(edges), 1)
    structural_native = max(len(nodes) + len(edges), 1)
    manifest = {
        "dataset": dataset_name,
        "method": method,
        "graph_seed": int(graph_seed),
        "support_node_ratio": float(support_node_ratio),
        "export_dir": str(export_dir),
        "artifact_manifest_path": str(export_dir / "gate21_7_external_tp_artifact_manifest.json"),
        "node_count": len(active_old_ids),
        "edge_count": len(kept_edges),
        "actual_support_node_ratio": float(selected_support / native_support),
        "actual_support_edge_ratio": float(len(kept_edges) / native_edges),
        "actual_structural_storage_ratio": float((len(selected_old_ids) + len(kept_edges)) / structural_native),
        "raw_hgb_text_byte_ratio": float(export_bytes / max(native_bytes, 1)),
        "relation_counts": relation_counts,
        "export_file_list_hash": _file_list_hash(export_dir),
        "compress_wall_time_seconds": float(time.perf_counter() - start),
        "export_wall_time_seconds": float(export_wall),
        "keeps_all_target_nodes": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
    }
    (export_dir / "gate21_7_external_tp_artifact_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_nodes(path: Path) -> tuple[dict[int, tuple[str, int, str]], dict[int, list[int]]]:
    nodes: dict[int, tuple[str, int, str]] = {}
    by_type: dict[int, list[int]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            old_id = int(parts[0])
            name = parts[1]
            type_id = int(parts[2])
            feat = parts[3] if len(parts) >= 4 else ""
            nodes[old_id] = (name, type_id, feat)
            by_type.setdefault(type_id, []).append(old_id)
    return nodes, by_type


def _read_edges(path: Path) -> list[tuple[int, int, int, float]]:
    edges: list[tuple[int, int, int, float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            edges.append((int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])))
    return edges


def _degree_scores(edges: Sequence[tuple[int, int, int, float]], node_count: int) -> np.ndarray:
    degree = np.zeros(int(node_count), dtype=np.float64)
    for src, dst, _rel, weight in edges:
        value = abs(float(weight)) if math.isfinite(float(weight)) else 1.0
        degree[int(src)] += value
        degree[int(dst)] += value
    return degree


def _select_nodes(
    *,
    method: str,
    by_type: Mapping[int, Sequence[int]],
    degree: np.ndarray,
    support_node_ratio: float,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for type_id in sorted(by_type):
        nodes = np.asarray(by_type[type_id], dtype=np.int64)
        if int(type_id) == 0:
            selected.extend(int(v) for v in nodes.tolist())
            continue
        keep = max(1, min(int(nodes.size), int(round(float(support_node_ratio) * int(nodes.size)))))
        if method == "Random-HG-TP":
            chosen = rng.choice(nodes, size=keep, replace=False)
        elif method == "Herding-HG-TP":
            order = np.argsort(-degree[nodes], kind="mergesort")
            chosen = nodes[order[:keep]]
        elif method == "KCenter-HG-TP":
            order = np.argsort(degree[nodes], kind="mergesort")
            positions = np.linspace(0, max(int(nodes.size) - 1, 0), keep).round().astype(np.int64)
            chosen = nodes[order[positions]]
        elif method == "GraphSparsify-TP":
            chosen = nodes
        else:
            raise ValueError(f"real TP HGB runner does not support method {method!r}")
        selected.extend(int(v) for v in np.sort(chosen).tolist())
    return selected


def _write_nodes(path: Path, nodes: Mapping[int, tuple[str, int, str]], selected_old_ids: Iterable[int]) -> dict[int, int]:
    selected = list(selected_old_ids)
    old_to_new = {int(old): int(i) for i, old in enumerate(selected)}
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        for new_id, old_id in enumerate(selected):
            name, type_id, feat = nodes[int(old_id)]
            line = f"{int(new_id)}\t{name}\t{int(type_id)}"
            handle.write(line + (f"\t{feat}\n" if feat != "" else "\n"))
    return old_to_new


def _sample_edges_by_relation(
    candidates: Sequence[tuple[int, int, int, float]],
    *,
    edge_ratio: float,
    seed: int,
    relation_order: Mapping[int, int],
) -> list[tuple[int, int, int, float]]:
    rng = np.random.default_rng(int(seed))
    by_relation: dict[int, list[tuple[int, int, int, float]]] = {}
    for edge in candidates:
        by_relation.setdefault(int(edge[2]), []).append(edge)
    kept: list[tuple[int, int, int, float]] = []
    for relation_id, group in sorted(by_relation.items(), key=lambda item: relation_order.get(int(item[0]), int(item[0]))):
        count = max(1, min(len(group), int(round(float(edge_ratio) * len(group)))))
        indices = rng.choice(np.arange(len(group)), size=count, replace=False)
        kept.extend(group[int(index)] for index in sorted(indices.tolist()))
    return kept


def _ensure_target_incidence(
    all_edges: Sequence[tuple[int, int, int, float]],
    candidate_edges: Sequence[tuple[int, int, int, float]],
    selected: set[int],
    *,
    target_nodes: Sequence[int],
) -> list[tuple[int, int, int, float]]:
    out = list(candidate_edges)
    incident = {int(src) for src, dst, _rel, _weight in out if int(src) in target_nodes}
    incident.update(int(dst) for src, dst, _rel, _weight in out if int(dst) in target_nodes)
    target_set = set(int(node) for node in target_nodes)
    seen_edges = {(int(src), int(dst), int(rel)) for src, dst, rel, _weight in out}
    for target in sorted(target_set - incident):
        for edge in all_edges:
            src, dst, rel, _weight = edge
            if int(src) != int(target) and int(dst) != int(target):
                continue
            selected.add(int(src))
            selected.add(int(dst))
            key = (int(src), int(dst), int(rel))
            if key not in seen_edges:
                out.append(edge)
                seen_edges.add(key)
            break
    return out


def _active_nodes_for_official_loader(
    edges: Sequence[tuple[int, int, int, float]],
    *,
    target_nodes: Sequence[int],
) -> list[int]:
    active = set(int(node) for node in target_nodes)
    for src, dst, _rel, _weight in edges:
        active.add(int(src))
        active.add(int(dst))
    return sorted(active)


def _write_link_edges(
    path: Path,
    edges: Sequence[tuple[int, int, int, float]],
    *,
    old_to_new: Mapping[int, int],
    relation_order: Mapping[int, int],
) -> list[tuple[int, int, int, float]]:
    candidates = sorted(
        [edge for edge in edges if edge[0] in old_to_new and edge[1] in old_to_new],
        key=lambda edge: (relation_order.get(int(edge[2]), int(edge[2])), int(edge[0]), int(edge[1])),
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        for src, dst, rel, weight in candidates:
            handle.write(f"{old_to_new[int(src)]}\t{old_to_new[int(dst)]}\t{int(rel)}\t{float(weight)}\n")
    return candidates


def _write_labels(src: Path, dst: Path, old_to_new: Mapping[int, int]) -> None:
    with Path(src).open(encoding="utf-8") as inp, Path(dst).open("w", encoding="utf-8", newline="") as out:
        for line in inp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            old_id = int(parts[0])
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            out.write(f"{new_id}\t{new_id}\t{parts[2]}\t{parts[3]}\n")


def _relation_counts(edges: Sequence[tuple[int, int, int, float]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for _src, _dst, relation_id, _weight in edges:
        key = str(int(relation_id))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _relation_order(edges: Sequence[tuple[int, int, int, float]]) -> dict[int, int]:
    order: dict[int, int] = {}
    for _src, _dst, relation_id, _weight in edges:
        rid = int(relation_id)
        if rid not in order:
            order[rid] = len(order)
    return order


def _with_epoch(command: Sequence[str], task_epochs: int) -> list[str]:
    out = list(command)
    if "--epoch" in out:
        out[out.index("--epoch") + 1] = str(int(task_epochs))
    return out


def _dir_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file()))


def _file_list_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        digest.update(f"{rel}\t{item.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _ratio_token(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")
