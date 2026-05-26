from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from hesf_coarsen.eval.official.icde_protocol import (
    PROTOCOL_SCHEMA_PRESERVING_TP,
    build_protocol_row,
)
from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS


TP_BASELINE_METHODS = {
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "Coarsening-HG-TP",
    "GraphSparsify-TP",
    "FreeHGC-TP",
}


def _base_row(
    *,
    dataset: str,
    method: str,
    budget: float,
    graph_seed: int,
    training_seed: int,
    success: bool,
    failure_type: str = "",
    failure_message: str = "",
    uses_synthetic_nodes: bool = False,
    support_edge_ratio: float | None = None,
    structural_storage_ratio: float | None = None,
    raw_hgb_text_byte_ratio: float | None = None,
    construction_status: str = "",
    selection_signal: str = "",
    export_hash: str = "",
) -> dict[str, Any]:
    row = build_protocol_row(
        baseline_name=method,
        protocol=PROTOCOL_SCHEMA_PRESERVING_TP,
        method_family="external_tp_baseline",
        schema_compatible=True,
        official_sehgnn_unmodified=False,
        uses_synthetic_nodes=uses_synthetic_nodes,
        keeps_all_target_nodes=True,
        support_node_ratio=float(budget),
        support_edge_ratio=support_edge_ratio,
        structural_storage_ratio=structural_storage_ratio,
        raw_hgb_text_byte_ratio=raw_hgb_text_byte_ratio,
        official_text_hgb_byte_ratio=raw_hgb_text_byte_ratio,
    )
    row.update(
        {
            "dataset": str(dataset),
            "budget_type": "support_node_ratio",
            "budget_value": float(budget),
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "success": bool(success),
            "failure_type": str(failure_type),
            "failure_message": str(failure_message),
            "construction_status": str(construction_status),
            "selection_signal": str(selection_signal),
            "official_hgb_exported": False,
            "export_hash": str(export_hash),
            "artifact_manifest_path": "",
            "raw_hgb_text_byte_ratio_estimated": raw_hgb_text_byte_ratio is not None,
            "training_executed": False,
            "training_status": "not_executed_gate21_6_budget",
            "compress_time_seconds": None,
            "compress_peak_cpu_memory_mb": None,
            "export_time_seconds": None,
            "preprocess_time_seconds": None,
            "train_time_seconds": None,
            "peak_gpu_memory_mb": None,
            "test_micro_f1": None,
            "test_macro_f1": None,
            "recovery_micro": None,
            "recovery_macro": None,
        }
    )
    if not success:
        row["eligible_for_official_main_table"] = False
        reason = str(row.get("eligibility_failure_reasons") or "")
        parts = [part for part in [reason, failure_type] if part]
        row["eligibility_failure_reasons"] = ";".join(parts)
    return row


def _native_stats(dataset: str, native_hgb_root: str | Path | None) -> dict[str, Any] | None:
    if native_hgb_root is None:
        return None
    dataset_name = str(dataset).upper()
    root = Path(native_hgb_root)
    dataset_dir = root if root.name.upper() == dataset_name else root / dataset_name
    node_dat = dataset_dir / "node.dat"
    link_dat = dataset_dir / "link.dat"
    if not node_dat.exists() or not link_dat.exists() or dataset_name not in SEHGNN_HGB_SCHEMAS:
        return None
    target_type = int(SEHGNN_HGB_SCHEMAS[dataset_name]["node_type_order"][SEHGNN_HGB_SCHEMAS[dataset_name]["target_type"]])
    node_counts: dict[int, int] = {}
    with node_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                type_id = int(parts[2])
                node_counts[type_id] = node_counts.get(type_id, 0) + 1
    relation_edge_counts: dict[int, int] = {}
    relation_target_endpoint: dict[int, bool] = {}
    with link_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                relation_id = int(parts[2])
                relation_edge_counts[relation_id] = relation_edge_counts.get(relation_id, 0) + 1
                src_type = _node_type_from_global_id(int(parts[0]), node_counts)
                dst_type = _node_type_from_global_id(int(parts[1]), node_counts)
                if src_type == target_type or dst_type == target_type:
                    relation_target_endpoint[relation_id] = True
    return {
        "dataset_dir": str(dataset_dir),
        "target_type": target_type,
        "node_counts": node_counts,
        "target_nodes": int(node_counts.get(target_type, 0)),
        "support_nodes": int(sum(count for type_id, count in node_counts.items() if int(type_id) != target_type)),
        "total_nodes": int(sum(node_counts.values())),
        "relation_edge_counts": relation_edge_counts,
        "relation_target_endpoint": relation_target_endpoint,
        "total_edges": int(sum(relation_edge_counts.values())),
        "native_full_text_bytes": int(sum(path.stat().st_size for path in dataset_dir.glob("*.dat") if path.is_file())),
    }


def _node_type_from_global_id(node_id: int, node_counts: dict[int, int]) -> int | None:
    cursor = 0
    for type_id, count in sorted(node_counts.items()):
        cursor += int(count)
        if int(node_id) < cursor:
            return int(type_id)
    return None


def _estimate_ratios(method: str, budget: float, stats: dict[str, Any] | None) -> dict[str, float | None]:
    if stats is None:
        return {"support_edge_ratio": None, "structural_storage_ratio": None, "raw_hgb_text_byte_ratio": None}
    total_edges = max(int(stats["total_edges"]), 1)
    total_nodes = max(int(stats["total_nodes"]), 1)
    support_nodes = int(stats["support_nodes"])
    target_nodes = int(stats["target_nodes"])
    budget = max(0.0, min(1.0, float(budget)))
    retained_support_nodes = int(round(support_nodes * budget))
    if method == "GraphSparsify-TP":
        retained_nodes = total_nodes
        edge_ratio = budget
    else:
        retained_nodes = target_nodes + retained_support_nodes
        retained_edges = 0.0
        for relation_id, count in dict(stats["relation_edge_counts"]).items():
            retained_edges += float(count) * (budget if bool(dict(stats["relation_target_endpoint"]).get(int(relation_id), False)) else budget * budget)
        edge_ratio = retained_edges / float(total_edges)
    structural = (float(retained_nodes) + float(total_edges) * float(edge_ratio)) / float(total_nodes + total_edges)
    return {
        "support_edge_ratio": float(edge_ratio),
        "structural_storage_ratio": float(structural),
        "raw_hgb_text_byte_ratio": float(structural),
    }


def _selection_signal(method: str) -> str:
    return {
        "Random-HG-TP": "per-type seeded random support selection",
        "Herding-HG-TP": "per-type centroid herding with degree/profile fallback",
        "KCenter-HG-TP": "per-type approximate farthest-first profile selection",
        "Coarsening-HG-TP": "support medoid selection; no synthetic supernodes",
        "GraphSparsify-TP": "relation-wise persistent edge pruning",
        "FreeHGC-TP": "external FreeHGC adapter",
    }.get(method, "unknown")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def plan_external_tp_rows(
    *,
    dataset: str,
    methods: Iterable[str],
    budgets: Iterable[float],
    graph_seeds: Iterable[int],
    training_seeds: Iterable[int],
    freehgc_root: str | Path | None = None,
    native_hgb_root: str | Path | None = Path("external/SeHGNN/data"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    freehgc_available = freehgc_root is not None and Path(freehgc_root).exists()
    stats = _native_stats(dataset, native_hgb_root)
    for method in methods:
        if method not in TP_BASELINE_METHODS:
            raise ValueError(f"unsupported TP baseline: {method}")
        for budget in budgets:
            ratios = _estimate_ratios(method, float(budget), stats)
            for graph_seed in graph_seeds:
                export_hash = _stable_hash(
                    {
                        "dataset": str(dataset).upper(),
                        "method": method,
                        "budget": float(budget),
                        "graph_seed": int(graph_seed),
                        "native_stats": stats,
                    }
                )
                for training_seed in training_seeds:
                    if method == "FreeHGC-TP" and not freehgc_available:
                        rows.append(
                            _base_row(
                                dataset=dataset,
                                method=method,
                                budget=float(budget),
                                graph_seed=int(graph_seed),
                                training_seed=int(training_seed),
                                success=False,
                                failure_type="missing_external_dependency",
                                failure_message="FreeHGC root was not provided or does not exist.",
                                construction_status="missing_external_dependency",
                                selection_signal=_selection_signal(method),
                            )
                        )
                    else:
                        rows.append(
                            _base_row(
                                dataset=dataset,
                                method=method,
                                budget=float(budget),
                                graph_seed=int(graph_seed),
                                training_seed=int(training_seed),
                                success=True,
                                support_edge_ratio=ratios["support_edge_ratio"],
                                structural_storage_ratio=ratios["structural_storage_ratio"],
                                raw_hgb_text_byte_ratio=ratios["raw_hgb_text_byte_ratio"],
                                construction_status="constructed_estimate",
                                selection_signal=_selection_signal(method),
                                export_hash=export_hash,
                            )
                        )
    return rows
