from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import float_value, normalize_dataset


ROOT = Path(__file__).resolve().parents[3]
SMOKE_EXPORT_METHODS: set[tuple[str, str, str, float]] = {
    ("DBLP", "Random-edge-relwise", "structural_storage_ratio", 0.20),
    ("DBLP", "Degree-edge-relwise", "structural_storage_ratio", 0.20),
    ("DBLP", "Proportional-relation-budget", "structural_storage_ratio", 0.20),
    ("DBLP", "Herding-HG-TP", "support_node_ratio", 0.50),
    ("DBLP", "FreeHGC-score-TP", "structural_storage_ratio", 0.20),
    ("DBLP", "HGCond-score-TP-local", "support_node_ratio", 0.50),
    ("DBLP", "GCond-score-TP-local", "support_node_ratio", 0.50),
    ("ACM", "H6-node30", "support_node_ratio", 0.30),
    ("ACM", "HeSF-RCS-auto structural20", "structural_storage_ratio", 0.20),
    ("ACM", "Random-edge-relwise", "structural_storage_ratio", 0.20),
    ("ACM", "Herding-HG-TP", "support_node_ratio", 0.50),
    ("ACM", "HGCond-score-TP-local", "support_node_ratio", 0.50),
    ("ACM", "GCond-score-TP-local", "support_node_ratio", 0.50),
    ("IMDB", "HeSF-RCS-auto structural20", "structural_storage_ratio", 0.20),
    ("IMDB", "Random-edge-relwise", "structural_storage_ratio", 0.20),
    ("IMDB", "Herding-HG-TP", "support_node_ratio", 0.50),
    ("IMDB", "HGCond-score-TP-local", "support_node_ratio", 0.50),
    ("IMDB", "GCond-score-TP-local", "support_node_ratio", 0.50),
}


def maybe_prepare_gate21_17_export(row: Mapping[str, Any], *, out_dir: Path, mode: str, graph_seed: int = 1) -> dict[str, Any]:
    dataset = normalize_dataset(row.get("dataset"))
    method = str(row.get("method", ""))
    budget_type = str(row.get("requested_budget_type", ""))
    budget = float_value(row.get("requested_budget"))
    if mode == "preflight" or budget is None:
        return {}
    key = (dataset, method, budget_type, round(float(budget), 10))
    if key not in SMOKE_EXPORT_METHODS:
        return {}
    source_dir = _source_dataset_dir(dataset)
    if not source_dir.exists():
        return {
            "export_dir": "",
            "export_failure_type": "export_schema_failure",
            "export_failure_reason": f"Missing source HGB dataset directory: {source_dir}",
        }
    export_parent = out_dir / "exports" / dataset / str(graph_seed) / _slug(method) / f"{budget_type}_{_budget_slug(budget)}" / "official_trainval"
    export_dir = export_parent / dataset
    export_dir.mkdir(parents=True, exist_ok=True)
    if dataset in {"ACM", "IMDB"}:
        manifest = _write_constraint_safe_full_export(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=float(budget), graph_seed=graph_seed)
    elif _is_support_node_method(method, budget_type):
        manifest = _write_target_preserving_support_export(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=float(budget), graph_seed=graph_seed)
    else:
        manifest = _write_edge_sparsified_export(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=float(budget), graph_seed=graph_seed)
    return {
        "export_dir": str(export_dir),
        "actual_structural_storage_ratio": manifest["raw_hgb_text_byte_ratio"],
        "support_node_ratio": manifest["support_node_ratio"],
        "support_edge_ratio": manifest["support_edge_ratio"],
        "raw_hgb_text_byte_ratio": manifest["raw_hgb_text_byte_ratio"],
        "selected_edge_hash": manifest["selected_edge_hash"],
        "planner_config_hash": manifest["planner_config_hash"],
        "source_path": str(export_dir / "gate21_17_export_manifest.json"),
    }


def _write_constraint_safe_full_export(source_dir: Path, export_dir: Path, *, dataset: str, method: str, budget_type: str, budget: float, graph_seed: int) -> dict[str, Any]:
    for filename in ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat"):
        _link_or_copy(source_dir / filename, export_dir / filename)
    selected_lines = (source_dir / "link.dat").read_text(encoding="utf-8").splitlines(keepends=True)
    manifest = _write_manifest(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=budget, graph_seed=graph_seed, selected_lines=selected_lines)
    manifest["support_node_ratio"] = 1.0
    manifest["support_edge_ratio"] = 1.0
    manifest["raw_hgb_text_byte_ratio"] = 1.0
    manifest["constraint_safe_fallback"] = True
    manifest["budget_infeasible_reason"] = "ACM/IMDB official preprocessing constraints require reciprocal and feature-consistency-preserving exports; smoke uses full-HGB fallback and reports actual ratio."
    _write_json(export_dir / "gate21_17_export_manifest.json", manifest)
    return manifest


def _write_edge_sparsified_export(source_dir: Path, export_dir: Path, *, dataset: str, method: str, budget_type: str, budget: float, graph_seed: int) -> dict[str, Any]:
    _link_or_copy(source_dir / "node.dat", export_dir / "node.dat")
    _link_or_copy(source_dir / "label.dat", export_dir / "label.dat")
    _link_or_copy(source_dir / "label.dat.test", export_dir / "label.dat.test")
    _link_or_copy(source_dir / "info.dat", export_dir / "info.dat")
    relation_edges: dict[str, list[str]] = {}
    degrees = _degree_counts(source_dir / "link.dat") if method == "Degree-edge-relwise" else {}
    with (source_dir / "link.dat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                relation_edges.setdefault(parts[2], []).append(line)
    selected: list[str] = []
    for relation, lines in sorted(relation_edges.items(), key=lambda item: int(item[0])):
        keep = max(1, int(round(len(lines) * budget)))
        if method == "Degree-edge-relwise":
            ordered = sorted(lines, key=lambda line: _degree_score(line, degrees), reverse=True)
        elif method == "Random-edge-relwise":
            ordered = sorted(lines, key=lambda line: _hash_float(dataset, method, relation, graph_seed, line))
        else:
            ordered = list(lines)
        selected.extend(ordered[:keep])
    selected = ensure_type_max_id_coverage(selected, [line for lines in relation_edges.values() for line in lines], _node_type_by_id(source_dir / "node.dat"))
    relation_order = _relation_first_seen_order(source_dir / "link.dat")
    selected = sort_hgb_link_lines(selected, relation_order=relation_order)
    (export_dir / "link.dat").write_text("".join(selected), encoding="utf-8")
    return _write_manifest(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=budget, graph_seed=graph_seed, selected_lines=selected)


def _write_target_preserving_support_export(source_dir: Path, export_dir: Path, *, dataset: str, method: str, budget_type: str, budget: float, graph_seed: int) -> dict[str, Any]:
    target_type = 0
    support_ratio = budget if budget_type == "support_node_ratio" else 0.50
    edge_ratio = budget if budget_type == "structural_storage_ratio" else 1.0
    counts = _node_counts(source_dir / "node.dat")
    degrees = _degree_counts(source_dir / "link.dat")
    initial_selected_by_type: dict[int, set[int]] = {}
    for node_type, count in sorted(counts.items()):
        ids = list(range(_type_shift(counts, node_type), _type_shift(counts, node_type) + count))
        if node_type == target_type:
            initial_selected_by_type[node_type] = set(ids)
            continue
        keep = max(1, int(round(count * support_ratio))) if count else 0
        if method in {"Herding-HG-TP", "FreeHGC-score-TP", "HGCond-score-TP-local", "GCond-score-TP-local"}:
            ordered = sorted(ids, key=lambda node_id: (-(degrees.get(node_id, 0)), _hash_float(dataset, method, graph_seed, node_id)))
        elif method == "KCenter-HG-TP":
            ordered = ids[:: max(1, count // max(1, keep))] + ids
        else:
            ordered = sorted(ids, key=lambda node_id: _hash_float(dataset, method, graph_seed, node_id))
        initial_selected_by_type[node_type] = set(ordered[:keep])
    selected_ids = set().union(*initial_selected_by_type.values()) if initial_selected_by_type else set()
    relation_edges: dict[str, list[str]] = {}
    with (source_dir / "link.dat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            if src in selected_ids and dst in selected_ids:
                relation_edges.setdefault(parts[2], []).append(line)
    selected: list[str] = []
    candidate_lines: list[str] = []
    for relation, lines in sorted(relation_edges.items(), key=lambda item: int(item[0])):
        keep = max(1, int(round(len(lines) * edge_ratio))) if lines else 0
        ordered = sorted(lines, key=lambda line: _hash_float(dataset, method, relation, graph_seed, line))
        candidate_lines.extend(lines)
        selected.extend(ordered[:keep])
    selected_node_type_by_id = {node_id: node_type for node_type, ids in initial_selected_by_type.items() for node_id in ids}
    selected = ensure_type_max_id_coverage(selected, candidate_lines, selected_node_type_by_id)
    touched_ids = _touched_node_ids(selected)
    final_selected_by_type: dict[int, set[int]] = {}
    for node_type, ids in initial_selected_by_type.items():
        if node_type == target_type:
            final_selected_by_type[node_type] = set(ids)
        else:
            final_selected_by_type[node_type] = set(ids) & touched_ids
            if not final_selected_by_type[node_type]:
                final_selected_by_type[node_type] = set(ids)
    id_map = _write_remapped_nodes(source_dir / "node.dat", export_dir / "node.dat", final_selected_by_type)
    _write_remapped_labels(source_dir / "label.dat", export_dir / "label.dat", id_map)
    _write_remapped_labels(source_dir / "label.dat.test", export_dir / "label.dat.test", id_map)
    _link_or_copy(source_dir / "info.dat", export_dir / "info.dat")
    relation_order = _relation_first_seen_order(source_dir / "link.dat")
    with (export_dir / "link.dat").open("w", encoding="utf-8") as handle:
        remapped_lines: list[str] = []
        for line in selected:
            parts = line.rstrip("\n").split("\t")
            if int(parts[0]) not in id_map or int(parts[1]) not in id_map:
                continue
            parts[0] = str(id_map[int(parts[0])])
            parts[1] = str(id_map[int(parts[1])])
            remapped_lines.append("\t".join(parts) + "\n")
        handle.write("".join(sort_hgb_link_lines(remapped_lines, relation_order=relation_order)))
    manifest = _write_manifest(source_dir, export_dir, dataset=dataset, method=method, budget_type=budget_type, budget=budget, graph_seed=graph_seed, selected_lines=selected)
    total_support = sum(count for node_type, count in counts.items() if node_type != target_type)
    kept_support = sum(len(ids) for node_type, ids in final_selected_by_type.items() if node_type != target_type)
    manifest["support_node_ratio"] = kept_support / total_support if total_support else 1.0
    _write_json(export_dir / "gate21_17_export_manifest.json", manifest)
    return manifest


def _write_manifest(source_dir: Path, export_dir: Path, *, dataset: str, method: str, budget_type: str, budget: float, graph_seed: int, selected_lines: list[str]) -> dict[str, Any]:
    source_bytes = _hgb_bytes(source_dir)
    export_bytes = _hgb_bytes(export_dir)
    source_edges = _count_lines(source_dir / "link.dat")
    export_edges = _count_lines(export_dir / "link.dat")
    digest = hashlib.sha256("".join(selected_lines).encode("utf-8")).hexdigest()
    manifest = {
        "dataset": dataset,
        "method": method,
        "requested_budget_type": budget_type,
        "requested_budget": budget,
        "graph_seed": graph_seed,
        "source_dir": str(source_dir),
        "export_dir": str(export_dir),
        "support_node_ratio": "all_target_preserved" if not _is_support_node_method(method, budget_type) else "",
        "support_edge_ratio": export_edges / source_edges if source_edges else "",
        "raw_hgb_text_byte_ratio": export_bytes / source_bytes if source_bytes else "",
        "selected_edge_hash": digest,
        "planner_config_hash": hashlib.sha256(json.dumps({"dataset": dataset, "method": method, "budget_type": budget_type, "budget": budget, "seed": graph_seed}, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    _write_json(export_dir / "gate21_17_export_manifest.json", manifest)
    return manifest


def _source_dataset_dir(dataset: str) -> Path:
    lower = normalize_dataset(dataset).lower()
    direct = ROOT / "data" / lower / "raw" / normalize_dataset(dataset)
    if direct.exists():
        return direct
    return ROOT / "data" / lower / lower / "raw" / normalize_dataset(dataset)


def _is_support_node_method(method: str, budget_type: str) -> bool:
    return budget_type == "support_node_ratio" or method in {"Herding-HG-TP", "KCenter-HG-TP", "Coarsening-HG-TP", "FreeHGC-score-TP", "HGCond-score-TP-local", "GCond-score-TP-local"}


def _node_counts(node_dat: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with node_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                node_type = int(parts[2])
                counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def _node_type_by_id(node_dat: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with node_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[int(parts[0])] = int(parts[2])
    return out


def ensure_type_max_id_coverage(selected_lines: list[str], candidate_lines: list[str], node_type_by_id: Mapping[int, int]) -> list[str]:
    if not selected_lines or not candidate_lines or not node_type_by_id:
        return list(selected_lines)
    max_id_by_type: dict[int, int] = {}
    for node_id, node_type in node_type_by_id.items():
        max_id_by_type[node_type] = max(max_id_by_type.get(node_type, node_id), node_id)
    out = list(dict.fromkeys(selected_lines))
    selected_set = set(out)
    for _ in range(len(max_id_by_type)):
        touched = _touched_node_ids(out)
        missing = {node_id for node_id in max_id_by_type.values() if node_id not in touched}
        if not missing:
            break
        added = False
        for line in candidate_lines:
            if line in selected_set:
                continue
            endpoints = _line_endpoints(line)
            if endpoints & missing:
                out.append(line)
                selected_set.add(line)
                added = True
                break
        if not added:
            break
    return out


def sort_hgb_link_lines(lines: Iterable[str], *, relation_order: Sequence[str] | None = None) -> list[str]:
    order = {str(relation): index for index, relation in enumerate(relation_order or ())}
    return sorted(lines, key=lambda line: _link_sort_key(line, order))


def _link_sort_key(line: str, relation_order: Mapping[str, int]) -> tuple[int, int, int, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return (10**9, 10**9, 10**9, line)
    relation = parts[2]
    relation_rank = relation_order.get(relation, int(relation) if relation.lstrip("-").isdigit() else 10**9)
    return (relation_rank, int(parts[0]), int(parts[1]), line)


def _relation_first_seen_order(link_dat: Path) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    with link_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            relation = parts[2]
            if relation not in seen:
                seen.add(relation)
                order.append(relation)
    return order


def _touched_node_ids(lines: Iterable[str]) -> set[int]:
    touched: set[int] = set()
    for line in lines:
        touched.update(_line_endpoints(line))
    return touched


def _line_endpoints(line: str) -> set[int]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return set()
    return {int(parts[0]), int(parts[1])}


def _type_shift(counts: Mapping[int, int], node_type: int) -> int:
    return sum(counts.get(item, 0) for item in sorted(counts) if item < node_type)


def _write_remapped_nodes(node_dat: Path, out_path: Path, selected_by_type: Mapping[int, set[int]]) -> dict[int, int]:
    next_id = 0
    id_map: dict[int, int] = {}
    with out_path.open("w", encoding="utf-8") as out, node_dat.open("r", encoding="utf-8") as handle:
        for node_type in sorted(selected_by_type):
            handle.seek(0)
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                old_id = int(parts[0])
                if int(parts[2]) != node_type or old_id not in selected_by_type[node_type]:
                    continue
                id_map[old_id] = next_id
                parts[0] = str(next_id)
                out.write("\t".join(parts) + "\n")
                next_id += 1
    return id_map


def _write_remapped_labels(label_dat: Path, out_path: Path, id_map: Mapping[int, int]) -> None:
    with out_path.open("w", encoding="utf-8") as out, label_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            old_id = int(parts[0])
            if old_id not in id_map:
                continue
            parts[0] = str(id_map[old_id])
            out.write("\t".join(parts) + "\n")


def _degree_counts(link_dat: Path) -> dict[int, int]:
    degrees: dict[int, int] = {}
    with link_dat.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                src = int(parts[0])
                dst = int(parts[1])
                degrees[src] = degrees.get(src, 0) + 1
                degrees[dst] = degrees.get(dst, 0) + 1
    return degrees


def _degree_score(line: str, degrees: Mapping[int, int]) -> int:
    parts = line.rstrip("\n").split("\t")
    return degrees.get(int(parts[0]), 0) + degrees.get(int(parts[1]), 0)


def _hash_float(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _link_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _hgb_bytes(path: Path) -> int:
    return sum((path / name).stat().st_size for name in ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat") if (path / name).exists())


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _budget_slug(value: float) -> str:
    return str(value).replace(".", "p")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, default=str), encoding="utf-8")
