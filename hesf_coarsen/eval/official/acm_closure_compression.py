from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


ACM_RELATIONS = {
    "PP": 0,
    "PP_r": 1,
    "PA": 2,
    "AP": 3,
    "PC": 4,
    "CP": 5,
    "PK": 6,
    "KP": 7,
}


def export_acm_closure_compressed(
    source_dir: str | Path,
    export_dir: str | Path,
    *,
    method: str,
    keyword_ratio: float,
    graph_seed: int = 1,
) -> dict[str, Any]:
    source = Path(source_dir)
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    nodes = _read_nodes(source / "node.dat")
    shifts = _type_shifts(nodes)
    counts = _type_counts(nodes)
    relations = _read_relations(source / "link.dat")
    k_count = counts.get(3, 0)
    keep_count = max(1, min(k_count, int(round(k_count * float(keyword_ratio))))) if k_count else 0
    selected_k_local = _select_keywords(relations.get(ACM_RELATIONS["PK"], []), shifts=shifts, k_count=k_count, keep_count=keep_count, method=method, graph_seed=graph_seed)
    selected_k_global = {shifts.get(3, 0) + local_id for local_id in selected_k_local}

    out_relations: dict[int, list[tuple[int, int, float]]] = {}
    for rel_id, edges in relations.items():
        if rel_id == ACM_RELATIONS["PK"]:
            out_relations[rel_id] = _coalesced_edges((src, dst, weight) for src, dst, weight in edges if dst in selected_k_global)
        elif rel_id == ACM_RELATIONS["KP"]:
            out_relations[rel_id] = _coalesced_edges((src, dst, weight) for src, dst, weight in edges if src in selected_k_global)
        else:
            out_relations[rel_id] = _coalesced_edges(edges)

    pk_local = {
        (src - shifts.get(0, 0), dst - shifts.get(3, 0))
        for src, dst, _ in out_relations.get(ACM_RELATIONS["PK"], [])
    }
    p_features = _feature_matrix(counts.get(0, 0), k_count, pk_local)
    a_features = _left_multiply_relation(
        out_relations.get(ACM_RELATIONS["AP"], []),
        right_features=p_features,
        left_shift=shifts.get(1, 0),
        right_shift=shifts.get(0, 0),
        left_count=counts.get(1, 0),
        width=k_count,
    )
    c_features = _left_multiply_relation(
        out_relations.get(ACM_RELATIONS["CP"], []),
        right_features=p_features,
        left_shift=shifts.get(2, 0),
        right_shift=shifts.get(0, 0),
        left_count=counts.get(2, 0),
        width=k_count,
    )
    feature_by_type = {0: p_features, 1: a_features, 2: c_features}
    _write_nodes(export / "node.dat", nodes, feature_by_type=feature_by_type, shifts=shifts)
    _write_relations(export / "link.dat", out_relations)
    for filename in ("label.dat", "label.dat.test", "info.dat"):
        shutil.copy2(source / filename, export / filename)

    manifest = _manifest(
        source,
        export,
        dataset="ACM",
        method=method,
        graph_seed=graph_seed,
        requested_budget_type="keyword_feature_ratio",
        requested_budget=keyword_ratio,
        selected_edge_lines=_relation_lines(out_relations),
    )
    source_pk = len(relations.get(ACM_RELATIONS["PK"], []))
    export_pk = len(out_relations.get(ACM_RELATIONS["PK"], []))
    source_edges = sum(len(edges) for edges in relations.values())
    export_edges = sum(len(edges) for edges in out_relations.values())
    manifest.update(
        {
            "constraint_safe_fallback": False,
            "keyword_feature_ratio": len(selected_k_local) / k_count if k_count else 1.0,
            "PK_edge_ratio": export_pk / source_pk if source_pk else 1.0,
            "actual_support_edge_ratio": export_edges / source_edges if source_edges else 1.0,
            "actual_support_node_ratio": 1.0,
            "semantic_structural_storage_ratio": export_pk / source_pk if source_pk else 1.0,
        }
    )
    _write_json(export / "gate21_18_export_manifest.json", manifest)
    return manifest


def audit_acm_closure_export(export_dir: str | Path, *, source_dir: str | Path | None = None) -> dict[str, Any]:
    export = Path(export_dir)
    source = Path(source_dir) if source_dir is not None else None
    if not (export / "node.dat").exists() or not (export / "link.dat").exists():
        return {
            "dataset": "ACM",
            "export_dir": str(export),
            "constraint_safe_fallback": False,
            "official_loader_preflight_pass": False,
            "failure_type": "export_schema_failure",
            "failure_reason": "Missing node.dat or link.dat.",
        }
    nodes = _read_nodes(export / "node.dat")
    shifts = _type_shifts(nodes)
    counts = _type_counts(nodes)
    relations = _read_relations(export / "link.dat")
    p_features = _features_by_type(nodes, 0, shifts=shifts, width=counts.get(3, 0))
    a_features = _features_by_type(nodes, 1, shifts=shifts, width=counts.get(3, 0))
    c_features = _features_by_type(nodes, 2, shifts=shifts, width=counts.get(3, 0))
    pk_local = {
        (src - shifts.get(0, 0), dst - shifts.get(3, 0))
        for src, dst, _ in relations.get(ACM_RELATIONS["PK"], [])
    }
    p_nonzero = {(row, col) for row, values in enumerate(p_features) for col, value in enumerate(values) if float(value) != 0.0}
    expected_a = _left_multiply_relation(
        relations.get(ACM_RELATIONS["AP"], []),
        right_features=p_features,
        left_shift=shifts.get(1, 0),
        right_shift=shifts.get(0, 0),
        left_count=counts.get(1, 0),
        width=counts.get(3, 0),
    )
    expected_c = _left_multiply_relation(
        relations.get(ACM_RELATIONS["CP"], []),
        right_features=p_features,
        left_shift=shifts.get(2, 0),
        right_shift=shifts.get(0, 0),
        left_count=counts.get(2, 0),
        width=counts.get(3, 0),
    )
    source_relations = _read_relations(source / "link.dat") if source is not None and (source / "link.dat").exists() else {}
    export_edges = sum(len(edges) for edges in relations.values())
    source_edges = sum(len(edges) for edges in source_relations.values()) if source_relations else export_edges
    source_pk = len(source_relations.get(ACM_RELATIONS["PK"], [])) if source_relations else len(relations.get(ACM_RELATIONS["PK"], []))
    export_pk = len(relations.get(ACM_RELATIONS["PK"], []))
    return {
        "dataset": "ACM",
        "export_dir": str(export),
        "constraint_safe_fallback": False,
        "P_nonzero_count": len(p_nonzero),
        "PK_edge_count": export_pk,
        "P_matches_PK": p_nonzero == pk_local,
        "A_matches_AP_PK": _matrix_equal(a_features, expected_a),
        "C_matches_CP_PK": _matrix_equal(c_features, expected_c),
        "PA_AP_reciprocal": _reciprocal_equal(relations.get(ACM_RELATIONS["PA"], []), relations.get(ACM_RELATIONS["AP"], [])),
        "PC_CP_reciprocal": _reciprocal_equal(relations.get(ACM_RELATIONS["PC"], []), relations.get(ACM_RELATIONS["CP"], [])),
        "PK_KP_reciprocal": _reciprocal_equal(relations.get(ACM_RELATIONS["PK"], []), relations.get(ACM_RELATIONS["KP"], [])),
        "keyword_feature_ratio": _keyword_ratio(nodes, p_nonzero),
        "PK_edge_ratio": export_pk / source_pk if source_pk else 1.0,
        "actual_support_edge_ratio": export_edges / source_edges if source_edges else 1.0,
        "official_loader_preflight_pass": bool(export_edges and p_nonzero == pk_local),
    }


def _select_keywords(
    pk_edges: list[tuple[int, int, float]],
    *,
    shifts: Mapping[int, int],
    k_count: int,
    keep_count: int,
    method: str,
    graph_seed: int,
) -> set[int]:
    degrees: dict[int, int] = {}
    for _, dst, _ in pk_edges:
        local = dst - shifts.get(3, 0)
        degrees[local] = degrees.get(local, 0) + 1
    all_keywords = list(range(k_count))
    lower = method.lower()
    if "random" in lower:
        ordered = sorted(all_keywords, key=lambda item: _hash_float("ACM", method, graph_seed, item))
    else:
        ordered = sorted(all_keywords, key=lambda item: (-(degrees.get(item, 0)), _hash_float("ACM", method, graph_seed, item)))
    return set(ordered[:keep_count])


def _read_nodes(path: Path) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            nodes.append(
                {
                    "id": int(parts[0]),
                    "name": parts[1],
                    "type": int(parts[2]),
                    "features": _parse_features(parts[3]) if len(parts) >= 4 else None,
                }
            )
    return nodes


def _write_nodes(path: Path, nodes: list[Mapping[str, Any]], *, feature_by_type: Mapping[int, list[list[float]]], shifts: Mapping[int, int]) -> None:
    lines: list[str] = []
    for node in nodes:
        node_id = int(node["id"])
        node_type = int(node["type"])
        values = None
        if node_type in feature_by_type:
            values = feature_by_type[node_type][node_id - shifts[node_type]]
        elif node.get("features") is not None:
            values = node.get("features")
        if values is None:
            lines.append(f"{node_id}\t{node['name']}\t{node_type}\n")
        else:
            lines.append(f"{node_id}\t{node['name']}\t{node_type}\t{_format_features(values)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _read_relations(path: Path) -> dict[int, list[tuple[int, int, float]]]:
    relations: dict[int, list[tuple[int, int, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            src, dst, rel, weight = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            relations.setdefault(rel, []).append((src, dst, weight))
    return relations


def _write_relations(path: Path, relations: Mapping[int, Iterable[tuple[int, int, float]]]) -> None:
    lines: list[str] = []
    for rel_id in sorted(relations):
        for src, dst, weight in sorted(relations[rel_id], key=lambda item: (item[0], item[1], item[2])):
            lines.append(f"{src}\t{dst}\t{rel_id}\t{_format_number(weight)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _feature_matrix(rows: int, cols: int, nonzero_pairs: set[tuple[int, int]]) -> list[list[float]]:
    matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row, col in nonzero_pairs:
        if 0 <= row < rows and 0 <= col < cols:
            matrix[row][col] = 1.0
    return matrix


def _left_multiply_relation(
    relation_edges: Iterable[tuple[int, int, float]],
    *,
    right_features: list[list[float]],
    left_shift: int,
    right_shift: int,
    left_count: int,
    width: int,
) -> list[list[float]]:
    out = [[0.0 for _ in range(width)] for _ in range(left_count)]
    for src, dst, _ in _coalesced_edges(relation_edges):
        left = src - left_shift
        right = dst - right_shift
        if left < 0 or left >= left_count or right < 0 or right >= len(right_features):
            continue
        for col, value in enumerate(right_features[right]):
            if value:
                out[left][col] = 1.0
    return out


def _features_by_type(nodes: list[Mapping[str, Any]], node_type: int, *, shifts: Mapping[int, int], width: int) -> list[list[float]]:
    count = sum(1 for node in nodes if int(node["type"]) == node_type)
    out = [[0.0 for _ in range(width)] for _ in range(count)]
    for node in nodes:
        if int(node["type"]) != node_type:
            continue
        values = node.get("features") or []
        local = int(node["id"]) - shifts[node_type]
        out[local] = [float(values[idx]) if idx < len(values) else 0.0 for idx in range(width)]
    return out


def _type_counts(nodes: list[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        node_type = int(node["type"])
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def _type_shifts(nodes: list[Mapping[str, Any]]) -> dict[int, int]:
    shifts: dict[int, int] = {}
    for node in nodes:
        node_type = int(node["type"])
        shifts.setdefault(node_type, int(node["id"]))
    return shifts


def _matrix_equal(left: list[list[float]], right: list[list[float]]) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            return False
        for left_value, right_value in zip(left_row, right_row):
            if abs(float(left_value) - float(right_value)) > 1e-6:
                return False
    return True


def _coalesced_edges(edges: Iterable[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, float]] = []
    for src, dst, weight in edges:
        key = (int(src), int(dst))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(src), int(dst), float(weight)))
    return out


def _reciprocal_equal(forward: Iterable[tuple[int, int, float]], reverse: Iterable[tuple[int, int, float]]) -> bool:
    return {(dst, src) for src, dst, _ in forward} == {(src, dst) for src, dst, _ in reverse}


def _keyword_ratio(nodes: list[Mapping[str, Any]], p_nonzero: set[tuple[int, int]]) -> float:
    k_count = sum(1 for node in nodes if int(node["type"]) == 3)
    selected = {col for _, col in p_nonzero}
    return len(selected) / k_count if k_count else 1.0


def _manifest(
    source: Path,
    export: Path,
    *,
    dataset: str,
    method: str,
    graph_seed: int,
    requested_budget_type: str,
    requested_budget: float,
    selected_edge_lines: list[str],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "graph_seed": graph_seed,
        "requested_budget_type": requested_budget_type,
        "requested_budget": requested_budget,
        "source_dir": str(source),
        "export_dir": str(export),
        "raw_hgb_text_byte_ratio": _hgb_bytes(export) / _hgb_bytes(source) if _hgb_bytes(source) else 1.0,
        "selected_edge_hash": hashlib.sha256("".join(selected_edge_lines).encode("utf-8")).hexdigest(),
        "planner_config_hash": hashlib.sha256(json.dumps({"dataset": dataset, "method": method, "budget": requested_budget, "seed": graph_seed}, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _relation_lines(relations: Mapping[int, Iterable[tuple[int, int, float]]]) -> list[str]:
    lines: list[str] = []
    for rel_id in sorted(relations):
        for src, dst, weight in sorted(relations[rel_id], key=lambda item: (item[0], item[1], item[2])):
            lines.append(f"{src}\t{dst}\t{rel_id}\t{_format_number(weight)}\n")
    return lines


def _parse_features(raw: str) -> list[float]:
    if not raw:
        return []
    return [float(item) for item in raw.split(",") if item != ""]


def _format_features(values: Iterable[float]) -> str:
    return ",".join(_format_number(float(value)) for value in values)


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _hash_float(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _hgb_bytes(path: Path) -> int:
    return sum((path / name).stat().st_size for name in ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat") if (path / name).exists())


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=2, default=str), encoding="utf-8")
