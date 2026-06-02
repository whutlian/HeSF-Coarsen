from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


IMDB_RELATIONS = {
    "MD": 0,
    "DM": 1,
    "MA": 2,
    "AM": 3,
    "MK": 4,
    "KM": 5,
}


def export_imdb_constraint_compressed(
    source_dir: str | Path,
    export_dir: str | Path,
    *,
    method: str,
    actor_ratio: float,
    keyword_ratio: float,
    graph_seed: int = 1,
) -> dict[str, Any]:
    source = Path(source_dir)
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    relations = _read_relations(source / "link.dat")
    max_ids = _max_node_id_by_type(source / "node.dat")
    ma_edges = relations.get(IMDB_RELATIONS["MA"], [])
    mk_edges = relations.get(IMDB_RELATIONS["MK"], [])
    ma_selected = _ensure_destination_coverage(
        _select_edges(ma_edges, ratio=actor_ratio, method=method, channel="MA", graph_seed=graph_seed),
        ma_edges,
        required_dst_ids={max_ids[2]} if 2 in max_ids else set(),
    )
    mk_selected = _ensure_destination_coverage(
        _select_edges(mk_edges, ratio=keyword_ratio, method=method, channel="MK", graph_seed=graph_seed),
        mk_edges,
        required_dst_ids={max_ids[3]} if 3 in max_ids else set(),
    )
    out_relations = {
        IMDB_RELATIONS["MD"]: list(relations.get(IMDB_RELATIONS["MD"], [])),
        IMDB_RELATIONS["DM"]: [(dst, src, weight) for src, dst, weight in relations.get(IMDB_RELATIONS["MD"], [])],
        IMDB_RELATIONS["MA"]: ma_selected,
        IMDB_RELATIONS["AM"]: [(dst, src, weight) for src, dst, weight in ma_selected],
        IMDB_RELATIONS["MK"]: mk_selected,
        IMDB_RELATIONS["KM"]: [(dst, src, weight) for src, dst, weight in mk_selected],
    }
    shutil.copy2(source / "node.dat", export / "node.dat")
    shutil.copy2(source / "label.dat", export / "label.dat")
    shutil.copy2(source / "label.dat.test", export / "label.dat.test")
    shutil.copy2(source / "info.dat", export / "info.dat")
    _write_relations(export / "link.dat", out_relations)
    for optional in ("meta.dat", "url.dat"):
        if (source / optional).exists():
            shutil.copy2(source / optional, export / optional)

    source_edges = sum(len(edges) for edges in relations.values())
    export_edges = sum(len(edges) for edges in out_relations.values())
    selected_lines = _relation_lines(out_relations)
    manifest = {
        "dataset": "IMDB",
        "method": method,
        "graph_seed": graph_seed,
        "requested_budget_type": "support_edge_ratio",
        "requested_budget": _mean_requested(actor_ratio, keyword_ratio),
        "source_dir": str(source),
        "export_dir": str(export),
        "constraint_safe_fallback": False,
        "actor_channel_ratio": len(ma_selected) / len(relations.get(IMDB_RELATIONS["MA"], [])) if relations.get(IMDB_RELATIONS["MA"]) else 1.0,
        "keyword_channel_ratio": len(mk_selected) / len(relations.get(IMDB_RELATIONS["MK"], [])) if relations.get(IMDB_RELATIONS["MK"]) else 1.0,
        "actual_support_edge_ratio": export_edges / source_edges if source_edges else 1.0,
        "actual_support_node_ratio": 1.0,
        "semantic_structural_storage_ratio": export_edges / source_edges if source_edges else 1.0,
        "raw_hgb_text_byte_ratio": _hgb_bytes(export) / _hgb_bytes(source) if _hgb_bytes(source) else 1.0,
        "selected_edge_hash": hashlib.sha256("".join(selected_lines).encode("utf-8")).hexdigest(),
        "planner_config_hash": hashlib.sha256(
            json.dumps(
                {"dataset": "IMDB", "method": method, "actor_ratio": actor_ratio, "keyword_ratio": keyword_ratio, "seed": graph_seed},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(export / "gate21_18_export_manifest.json", manifest)
    return manifest


def audit_imdb_constraint_export(export_dir: str | Path, *, source_dir: str | Path | None = None) -> dict[str, Any]:
    export = Path(export_dir)
    source = Path(source_dir) if source_dir is not None else None
    if not (export / "node.dat").exists() or not (export / "link.dat").exists():
        return {
            "dataset": "IMDB",
            "export_dir": str(export),
            "constraint_safe_fallback": False,
            "official_loader_preflight_pass": False,
            "failure_type": "export_schema_failure",
            "failure_reason": "Missing node.dat or link.dat.",
        }
    movie_ids = _node_ids_by_type(export / "node.dat", 0)
    relations = _read_relations(export / "link.dat")
    source_relations = _read_relations(source / "link.dat") if source is not None and (source / "link.dat").exists() else {}
    md_counts = Counter(src for src, _, _ in relations.get(IMDB_RELATIONS["MD"], []))
    without = sum(1 for movie in movie_ids if md_counts.get(movie, 0) == 0)
    multi = sum(1 for movie, count in md_counts.items() if count > 1)
    export_edges = sum(len(edges) for edges in relations.values())
    source_edges = sum(len(edges) for edges in source_relations.values()) if source_relations else export_edges
    return {
        "dataset": "IMDB",
        "export_dir": str(export),
        "constraint_safe_fallback": False,
        "MD_DM_reciprocal": _reciprocal_equal(relations.get(IMDB_RELATIONS["MD"], []), relations.get(IMDB_RELATIONS["DM"], [])),
        "MA_AM_reciprocal": _reciprocal_equal(relations.get(IMDB_RELATIONS["MA"], []), relations.get(IMDB_RELATIONS["AM"], [])),
        "MK_KM_reciprocal": _reciprocal_equal(relations.get(IMDB_RELATIONS["MK"], []), relations.get(IMDB_RELATIONS["KM"], [])),
        "movie_single_director_constraint_pass": without == 0 and multi == 0,
        "num_movies_without_director": without,
        "num_movies_multi_director": multi,
        "actual_support_edge_ratio": export_edges / source_edges if source_edges else 1.0,
        "semantic_structural_storage_ratio": export_edges / source_edges if source_edges else 1.0,
        "official_loader_preflight_pass": bool(movie_ids and without == 0 and multi == 0),
    }


def _select_edges(
    edges: list[tuple[int, int, float]],
    *,
    ratio: float,
    method: str,
    channel: str,
    graph_seed: int,
) -> list[tuple[int, int, float]]:
    if not edges:
        return []
    keep = max(1, min(len(edges), int(round(len(edges) * float(ratio)))))
    lower = method.lower()
    if "degree" in lower or "hesf" in lower:
        degrees: dict[int, int] = {}
        for src, dst, _ in edges:
            degrees[src] = degrees.get(src, 0) + 1
            degrees[dst] = degrees.get(dst, 0) + 1
        ordered = sorted(edges, key=lambda item: (-(degrees.get(item[0], 0) + degrees.get(item[1], 0)), _hash_float("IMDB", method, channel, graph_seed, item[0], item[1])))
    else:
        ordered = sorted(edges, key=lambda item: _hash_float("IMDB", method, channel, graph_seed, item[0], item[1]))
    return ordered[:keep]


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
    path.write_text("".join(_relation_lines(relations)), encoding="utf-8")


def _relation_lines(relations: Mapping[int, Iterable[tuple[int, int, float]]]) -> list[str]:
    lines: list[str] = []
    for rel_id in sorted(relations):
        for src, dst, weight in sorted(relations[rel_id], key=lambda item: (item[0], item[1], item[2])):
            lines.append(f"{src}\t{dst}\t{rel_id}\t{_format_number(weight)}\n")
    return lines


def _node_ids_by_type(path: Path, node_type: int) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and int(parts[2]) == node_type:
                ids.add(int(parts[0]))
    return ids


def _max_node_id_by_type(path: Path) -> dict[int, int]:
    max_ids: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                node_id = int(parts[0])
                node_type = int(parts[2])
                max_ids[node_type] = max(max_ids.get(node_type, node_id), node_id)
    return max_ids


def _ensure_destination_coverage(
    selected: list[tuple[int, int, float]],
    candidates: list[tuple[int, int, float]],
    *,
    required_dst_ids: set[int],
) -> list[tuple[int, int, float]]:
    out = _coalesced_edges(selected)
    selected_pairs = {(src, dst) for src, dst, _ in out}
    touched = {dst for _, dst, _ in out}
    for required in sorted(required_dst_ids):
        if required in touched:
            continue
        for src, dst, weight in candidates:
            if dst != required or (src, dst) in selected_pairs:
                continue
            out.append((src, dst, weight))
            selected_pairs.add((src, dst))
            touched.add(dst)
            break
    return out


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


def _mean_requested(actor_ratio: float, keyword_ratio: float) -> float:
    return (float(actor_ratio) + float(keyword_ratio)) / 2.0


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
