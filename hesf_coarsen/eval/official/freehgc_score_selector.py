from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.stage_report_executor import (
    ensure_type_max_id_coverage,
    sort_hgb_link_lines,
)
from hesf_coarsen.eval.official.stage_report_protocol import normalize_dataset


FREEHGC_SELECTOR_FIELDS = (
    "dataset",
    "method",
    "requested_budget_type",
    "requested_budget",
    "actual_semantic_structural_ratio",
    "support_edge_ratio",
    "support_node_ratio",
    "raw_hgb_text_byte_ratio",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "recovery_micro",
    "recovery_macro",
    "selected_edge_hash",
    "selected_edge_hash_by_relation",
    "planner_config_hash",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "selector_uses_test_labels",
    "uses_test_for_selection",
    "failure_type",
    "failure_reason",
)


def build_freehgc_score_selector_plan_rows(
    *,
    dataset: str = "DBLP",
    budgets: Iterable[float] = (0.16, 0.20),
) -> list[dict[str, Any]]:
    name = normalize_dataset(dataset)
    out: list[dict[str, Any]] = []
    for budget in budgets:
        value = float(budget)
        out.append(
            {
                "dataset": name,
                "method": f"FreeHGC-score-as-selector structural{int(round(value * 100)):02d}",
                "method_family": "selector_probe",
                "planner_backend": "DBLPRelationChannelPlanner",
                "planner_mode": "freehgc_score_selector_proxy",
                "requested_budget_type": "structural_storage_ratio",
                "requested_budget": value,
                "eligible_for_main_table": True,
                "eligible_for_compression_claim": True,
                "official_sehgnn_unmodified": True,
                "selector_uses_test_labels": False,
                "uses_test_for_selection": False,
                "training_executed": False,
                "success": False,
                "failure_type": "implemented_pending_official_training",
                "failure_reason": "",
            }
        )
    return out


def build_dblp_freehgc_score_selector_export(
    *,
    source_dir: str | Path,
    export_dir: str | Path,
    budget: float,
    graph_seed: int = 1,
) -> dict[str, Any]:
    source = Path(source_dir)
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    for filename in ("node.dat", "label.dat", "label.dat.test", "info.dat"):
        shutil.copy2(source / filename, export / filename)
    edges = _read_link_lines(source / "link.dat")
    node_types = _node_type_by_id(source / "node.dat")
    degrees = _degree_counts(edges)
    relation_groups: dict[str, list[str]] = {}
    for line in edges:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            relation_groups.setdefault(parts[2], []).append(line)

    # FreeHGC-style proxy: AP/PV skeleton pressure first, PA/VP feedback second,
    # sparse PT/TP only when budget allows. This is local and train/test-label free.
    keep_counts = _relation_keep_counts(relation_groups, float(budget))
    selected: list[str] = []
    selected_by_relation: dict[str, int] = {}
    for relation, lines in sorted(relation_groups.items(), key=lambda item: int(item[0])):
        keep = max(0, min(len(lines), int(keep_counts.get(relation, 0)))) if lines else 0
        ordered = sorted(
            lines,
            key=lambda line: (
                -_edge_score(line, degrees=degrees, node_types=node_types, relation=relation),
                _hash_float("freehgc_selector", budget, graph_seed, line),
            ),
        )
        chosen = ordered[:keep]
        selected.extend(chosen)
        selected_by_relation[relation] = len(chosen)
    selected = ensure_type_max_id_coverage(selected, edges, node_types)
    selected = sort_hgb_link_lines(selected, relation_order=_relation_first_seen_order(source / "link.dat"))
    (export / "link.dat").write_text("".join(selected), encoding="utf-8")

    source_edges = max(len(edges), 1)
    semantic_ratio = len(selected) / source_edges
    manifest = {
        "dataset": "DBLP",
        "method": f"FreeHGC-score-as-selector structural{int(round(float(budget) * 100)):02d}",
        "requested_budget_type": "structural_storage_ratio",
        "requested_budget": float(budget),
        "graph_seed": int(graph_seed),
        "source_dir": str(source),
        "export_dir": str(export),
        "actual_semantic_structural_ratio": semantic_ratio,
        "semantic_structural_storage_ratio": semantic_ratio,
        "support_edge_ratio": semantic_ratio,
        "actual_support_edge_ratio": semantic_ratio,
        "support_node_ratio": 1.0,
        "raw_hgb_text_byte_ratio": _hgb_bytes(export) / max(_hgb_bytes(source), 1),
        "selected_edge_hash": hashlib.sha256("".join(selected).encode("utf-8")).hexdigest(),
        "selected_edge_hash_by_relation": json.dumps(selected_by_relation, sort_keys=True),
        "planner_config_hash": hashlib.sha256(
            json.dumps({"method": "freehgc_score_selector_proxy", "budget": float(budget), "graph_seed": int(graph_seed)}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "selector_uses_test_labels": False,
        "uses_test_for_selection": False,
        "constraint_safe_fallback": False,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
    }
    (export / "gate21_20_freehgc_score_selector_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _relation_keep_counts(relation_groups: Mapping[str, list[str]], budget: float) -> dict[str, int]:
    total_edges = sum(len(lines) for lines in relation_groups.values())
    relation_ids = sorted(relation_groups, key=lambda item: int(item))
    target = max(len(relation_ids), min(total_edges, int(round(total_edges * float(budget)))))
    counts = {relation: 1 if relation_groups[relation] else 0 for relation in relation_ids}
    remaining = max(0, target - sum(counts.values()))
    weights = _relation_weights(float(budget))
    while remaining > 0:
        candidates = [
            (
                weights.get(relation, 0.05) * (len(relation_groups[relation]) - counts[relation]),
                -int(relation),
                relation,
            )
            for relation in relation_ids
            if counts[relation] < len(relation_groups[relation])
        ]
        if not candidates:
            break
        _score, _tie, relation = max(candidates)
        counts[relation] += 1
        remaining -= 1
    return counts


def _relation_weights(budget: float) -> dict[str, float]:
    # DBLP HGB relation ids are AP, PA, PT, TP, PV, VP in first-seen order.
    if budget <= 0.16:
        return {"0": 0.34, "1": 0.16, "2": 0.02, "3": 0.02, "4": 0.34, "5": 0.12}
    return {"0": 0.30, "1": 0.18, "2": 0.04, "3": 0.04, "4": 0.30, "5": 0.14}


def _edge_score(line: str, *, degrees: Mapping[int, int], node_types: Mapping[int, int], relation: str) -> float:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return 0.0
    src = int(parts[0])
    dst = int(parts[1])
    target_bonus = 2.0 if node_types.get(src) == 0 or node_types.get(dst) == 0 else 0.0
    relation_bonus = 2.0 if relation in {"0", "4"} else 0.75 if relation in {"1", "5"} else 0.25
    return float(degrees.get(src, 0) + degrees.get(dst, 0)) + target_bonus + relation_bonus


def _read_link_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _node_type_by_id(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[int(parts[0])] = int(parts[2])
    return out


def _degree_counts(lines: Iterable[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        src, dst = int(parts[0]), int(parts[1])
        out[src] = out.get(src, 0) + 1
        out[dst] = out.get(dst, 0) + 1
    return out


def _relation_first_seen_order(path: Path) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] not in seen:
                seen.add(parts[2])
                order.append(parts[2])
    return order


def _hgb_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).glob("*.dat") if item.is_file())


def _hash_float(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)
