from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS, supported_sehgnn_hgb_dataset


DBLP_RELATION_ENDPOINTS = {
    "AP": (0, 1),
    "PA": (1, 0),
    "PT": (1, 2),
    "PV": (1, 3),
    "TP": (2, 1),
    "VP": (3, 1),
}


def compute_hgb_coverage_diagnostics_v2(
    export_dir: str | Path,
    *,
    dataset: str = "DBLP",
    method: str,
    graph_seed: int,
    relation_keep_plan: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    if dataset_name != "DBLP":
        raise ValueError("Gate21.7 coverage diagnostics v2 currently implement DBLP AP/PV semantics")

    root = Path(export_dir)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    nodes = _parse_node_dat(root / "node.dat")
    links = _parse_link_dat(root / "link.dat", schema=schema, node_type_by_id=nodes)
    trainval_labels = _parse_label_dat(root / "label.dat")
    test_labels = _parse_label_dat(root / "label.dat.test")

    target_type = int(schema["node_type_order"]["A"])
    paper_type = int(schema["node_type_order"]["P"])
    venue_type = int(schema["node_type_order"]["V"])
    target_authors = {node_id for node_id, node_type in nodes.items() if node_type == target_type}
    papers = {node_id for node_id, node_type in nodes.items() if node_type == paper_type}
    venues = {node_id for node_id, node_type in nodes.items() if node_type == venue_type}

    edges_by_name = links["edges_by_name"]
    ap_edges = edges_by_name.get("AP", [])
    pa_edges = edges_by_name.get("PA", [])
    pv_edges = edges_by_name.get("PV", [])
    vp_edges = edges_by_name.get("VP", [])

    ap_papers_by_author: dict[int, set[int]] = defaultdict(set)
    papers_by_author: dict[int, set[int]] = defaultdict(set)
    for author, paper in ap_edges:
        if author in target_authors and paper in papers:
            ap_papers_by_author[author].add(paper)
            papers_by_author[author].add(paper)
    pa_authors = set()
    for paper, author in pa_edges:
        if author in target_authors and paper in papers:
            pa_authors.add(author)
            papers_by_author[author].add(paper)

    pv_venues_by_paper: dict[int, set[int]] = defaultdict(set)
    venues_by_paper: dict[int, set[int]] = defaultdict(set)
    for paper, venue in pv_edges:
        if paper in papers and venue in venues:
            pv_venues_by_paper[paper].add(venue)
            venues_by_paper[paper].add(venue)
    for venue, paper in vp_edges:
        if paper in papers and venue in venues:
            venues_by_paper[paper].add(venue)

    ap_degrees = [len(ap_papers_by_author.get(author, set())) for author in sorted(target_authors)]
    papers_reached = set().union(*papers_by_author.values()) if papers_by_author else set()
    authors_reaching_paper = {author for author, reached in papers_by_author.items() if reached}
    authors_reaching_venue_ap_pv = {
        author
        for author, reached in ap_papers_by_author.items()
        if any(pv_venues_by_paper.get(paper) for paper in reached)
    }
    authors_reaching_venue_any = {
        author
        for author, reached in papers_by_author.items()
        if any(venues_by_paper.get(paper) for paper in reached)
    }
    venues_reached = {
        venue
        for paper in papers_reached
        for venue in venues_by_paper.get(int(paper), set())
    }
    papers_with_pv = {paper for paper, _venue in pv_edges if paper in papers}
    labels_by_venue = _class_proxy_coverage_by_venue(trainval_labels, papers_by_author, venues_by_paper)

    num_target = len(target_authors)
    num_reached_papers_with_pv = len(papers_reached & papers_with_pv)
    return {
        "dataset": dataset_name,
        "method": str(method),
        "graph_seed": int(graph_seed),
        "relation_keep_plan": dict(relation_keep_plan or {}),
        "num_target_authors": int(num_target),
        "num_target_authors_with_AP_out_edge": int(sum(1 for value in ap_degrees if value > 0)),
        "num_target_authors_with_PA_in_edge": int(len(pa_authors)),
        "fraction_target_authors_with_AP_edge": _fraction(sum(1 for value in ap_degrees if value > 0), num_target),
        "fraction_target_authors_reaching_paper": _fraction(len(authors_reaching_paper), num_target),
        "fraction_target_authors_reaching_venue_via_AP_PV": _fraction(len(authors_reaching_venue_ap_pv), num_target),
        "fraction_target_authors_reaching_venue_via_AP_PV_or_PA_VP": _fraction(len(authors_reaching_venue_any), num_target),
        "num_isolated_target_authors": int(sum(1 for author in target_authors if not papers_by_author.get(author))),
        "mean_AP_degree_per_author": float(mean(ap_degrees)) if ap_degrees else 0.0,
        "median_AP_degree_per_author": float(median(ap_degrees)) if ap_degrees else 0.0,
        "p10_AP_degree_per_author": _percentile(ap_degrees, 0.10),
        "p90_AP_degree_per_author": _percentile(ap_degrees, 0.90),
        "num_papers_reached_from_target_authors": int(len(papers_reached)),
        "num_papers_with_PV_edge": int(len(papers_with_pv)),
        "fraction_reached_papers_with_PV_edge": _fraction(num_reached_papers_with_pv, len(papers_reached)),
        "num_venues_reached": int(len(venues_reached)),
        "venue_coverage_fraction": _fraction(len(venues_reached), len(venues)),
        "paper_degree_quantiles_after_pruning": _quantile_summary(_paper_pv_degrees(papers, pv_venues_by_paper)),
        "venue_degree_quantiles_after_pruning": _quantile_summary(_venue_degrees(venues, pv_edges, vp_edges)),
        "class_proxy_coverage_by_venue": labels_by_venue,
        "coverage_AP_edge_count": int(len(ap_edges)),
        "coverage_PV_edge_count": int(len(pv_edges)),
        "coverage_relation_edge_count_by_name": dict(links["edge_count_by_name"]),
        "node_count_by_type": dict(Counter(nodes.values())),
        "node_type_offsets_match_node_dat_counts": _node_type_offsets_match_node_dat_counts(nodes),
        "relation_direction_matches_official_relation_name": bool(links["relation_direction_matches_official_relation_name"]),
        "relation_direction_failure_count": int(links["relation_direction_failure_count"]),
        "label_dat_trainval_count": int(len(trainval_labels)),
        "label_dat_test_count": int(len(test_labels)),
        "coverage_used_test_labels": False,
    }


def coverage_sanity_assertion_rows(
    coverage_row: Mapping[str, Any],
    *,
    relation_retention_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ap_retained = _relation_retained_edges(relation_retention_rows, "AP")
    pv_retained = _relation_retained_edges(relation_retention_rows, "PV")
    relation_keep_plan = _coerce_keep_plan(coverage_row.get("relation_keep_plan", {}))
    method = str(coverage_row.get("method", ""))
    ap100 = _plan_keeps_relation(method, relation_keep_plan, "AP")
    pv100 = _plan_keeps_relation(method, relation_keep_plan, "PV")

    rows: list[dict[str, Any]] = []
    _append_assertion(
        rows,
        "ap_retained_implies_positive_mean_ap_degree",
        bool((ap_retained or 0) <= 0 or float(coverage_row.get("mean_AP_degree_per_author", 0.0) or 0.0) > 0.0),
        observed=coverage_row.get("mean_AP_degree_per_author", 0.0),
        expected=">0 when AP retained_edges > 0",
    )
    _append_assertion(
        rows,
        "ap100_implies_reaches_paper",
        bool(not ap100 or float(coverage_row.get("fraction_target_authors_reaching_paper", 0.0) or 0.0) > 0.0),
        observed=coverage_row.get("fraction_target_authors_reaching_paper", 0.0),
        expected=">0 when AP100",
    )
    _append_assertion(
        rows,
        "ap100_pv100_implies_reaches_venue",
        bool(not (ap100 and pv100) or float(coverage_row.get("fraction_target_authors_reaching_venue_via_AP_PV", 0.0) or 0.0) > 0.0),
        observed=coverage_row.get("fraction_target_authors_reaching_venue_via_AP_PV", 0.0),
        expected=">0 when AP100 and PV100",
    )
    _append_assertion(
        rows,
        "coverage_ap_edge_count_matches_relation_retention",
        bool(ap_retained is not None and int(coverage_row.get("coverage_AP_edge_count", -1)) == int(ap_retained)),
        observed=coverage_row.get("coverage_AP_edge_count", ""),
        expected=ap_retained if ap_retained is not None else "relation_retention_AP_required",
    )
    _append_assertion(
        rows,
        "coverage_pv_edge_count_matches_relation_retention",
        bool(pv_retained is not None and int(coverage_row.get("coverage_PV_edge_count", -1)) == int(pv_retained)),
        observed=coverage_row.get("coverage_PV_edge_count", ""),
        expected=pv_retained if pv_retained is not None else "relation_retention_PV_required",
    )
    _append_assertion(
        rows,
        "node_type_offsets_match_node_dat_counts",
        bool(coverage_row.get("node_type_offsets_match_node_dat_counts", False)),
        observed=coverage_row.get("node_count_by_type", {}),
        expected="node.dat-derived contiguous type blocks",
    )
    _append_assertion(
        rows,
        "relation_direction_matches_official_relation_name",
        bool(coverage_row.get("relation_direction_matches_official_relation_name", False)),
        observed=coverage_row.get("relation_direction_failure_count", ""),
        expected=0,
    )
    _append_assertion(
        rows,
        "isolated_target_authors_not_above_target_count",
        int(coverage_row.get("num_isolated_target_authors", 0) or 0) <= int(coverage_row.get("num_target_authors", 0) or 0),
        observed=coverage_row.get("num_isolated_target_authors", 0),
        expected=f"<= {coverage_row.get('num_target_authors', 0)}",
    )
    return rows


def coverage_semantic_validation_pass(assertion_rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(assertion_rows) and all(bool(row.get("pass", False)) for row in assertion_rows)


def write_gate21_7_coverage_outputs(
    output_dir: str | Path,
    *,
    coverage_rows: Sequence[Mapping[str, Any]],
    assertion_rows: Sequence[Mapping[str, Any]],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "gate21_7_coverage_diagnostics_v2.csv", coverage_rows)
    _write_csv(root / "gate21_7_coverage_sanity_assertions.csv", assertion_rows)


def _parse_node_dat(path: Path) -> dict[int, int]:
    nodes: dict[int, int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"node.dat row has fewer than 3 fields: {line!r}")
            node_id = int(parts[0])
            node_type = int(parts[2])
            if node_id in nodes:
                raise ValueError(f"duplicate node.dat id: {node_id}")
            nodes[node_id] = node_type
    return nodes


def _parse_link_dat(path: Path, *, schema: Mapping[str, Any], node_type_by_id: Mapping[int, int]) -> dict[str, Any]:
    relation_name_by_id = {int(relation_id): name for name, relation_id in schema["relation_id_order"].items()}
    edges_by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    edge_count_by_name: Counter[str] = Counter()
    direction_failures = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"link.dat row has fewer than 3 fields: {line!r}")
            src = int(parts[0])
            dst = int(parts[1])
            relation_id = int(parts[2])
            relation_name = relation_name_by_id.get(relation_id, str(relation_id))
            edges_by_name[relation_name].append((src, dst))
            edge_count_by_name[relation_name] += 1
            expected = DBLP_RELATION_ENDPOINTS.get(relation_name)
            if expected is not None:
                src_type = node_type_by_id.get(src)
                dst_type = node_type_by_id.get(dst)
                if src_type != expected[0] or dst_type != expected[1]:
                    direction_failures += 1
    return {
        "edges_by_name": dict(edges_by_name),
        "edge_count_by_name": dict(edge_count_by_name),
        "relation_direction_matches_official_relation_name": direction_failures == 0,
        "relation_direction_failure_count": direction_failures,
    }


def _parse_label_dat(path: Path) -> dict[int, tuple[int, ...]]:
    labels: dict[int, tuple[int, ...]] = {}
    if not Path(path).exists():
        return labels
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"label.dat row has fewer than 4 fields: {line!r}")
            node_id = int(parts[0])
            labels[node_id] = tuple(int(value) for value in parts[3].split(",") if value != "")
    return labels


def _class_proxy_coverage_by_venue(
    trainval_labels: Mapping[int, tuple[int, ...]],
    papers_by_author: Mapping[int, set[int]],
    venues_by_paper: Mapping[int, set[int]],
) -> dict[int, int]:
    labels_by_venue: dict[int, set[int]] = defaultdict(set)
    for author, labels in trainval_labels.items():
        for paper in papers_by_author.get(int(author), set()):
            for venue in venues_by_paper.get(int(paper), set()):
                labels_by_venue[int(venue)].update(int(label) for label in labels)
    return {venue: len(labels) for venue, labels in sorted(labels_by_venue.items())}


def _node_type_offsets_match_node_dat_counts(nodes: Mapping[int, int]) -> bool:
    counts = Counter(nodes.values())
    total = sum(int(count) for count in counts.values())
    if total != len(nodes):
        return False
    offset = 0
    for type_id in sorted(counts):
        ids = {node_id for node_id, node_type in nodes.items() if node_type == type_id}
        expected_ids = set(range(offset, offset + int(counts[type_id])))
        if ids != expected_ids:
            return False
        offset += int(counts[type_id])
    return True


def _relation_retained_edges(rows: Sequence[Mapping[str, Any]], relation_name: str) -> int | None:
    for row in rows:
        name = row.get("relation_name", row.get("loaded_relation_name", row.get("canonical_relation_name", "")))
        if str(name) != relation_name:
            continue
        for key in ("retained_edges", "retained_edge_count", "edge_count_after", "loaded_edge_count", "actual_edges"):
            if key in row and row[key] not in {"", None}:
                return int(row[key])
    return None


def _append_assertion(
    rows: list[dict[str, Any]],
    assertion_name: str,
    pass_flag: bool,
    *,
    observed: Any,
    expected: Any,
) -> None:
    rows.append(
        {
            "assertion_name": assertion_name,
            "pass": bool(pass_flag),
            "observed": observed,
            "expected": expected,
            "failure_message": "" if pass_flag else f"{assertion_name} failed",
        }
    )


def _coerce_keep_plan(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, Mapping):
            return {str(k): float(v) for k, v in data.items()}
    return {}


def _plan_keeps_relation(method: str, relation_keep_plan: Mapping[str, float], relation_name: str) -> bool:
    if float(relation_keep_plan.get(relation_name, 0.0) or 0.0) >= 1.0:
        return True
    return bool(re.search(fr"{re.escape(relation_name)}100\b", method))


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count / total)


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * float(percentile)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    weight = index - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _quantile_summary(values: Sequence[int]) -> dict[str, float]:
    return {
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
    }


def _paper_pv_degrees(papers: set[int], venues_by_paper: Mapping[int, set[int]]) -> list[int]:
    return [len(venues_by_paper.get(int(paper), set())) for paper in sorted(papers)]


def _venue_degrees(venues: set[int], pv_edges: Sequence[tuple[int, int]], vp_edges: Sequence[tuple[int, int]]) -> list[int]:
    counts: Counter[int] = Counter()
    for _paper, venue in pv_edges:
        if venue in venues:
            counts[int(venue)] += 1
    for venue, _paper in vp_edges:
        if venue in venues:
            counts[int(venue)] += 1
    return [int(counts.get(int(venue), 0)) for venue in sorted(venues)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True)
    return value
