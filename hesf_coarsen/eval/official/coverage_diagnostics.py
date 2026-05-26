from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence


def _pairs(values: Sequence[tuple[int, int]] | None) -> list[tuple[int, int]]:
    return [(int(src), int(dst)) for src, dst in (values or [])]


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count / total)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * float(percentile)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    weight = index - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _bucket_counts(values: list[int]) -> dict[str, int]:
    buckets = {"zero": 0, "one": 0, "two_to_five": 0, "gt_five": 0}
    for value in values:
        if value <= 0:
            buckets["zero"] += 1
        elif value == 1:
            buckets["one"] += 1
        elif value <= 5:
            buckets["two_to_five"] += 1
        else:
            buckets["gt_five"] += 1
    return buckets


def compute_apv_coverage_diagnostics(
    *,
    method: str,
    graph_seed: int,
    relation_keep_plan: Mapping[str, float],
    num_authors: int,
    num_papers: int,
    num_venues: int,
    num_terms: int,
    relations: Mapping[str, Sequence[tuple[int, int]]],
    train_labels: Mapping[int, int] | None = None,
    val_labels: Mapping[int, int] | None = None,
    test_labels: Mapping[int, int] | None = None,
    graph_seed_1_relations: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    if test_labels:
        raise ValueError("coverage diagnostics must not receive test labels")

    ap = _pairs(relations.get("AP"))
    pa = _pairs(relations.get("PA"))
    pv = _pairs(relations.get("PV"))
    pt = _pairs(relations.get("PT"))

    papers_by_author: dict[int, set[int]] = defaultdict(set)
    for author, paper in ap:
        if 0 <= author < num_authors and 0 <= paper < num_papers:
            papers_by_author[author].add(paper)
    for paper, author in pa:
        if 0 <= author < num_authors and 0 <= paper < num_papers:
            papers_by_author[author].add(paper)

    venues_by_paper: dict[int, set[int]] = defaultdict(set)
    for paper, venue in pv:
        if 0 <= paper < num_papers and 0 <= venue < num_venues:
            venues_by_paper[paper].add(venue)

    terms_by_paper: dict[int, set[int]] = defaultdict(set)
    for paper, term in pt:
        if 0 <= paper < num_papers and 0 <= term < num_terms:
            terms_by_paper[paper].add(term)

    ap_authors = {author for author, _paper in ap if 0 <= author < num_authors}
    pa_authors = {author for _paper, author in pa if 0 <= author < num_authors}
    reached_papers = set().union(*papers_by_author.values()) if papers_by_author else set()
    reached_venues = {
        venue
        for paper in reached_papers
        for venue in venues_by_paper.get(int(paper), set())
    }
    reached_terms = {
        term
        for paper in reached_papers
        for term in terms_by_paper.get(int(paper), set())
    }
    authors_reaching_venue = {
        author
        for author, papers in papers_by_author.items()
        if any(venues_by_paper.get(paper) for paper in papers)
    }
    authors_reaching_term = {
        author
        for author, papers in papers_by_author.items()
        if any(terms_by_paper.get(paper) for paper in papers)
    }

    ap_degrees = [len(papers_by_author.get(author, set())) for author in range(num_authors)]
    pv_degrees = [len(venues_by_paper.get(paper, set())) for paper in range(num_papers)]
    known_labels = dict(train_labels or {})
    known_labels.update({int(node): int(label) for node, label in (val_labels or {}).items()})
    class_proxy_by_venue: dict[int, set[int]] = defaultdict(set)
    for author, label in known_labels.items():
        for paper in papers_by_author.get(int(author), set()):
            for venue in venues_by_paper.get(paper, set()):
                class_proxy_by_venue[int(venue)].add(int(label))

    current_edges = set(ap) | set(pa) | set(pv) | set(pt)
    seed1_edges: set[tuple[int, int]] = set()
    if graph_seed_1_relations is not None:
        for name in ["AP", "PA", "PV", "PT"]:
            seed1_edges.update(_pairs(graph_seed_1_relations.get(name)))
    union = current_edges | seed1_edges
    edge_jaccard = 1.0 if not union else float(len(current_edges & seed1_edges) / len(union))

    warnings: list[str] = []
    if _fraction(len(ap_authors), num_authors) < 1.0:
        warnings.append("ap_author_coverage_lt_1")
    if _fraction(len(reached_venues), num_venues) < 1.0 and num_venues > 0:
        warnings.append("venue_coverage_lt_1")

    return {
        "method": str(method),
        "graph_seed": int(graph_seed),
        "relation_keep_plan": dict(relation_keep_plan),
        "num_target_authors": int(num_authors),
        "fraction_target_authors_with_AP_edge": _fraction(len(ap_authors), num_authors),
        "fraction_target_authors_with_PA_edge": _fraction(len(pa_authors), num_authors),
        "fraction_target_authors_reaching_paper": _fraction(len(papers_by_author), num_authors),
        "fraction_target_authors_reaching_venue": _fraction(len(authors_reaching_venue), num_authors),
        "fraction_target_authors_reaching_term": _fraction(len(authors_reaching_term), num_authors),
        "num_isolated_target_authors": int(sum(1 for degree in ap_degrees if degree == 0)),
        "mean_AP_degree_per_author": float(mean(ap_degrees)) if ap_degrees else 0.0,
        "median_AP_degree_per_author": float(median(ap_degrees)) if ap_degrees else 0.0,
        "p10_AP_degree_per_author": _percentile(ap_degrees, 0.10),
        "p90_AP_degree_per_author": _percentile(ap_degrees, 0.90),
        "mean_PV_degree_per_paper": float(mean(pv_degrees)) if pv_degrees else 0.0,
        "venue_coverage_count": int(len(reached_venues)),
        "venue_coverage_fraction": _fraction(len(reached_venues), num_venues),
        "paper_coverage_count": int(len(reached_papers)),
        "paper_coverage_fraction": _fraction(len(reached_papers), num_papers),
        "edge_jaccard_vs_graph_seed_1": edge_jaccard,
        "class_proxy_coverage_by_venue": {venue: len(labels) for venue, labels in sorted(class_proxy_by_venue.items())},
        "class_proxy_coverage_by_paper_bucket": _bucket_counts(pv_degrees),
        "degree_bucket_coverage_author": _bucket_counts(ap_degrees),
        "degree_bucket_coverage_paper": _bucket_counts(pv_degrees),
        "coverage_warning_flags": ";".join(warnings),
        "coverage_used_test_labels": False,
    }
