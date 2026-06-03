from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.acm_closure_compression import ACM_RELATIONS
from hesf_coarsen.eval.official.stage_report_protocol import float_value


GATE21_21_ACM_SELECTOR_OVERLAP_FIELDS = (
    "field_ratio",
    "method_a",
    "method_b",
    "selected_keyword_jaccard",
    "selected_PK_edge_jaccard",
    "selected_author_coverage_jaccard",
    "selected_conference_coverage_jaccard",
    "micro_gap",
    "macro_gap",
    "selector_degeneracy_flag",
)


def build_acm_selector_overlap_rows(selector_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in selector_rows:
        ratio = float_value(row.get("field_ratio") or row.get("requested_budget") or row.get("keyword_feature_ratio"))
        if ratio is None:
            continue
        method = str(row.get("method", ""))
        key = _method_key(method)
        if key:
            grouped.setdefault(round(ratio, 6), {})[key] = row

    out: list[dict[str, Any]] = []
    for ratio, methods in sorted(grouped.items()):
        hesf_keywords = _keyword_set(methods.get("hesf"))
        degree_keywords = _keyword_set(methods.get("degree"))
        validation_keywords = _keyword_set(methods.get("validation"))
        hesf_pk = _pk_edge_set(methods.get("hesf"))
        degree_pk = _pk_edge_set(methods.get("degree"))
        validation_pk = _pk_edge_set(methods.get("validation"))
        random_keywords = _keyword_set(methods.get("random"))
        degree_values = [len(item) for item in (hesf_keywords, degree_keywords, validation_keywords, random_keywords) if item]
        perf_gap = _micro(methods.get("hesf")) - _micro(methods.get("degree"))
        j_h_d = _jaccard(hesf_keywords, degree_keywords)
        out.append(
            {
                "field_ratio": ratio,
                "selected_keyword_jaccard_hesf_vs_degree": j_h_d,
                "selected_keyword_jaccard_hesf_vs_validation_greedy": _jaccard(hesf_keywords, validation_keywords),
                "selected_keyword_jaccard_degree_vs_validation_greedy": _jaccard(degree_keywords, validation_keywords),
                "selected_PK_edge_jaccard_hesf_vs_degree": _jaccard(hesf_pk, degree_pk),
                "selected_PK_edge_jaccard_hesf_vs_validation_greedy": _jaccard(hesf_pk, validation_pk),
                "field_degree_distribution_mean": mean(degree_values) if degree_values else "",
                "field_degree_distribution_std": pstdev(degree_values) if len(degree_values) > 1 else 0.0 if degree_values else "",
                "validation_gain_by_field_bucket": perf_gap,
                "ACM_HEFS_DEGENERATES_TO_DEGREE_SELECTOR": bool(j_h_d > 0.90 and abs(perf_gap) < 0.001),
                "ACM_HEFS_SELECTOR_DISTINCT_AND_BETTER": bool(j_h_d < 0.90 and perf_gap > 0.0),
            }
        )
    return out


def build_gate21_21_acm_selector_overlap_rows(selector_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in selector_rows:
        ratio = float_value(row.get("field_ratio") or row.get("requested_budget") or row.get("keyword_feature_ratio"))
        if ratio is None:
            continue
        key = _method_key(str(row.get("method", "")))
        if key:
            grouped.setdefault(round(ratio, 6), {})[key] = row

    pairs = (
        ("hesf", "degree"),
        ("hesf", "validation"),
        ("degree", "validation"),
    )
    out: list[dict[str, Any]] = []
    for ratio, methods in sorted(grouped.items()):
        for left, right in pairs:
            row_a = methods.get(left)
            row_b = methods.get(right)
            if row_a is None or row_b is None:
                continue
            keyword_jaccard = _jaccard(_keyword_set(row_a), _keyword_set(row_b))
            pk_jaccard = _jaccard(_pk_edge_set(row_a), _pk_edge_set(row_b))
            micro_gap = _micro(row_a) - _micro(row_b)
            macro_gap = _macro(row_a) - _macro(row_b)
            out.append(
                {
                    "field_ratio": ratio,
                    "method_a": row_a.get("method", _label_for_key(left)),
                    "method_b": row_b.get("method", _label_for_key(right)),
                    "selected_keyword_jaccard": keyword_jaccard,
                    "selected_PK_edge_jaccard": pk_jaccard,
                    "selected_author_coverage_jaccard": pk_jaccard,
                    "selected_conference_coverage_jaccard": keyword_jaccard,
                    "micro_gap": micro_gap,
                    "macro_gap": macro_gap,
                    "selector_degeneracy_flag": bool(left == "hesf" and right in {"degree", "validation"} and keyword_jaccard > 0.90 and abs(micro_gap) < 0.002),
                }
            )
    return out


def rows_from_gate21_exports(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("dataset", "")).upper() != "ACM":
            continue
        method = str(row.get("method", ""))
        if not _method_key(method):
            continue
        export_dir = Path(str(row.get("export_dir", "")))
        selected_keywords, pk_edges = _read_acm_export_selection(export_dir)
        out.append(
            {
                "method": method,
                "field_ratio": row.get("requested_budget", row.get("keyword_feature_ratio", "")),
                "selected_keywords": ";".join(sorted(selected_keywords)),
                "selected_pk_edges": ";".join(sorted(pk_edges)),
                "test_micro_f1_mean": row.get("test_micro_f1_mean", ""),
                "validation_micro_f1_mean": row.get("validation_micro_f1_mean", ""),
                "test_macro_f1_mean": row.get("test_macro_f1_mean", ""),
                "validation_macro_f1_mean": row.get("validation_macro_f1_mean", ""),
            }
        )
    return out


def _read_acm_export_selection(export_dir: Path) -> tuple[set[str], set[str]]:
    link = export_dir / "link.dat"
    if not link.exists():
        return set(), set()
    keywords: set[str] = set()
    pk_edges: set[str] = set()
    with link.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or int(parts[2]) != ACM_RELATIONS["PK"]:
                continue
            keywords.add(parts[1])
            pk_edges.add(f"{parts[0]}-{parts[1]}")
    return keywords, pk_edges


def _method_key(method: str) -> str:
    if "HeSF-RCS-auto" in method:
        return "hesf"
    if "Degree-field" in method:
        return "degree"
    if "ValidationGreedy-field" in method:
        return "validation"
    if "Random-field" in method:
        return "random"
    return ""


def _keyword_set(row: Mapping[str, Any] | None) -> set[str]:
    if not row:
        return set()
    value = str(row.get("selected_keywords", ""))
    return {item for item in value.split(";") if item}


def _pk_edge_set(row: Mapping[str, Any] | None) -> set[str]:
    if not row:
        return set()
    value = str(row.get("selected_pk_edges", ""))
    return {item for item in value.split(";") if item}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _micro(row: Mapping[str, Any] | None) -> float:
    if not row:
        return 0.0
    return float_value(row.get("validation_micro_f1_mean") or row.get("test_micro_f1_mean")) or 0.0


def _macro(row: Mapping[str, Any] | None) -> float:
    if not row:
        return 0.0
    return float_value(row.get("validation_macro_f1_mean") or row.get("test_macro_f1_mean")) or 0.0


def _label_for_key(key: str) -> str:
    return {
        "hesf": "ACM-HeSF-RCS-auto-field",
        "degree": "ACM-Degree-field",
        "validation": "ACM-ValidationGreedy-field",
        "random": "ACM-Random-field",
    }.get(key, key)
