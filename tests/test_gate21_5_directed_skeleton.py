from __future__ import annotations

from hesf_coarsen.eval.official.relation_budget_allocator import RelationStats


def _stats() -> list[RelationStats]:
    return [
        RelationStats(0, "AP", "AP_PA", "A", "P", 100, 100, 1),
        RelationStats(1, "PA", "AP_PA", "P", "A", 100, 100, 1),
        RelationStats(2, "PT", "PT_TP", "P", "T", 1000, 1000, 1),
        RelationStats(3, "PV", "PV_VP", "P", "V", 20, 20, 1),
        RelationStats(4, "TP", "PT_TP", "T", "P", 1000, 1000, 1),
        RelationStats(5, "VP", "PV_VP", "V", "P", 20, 20, 1),
    ]


def test_directed_relation_spec_parser_supports_independent_directions() -> None:
    from hesf_coarsen.eval.official.directed_relation_skeleton import parse_directed_relation_spec

    spec = parse_directed_relation_spec("AP100-PA00-PV100-VP00-PTTP00")

    assert spec.retention_by_relation == {"AP": 1.0, "PA": 0.0, "PT": 0.0, "TP": 0.0, "PV": 1.0, "VP": 0.0}
    assert spec.canonical_spec == "AP100-PA00-PV100-VP00-PTTP00"
    assert spec.canonical_method == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"
    assert spec.deterministic is True


def test_directed_relation_spec_parser_supports_partial_reverse_rescue() -> None:
    from hesf_coarsen.eval.official.directed_relation_skeleton import parse_directed_relation_spec

    spec = parse_directed_relation_spec("AP100-PA50-PV100-VP50-PTTP10")

    assert spec.retention_by_relation["AP"] == 1.0
    assert spec.retention_by_relation["PA"] == 0.5
    assert spec.retention_by_relation["PV"] == 1.0
    assert spec.retention_by_relation["VP"] == 0.5
    assert spec.retention_by_relation["PT"] == 0.1
    assert spec.retention_by_relation["TP"] == 0.1


def test_directed_min_edge_preservation_marks_zero_relations_not_dropped() -> None:
    from hesf_coarsen.eval.official.directed_relation_skeleton import allocate_directed_relation_budget

    rows = allocate_directed_relation_budget(_stats(), "AP100-PA00-PV100-VP00-PTTP00", schema_min_edges=True)
    by_name = {row["relation_name"]: row for row in rows}

    assert by_name["AP"]["retained_edge_count"] == 100
    assert by_name["PV"]["retained_edge_count"] == 20
    for relation in ("PA", "PT", "TP", "VP"):
        assert by_name[relation]["retained_edge_count"] == 1
        assert by_name[relation]["min_edges_constraint_active"] is True
        assert by_name[relation]["relation_dropped_flag"] is False


def test_zero_edge_diagnostic_is_not_main_decision_eligible() -> None:
    from hesf_coarsen.eval.official.directed_relation_skeleton import allocate_directed_relation_budget

    rows = allocate_directed_relation_budget(_stats(), "AP100-PA00-PV100-VP00-PTTP00", schema_min_edges=False)
    by_name = {row["relation_name"]: row for row in rows}

    assert by_name["PA"]["retained_edge_count"] == 0
    assert by_name["PA"]["relation_dropped_flag"] is True
    assert by_name["PA"]["eligible_for_main_decision"] is False
