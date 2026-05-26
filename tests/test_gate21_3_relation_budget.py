from __future__ import annotations

from hesf_coarsen.eval.official.relation_budget_allocator import RelationStats


def _stats() -> list[RelationStats]:
    return [
        RelationStats(0, "AP", "AP_PA", "A", "P", 100, 10, 1),
        RelationStats(1, "PA", "AP_PA", "P", "A", 100, 10, 1),
        RelationStats(2, "PT", "PT_TP", "P", "T", 100, 20, 1),
        RelationStats(3, "PV", "PV_VP", "P", "V", 100, 4, 1),
        RelationStats(4, "TP", "PT_TP", "T", "P", 100, 20, 1),
        RelationStats(5, "VP", "PV_VP", "V", "P", 100, 4, 1),
    ]


def test_parse_relation_channel_spec_APPA100_PVVP100_PTTP30() -> None:
    from hesf_coarsen.eval.official.relation_budget_allocator import parse_relation_channel_spec

    parsed = parse_relation_channel_spec("APPA100-PVVP100-PTTP30")

    assert parsed.retention_by_relation == {"AP": 1.0, "PA": 1.0, "PV": 1.0, "VP": 1.0, "PT": 0.30, "TP": 0.30}
    assert all(spec.sampling_strategy == "random" for spec in parsed.pair_specs)


def test_pairgrid_actual_budget_matches_requested() -> None:
    from hesf_coarsen.eval.official.relation_budget_allocator import allocate_relation_channel_spec, parse_relation_channel_spec

    allocation = allocate_relation_channel_spec(_stats(), parse_relation_channel_spec("APPA100-PVVP100-PTTP30"))
    actual = {row.relation_name: row.actual_edges for row in allocation}

    assert actual == {"AP": 10, "PA": 10, "PT": 6, "PV": 4, "TP": 6, "VP": 4}
    assert sum(actual.values()) == 40


def test_direction_specific_budget_parse() -> None:
    from hesf_coarsen.eval.official.relation_budget_allocator import parse_relation_channel_spec

    parsed = parse_relation_channel_spec("AP100-PA50-PT50-TP20-PV100-VP100")

    assert parsed.retention_by_relation["AP"] == 1.0
    assert parsed.retention_by_relation["PA"] == 0.50
    assert parsed.retention_by_relation["PT"] == 0.50
    assert parsed.retention_by_relation["TP"] == 0.20


def test_min_edges_constraint_records_flag() -> None:
    from hesf_coarsen.eval.official.relation_budget_allocator import allocate_relation_channel_spec, parse_relation_channel_spec

    allocation = allocate_relation_channel_spec(_stats(), parse_relation_channel_spec("APPA00-PVVP00-PTTP00"), min_edges_per_relation=1)

    assert all(row.actual_edges == 1 for row in allocation)
    assert all(row.min_edges_constraint_active for row in allocation)
