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


def test_apv_skeleton_alias_keeps_machine_readable_spec() -> None:
    from hesf_coarsen.eval.official.relation_channel_skeleton import canonicalize_gate21_4_method

    spec = canonicalize_gate21_4_method("APV-skeleton")

    assert spec["method"] == "H6-APV-skeleton"
    assert spec["canonical_method"] == "H6-relgrid-APPA100-PVVP100-PTTP00"
    assert spec["relation_channel_spec"] == "APPA100-PVVP100-PTTP00"
    assert spec["budget_strategy"] == "relation_channel_skeleton"
    assert spec["edge_score_strategy"] == "keep_full_APPA_PVVP_min_schema_PTTP"


def test_apv_skeleton_retains_min_schema_term_edges() -> None:
    from hesf_coarsen.eval.official.relation_channel_skeleton import allocate_skeleton_budget

    rows = allocate_skeleton_budget(_stats(), "APPA100-PVVP100-PTTP00", min_edges_per_relation=1)

    retained = {row.relation_name: row.actual_edges for row in rows}
    assert retained == {"AP": 100, "PA": 100, "PT": 1, "PV": 20, "TP": 1, "VP": 20}
    assert {row.relation_name for row in rows if row.min_edges_constraint_active} == {"PT", "TP"}


def test_pttp05_uses_rounded_candidate_budget() -> None:
    from hesf_coarsen.eval.official.relation_channel_skeleton import allocate_skeleton_budget

    rows = allocate_skeleton_budget(_stats(), "APPA100-PVVP100-PTTP05", min_edges_per_relation=1)

    retained = {row.relation_name: row.actual_edges for row in rows}
    assert retained["PT"] == 50
    assert retained["TP"] == 50


def test_directionality_spec_detected() -> None:
    from hesf_coarsen.eval.official.relation_channel_skeleton import canonicalize_gate21_4_method

    spec = canonicalize_gate21_4_method("AP100-PA00-PV100-VP100-PT00-TP00")

    assert spec["is_directionality_ablation"] is True
    assert spec["relation_channel_spec"] == "AP100-PA00-PV100-VP100-PT00-TP00"
