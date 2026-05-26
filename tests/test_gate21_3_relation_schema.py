from __future__ import annotations

from tests.gate21_3_helpers import tiny_dblp_graph, write_minimal_hgb_dir


def test_dblp_canonical_relation_pairs() -> None:
    from hesf_coarsen.eval.official.relation_schema import DBLP_RELATION_KEYS, build_relation_keys

    keys = build_relation_keys(tiny_dblp_graph(), dataset="DBLP")

    assert [key.official_relation_name for key in keys] == ["AP", "PA", "PT", "PV", "TP", "VP"]
    assert [key.official_relation_id for key in DBLP_RELATION_KEYS] == [0, 1, 2, 3, 4, 5]
    assert {key.relation_pair_name for key in keys} == {"AP_PA", "PT_TP", "PV_VP"}
    assert next(key for key in keys if key.official_relation_name == "TP").reciprocal_official_relation_name == "PT"


def test_relation_key_reciprocal_consistency() -> None:
    from hesf_coarsen.eval.official.relation_schema import assert_relation_key_reciprocals

    assert_relation_key_reciprocals(build_graph := tiny_dblp_graph(), dataset="DBLP")
    assert build_graph.relation_specs[0].name == "AP"


def test_official_relation_order_validation(tmp_path) -> None:
    from hesf_coarsen.eval.official.relation_schema import validate_hgb_relation_order

    hgb = tmp_path / "DBLP"
    counts = {0: 2, 1: 2, 2: 1, 3: 3, 4: 1, 5: 3}
    write_minimal_hgb_dir(hgb, relation_counts=counts)

    report = validate_hgb_relation_order(dataset="DBLP", dataset_dir=hgb, hgb_export_edge_counts={str(k): v for k, v in counts.items()})

    assert report["relation_order_matches_official"] is True
    assert report["node_type_order_matches_official"] is True
    assert report["link_dat_relation_counts_match_export_audit"] is True
    assert report["edge_count_by_relation"]["3"] == 3
