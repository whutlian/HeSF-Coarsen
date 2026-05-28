from __future__ import annotations


def test_gate21_11_coverage_diagnostics_require_distributional_fields() -> None:
    from hesf_coarsen.eval.official.coverage_diagnostics import gate21_11_distributional_coverage_ready

    ready = {
        "fraction_target_authors_with_AP_edge": 1.0,
        "fraction_target_authors_reaching_venue_via_AP_PV": 0.9,
        "venue_degree_bucket_coverage_json": "{}",
        "paper_degree_bucket_coverage_json": "{}",
        "author_degree_bucket_coverage_json": "{}",
        "per_class_venue_coverage_json": "{}",
        "per_class_paper_coverage_json": "{}",
        "venue_class_proxy_purity_trainval": 0.5,
        "paper_class_proxy_purity_trainval": 0.5,
        "coverage_edge_count_matches_relation_retention": True,
        "node_type_offsets_match_node_dat_counts": True,
        "relation_direction_matches_official_relation_name": True,
    }

    assert gate21_11_distributional_coverage_ready([ready])
    not_ready = dict(ready)
    not_ready["per_class_venue_coverage_json"] = ""
    assert not gate21_11_distributional_coverage_ready([not_ready])
