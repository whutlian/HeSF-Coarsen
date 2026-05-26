from __future__ import annotations


def test_ap_pv_coverage_uses_structure_and_train_val_label_proxies_only() -> None:
    from hesf_coarsen.eval.official.coverage_diagnostics import compute_apv_coverage_diagnostics

    row = compute_apv_coverage_diagnostics(
        method="toy-apv",
        graph_seed=1,
        relation_keep_plan={"AP": 1.0, "PV": 1.0},
        num_authors=3,
        num_papers=4,
        num_venues=2,
        num_terms=2,
        relations={
            "AP": [(0, 0), (0, 1), (1, 2)],
            "PA": [(0, 0), (1, 0), (2, 1)],
            "PV": [(0, 0), (1, 1), (2, 1)],
            "PT": [(0, 0), (2, 1)],
        },
        train_labels={0: 1},
        val_labels={1: 2},
    )

    assert row["num_target_authors"] == 3
    assert row["fraction_target_authors_with_AP_edge"] == 2 / 3
    assert row["fraction_target_authors_reaching_paper"] == 2 / 3
    assert row["fraction_target_authors_reaching_venue"] == 2 / 3
    assert row["paper_coverage_count"] == 3
    assert row["paper_coverage_fraction"] == 3 / 4
    assert row["venue_coverage_count"] == 2
    assert row["venue_coverage_fraction"] == 1.0
    assert row["class_proxy_coverage_by_venue"] == {0: 1, 1: 2}
    assert row["coverage_used_test_labels"] is False


def test_coverage_diagnostics_reject_test_labels() -> None:
    from hesf_coarsen.eval.official.coverage_diagnostics import compute_apv_coverage_diagnostics

    try:
        compute_apv_coverage_diagnostics(
            method="bad",
            graph_seed=1,
            relation_keep_plan={},
            num_authors=1,
            num_papers=1,
            num_venues=0,
            num_terms=0,
            relations={},
            train_labels={},
            val_labels={},
            test_labels={0: 1},
        )
    except ValueError as exc:
        assert "test labels" in str(exc)
    else:
        raise AssertionError("coverage diagnostics accepted test labels")
