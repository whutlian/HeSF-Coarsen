from __future__ import annotations

import tempfile
from pathlib import Path


def _write_toy_dblp_hgb_export(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "node.dat").write_text(
        "\n".join(
            [
                "0\tA0\t0\t1,0",
                "1\tA1\t0\t0,1",
                "2\tP0\t1\t1,0",
                "3\tP1\t1\t0,1",
                "4\tT0\t2",
                "5\tV0\t3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (export_dir / "link.dat").write_text(
        "\n".join(
            [
                "0\t2\t0\t1.0",
                "2\t0\t1\t1.0",
                "2\t5\t3\t1.0",
                "5\t2\t5\t1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (export_dir / "label.dat").write_text("0\tA0\t0\t1\n", encoding="utf-8")
    (export_dir / "label.dat.test").write_text("1\tA1\t0\t0\n", encoding="utf-8")


def test_ap_retention_sanity_uses_real_hgb_node_link_schema_and_labels() -> None:
    from hesf_coarsen.eval.official.coverage_diagnostics_v2 import (
        compute_hgb_coverage_diagnostics_v2,
        coverage_sanity_assertion_rows,
        coverage_semantic_validation_pass,
    )

    with tempfile.TemporaryDirectory(dir=Path.cwd() / "build") as workdir:
        export_dir = Path(workdir) / "DBLP"
        _write_toy_dblp_hgb_export(export_dir)
        row = compute_hgb_coverage_diagnostics_v2(
            export_dir,
            dataset="DBLP",
            method="H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
            graph_seed=1,
            relation_keep_plan={"AP": 1.0, "PV": 1.0},
        )
    relation_retention_rows = [
        {"relation_name": "AP", "retained_edges": 1},
        {"relation_name": "PV", "retained_edges": 1},
    ]

    assertions = coverage_sanity_assertion_rows(row, relation_retention_rows=relation_retention_rows)

    assert row["num_target_authors"] == 2
    assert row["coverage_AP_edge_count"] == 1
    assert row["coverage_PV_edge_count"] == 1
    assert row["mean_AP_degree_per_author"] > 0
    assert row["fraction_target_authors_reaching_paper"] > 0
    assert row["fraction_target_authors_reaching_venue_via_AP_PV"] > 0
    assert row["num_venues_reached"] == 1
    assert row["label_dat_trainval_count"] == 1
    assert row["class_proxy_coverage_by_venue"] == {5: 1}
    assert coverage_semantic_validation_pass(assertions) is True

    by_name = {assertion["assertion_name"]: assertion for assertion in assertions}
    assert by_name["ap_retained_implies_positive_mean_ap_degree"]["pass"] is True
    assert by_name["ap100_implies_reaches_paper"]["pass"] is True
    assert by_name["ap100_pv100_implies_reaches_venue"]["pass"] is True
    assert by_name["coverage_ap_edge_count_matches_relation_retention"]["pass"] is True
    assert by_name["coverage_pv_edge_count_matches_relation_retention"]["pass"] is True
    assert by_name["node_type_offsets_match_node_dat_counts"]["pass"] is True
    assert by_name["relation_direction_matches_official_relation_name"]["pass"] is True
    assert by_name["isolated_target_authors_not_above_target_count"]["pass"] is True


def test_relation_retention_mismatch_fails_coverage_semantic_validation() -> None:
    from hesf_coarsen.eval.official.coverage_diagnostics_v2 import (
        compute_hgb_coverage_diagnostics_v2,
        coverage_sanity_assertion_rows,
        coverage_semantic_validation_pass,
    )

    with tempfile.TemporaryDirectory(dir=Path.cwd() / "build") as workdir:
        export_dir = Path(workdir) / "DBLP"
        _write_toy_dblp_hgb_export(export_dir)
        row = compute_hgb_coverage_diagnostics_v2(
            export_dir,
            dataset="DBLP",
            method="H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
            graph_seed=1,
            relation_keep_plan={"AP": 1.0, "PV": 1.0},
        )

    assertions = coverage_sanity_assertion_rows(
        row,
        relation_retention_rows=[
            {"relation_name": "AP", "retained_edges": 2},
            {"relation_name": "PV", "retained_edges": 1},
        ],
    )

    by_name = {assertion["assertion_name"]: assertion for assertion in assertions}
    assert by_name["coverage_ap_edge_count_matches_relation_retention"]["pass"] is False
    assert coverage_semantic_validation_pass(assertions) is False
