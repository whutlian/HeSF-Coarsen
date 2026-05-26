from __future__ import annotations

from pathlib import Path


def test_loaded_relation_audit_reads_official_link_dat_counts_and_hashes(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.metapath_channel_audit import loaded_relation_audit_rows

    export_dir = tmp_path / "DBLP"
    export_dir.mkdir()
    (export_dir / "link.dat").write_text("0\t1\t0\t1.0\n1\t2\t3\t1.0\n", encoding="utf-8")

    rows = loaded_relation_audit_rows(
        dataset="DBLP",
        method="H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
        canonical_method="H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
        graph_seed=1,
        training_seed=1,
        export_dir=export_dir,
        expected_relation_counts={"0": 1, "3": 1},
    )

    by_name = {row["loaded_relation_name"]: row for row in rows}
    assert by_name["AP"]["loaded_edge_count"] == 1
    assert by_name["PV"]["loaded_edge_count"] == 1
    assert by_name["AP"]["loaded_edge_hash"]
    assert by_name["AP"]["loaded_count_matches_expected"] is True


def test_cache_sanity_detects_link_dat_perturbation(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.metapath_channel_audit import cache_sanity_row

    export_dir = tmp_path / "DBLP"
    export_dir.mkdir()
    (export_dir / "node.dat").write_text("0\tn0\t0\t1,0\n", encoding="utf-8")
    (export_dir / "link.dat").write_text("0\t1\t0\t1.0\n", encoding="utf-8")

    row = cache_sanity_row(export_dir)

    assert row["link_dat_hash_changed"] is True
    assert row["loaded_link_hash_inside_sehgnn_runner_changed"] is True
    assert row["loaded_relation_hash_changed"] is True
