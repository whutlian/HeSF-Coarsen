from __future__ import annotations

from pathlib import Path


def test_structural_budget_uses_semantic_ratio_not_raw_bytes() -> None:
    from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth

    row = annotate_budget_truth(
        {
            "requested_budget_type": "structural_storage_ratio",
            "requested_budget": 0.20,
            "semantic_structural_storage_ratio": 0.97,
            "raw_hgb_text_byte_ratio": 0.20,
            "actual_support_edge_ratio": 0.20,
        },
        tolerance=0.02,
    )

    assert row["budget_match_for_requested_metric"] is False
    assert row["budget_metric_used_for_match"] == "semantic_structural_storage_ratio"
    assert row["budget_match_failure_type"] == "budget_mismatch"
    assert "semantic_structural_storage_ratio" in row["budget_match_failure_reason"]


def test_full_fallback_rows_are_excluded_from_gate21_18_main_table() -> None:
    from hesf_coarsen.eval.official.gate21_18_decision import gate21_18_decision

    decision = gate21_18_decision(
        main_rows=[
            {
                "dataset": "ACM",
                "method": "ACM-Random-field20",
                "constraint_safe_fallback": True,
                "eligible_for_main_table": False,
                "eligible_for_compression_claim": False,
                "selected_edge_hash": "full_graph_hash",
                "success": True,
                "training_executed": True,
                "test_micro_f1_mean": 0.5,
                "test_macro_f1_mean": 0.4,
            }
        ],
        fallback_rows=[
            {
                "dataset": "ACM",
                "method": "ACM-Random-field20",
                "constraint_safe_fallback": True,
                "selected_edge_hash": "full_graph_hash",
            }
        ],
    )

    assert decision["NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"] is True
    assert decision["FULL_HASH_ROWS_ONLY_IN_SANITY_TABLE"] is True


def test_acm_closure_compression_exports_nonfallback_consistent_graph(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.acm_closure_compression import (
        audit_acm_closure_export,
        export_acm_closure_compressed,
    )

    source_dir = tmp_path / "source" / "ACM"
    export_dir = tmp_path / "export" / "ACM"
    _write_synthetic_acm(source_dir)

    manifest = export_acm_closure_compressed(
        source_dir,
        export_dir,
        method="coverage_greedy",
        keyword_ratio=0.50,
        graph_seed=1,
    )
    audit = audit_acm_closure_export(export_dir, source_dir=source_dir)

    assert manifest["constraint_safe_fallback"] is False
    assert 0.0 < manifest["keyword_feature_ratio"] < 1.0
    assert 0.0 < manifest["PK_edge_ratio"] < 1.0
    assert audit["P_matches_PK"] is True
    assert audit["A_matches_AP_PK"] is True
    assert audit["C_matches_CP_PK"] is True
    assert audit["PK_KP_reciprocal"] is True
    author_rows = _read_node_rows_by_type(export_dir / "node.dat", 1)
    conference_rows = _read_node_rows_by_type(export_dir / "node.dat", 2)
    assert max(author_rows[2]) <= 1.0
    assert max(conference_rows[3]) <= 1.0


def test_imdb_constraint_compression_exports_nonfallback_consistent_graph(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.imdb_constraint_compression import (
        audit_imdb_constraint_export,
        export_imdb_constraint_compressed,
    )

    source_dir = tmp_path / "source" / "IMDB"
    export_dir = tmp_path / "export" / "IMDB"
    _write_synthetic_imdb(source_dir)

    manifest = export_imdb_constraint_compressed(
        source_dir,
        export_dir,
        method="random",
        actor_ratio=0.50,
        keyword_ratio=0.50,
        graph_seed=1,
    )
    audit = audit_imdb_constraint_export(export_dir, source_dir=source_dir)

    assert manifest["constraint_safe_fallback"] is False
    assert 0.0 < manifest["actual_support_edge_ratio"] < 1.0
    assert 0.0 < manifest["semantic_structural_storage_ratio"] < 1.0
    assert audit["MD_DM_reciprocal"] is True
    assert audit["MA_AM_reciprocal"] is True
    assert audit["MK_KM_reciprocal"] is True
    assert audit["movie_single_director_constraint_pass"] is True
    assert max(_touched_destination_ids(export_dir / "link.dat", relation_id=2)) == 5
    assert max(_touched_destination_ids(export_dir / "link.dat", relation_id=4)) == 7


def test_gate21_18_rep_selection_accepts_dataset_prefixed_hesf_auto_rows() -> None:
    from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_18_representatives

    reps = select_gate21_18_representatives(
        [
            {
                "dataset": "ACM",
                "method": "ACM-HeSF-RCS-auto-field20",
                "eligible_for_main_table": True,
                "success": True,
                "training_executed": True,
                "validation_micro_f1_mean": 0.96,
                "validation_macro_f1_mean": 0.95,
                "test_micro_f1_mean": 0.94,
                "test_macro_f1_mean": 0.93,
            }
        ],
        datasets=["ACM"],
    )

    main = next(row for row in reps if row["method"] == "HeSF-RCS-Rep-Validated")
    assert main["source_method"] == "ACM-HeSF-RCS-auto-field20"
    assert main["selection_source"] == "actual_validation"
    assert main["uses_test_for_selection"] is False


def _write_synthetic_acm(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        (0, "p0", 0, "1,1,0"),
        (1, "p1", 0, "0,1,1"),
        (2, "a0", 1, "1,2,1"),
        (3, "c0", 2, "1,2,1"),
        (4, "k0", 3, None),
        (5, "k1", 3, None),
        (6, "k2", 3, None),
    ]
    _write_node_dat(path / "node.dat", rows)
    _write_lines(
        path / "link.dat",
        [
            "0\t1\t0\t1\n",
            "1\t0\t1\t1\n",
            "0\t2\t2\t1\n",
            "2\t0\t3\t1\n",
            "2\t0\t3\t1\n",
            "1\t2\t2\t1\n",
            "2\t1\t3\t1\n",
            "0\t3\t4\t1\n",
            "3\t0\t5\t1\n",
            "1\t3\t4\t1\n",
            "3\t1\t5\t1\n",
            "0\t4\t6\t1\n",
            "4\t0\t7\t1\n",
            "0\t5\t6\t1\n",
            "5\t0\t7\t1\n",
            "1\t5\t6\t1\n",
            "5\t1\t7\t1\n",
            "1\t6\t6\t1\n",
            "6\t1\t7\t1\n",
        ],
    )
    _write_lines(path / "label.dat", ["0\tp0\t0\t0\n"])
    _write_lines(path / "label.dat.test", ["1\tp1\t0\t1\n"])
    (path / "info.dat").write_text("synthetic ACM\n", encoding="utf-8")


def _write_synthetic_imdb(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        (0, "m0", 0, "0.1,0.2"),
        (1, "m1", 0, "0.3,0.4"),
        (2, "d0", 1, "1,0"),
        (3, "d1", 1, "0,1"),
        (4, "a0", 2, "1,0"),
        (5, "a1", 2, "0,1"),
        (6, "k0", 3, None),
        (7, "k1", 3, None),
    ]
    _write_node_dat(path / "node.dat", rows)
    _write_lines(
        path / "link.dat",
        [
            "0\t2\t0\t1\n",
            "2\t0\t1\t1\n",
            "1\t3\t0\t1\n",
            "3\t1\t1\t1\n",
            "0\t4\t2\t1\n",
            "4\t0\t3\t1\n",
            "1\t5\t2\t1\n",
            "5\t1\t3\t1\n",
            "0\t6\t4\t1\n",
            "6\t0\t5\t1\n",
            "1\t7\t4\t1\n",
            "7\t1\t5\t1\n",
        ],
    )
    _write_lines(path / "label.dat", ["0\tm0\t0\t0\n"])
    _write_lines(path / "label.dat.test", ["1\tm1\t0\t1\n"])
    (path / "info.dat").write_text("synthetic IMDB\n", encoding="utf-8")


def _write_node_dat(path: Path, rows: list[tuple[int, str, int, str | None]]) -> None:
    lines = []
    for node_id, name, node_type, attr in rows:
        if attr is None:
            lines.append(f"{node_id}\t{name}\t{node_type}\n")
        else:
            lines.append(f"{node_id}\t{name}\t{node_type}\t{attr}\n")
    _write_lines(path, lines)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _read_node_rows_by_type(path: Path, node_type: int) -> dict[int, list[float]]:
    rows: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and int(parts[2]) == node_type:
                rows[int(parts[0])] = [float(value) for value in parts[3].split(",")]
    return rows


def _touched_destination_ids(path: Path, relation_id: int) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and int(parts[2]) == relation_id:
                ids.add(int(parts[1]))
    return ids
