from __future__ import annotations


def test_gate21_20_rep_selection_keeps_hesf_pool_separate() -> None:
    from hesf_coarsen.eval.official.rep_selection import select_gate21_20_representatives

    rows = [
        _ready_row("DBLP", "GCond-score-TP-local", "external_tp_baseline", validation_micro=0.90, validation_macro=0.89, test_micro=0.88, test_macro=0.87),
        _ready_row("DBLP", "HeSF-RCS-auto structural16", "schema_preserving_rcs", validation_micro=0.81, validation_macro=0.80, test_micro=0.95, test_macro=0.94),
        _ready_row("DBLP", "HeSF-RCS-auto structural12", "schema_preserving_rcs", validation_micro=0.80, validation_macro=0.79, test_micro=0.94, test_macro=0.93),
    ]

    reps = select_gate21_20_representatives(rows, datasets=["DBLP"])
    hesf = _rep(reps, "DBLP", "HeSF-RCS-Rep-Validated")
    best = _rep(reps, "DBLP", "Best-Compressed-Validated")
    oracle = _rep(reps, "DBLP", "TestOracle-Best")

    assert hesf["selected_method"] == "HeSF-RCS-auto structural16"
    assert str(hesf["selected_method_family"]).startswith("schema_preserving")
    assert hesf["candidate_pool"] == "hesf_rcs_only"
    assert hesf["uses_test_for_selection"] is False
    assert hesf["eligible_for_main_decision"] is True
    assert best["selected_method"] == "GCond-score-TP-local"
    assert best["candidate_pool"] == "all_compressed"
    assert oracle["uses_test_for_selection"] is True
    assert oracle["eligible_for_main_decision"] is False


def test_gate21_20_rep_selection_reports_missing_hesf_validation() -> None:
    from hesf_coarsen.eval.official.rep_selection import select_gate21_20_representatives

    rows = [_ready_row("DBLP", "HeSF-RCS-auto structural16", "schema_preserving_rcs", validation_micro="", validation_macro="", test_micro=0.95, test_macro=0.94)]
    reps = select_gate21_20_representatives(rows, datasets=["DBLP"])
    hesf = _rep(reps, "DBLP", "HeSF-RCS-Rep-Validated")

    assert hesf["selected_method"] == ""
    assert hesf["selection_reason"] == "missing_real_validation_metric"
    assert hesf["uses_test_for_selection"] is False
    assert hesf["eligible_for_main_decision"] is False


def test_gate21_20_acm_overlap_and_imdb_upgrade_contracts() -> None:
    from hesf_coarsen.eval.official.acm_selector_overlap import build_acm_selector_overlap_rows
    from hesf_coarsen.eval.official.imdb_planner_upgrade import build_imdb_hesf_upgrade_rows

    acm_rows = [
        {"field_ratio": 0.20, "method": "ACM-HeSF-RCS-auto-field20", "selected_keywords": "1;2;3", "selected_pk_edges": "p1-k1;p2-k2"},
        {"field_ratio": 0.20, "method": "ACM-Degree-field20", "selected_keywords": "1;2;3", "selected_pk_edges": "p1-k1;p2-k2"},
        {"field_ratio": 0.20, "method": "ACM-ValidationGreedy-field20", "selected_keywords": "1;2;4", "selected_pk_edges": "p1-k1;p3-k4"},
    ]
    overlap = build_acm_selector_overlap_rows(acm_rows)
    assert overlap
    assert 0.0 <= float(overlap[0]["selected_keyword_jaccard_hesf_vs_degree"]) <= 1.0
    assert "selected_PK_edge_jaccard_hesf_vs_validation_greedy" in overlap[0]
    assert "ACM_HEFS_DEGENERATES_TO_DEGREE_SELECTOR" in overlap[0]

    gate19_rows = [
        _ready_row("IMDB", "IMDB-ValidationGreedy-channel40", "relation_structural_baseline", validation_micro=0.68, validation_macro=0.65, test_micro=0.68, test_macro=0.64, channel_edge_ratio=0.40, semantic=0.46),
        _ready_row("IMDB", "IMDB-ValidationGreedy-channel50", "relation_structural_baseline", validation_micro=0.69, validation_macro=0.66, test_micro=0.69, test_macro=0.66, channel_edge_ratio=0.50, semantic=0.56),
    ]
    upgraded = build_imdb_hesf_upgrade_rows(gate19_rows, budgets=[0.40, 0.50])
    methods = {row["method"] for row in upgraded}
    assert {"IMDB-HeSF-RCS-channel40", "IMDB-HeSF-RCS-channel50"}.issubset(methods)
    for row in upgraded:
        assert row["MD_keep"] == 1.0
        assert row["constraint_pass"] is True
        assert row["official_sehgnn_unmodified"] is True


def test_gate21_20_freehgc_selector_and_final_tables_contracts() -> None:
    from hesf_coarsen.eval.official.final_stage_report_tables import build_best_method_comparison, build_frontier_rows
    from hesf_coarsen.eval.official.freehgc_score_selector import build_freehgc_score_selector_plan_rows

    selector_rows = build_freehgc_score_selector_plan_rows(dataset="DBLP", budgets=[0.16, 0.20])
    assert {row["method"] for row in selector_rows} == {
        "FreeHGC-score-as-selector structural16",
        "FreeHGC-score-as-selector structural20",
    }
    assert all(row["eligible_for_main_table"] for row in selector_rows)

    rows = [
        _ready_row("DBLP", "Full-native-SeHGNN", "full_fidelity_baseline", validation_micro="", validation_macro="", test_micro=0.95, test_macro=0.94, semantic=1.0),
        _ready_row("DBLP", "HeSF-RCS-auto structural16", "schema_preserving_rcs", validation_micro=0.81, validation_macro=0.80, test_micro=0.94, test_macro=0.93, semantic=0.16),
        _ready_row("DBLP", "Random-edge-relwise", "relation_structural_baseline", validation_micro=0.70, validation_macro=0.69, test_micro=0.78, test_macro=0.77, semantic=0.20),
    ]
    comparison = build_best_method_comparison(rows, rep_rows=[{"dataset": "DBLP", "rep_type": "HeSF-RCS-Rep-Validated", "selected_method": "HeSF-RCS-auto structural16", "uses_test_for_selection": False}])
    assert any(row["role"] == "Best HeSF-RCS-Rep-Validated" and row["method"] == "HeSF-RCS-auto structural16" for row in comparison)
    frontier = build_frontier_rows(rows)
    assert {"pareto_frontier_flag", "micro_mean", "macro_mean"}.issubset(frontier[0])


def test_gate21_20_decision_flags_enforce_final_rules() -> None:
    from hesf_coarsen.eval.official.gate21_20_decision import GATE21_20_DECISION_FLAGS, gate21_20_decision

    rows = []
    for dataset in ("DBLP", "ACM", "IMDB"):
        rows.extend(
            [
                _ready_row(dataset, "Full-native-SeHGNN", "full_fidelity_baseline", validation_micro="", validation_macro="", semantic=1.0, support_edge=1.0),
                _ready_row(dataset, "Export-full-SeHGNN", "full_fidelity_baseline", validation_micro="", validation_macro="", semantic=1.0, support_edge=1.0),
            ]
        )
    rows.extend(
        [
            _ready_row("DBLP", "HeSF-RCS-auto structural16", "schema_preserving_rcs"),
            _ready_row("DBLP", "FreeHGC-score-as-selector structural16", "selector_probe"),
            _ready_row("DBLP", "FreeHGC-score-as-selector structural20", "selector_probe"),
            _ready_row("ACM", "ACM-HeSF-RCS-auto-field20", "schema_preserving_rcs"),
            _ready_row("IMDB", "IMDB-HeSF-RCS-channel50", "hesf_rcs", channel_edge_ratio=0.50),
        ]
    )
    rep_rows = [
        {
            "dataset": "DBLP",
            "rep_type": "HeSF-RCS-Rep-Validated",
            "selected_method": "HeSF-RCS-auto structural16",
            "selected_method_family": "schema_preserving_rcs",
            "uses_test_for_selection": False,
            "eligible_for_main_decision": True,
        }
    ]
    robustness = [
        {"dataset": "DBLP", "method": "HeSF-RCS-auto structural16", "training_executed_count": 3, "training_seed_count": 3, "failure_count": 0},
        {"dataset": "ACM", "method": "ACM-HeSF-RCS-auto-field20", "training_executed_count": 3, "training_seed_count": 3, "failure_count": 0},
        {"dataset": "IMDB", "method": "IMDB-HeSF-RCS-channel50", "training_executed_count": 3, "training_seed_count": 3, "failure_count": 0},
    ]
    decision = gate21_20_decision(
        main_rows=rows,
        rep_rows=rep_rows,
        robustness_rows=robustness,
        acm_overlap_rows=[{"field_ratio": 0.20}],
        imdb_upgrade_rows=[{"method": "IMDB-HeSF-RCS-channel50"}],
        freehgc_selector_rows=[{"method": "FreeHGC-score-as-selector structural16"}, {"method": "FreeHGC-score-as-selector structural20"}],
        datasets=["DBLP", "ACM", "IMDB"],
    )
    assert set(GATE21_20_DECISION_FLAGS).issubset(decision)
    assert decision["HESF_RCS_REP_CANDIDATE_POOL_PASS"] is True
    assert decision["HESF_RCS_REP_NO_TEST_LEAKAGE"] is True
    assert decision["FREEHGC_SCORE_AS_SELECTOR_READY"] is True
    assert decision["ACM_SELECTOR_OVERLAP_READY"] is True
    assert decision["IMDB_HEFS_UPGRADED_PLANNER_READY"] is True
    assert decision["STAGE_REPORT_QUICK_ROBUSTNESS_READY"] is True


def _rep(rows: list[dict[str, object]], dataset: str, rep_type: str) -> dict[str, object]:
    return next(row for row in rows if row["dataset"] == dataset and row["rep_type"] == rep_type)


def _ready_row(
    dataset: str,
    method: str,
    family: str,
    *,
    validation_micro: float | str = 0.80,
    validation_macro: float | str = 0.79,
    test_micro: float | str = 0.82,
    test_macro: float | str = 0.81,
    semantic: float = 0.20,
    support_edge: float = 0.20,
    channel_edge_ratio: float | str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "method_family": family,
        "requested_budget_type": "channel_edge_ratio" if channel_edge_ratio != "" else "structural_storage_ratio",
        "requested_budget": channel_edge_ratio if channel_edge_ratio != "" else semantic,
        "semantic_structural_storage_ratio": semantic,
        "actual_support_edge_ratio": support_edge,
        "support_edge_ratio": support_edge,
        "channel_edge_ratio": channel_edge_ratio,
        "raw_hgb_text_byte_ratio": 0.9,
        "test_micro_f1_mean": test_micro,
        "test_macro_f1_mean": test_macro,
        "validation_micro_f1_mean": validation_micro,
        "validation_macro_f1_mean": validation_macro,
        "schema_compatible": True,
        "target_preserving": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "training_executed": True,
        "constraint_safe_fallback": False,
        "eligible_for_compression_claim": family != "full_fidelity_baseline",
        "eligible_for_main_table": True,
        "eligible_for_main_decision": True,
        "success": True,
        "failure_type": "",
        "failure_reason": "",
    }
