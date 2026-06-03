from __future__ import annotations


def test_gate21_19_planner_backends_emit_required_frontier_plans() -> None:
    from hesf_coarsen.eval.official.gate21_19_planner_backends import (
        ACMClosureFieldPlanner,
        DatasetPlannerBackend,
        IMDBConstraintChannelPlanner,
        Plan,
    )

    acm_backend = ACMClosureFieldPlanner()
    imdb_backend = IMDBConstraintChannelPlanner()
    assert isinstance(acm_backend, DatasetPlannerBackend)
    assert isinstance(imdb_backend, DatasetPlannerBackend)

    acm_plans = acm_backend.candidate_plans(
        budgets=[0.10, 0.15, 0.20, 0.30],
        modes=["coverage_greedy", "field_degree", "random", "validation_greedy"],
    )
    imdb_plans = imdb_backend.candidate_plans(
        budgets=[0.20, 0.30, 0.40, 0.50],
        modes=["degree", "random", "validation_greedy", "mdfull_mix"],
    )

    acm_methods = {plan.method for plan in acm_plans}
    assert {
        "ACM-HeSF-RCS-auto-field20",
        "ACM-Degree-field20",
        "ACM-Random-field20",
        "ACM-ValidationGreedy-field20",
        "ACM-HeSF-RCS-auto-field10",
        "ACM-Degree-field30",
    }.issubset(acm_methods)

    imdb_methods = {plan.method for plan in imdb_plans}
    assert {
        "IMDB-HeSF-RCS-auto structural20",
        "IMDB-HeSF-RCS-auto structural30",
        "IMDB-Random-channel20",
        "IMDB-Degree-channel20",
        "IMDB-MDfull-MA20-MK50",
        "IMDB-MDfull-MA50-MK50",
        "IMDB-MDfull-MA00-MK100",
        "IMDB-ValidationGreedy-channel30",
    }.issubset(imdb_methods)

    for plan in [*acm_plans, *imdb_plans]:
        assert isinstance(plan, Plan)
        assert plan.dataset in {"ACM", "IMDB"}
        assert plan.planner_backend
        assert plan.planner_mode
        assert plan.requested_budget_type


def test_gate21_19_rep_selection_uses_validation_metrics_only() -> None:
    from hesf_coarsen.eval.official.validation_metric_resolver import select_gate21_19_representatives

    rows = [
        {
            "dataset": "IMDB",
            "method": "IMDB-HeSF-RCS-auto structural20",
            "method_family": "schema_preserving_rcs",
            "eligible_for_main_table": True,
            "eligible_for_compression_claim": True,
            "success": True,
            "training_executed": True,
            "validation_micro_f1_mean": 0.66,
            "validation_macro_f1_mean": 0.63,
            "test_micro_f1_mean": 0.69,
            "test_macro_f1_mean": 0.66,
        },
        {
            "dataset": "IMDB",
            "method": "IMDB-MDfull-MA20-MK50",
            "method_family": "schema_preserving_rcs",
            "eligible_for_main_table": True,
            "eligible_for_compression_claim": True,
            "success": True,
            "training_executed": True,
            "validation_micro_f1_mean": 0.68,
            "validation_macro_f1_mean": 0.65,
            "test_micro_f1_mean": 0.67,
            "test_macro_f1_mean": 0.64,
        },
    ]

    reps = select_gate21_19_representatives(rows, datasets=["IMDB"])
    main = next(row for row in reps if row["method"] == "HeSF-RCS-Rep-Validated")
    oracle = next(row for row in reps if row["method"] == "HeSF-RCS-TestOracleRep")

    assert main["source_method"] == "IMDB-MDfull-MA20-MK50"
    assert main["selection_source"] == "actual_validation"
    assert main["uses_test_for_selection"] is False
    assert main["eligible_for_decision"] is True
    assert oracle["uses_test_for_selection"] is True
    assert oracle["eligible_for_main_table"] is False
    assert oracle["eligible_for_decision"] is False


def test_gate21_19_decision_flags_and_rules() -> None:
    from hesf_coarsen.eval.official.gate21_19_decision import GATE21_19_DECISION_FLAGS, gate21_19_decision

    rows = []
    for dataset in ("DBLP", "ACM", "IMDB"):
        rows.extend(
            [
                _ready_row(dataset, "Full-native-SeHGNN", method_family="full_fidelity_baseline", eligible_for_compression_claim=False),
                _ready_row(dataset, "Export-full-SeHGNN", method_family="full_fidelity_baseline", eligible_for_compression_claim=False),
            ]
        )
    rows.extend(
        [
            _ready_row("DBLP", "HeSF-RCS-auto structural12", planner_backend="DBLPRelationChannelPlanner", planner_mode="relation_channel", requested_budget_type="structural_storage_ratio", requested_budget=0.12),
            _ready_row("DBLP", "Random-edge-relwise", method_family="relation_structural_baseline", planner_backend="DBLPRelationChannelPlanner", planner_mode="random", requested_budget_type="support_edge_ratio", requested_budget=0.20),
            _ready_row("DBLP", "Herding-HG-TP", method_family="external_tp_baseline", planner_backend="ExternalTPLocalPlanner", planner_mode="herding", requested_budget_type="support_node_ratio", requested_budget=0.50),
            _ready_row("DBLP", "FreeHGC-score-TP-local", method_family="external_tp_baseline", planner_backend="ExternalTPLocalPlanner", planner_mode="freehgc_score", requested_budget_type="support_edge_ratio", requested_budget=0.20),
            _ready_row("ACM", "ACM-HeSF-RCS-auto-field20", planner_backend="ACMClosureFieldPlanner", planner_mode="coverage_greedy", requested_budget_type="keyword_feature_ratio", requested_budget=0.20, keyword_feature_ratio=0.20),
            _ready_row("ACM", "ACM-Degree-field20", method_family="relation_structural_baseline", planner_backend="ACMClosureFieldPlanner", planner_mode="field_degree", requested_budget_type="keyword_feature_ratio", requested_budget=0.20, keyword_feature_ratio=0.20),
            _ready_row("ACM", "ACM-Random-field20", method_family="relation_structural_baseline", planner_backend="ACMClosureFieldPlanner", planner_mode="random", requested_budget_type="keyword_feature_ratio", requested_budget=0.20, keyword_feature_ratio=0.20),
            _ready_row("ACM", "ACM-ValidationGreedy-field20", planner_backend="ACMClosureFieldPlanner", planner_mode="validation_greedy", requested_budget_type="keyword_feature_ratio", requested_budget=0.20, keyword_feature_ratio=0.20),
            _ready_row("IMDB", "IMDB-HeSF-RCS-auto structural20", planner_backend="IMDBConstraintChannelPlanner", planner_mode="degree", requested_budget_type="structural_storage_ratio", requested_budget=0.20),
            _ready_row("IMDB", "IMDB-HeSF-RCS-auto structural30", planner_backend="IMDBConstraintChannelPlanner", planner_mode="degree", requested_budget_type="structural_storage_ratio", requested_budget=0.30),
            _ready_row("IMDB", "IMDB-Random-channel20", method_family="relation_structural_baseline", planner_backend="IMDBConstraintChannelPlanner", planner_mode="random", requested_budget_type="channel_edge_ratio", requested_budget=0.20, channel_edge_ratio=0.20),
            _ready_row("IMDB", "IMDB-Degree-channel20", method_family="relation_structural_baseline", planner_backend="IMDBConstraintChannelPlanner", planner_mode="degree", requested_budget_type="channel_edge_ratio", requested_budget=0.20, channel_edge_ratio=0.20),
            _ready_row("IMDB", "IMDB-MDfull-MA20-MK50", planner_backend="IMDBConstraintChannelPlanner", planner_mode="mdfull_mix", requested_budget_type="channel_edge_ratio", requested_budget=0.50, channel_edge_ratio=0.50),
            _ready_row("IMDB", "IMDB-ValidationGreedy-channel30", planner_backend="IMDBConstraintChannelPlanner", planner_mode="validation_greedy", requested_budget_type="channel_edge_ratio", requested_budget=0.30, channel_edge_ratio=0.30),
            _ready_row("ACM", "HeSF-RCS-Rep-Validated", planner_backend="ACMClosureFieldPlanner", planner_mode="validation_rep", requested_budget_type="keyword_feature_ratio", requested_budget=0.20, selection_source="actual_validation", uses_test_for_selection=False),
            _ready_row("IMDB", "HeSF-RCS-Rep-Validated", planner_backend="IMDBConstraintChannelPlanner", planner_mode="validation_rep", requested_budget_type="channel_edge_ratio", requested_budget=0.30, selection_source="actual_validation", uses_test_for_selection=False),
        ]
    )

    decision = gate21_19_decision(main_rows=rows, datasets=["DBLP", "ACM", "IMDB"], mode="smoke")
    assert set(GATE21_19_DECISION_FLAGS).issubset(decision)
    for flag in GATE21_19_DECISION_FLAGS:
        if flag != "STAGE_REPORT_QUICK_ROBUSTNESS_READY":
            assert decision[flag] is True, flag

    bad = list(rows)
    bad.append(
        _ready_row(
            "ACM",
            "ACM-Fallback-field20",
            constraint_safe_fallback=True,
            eligible_for_compression_claim=True,
            planner_backend="ACMClosureFieldPlanner",
            planner_mode="fallback",
        )
    )
    bad_decision = gate21_19_decision(main_rows=bad, datasets=["DBLP", "ACM", "IMDB"], mode="smoke")
    assert bad_decision["NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"] is False
    assert bad_decision["STAGE_REPORT_SMOKE_READY"] is False


def test_gate21_19_main_fields_match_prompt_contract() -> None:
    from experiments.scripts.run_gate21_19_multidataset_frontier import GATE21_19_MAIN_FIELDS

    required = {
        "dataset",
        "method",
        "method_family",
        "planner_backend",
        "planner_mode",
        "requested_budget_type",
        "requested_budget",
        "actual_support_edge_ratio",
        "semantic_structural_storage_ratio",
        "raw_hgb_text_byte_ratio",
        "keyword_feature_ratio",
        "channel_edge_ratio",
        "support_node_ratio",
        "test_micro_f1_mean",
        "test_micro_f1_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "validation_micro_f1_mean",
        "validation_macro_f1_mean",
        "recovery_vs_native_full_micro",
        "recovery_vs_native_full_macro",
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "constraint_safe_fallback",
        "eligible_for_compression_claim",
        "eligible_for_main_table",
        "success",
        "failure_type",
        "failure_reason",
    }
    assert required.issubset(set(GATE21_19_MAIN_FIELDS))


def _ready_row(
    dataset: str,
    method: str,
    *,
    method_family: str = "schema_preserving_rcs",
    planner_backend: str = "",
    planner_mode: str = "",
    requested_budget_type: str = "structural_storage_ratio",
    requested_budget: float = 0.20,
    actual_support_edge_ratio: float = 0.20,
    semantic_structural_storage_ratio: float = 0.20,
    raw_hgb_text_byte_ratio: float = 0.90,
    keyword_feature_ratio: float | str = "",
    channel_edge_ratio: float | str = "",
    selection_source: str = "",
    uses_test_for_selection: bool = False,
    constraint_safe_fallback: bool = False,
    eligible_for_compression_claim: bool = True,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "method_family": method_family,
        "planner_backend": planner_backend,
        "planner_mode": planner_mode,
        "requested_budget_type": requested_budget_type,
        "requested_budget": requested_budget,
        "actual_support_edge_ratio": actual_support_edge_ratio,
        "semantic_structural_storage_ratio": semantic_structural_storage_ratio,
        "raw_hgb_text_byte_ratio": raw_hgb_text_byte_ratio,
        "keyword_feature_ratio": keyword_feature_ratio,
        "channel_edge_ratio": channel_edge_ratio,
        "support_node_ratio": 0.50 if requested_budget_type == "support_node_ratio" else "",
        "test_micro_f1_mean": 0.80,
        "test_macro_f1_mean": 0.78,
        "validation_micro_f1_mean": 0.79,
        "validation_macro_f1_mean": 0.77,
        "schema_compatible": True,
        "target_preserving": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "training_executed": True,
        "constraint_safe_fallback": constraint_safe_fallback,
        "eligible_for_compression_claim": eligible_for_compression_claim,
        "eligible_for_main_table": True,
        "eligible_for_decision": not uses_test_for_selection,
        "success": True,
        "failure_type": "",
        "failure_reason": "",
        "selection_source": selection_source,
        "uses_test_for_selection": uses_test_for_selection,
    }
