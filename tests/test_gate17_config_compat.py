from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def test_gate17_1_selector_and_background_names_instantiate():
    selectors = [
        "teacher_topk",
        "teacher_diverse_topk",
        "validation_greedy",
        "validation_proxy_diverse",
        "true_validation_block_greedy",
        "sensitivity_block_selector",
        "real_validation_block_greedy",
        "real_occlusion_block_selector",
        "occlusion_plus_dblp_prototype",
        "mlp_importance",
        "hybrid_teacher_response",
    ]
    backgrounds = [
        "drop",
        "dummy",
        "typed_background",
        "class_anchor_relation_prototype",
        "dblp_aware_prototype",
    ]

    for selector in selectors:
        for background in backgrounds:
            cfg = SupportSelectorConfig(selector=selector, background_strategy=background)
            assert cfg.selector == selector
            assert cfg.background_strategy == background


def test_gate17_1_prompt_config_fields_are_constructor_compatible():
    cfg = SupportSelectorConfig(
        selector="real_validation_block_greedy",
        background_strategy="dblp_aware_prototype",
        block_key_mode="class_anchor_relation",
        candidate_pool_size=8,
        short_eval_epochs=3,
        max_validation_greedy_steps=3,
        occlusion_candidate_pool_size=8,
        occlusion_short_eval_epochs=3,
        occlusion_short_patience=1,
        max_members_per_prototype=512,
        split_large_prototype_by_degree=True,
        split_large_prototype_by_anchor=True,
        split_large_prototype_by_relation=True,
        force_raw_bridge_nodes=True,
        min_raw_bridge_per_relation_channel=1,
        min_prototype_per_class=1,
        min_prototype_per_relation_channel=1,
        rare_class_never_fallback=True,
    )

    assert cfg.block_key_mode == "class_anchor_relation"
    assert cfg.force_raw_bridge_nodes is True
    assert cfg.min_raw_bridge_per_relation_channel == 1
    assert cfg.min_prototype_per_class == 1
    assert cfg.min_prototype_per_relation_channel == 1
    assert cfg.rare_class_never_fallback is True
