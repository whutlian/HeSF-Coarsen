from __future__ import annotations


def test_feature_ablation_ready_requires_methods_transforms_and_settings_with_task_metrics() -> None:
    from hesf_coarsen.eval.official.feature_ablation_task_runner import feature_ablation_ready

    methods = ["full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"]
    transforms = ["raw", "zero-paper-preserve-dim", "zero-term-preserve-dim", "zero-venue-preserve-dim", "zero-all-support-preserve-dim", "paper-only-preserve-original-dims", "term-only-preserve-original-dims", "paper-random-projection64", "paper-pca64"]
    settings = ["default", "no_label_feats", "num_feature_hops_0", "num_label_hops_0", "feature_only_mlp_adapter"]
    rows = [
        {"method": method, "feature_transform": transform, "label_graph_setting": setting, "training_executed": True, "test_micro_f1": 0.9, "test_macro_f1": 0.88}
        for method in methods
        for transform in transforms
        for setting in settings
    ]

    assert feature_ablation_ready(rows) is True
    assert feature_ablation_ready(rows[:-1]) is False
