from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SupportFeatureConfig:
    include_raw_feature: bool = True
    include_degree_profile: bool = True
    include_relation_profile: bool = True
    include_class_footprint: bool = True
    include_anchor_distribution: bool = True
    include_target_response_signature: bool = True
    include_relation_response_signature: bool = True
    footprint_mode: Literal[
        "onehop_train",
        "twohop_propagated",
        "teacher_soft",
        "hybrid",
    ] = "hybrid"


@dataclass(frozen=True)
class TeacherConfig:
    enabled: bool = True
    model: Literal[
        "hettree_lite",
        "sehgnn_lite",
        "official_sehgnn",
        "official_hettree",
    ] = "hettree_lite"
    require_official_for_paper_claim: bool = False
    tune_full_graph_lite: bool = True
    save_logits: bool = True
    save_embeddings: bool = True
    epochs_grid: tuple[int, ...] = (100, 200, 300)
    hidden_dim_grid: tuple[int, ...] = (64, 128)
    lr_grid: tuple[float, ...] = (0.001, 0.003, 0.005)
    dropout_grid: tuple[float, ...] = (0.1, 0.25, 0.5)
    weight_decay_grid: tuple[float, ...] = (1.0e-5, 1.0e-4)
    restarts: int = 3
    patience: int = 30
    monitor: str = "projected_val_macro_f1"
    proxy_logits_mode: Literal[
        "disabled",
        "diagnostic_only",
        "fallback_if_no_trained_logits",
    ] = "diagnostic_only"


@dataclass(frozen=True)
class SupportSelectorConfig:
    selector: Literal[
        "teacher_topk",
        "teacher_diverse_topk",
        "validation_greedy",
        "validation_proxy_diverse",
        "true_validation_block_greedy",
        "real_validation_block_greedy",
        "real_occlusion_block_selector",
        "occlusion_plus_dblp_prototype",
        "h6_seeded_occlusion",
        "h6_seeded_lossy_prototype",
        "dblp_aware_prototype",
        "sensitivity_block_selector",
        "mlp_importance",
        "hybrid_teacher_response",
    ] = "teacher_diverse_topk"
    support_ratios: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
    class_balance: bool = True
    anchor_diversity: bool = True
    max_context_collision_js: float = 0.35
    allow_background_bucket: bool = True
    background_strategy: Literal[
        "drop",
        "dummy",
        "typed_background",
        "class_anchor_relation_prototype",
        "dblp_aware_prototype",
    ] = "class_anchor_relation_prototype"
    candidate_pool_size: int = 16
    short_eval_epochs: int = 5
    warm_start: bool = False
    min_gain: float = -1.0
    allow_proxy_fill: bool = True
    neutral_fill: bool = False
    neutral_fill_max_drop: float = 1.0e-4
    max_validation_greedy_steps: int = 3
    block_key_mode: Literal["class_anchor_relation", "default", "dblp_aware"] = "class_anchor_relation"
    occlusion_candidate_pool_size: int = 8
    occlusion_short_eval_epochs: int = 3
    occlusion_short_patience: int = 2
    occlusion_cache_enabled: bool = True
    alpha_teacher_kl: float = 0.1
    beta_margin: float = 0.2
    gamma_class_recall: float = 0.2
    primary_occlusion_term: str = "validation_cross_entropy_delta"
    min_occlusion_importance: float = 1.0e-6
    allow_negative_occlusion_fill: bool = False
    neutral_fill_max_task_drop: float = 1.0e-4
    max_prototypes_per_type: int = 64
    max_prototypes_per_class_anchor_relation: int = 4
    min_nodes_per_prototype: int = 1
    max_members_per_prototype: int = 512
    split_large_prototype_by_degree: bool = True
    split_large_prototype_by_anchor: bool = True
    split_large_prototype_by_relation: bool = True
    force_raw_bridge_nodes: bool = False
    force_raw_keep_high_degree_bridges: bool = False
    residual_prototype_mode: Literal["none", "lossy_topk", "full_upperbound"] = "none"
    prototype_budget_fraction: float = 0.10
    max_represented_support_ratio_slack: float = 0.10
    prototype_member_budget_total: int | None = None
    prototype_edge_mass_scale: float = 0.25
    meta_path_channel_source: str = "semantic_tree_path_id"
    raw_bridge_mode: Literal[
        "off",
        "budgeted_cap",
        "importance_threshold_budgeted",
        "free_raw_diagnostic",
    ] = "off"
    min_raw_bridge_per_relation_channel: int = 1
    min_prototype_per_class: int = 1
    min_prototype_per_relation_channel: int = 1
    rare_class_never_fallback: bool = True
    rare_class_min_prototypes: int = 1
    per_relation_min_prototypes: int = 1
    per_anchor_min_prototypes: int = 1
    prototype_feature_aggregation: Literal["mean", "degree_weighted_mean"] = "degree_weighted_mean"
    prototype_edge_aggregation: Literal["sum", "mean"] = "sum"


@dataclass(frozen=True)
class SelectionRegularizerConfig:
    lambda_response: float = 0.05
    lambda_diversity: float = 0.20
    lambda_budget: float = 1.0
    response_is_auxiliary: bool = True


@dataclass(frozen=True)
class Gate15Config:
    target_node_type: int
    keep_all_target_nodes: bool = True
    support_only: bool = True
    feature: SupportFeatureConfig = field(default_factory=SupportFeatureConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    selector: SupportSelectorConfig = field(default_factory=SupportSelectorConfig)
    regularizer: SelectionRegularizerConfig = field(default_factory=SelectionRegularizerConfig)
