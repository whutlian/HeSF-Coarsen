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


@dataclass(frozen=True)
class SupportSelectorConfig:
    selector: Literal[
        "teacher_topk",
        "teacher_diverse_topk",
        "validation_greedy",
        "mlp_importance",
        "hybrid_teacher_response",
    ] = "teacher_diverse_topk"
    support_ratios: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
    class_balance: bool = True
    anchor_diversity: bool = True
    max_context_collision_js: float = 0.35
    allow_background_bucket: bool = True
    background_strategy: Literal["drop", "dummy", "typed_background"] = "typed_background"


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
