from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TargetConditionedSpecConfig:
    temperatures: tuple[float, ...] = (0.25, 1.0, 4.0)
    probe_source: Literal["train_labels", "teacher_soft", "hybrid"] = "train_labels"
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class RelationResponseConfig:
    enabled: bool = True
    use_target_support_relations_only: bool = True
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class SupportCoverageConfig:
    anchor_source: Literal["all_train_targets", "train_plus_confident"] = "all_train_targets"
    topk: int = 32
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class SupportPurityConfig:
    enabled: bool = True
    footprint_source: Literal["onehop_train_labels", "teacher_probs"] = "onehop_train_labels"
    js_merge_block_threshold: float = 0.35


@dataclass(frozen=True)
class TaskFirstScoringConfig:
    lambda_target_spec: float = 1.0
    lambda_rel_response: float = 0.5
    lambda_support_coverage: float = 0.75
    lambda_support_purity: float = 0.75
    lambda_feat: float = 0.1
    normalization: Literal["p95", "none"] = "p95"


@dataclass(frozen=True)
class TaskFirstConfig:
    target_node_type: int
    keep_all_target_nodes: bool = True
    support_only_coarsening: bool = True
    same_type_only: bool = True
    same_partition_only: bool = True
    target_spec: TargetConditionedSpecConfig = field(default_factory=TargetConditionedSpecConfig)
    relation_response: RelationResponseConfig = field(default_factory=RelationResponseConfig)
    support_coverage: SupportCoverageConfig = field(default_factory=SupportCoverageConfig)
    support_purity: SupportPurityConfig = field(default_factory=SupportPurityConfig)
    scoring: TaskFirstScoringConfig = field(default_factory=TaskFirstScoringConfig)
