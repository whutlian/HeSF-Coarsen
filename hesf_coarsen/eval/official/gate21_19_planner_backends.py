"""Gate21.19 dataset-specific planner backends.

The backends keep the planning surface explicit: DBLP is relation-channel
selection, ACM is closure-field selection, and IMDB is constrained channel
selection.  They produce runnable plans that the Gate21.19 runner can either
match to previous official runs or export into official HGB/SeHGNN datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Plan:
    dataset: str
    method: str
    method_family: str
    planner_backend: str
    planner_mode: str
    requested_budget_type: str
    requested_budget: float
    seed: int = 1
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    plan: Plan
    export_dir: Path
    manifest_path: Path | None
    success: bool
    failure_type: str = ""
    failure_reason: str = ""


class DatasetPlannerBackend(ABC):
    """Interface implemented by each dataset-specific planner backend."""

    dataset: str
    planner_backend: str

    @abstractmethod
    def candidate_plans(
        self,
        budgets: Iterable[float] | None = None,
        modes: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> list[Plan]:
        raise NotImplementedError

    @abstractmethod
    def export_plan(self, plan: Plan, export_dir: Path) -> ExportResult:
        raise NotImplementedError


def _budget_tag(value: float, width: int = 2) -> str:
    return f"{int(round(float(value) * 100)):0{width}d}"


def _sorted_unique(values: Iterable[float] | None, default: Iterable[float]) -> list[float]:
    raw = default if values is None else values
    return sorted({round(float(v), 6) for v in raw})


def _seed_values(seeds: Iterable[int] | None) -> list[int]:
    raw = [1] if seeds is None else list(seeds)
    return sorted({int(seed) for seed in raw})


class DBLPRelationChannelPlanner(DatasetPlannerBackend):
    dataset = "DBLP"
    planner_backend = "DBLPRelationChannelPlanner"

    def candidate_plans(
        self,
        budgets: Iterable[float] | None = None,
        modes: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> list[Plan]:
        budget_values = _sorted_unique(budgets, [0.12, 0.16, 0.20, 0.30])
        requested_modes = set(modes or ["all"])
        seed_values = _seed_values(seeds)
        plans: list[Plan] = []

        def wants(*names: str) -> bool:
            return "all" in requested_modes or any(name in requested_modes for name in names)

        for seed in seed_values:
            if wants("structural", "hesf"):
                for budget in budget_values:
                    if budget in {0.12, 0.16, 0.20, 0.30}:
                        plans.append(
                            Plan(
                                dataset=self.dataset,
                                method=f"HeSF-RCS-auto-structural{_budget_tag(budget)}",
                                method_family="hesf_rcs",
                                planner_backend=self.planner_backend,
                                planner_mode="structural_rcs",
                                requested_budget_type="structural_storage_ratio",
                                requested_budget=budget,
                                seed=seed,
                                params={"structural_ratio": budget},
                            )
                        )
            if wants("support_edge", "baselines"):
                for method, mode in [
                    ("Random-edge-relwise-support_edge20", "random_relwise"),
                    ("Degree-edge-relwise-support_edge20", "degree_relwise"),
                    ("Proportional-relation-budget-support_edge20", "proportional_relation_budget"),
                ]:
                    plans.append(
                        Plan(
                            dataset=self.dataset,
                            method=method,
                            method_family="local_structural_baseline",
                            planner_backend=self.planner_backend,
                            planner_mode=mode,
                            requested_budget_type="actual_support_edge_ratio",
                            requested_budget=0.20,
                            seed=seed,
                            params={"support_edge_ratio": 0.20},
                        )
                    )
            if wants("external_tp", "baselines"):
                for method in [
                    "Herding-HG-TP-support_node50",
                    "HGCond-score-TP-local-support_node50",
                    "GCond-score-TP-local-support_node50",
                    "FreeHGC-score-TP-local-support_edge20",
                    "KCenter-HG-TP-support_node50",
                    "Random-HG-TP-support_node50",
                    "GraphSparsify-TP-support_edge20",
                ]:
                    budget_type = "actual_support_edge_ratio" if "edge20" in method else "support_node_ratio"
                    plans.append(
                        Plan(
                            dataset=self.dataset,
                            method=method,
                            method_family="external_tp_baseline",
                            planner_backend=self.planner_backend,
                            planner_mode="external_tp_local",
                            requested_budget_type=budget_type,
                            requested_budget=0.20 if budget_type == "actual_support_edge_ratio" else 0.50,
                            seed=seed,
                            params={budget_type: 0.20 if budget_type == "actual_support_edge_ratio" else 0.50},
                        )
                    )
            if wants("freehgc_selector_probe"):
                for budget in [0.16, 0.20]:
                    plans.append(
                        Plan(
                            dataset=self.dataset,
                            method=f"FreeHGC-score-as-selector-structural{_budget_tag(budget)}",
                            method_family="selector_probe",
                            planner_backend=self.planner_backend,
                            planner_mode="freehgc_score_selector",
                            requested_budget_type="structural_storage_ratio",
                            requested_budget=budget,
                            seed=seed,
                            params={"structural_ratio": budget},
                        )
                    )
        return plans

    def export_plan(self, plan: Plan, export_dir: Path) -> ExportResult:
        return ExportResult(
            plan=plan,
            export_dir=export_dir,
            manifest_path=None,
            success=False,
            failure_type="not_applicable",
            failure_reason="DBLP Gate21.19 plans are sourced from existing official DBLP runs.",
        )


class ACMClosureFieldPlanner(DatasetPlannerBackend):
    dataset = "ACM"
    planner_backend = "ACMClosureFieldPlanner"

    def __init__(
        self,
        *,
        source_dir: str | Path | None = None,
        sehgnn_repo: str | Path | None = None,
        graph_seed: int = 1,
    ) -> None:
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.sehgnn_repo = Path(sehgnn_repo) if sehgnn_repo is not None else None
        self.graph_seed = int(graph_seed)

    def candidate_plans(
        self,
        budgets: Iterable[float] | None = None,
        modes: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> list[Plan]:
        budget_values = _sorted_unique(budgets, [0.30, 0.20, 0.15, 0.10])
        requested_modes = set(modes or ["coverage_greedy", "field_degree", "random", "validation_greedy"])
        seed_values = _seed_values(seeds)
        plans: list[Plan] = []

        mode_specs = {
            "coverage_greedy": ("ACM-HeSF-RCS-auto-field", "hesf_rcs", "keyword_feature_ratio"),
            "field_degree": ("ACM-Degree-field", "field_baseline", "keyword_feature_ratio"),
            "random": ("ACM-Random-field", "field_baseline", "keyword_feature_ratio"),
            "validation_greedy": ("ACM-ValidationGreedy-field", "field_baseline", "keyword_feature_ratio"),
            "cost_normalized_validation_delta": (
                "ACM-CostNormValidationDelta-field",
                "field_baseline",
                "keyword_feature_ratio",
            ),
        }
        for seed in seed_values:
            for budget in budget_values:
                for mode, (prefix, family, budget_type) in mode_specs.items():
                    if mode not in requested_modes:
                        continue
                    plans.append(
                        Plan(
                            dataset=self.dataset,
                            method=f"{prefix}{_budget_tag(budget)}",
                            method_family=family,
                            planner_backend=self.planner_backend,
                            planner_mode=mode,
                            requested_budget_type=budget_type,
                            requested_budget=budget,
                            seed=seed,
                            params={"keyword_feature_ratio": budget},
                        )
                    )
        return plans

    def export_plan(self, plan: Plan, export_dir: Path) -> ExportResult:
        if self.source_dir is None or self.sehgnn_repo is None:
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=None,
                success=False,
                failure_type="missing_export_configuration",
                failure_reason="ACM exporter requires source_dir and sehgnn_repo.",
            )
        from hesf_coarsen.eval.official.acm_closure_compression import export_acm_closure_compressed

        method = {
            "coverage_greedy": "coverage_greedy",
            "field_degree": "degree",
            "random": "random",
            "validation_greedy": "validation_greedy",
            "cost_normalized_validation_delta": "cost_normalized_validation_delta",
        }.get(plan.planner_mode, "degree")
        try:
            manifest = export_acm_closure_compressed(
                source_dir=self.source_dir,
                export_dir=export_dir,
                keyword_ratio=float(plan.params["keyword_feature_ratio"]),
                method=method,
                graph_seed=self.graph_seed,
            )
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=export_dir / "gate21_18_export_manifest.json" if manifest else None,
                success=True,
            )
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=None,
                success=False,
                failure_type=type(exc).__name__,
                failure_reason=str(exc),
            )


class IMDBConstraintChannelPlanner(DatasetPlannerBackend):
    dataset = "IMDB"
    planner_backend = "IMDBConstraintChannelPlanner"

    def __init__(
        self,
        *,
        source_dir: str | Path | None = None,
        sehgnn_repo: str | Path | None = None,
        graph_seed: int = 1,
    ) -> None:
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.sehgnn_repo = Path(sehgnn_repo) if sehgnn_repo is not None else None
        self.graph_seed = int(graph_seed)

    def candidate_plans(
        self,
        budgets: Iterable[float] | None = None,
        modes: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> list[Plan]:
        budget_values = _sorted_unique(budgets, [0.20, 0.30, 0.40, 0.50])
        requested_modes = set(
            modes
            or [
                "hesf_structural",
                "random_channel",
                "degree_channel",
                "mdfull_mix",
                "validation_greedy",
            ]
        )
        seed_values = _seed_values(seeds)
        plans: list[Plan] = []

        def add(
            *,
            seed: int,
            method: str,
            family: str,
            mode: str,
            budget_type: str,
            budget: float,
            actor_ratio: float,
            keyword_ratio: float,
            export_method: str,
        ) -> None:
            plans.append(
                Plan(
                    dataset=self.dataset,
                    method=method,
                    method_family=family,
                    planner_backend=self.planner_backend,
                    planner_mode=mode,
                    requested_budget_type=budget_type,
                    requested_budget=budget,
                    seed=seed,
                    params={
                        "actor_ratio": actor_ratio,
                        "keyword_ratio": keyword_ratio,
                        "export_method": export_method,
                    },
                )
            )

        for seed in seed_values:
            wants_hesf = "all" in requested_modes or "hesf_structural" in requested_modes or "degree" in requested_modes
            wants_random = "all" in requested_modes or "random_channel" in requested_modes or "random" in requested_modes
            wants_degree = "all" in requested_modes or "degree_channel" in requested_modes or "degree" in requested_modes
            wants_mdfull = "all" in requested_modes or "mdfull_mix" in requested_modes
            wants_validation = "all" in requested_modes or "validation_greedy" in requested_modes

            if wants_hesf:
                add(
                    seed=seed,
                    method="IMDB-HeSF-RCS-auto structural30",
                    family="hesf_rcs",
                    mode="hesf_structural",
                    budget_type="structural_storage_ratio",
                    budget=0.30,
                    actor_ratio=0.15,
                    keyword_ratio=0.15,
                    export_method="coverage_greedy",
                )
                add(
                    seed=seed,
                    method="IMDB-HeSF-RCS-auto structural20",
                    family="hesf_rcs",
                    mode="hesf_structural",
                    budget_type="structural_storage_ratio",
                    budget=0.20,
                    actor_ratio=0.05,
                    keyword_ratio=0.05,
                    export_method="coverage_greedy",
                )
            if wants_random:
                add(
                    seed=seed,
                    method="IMDB-Random-channel20",
                    family="channel_baseline",
                    mode="random_channel",
                    budget_type="channel_edge_ratio",
                    budget=0.20,
                    actor_ratio=0.20,
                    keyword_ratio=0.20,
                    export_method="random",
                )
            if wants_degree:
                add(
                    seed=seed,
                    method="IMDB-Degree-channel20",
                    family="channel_baseline",
                    mode="degree_channel",
                    budget_type="channel_edge_ratio",
                    budget=0.20,
                    actor_ratio=0.20,
                    keyword_ratio=0.20,
                    export_method="degree",
                )
            if wants_mdfull:
                for actor, keyword in [
                    (0.50, 0.20),
                    (0.20, 0.50),
                    (0.50, 0.50),
                    (0.75, 0.25),
                    (0.25, 0.75),
                    (1.00, 0.00),
                    (0.00, 1.00),
                ]:
                    add(
                        seed=seed,
                        method=f"IMDB-MDfull-MA{_budget_tag(actor)}-MK{_budget_tag(keyword)}",
                        family="channel_baseline",
                        mode="mdfull_mix",
                        budget_type="channel_edge_ratio",
                        budget=max(actor, keyword),
                        actor_ratio=actor,
                        keyword_ratio=keyword,
                        export_method="degree",
                    )
            if wants_validation:
                for budget in budget_values:
                    add(
                        seed=seed,
                        method=f"IMDB-ValidationGreedy-channel{_budget_tag(budget)}",
                        family="channel_baseline",
                        mode="validation_greedy",
                        budget_type="channel_edge_ratio",
                        budget=budget,
                        actor_ratio=budget,
                        keyword_ratio=budget,
                        export_method="validation_greedy",
                    )
        return plans

    def export_plan(self, plan: Plan, export_dir: Path) -> ExportResult:
        if self.source_dir is None or self.sehgnn_repo is None:
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=None,
                success=False,
                failure_type="missing_export_configuration",
                failure_reason="IMDB exporter requires source_dir and sehgnn_repo.",
            )
        from hesf_coarsen.eval.official.imdb_constraint_compression import export_imdb_constraint_compressed

        try:
            manifest = export_imdb_constraint_compressed(
                source_dir=self.source_dir,
                export_dir=export_dir,
                actor_ratio=float(plan.params["actor_ratio"]),
                keyword_ratio=float(plan.params["keyword_ratio"]),
                method=str(plan.params.get("export_method", "degree")),
                graph_seed=self.graph_seed,
            )
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=export_dir / "gate21_18_export_manifest.json" if manifest else None,
                success=True,
            )
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics
            return ExportResult(
                plan=plan,
                export_dir=export_dir,
                manifest_path=None,
                success=False,
                failure_type=type(exc).__name__,
                failure_reason=str(exc),
            )
