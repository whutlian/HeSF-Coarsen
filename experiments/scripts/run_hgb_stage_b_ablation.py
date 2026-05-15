from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import (
    repo_root,
    run_subprocess_with_log,
    write_command_metadata,
    write_config_snapshot,
)
from hesf_coarsen.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class StageBAblationConfig:
    run_name: str
    dataset: str
    variant: str
    target_ratio: float
    seed: int
    candidate_source: str
    sketch_dim: int
    sketch_order: int
    sketch_method: str
    matching_method: str
    max_cluster_size: int
    relation_weighting_method: str
    metapath_preset: str
    metapath_operator_weight_total: float
    experiment_block: str
    unique_run_key: str
    config: dict


def _candidate_flags(source: str) -> dict:
    return {
        "enable_onehop": "onehop" in source,
        "enable_capped_twohop": "twohop" in source,
        "enable_bucket": "bucket" in source,
        "enable_partition_ann": source.endswith("_ann") or "ann" in source,
    }


def _configure_m0_mainline(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "uniform"
    cfg.setdefault("sketch", {})["method"] = "chebyshev_heat"
    cfg.setdefault("sketch", {})["dim"] = 16
    cfg.setdefault("metapath_sketch", {})["preset"] = "off"
    cfg["metapath_sketch"]["enabled"] = False
    cfg["metapath_sketch"]["operator_weight_total"] = 0.0
    cfg["metapath_sketch"]["paths"] = []
    cfg["metapath_sketch"]["auto_paths"] = False
    cfg.setdefault("scoring", {})["lambda_conv"] = 0.5
    return cfg


def _configure_g3_true_cumulative_repair(config: dict, objective: str) -> dict:
    cfg = deepcopy(config)
    objective_name = str(objective).lower().replace("-", "_")
    if objective_name in {"spectral", "fixed", "cumulative"}:
        objective_name = "energy"
    cfg.setdefault("coarsening", {})["matching_method"] = "greedy_cluster"
    cfg["coarsening"]["max_cluster_size"] = 4
    guard = cfg.setdefault("coarsening", {}).setdefault("cumulative_guard", {})
    guard.update(
        {
            "enabled": True,
            "probe_count": 32,
            "max_cumulative_dee": 0.40,
            "max_cumulative_sipe": 0.75,
            "repair_bad_clusters": True,
            "repair_strategy": "split_local_swap_accept",
            "accept_only_if_cumulative_improves": True,
            "accept_metric": "true_cumulative",
            "objective": objective_name,
            "repair_objective": objective_name,
        }
    )
    return cfg


def _variant_config(config: dict, variant: str) -> dict:
    cfg = deepcopy(config)
    if variant == "base" or variant in {
        "C0",
        "C1",
        "C2",
        "C3",
        "D0",
        "D1",
        "D2",
        "D3",
    }:
        return cfg
    if variant in {
        "M0-repeat",
        "M0-conv0.35",
        "M0-conv0.65",
        "M0-relation-guard",
    }:
        cfg = _configure_m0_mainline(cfg)
        if variant == "M0-conv0.35":
            cfg["scoring"]["lambda_conv"] = 0.35
        elif variant == "M0-conv0.65":
            cfg["scoring"]["lambda_conv"] = 0.65
        elif variant == "M0-relation-guard":
            cfg.setdefault("scoring", {}).setdefault("relation_guard", {}).update(
                {
                    "enabled": True,
                    "max_ree_increase": 0.02,
                    "max_relation_profile_drift": 0.05,
                }
            )
        return cfg
    if variant in {"P0", "P1", "P2", "P3"}:
        cfg = _configure_m0_mainline(cfg)
        cfg.setdefault("coarsening", {})["matching_method"] = (
            "mutual_best" if variant == "P0" else "greedy_cluster"
        )
        cfg["coarsening"]["max_cluster_size"] = {"P0": 2, "P1": 3, "P2": 4, "P3": 4}[variant]
        if variant == "P3":
            cfg.setdefault("scoring", {})["lambda_conv"] = 0.35
        return cfg
    if variant in {"G3-fixed", "G3-energy", "G3-task", "G3-relation"}:
        objective = {
            "G3-fixed": "energy",
            "G3-energy": "energy",
            "G3-task": "task",
            "G3-relation": "relation",
        }[variant]
        return _configure_g3_true_cumulative_repair(cfg, objective)
    if variant in {"M0", "M1", "M2", "M3", "M4", "M5"}:
        cfg = _configure_m0_mainline(cfg)
        if variant == "M1":
            cfg["scoring"]["lambda_conv"] = 0.25
        if variant == "M2":
            cfg["scoring"]["lambda_conv"] = 0.75
        if variant == "M3":
            cfg["fusion"]["relation_weighting"]["method"] = "capped_inverse_sqrt_energy"
            cfg["fusion"]["relation_weighting"]["gamma"] = 0.5
            cfg["fusion"]["relation_weighting"].setdefault("weight_clip_min", 0.05)
            cfg["fusion"]["relation_weighting"].setdefault("weight_clip_max", 0.25)
        if variant == "M4":
            cfg["sketch"]["method"] = "lazy"
            cfg["sketch"]["dim"] = 32
        if variant == "M5":
            cfg["metapath_sketch"]["enabled"] = True
            cfg["metapath_sketch"]["preset"] = "canonical"
            cfg["metapath_sketch"]["operator_weight_total"] = 0.1
        return cfg
    if variant in {"G0", "G1", "G2", "G3", "G4"}:
        cfg.setdefault("coarsening", {})["matching_method"] = "greedy_cluster"
        cfg["coarsening"]["max_cluster_size"] = 3 if variant == "G4" else 4
        cfg.setdefault("coarsening", {}).setdefault("cumulative_guard", {})
        guard = cfg["coarsening"]["cumulative_guard"]
        guard.update(
            {
                "enabled": variant in {"G1", "G2", "G3"},
                "probe_count": 32,
                "max_cumulative_dee": 0.40,
                "max_cumulative_sipe": 0.75,
                "repair_bad_clusters": variant in {"G1", "G2", "G3"},
                "accept_only_if_cumulative_improves": variant == "G3",
            }
        )
        if variant == "G0":
            guard["repair_strategy"] = "off"
        elif variant == "G1":
            guard["repair_strategy"] = "current"
        elif variant == "G2":
            guard["repair_strategy"] = "split_high_spread"
        elif variant == "G3":
            guard["repair_strategy"] = "split_local_swap_accept"
        else:
            guard["repair_strategy"] = "off"
        return cfg
    if variant in {"S0", "S1", "S2", "S3"}:
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "uniform"
        cfg.setdefault("metapath_sketch", {})["enabled"] = False
        cfg["metapath_sketch"]["preset"] = "off"
        cfg["metapath_sketch"]["operator_weight_total"] = 0.0
        cfg.setdefault("scoring", {})["lambda_conv"] = 0.5
        cfg.setdefault("sketch", {})["method"] = "chebyshev_heat"
        cfg["sketch"]["dim"] = 16
        if variant in {"S1", "S3"}:
            cfg["sketch"]["method"] = "lazy"
            cfg["sketch"]["dim"] = 32
        if variant in {"S2", "S3"}:
            cfg.setdefault("candidates", {}).setdefault("quotas", {})
            cfg["candidates"]["quotas"]["enforce_on"] = "selected_matches"
            cfg["candidates"]["quotas"]["bucket_min_fraction"] = 0.30
            cfg["candidates"]["quotas"]["twohop_max_fraction"] = 0.70
            cfg["candidates"]["quotas"]["fallback_max_fraction"] = 0.02
        return cfg
    if variant in {"A0", "A2", "A4", "A5", "V0", "V1", "V2", "V3", "V4", "V5"}:
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "uniform"
    if variant in {"A1", "A3"}:
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "inverse_sqrt_energy"
        cfg["fusion"]["relation_weighting"]["gamma"] = 0.5
    if variant in {"A0", "A1", "A4", "V0", "V1", "V4", "V5"}:
        cfg.setdefault("metapath_sketch", {})["preset"] = "off"
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.0
    if variant in {"A2", "A3", "A5", "V2", "V3"}:
        cfg.setdefault("metapath_sketch", {})["preset"] = "canonical"
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.1
    if variant in {"A4", "A5", "V4", "V5"}:
        cfg.setdefault("sketch", {})["method"] = "lazy"
        cfg.setdefault("sketch", {})["dim"] = 32
    if variant in {"A0", "A1", "A2", "A3", "V0", "V1", "V2", "V3"}:
        cfg.setdefault("sketch", {})["method"] = "chebyshev_heat"
        cfg.setdefault("sketch", {})["dim"] = 16
    if variant in {"V0", "V2", "V4"}:
        cfg.setdefault("scoring", {})["lambda_conv"] = 0.0
    if variant in {"V1", "V3", "V5"}:
        cfg.setdefault("scoring", {})["lambda_conv"] = 0.5
    if variant in {"A0", "A1", "A2", "A3", "A4", "A5", "V0", "V1", "V2", "V3", "V4", "V5"}:
        return cfg
    if variant == "C1-stop":
        cfg.setdefault("coarsening", {})["matching_method"] = "mutual_best"
        cfg.setdefault("coarsening", {})["max_cluster_size"] = 2
        cfg["coarsening"]["max_levels"] = 6
        return cfg
    if variant == "C2-repeat":
        cfg.setdefault("coarsening", {})["matching_method"] = "greedy_cluster"
        cfg.setdefault("coarsening", {})["max_cluster_size"] = 4
        return cfg
    if variant == "C2-size3":
        cfg.setdefault("coarsening", {})["matching_method"] = "greedy_cluster"
        cfg.setdefault("coarsening", {})["max_cluster_size"] = 3
        cfg["coarsening"]["max_levels"] = 5
        return cfg
    if variant == "C2-repair":
        cfg.setdefault("coarsening", {})["matching_method"] = "greedy_cluster"
        cfg.setdefault("coarsening", {})["max_cluster_size"] = 4
        cfg.setdefault("coarsening", {}).setdefault("cumulative_guard", {})
        cfg["coarsening"]["cumulative_guard"].update(
            {
                "enabled": True,
                "probe_count": 32,
                "max_cumulative_dee": 0.35,
                "max_cumulative_sipe": 0.70,
                "repair_bad_clusters": True,
            }
        )
        return cfg
    if variant == "uniform_weight":
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "uniform"
        cfg.setdefault("metapath_sketch", {}).setdefault("weighting", {})["method"] = "uniform"
        return cfg
    if variant == "clipped_inverse_weight":
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "clipped_inverse_energy"
        return cfg
    if variant == "inverse_sqrt_weight":
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "inverse_sqrt_energy"
        return cfg
    if variant == "no_metapath":
        cfg.setdefault("metapath_sketch", {})["enabled"] = False
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.0
        return cfg
    if variant == "lazy_no_metapath":
        cfg.setdefault("sketch", {})["method"] = "lazy"
        cfg.setdefault("metapath_sketch", {})["enabled"] = False
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.0
        return cfg
    if variant == "no_conv":
        cfg.setdefault("scoring", {})["lambda_conv"] = 0.0
        return cfg
    raise ValueError(f"unsupported Stage B variant: {variant}")


def _canonical_metapaths(dataset: str) -> list[dict]:
    name = dataset.upper()
    if name == "ACM":
        return [
            {
                "name": "paper_author_paper",
                "start_type": "paper",
                "end_type": "paper",
                "steps": [
                    {"relation_id": 2, "direction": "forward"},
                    {"relation_id": 3, "direction": "forward"},
                ],
            },
            {
                "name": "paper_subject_paper",
                "start_type": "paper",
                "end_type": "paper",
                "steps": [
                    {"relation_id": 4, "direction": "forward"},
                    {"relation_id": 5, "direction": "forward"},
                ],
            },
            {
                "name": "paper_term_paper",
                "start_type": "paper",
                "end_type": "paper",
                "steps": [
                    {"relation_id": 6, "direction": "forward"},
                    {"relation_id": 7, "direction": "forward"},
                ],
            },
        ]
    if name == "DBLP":
        return [
            {
                "name": "author_paper_author",
                "start_type": "author",
                "end_type": "author",
                "steps": [
                    {"relation_id": 0, "direction": "forward"},
                    {"relation_id": 3, "direction": "forward"},
                ],
            }
        ]
    if name == "IMDB":
        return [
            {
                "name": "movie_director_movie",
                "start_type": "movie",
                "end_type": "movie",
                "steps": [
                    {"relation_id": 0, "direction": "forward"},
                    {"relation_id": 1, "direction": "forward"},
                ],
            },
            {
                "name": "movie_actor_movie",
                "start_type": "movie",
                "end_type": "movie",
                "steps": [
                    {"relation_id": 2, "direction": "forward"},
                    {"relation_id": 3, "direction": "forward"},
                ],
            },
            {
                "name": "movie_keyword_movie",
                "start_type": "movie",
                "end_type": "movie",
                "steps": [
                    {"relation_id": 4, "direction": "forward"},
                    {"relation_id": 5, "direction": "forward"},
                ],
            },
        ]
    return []


def _apply_metapath_preset(config: dict, dataset: str, preset: str, operator_weight_total: float) -> dict:
    cfg = deepcopy(config)
    meta = cfg.setdefault("metapath_sketch", {})
    preset = str(preset).lower()
    if preset in {"off", "none", "false"} or float(operator_weight_total) <= 0.0:
        meta["enabled"] = False
        meta["operator_weight_total"] = 0.0
        meta["paths"] = []
        meta["auto_paths"] = False
        return cfg
    meta["enabled"] = True
    meta["operator_weight_total"] = float(operator_weight_total)
    if preset in {"canonical", "dataset_canonical"}:
        meta["paths"] = _canonical_metapaths(dataset)
        meta["auto_paths"] = False
    elif preset in {"auto", "auto_paths"}:
        meta["paths"] = []
        meta["auto_paths"] = True
    else:
        raise ValueError(f"unsupported metapath preset: {preset}")
    return cfg


def _unique_run_key(
    *,
    experiment_block: str,
    dataset: str,
    variant: str,
    target_ratio: float,
    seed: int,
    candidate_source: str,
    sketch_dim: int,
    sketch_order: int,
    sketch_method: str,
    max_levels: int,
    candidate_k: int,
    matching_method: str,
    max_cluster_size: int,
    relation_weighting_method: str,
    metapath_preset: str,
    metapath_operator_weight_total: float,
    lambda_conv: float,
) -> str:
    parts = [
        experiment_block,
        dataset,
        variant,
        f"r={float(target_ratio):.6g}",
        f"seed={int(seed)}",
        f"src={candidate_source}",
        f"d={int(sketch_dim)}",
        f"o={int(sketch_order)}",
        f"m={sketch_method}",
        f"L={int(max_levels)}",
        f"K={int(candidate_k)}",
        f"match={matching_method}",
        f"c={int(max_cluster_size)}",
        f"rw={relation_weighting_method}",
        f"mp={metapath_preset}:{float(metapath_operator_weight_total):.6g}",
        f"lc={float(lambda_conv):.6g}",
    ]
    text = "|".join(parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{experiment_block}:{dataset}:{variant}:{digest}"


def generate_stage_b_configs(
    *,
    datasets: Iterable[str],
    target_ratios: Iterable[float],
    max_levels: int,
    candidate_sources: Iterable[str],
    candidate_k: int,
    sketch_dims: Iterable[int],
    sketch_orders: Iterable[int],
    seeds: Iterable[int],
    variants: Iterable[str],
    experiment_block: str = "stage_b_ablation",
    normalization: str = "p95",
    normalization_scope: str = "level",
    matching_methods: Iterable[str] = ("mutual_best",),
    max_cluster_sizes: Iterable[int] = (2,),
    sketch_methods: Iterable[str] | None = None,
    relation_weighting_methods: Iterable[str] | None = None,
    metapath_presets: Iterable[str] | None = None,
    metapath_operator_weight_totals: Iterable[float] | None = None,
    lambda_conv: float | None = None,
    spectral_baseline_max_nodes: int | None = None,
    spectral_exact_eigenvalue_max_nodes: int | None = None,
    cumulative_spectral_exact_eigenvalue_max_nodes: int | None = None,
    baseline_task_eval: bool = False,
    baseline_task_epochs: int = 20,
    baseline_task_refine_epochs: int = 3,
    baseline_task_hidden_dim: int = 32,
    baseline_task_device: str = "auto",
) -> Iterable[StageBAblationConfig]:
    sketch_methods = tuple(sketch_methods or ["chebyshev_heat"])
    relation_weighting_methods = tuple(relation_weighting_methods or [""])
    metapath_presets = tuple(metapath_presets or [""])
    metapath_operator_weight_totals = tuple(metapath_operator_weight_totals or [-1.0])
    for (
        dataset,
        target_ratio,
        source,
        sketch_dim,
        sketch_order,
        sketch_method,
        seed,
        variant,
        matching_method,
        max_cluster_size,
        relation_weighting_method,
        metapath_preset,
        metapath_operator_weight_total,
    ) in product(
        datasets,
        target_ratios,
        candidate_sources,
        sketch_dims,
        sketch_orders,
        sketch_methods,
        seeds,
        variants,
        matching_methods,
        max_cluster_sizes,
        relation_weighting_methods,
        metapath_presets,
        metapath_operator_weight_totals,
    ):
        config = deepcopy(DEFAULT_CONFIG)
        config["seed"] = int(seed)
        config["coarsening"] = dict(
            config["coarsening"],
            target_ratio=float(target_ratio),
            max_levels=int(max_levels),
            matching_method=str(matching_method),
            max_cluster_size=int(max_cluster_size),
        )
        config["sketch"] = dict(
            config["sketch"],
            dim=int(sketch_dim),
            order=int(sketch_order),
            method=str(sketch_method),
        )
        config["candidates"] = dict(
            config["candidates"],
            total_budget_K=int(candidate_k),
            twohop_budget_K2=max(1, int(candidate_k) // 2),
            ann_budget_K=int(candidate_k),
            enable_fallback=True,
            fallback_penalty=1.0e6,
            fallback_max_fraction=0.05,
            **_candidate_flags(source),
        )
        config["scoring"] = dict(
            config["scoring"],
            normalization=str(normalization),
            normalization_scope=str(normalization_scope),
            lambda_spec=1.0,
            lambda_rel=0.5,
            lambda_feat=0.2,
            lambda_conv=0.5 if lambda_conv is None else float(lambda_conv),
            lambda_boundary=0.2,
        )
        config["diagnostics"] = dict(config["diagnostics"], enable_large_graph_envelope=True)
        config["diagnostics"].setdefault(
            "cumulative_spectral_baselines",
            ["random", "heavy_edge", "graphzoom_style", "convmatch_style"],
        )
        if baseline_task_eval:
            config["diagnostics"]["cumulative_spectral_baseline_task_eval"] = True
            config["diagnostics"]["cumulative_spectral_baseline_task_eval_params"] = {
                "epochs": int(baseline_task_epochs),
                "refine_epochs": int(baseline_task_refine_epochs),
                "hidden_dim": int(baseline_task_hidden_dim),
                "device": str(baseline_task_device),
            }
        if spectral_baseline_max_nodes is not None:
            config["diagnostics"]["spectral_baseline_max_nodes"] = int(spectral_baseline_max_nodes)
        if spectral_exact_eigenvalue_max_nodes is not None:
            config["diagnostics"]["spectral_exact_eigenvalue_max_nodes"] = int(
                spectral_exact_eigenvalue_max_nodes
            )
        if cumulative_spectral_exact_eigenvalue_max_nodes is not None:
            config["diagnostics"]["cumulative_spectral_exact_eigenvalue_max_nodes"] = int(
                cumulative_spectral_exact_eigenvalue_max_nodes
            )
        config = _variant_config(config, variant)
        if relation_weighting_method:
            config.setdefault("fusion", {}).setdefault("relation_weighting", {})[
                "method"
            ] = str(relation_weighting_method)
            if str(relation_weighting_method) == "inverse_sqrt_energy":
                config["fusion"]["relation_weighting"]["gamma"] = 0.5
            if str(relation_weighting_method) == "clipped_inverse_energy":
                config["fusion"]["relation_weighting"].setdefault("weight_clip_max", 0.25)
                config["fusion"]["relation_weighting"].setdefault("weight_clip_min", 0.0)
        effective_relation_weighting = str(
            config.get("fusion", {}).get("relation_weighting", {}).get("method", "")
        )
        effective_lambda_conv = float(config.get("scoring", {}).get("lambda_conv", 0.0))
        effective_sketch_dim = int(config.get("sketch", {}).get("dim", sketch_dim))
        effective_sketch_method = str(config.get("sketch", {}).get("method", sketch_method))
        inferred_metapath_preset = str(
            metapath_preset
            or config.get("metapath_sketch", {}).get("preset", "")
        )
        if inferred_metapath_preset:
            total = (
                float(metapath_operator_weight_total)
                if float(metapath_operator_weight_total) >= 0.0
                else float(config.get("metapath_sketch", {}).get("operator_weight_total", 0.0))
            )
            config = _apply_metapath_preset(config, dataset, inferred_metapath_preset, total)
        effective_metapath_total = float(
            config.get("metapath_sketch", {}).get("operator_weight_total", 0.0)
        )
        effective_metapath_preset = str(inferred_metapath_preset or "default")
        effective_coarsening = config.get("coarsening", {})
        effective_max_levels = int(effective_coarsening.get("max_levels", max_levels))
        effective_matching_method = str(
            effective_coarsening.get("matching_method", matching_method)
        )
        effective_max_cluster_size = int(
            effective_coarsening.get("max_cluster_size", max_cluster_size)
        )
        ratio_token = str(float(target_ratio)).replace(".", "p")
        method_token = effective_sketch_method.replace("chebyshev_heat", "cheb")
        match_token = effective_matching_method
        rw_token = effective_relation_weighting.replace("inverse_energy", "inv")
        mp_token = effective_metapath_preset
        run_name = (
            f"stageB_{dataset}_{variant}_r{ratio_token}_L{effective_max_levels}_"
            f"d{int(effective_sketch_dim)}_K{int(candidate_k)}_{source}_{method_token}_"
            f"{match_token}_c{effective_max_cluster_size}_{rw_token}_mp{mp_token}"
            f"{str(effective_metapath_total).replace('.', 'p')}_"
            f"lc{str(effective_lambda_conv).replace('.', 'p')}_seed{int(seed)}"
        )
        unique_key = _unique_run_key(
            experiment_block=experiment_block,
            dataset=dataset,
            variant=variant,
            target_ratio=float(target_ratio),
            seed=int(seed),
            candidate_source=source,
            sketch_dim=int(effective_sketch_dim),
            sketch_order=int(sketch_order),
            sketch_method=effective_sketch_method,
            max_levels=effective_max_levels,
            candidate_k=int(candidate_k),
            matching_method=effective_matching_method,
            max_cluster_size=effective_max_cluster_size,
            relation_weighting_method=effective_relation_weighting,
            metapath_preset=effective_metapath_preset,
            metapath_operator_weight_total=effective_metapath_total,
            lambda_conv=effective_lambda_conv,
        )
        yield StageBAblationConfig(
            run_name=run_name,
            dataset=dataset,
            variant=variant,
            target_ratio=float(target_ratio),
            seed=int(seed),
            candidate_source=source,
            sketch_dim=effective_sketch_dim,
            sketch_order=int(sketch_order),
            sketch_method=effective_sketch_method,
            matching_method=effective_matching_method,
            max_cluster_size=effective_max_cluster_size,
            relation_weighting_method=effective_relation_weighting,
            metapath_preset=effective_metapath_preset,
            metapath_operator_weight_total=effective_metapath_total,
            experiment_block=experiment_block,
            unique_run_key=unique_key,
            config=config,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused Stage B HGB ablations.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_stageB"))
    parser.add_argument("--target-ratio", type=float, action="append", dest="target_ratio")
    parser.add_argument("--target-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--candidate-source", default="onehop_twohop_bucket")
    parser.add_argument("--candidate-sources", nargs="+", default=None)
    parser.add_argument("--candidate-K", "--candidate-k", type=int, default=8, dest="candidate_K")
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--sketch-dims", type=int, nargs="+", default=None)
    parser.add_argument("--sketch-order", type=int, default=5)
    parser.add_argument("--sketch-orders", type=int, nargs="+", default=None)
    parser.add_argument("--sketch-method", default=None)
    parser.add_argument("--sketch-methods", nargs="+", default=None)
    parser.add_argument("--normalization", default="p95")
    parser.add_argument("--normalization-scope", default="level")
    parser.add_argument("--matching-method", default=None)
    parser.add_argument("--matching-methods", nargs="+", default=None)
    parser.add_argument("--max-cluster-size", type=int, default=None)
    parser.add_argument("--max-cluster-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--relation-weighting-method", default=None)
    parser.add_argument("--relation-weighting-methods", nargs="+", default=None)
    parser.add_argument("--metapath-preset", default=None, choices=["off", "canonical", "auto"])
    parser.add_argument("--metapath-presets", nargs="+", default=None)
    parser.add_argument("--metapath-operator-weight-total", type=float, default=None)
    parser.add_argument("--metapath-operator-weight-totals", type=float, nargs="+", default=None)
    parser.add_argument("--lambda-conv", type=float, default=None)
    parser.add_argument("--spectral-baseline-max-nodes", type=int, default=None)
    parser.add_argument("--spectral-exact-eigenvalue-max-nodes", type=int, default=None)
    parser.add_argument("--cumulative-spectral-exact-eigenvalue-max-nodes", type=int, default=None)
    parser.add_argument("--baseline-task-eval", action="store_true")
    parser.add_argument("--baseline-task-epochs", type=int, default=20)
    parser.add_argument("--baseline-task-refine-epochs", type=int, default=3)
    parser.add_argument("--baseline-task-hidden-dim", type=int, default=32)
    parser.add_argument("--baseline-task-device", default="auto")
    parser.add_argument("--experiment-block", default="stage_b_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "uniform_weight", "no_metapath", "lazy_no_metapath", "no_conv"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-backend", choices=["auto", "plain", "tqdm"], default="plain")
    parser.add_argument("--progress-interval", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _target_ratios(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.target_ratios:
        values.extend(float(value) for value in args.target_ratios)
    if args.target_ratio:
        values.extend(float(value) for value in args.target_ratio)
    return values or [0.5]


def _candidate_sources(args: argparse.Namespace) -> list[str]:
    return list(args.candidate_sources or [args.candidate_source])


def _sketch_dims(args: argparse.Namespace) -> list[int]:
    return [int(value) for value in (args.sketch_dims or [args.sketch_dim])]


def _sketch_orders(args: argparse.Namespace) -> list[int]:
    return [int(value) for value in (args.sketch_orders or [args.sketch_order])]


def _sketch_methods(args: argparse.Namespace) -> list[str] | None:
    values = args.sketch_methods or ([args.sketch_method] if args.sketch_method else None)
    return None if values is None else [str(value) for value in values]


def _matching_methods(args: argparse.Namespace) -> list[str]:
    values = args.matching_methods or ([args.matching_method] if args.matching_method else None)
    return [str(value) for value in (values or ["mutual_best"])]


def _max_cluster_sizes(args: argparse.Namespace) -> list[int]:
    values = args.max_cluster_sizes or ([args.max_cluster_size] if args.max_cluster_size else None)
    return [int(value) for value in (values or [2])]


def _relation_weighting_methods(args: argparse.Namespace) -> list[str] | None:
    values = args.relation_weighting_methods or (
        [args.relation_weighting_method] if args.relation_weighting_method else None
    )
    return None if values is None else [str(value) for value in values]


def _metapath_presets(args: argparse.Namespace) -> list[str] | None:
    values = args.metapath_presets or ([args.metapath_preset] if args.metapath_preset else None)
    return None if values is None else [str(value) for value in values]


def _metapath_operator_weight_totals(args: argparse.Namespace) -> list[float] | None:
    values = args.metapath_operator_weight_totals or (
        [args.metapath_operator_weight_total]
        if args.metapath_operator_weight_total is not None
        else None
    )
    return None if values is None else [float(value) for value in values]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_stage_b_configs(
        datasets=args.datasets,
        target_ratios=_target_ratios(args),
        max_levels=args.max_levels,
        candidate_sources=_candidate_sources(args),
        candidate_k=args.candidate_K,
        sketch_dims=_sketch_dims(args),
        sketch_orders=_sketch_orders(args),
        seeds=args.seeds,
        variants=args.variants,
        experiment_block=args.experiment_block,
        normalization=args.normalization,
        normalization_scope=args.normalization_scope,
        matching_methods=_matching_methods(args),
        max_cluster_sizes=_max_cluster_sizes(args),
        sketch_methods=_sketch_methods(args),
        relation_weighting_methods=_relation_weighting_methods(args),
        metapath_presets=_metapath_presets(args),
        metapath_operator_weight_totals=_metapath_operator_weight_totals(args),
        lambda_conv=args.lambda_conv,
        spectral_baseline_max_nodes=args.spectral_baseline_max_nodes,
        spectral_exact_eigenvalue_max_nodes=args.spectral_exact_eigenvalue_max_nodes,
        cumulative_spectral_exact_eigenvalue_max_nodes=args.cumulative_spectral_exact_eigenvalue_max_nodes,
        baseline_task_eval=args.baseline_task_eval,
        baseline_task_epochs=args.baseline_task_epochs,
        baseline_task_refine_epochs=args.baseline_task_refine_epochs,
        baseline_task_hidden_dim=args.baseline_task_hidden_dim,
        baseline_task_device=args.baseline_task_device,
    ):
        graph_dir = (
            (args.graph_root / item.dataset.lower())
            if args.graph_root
            else (args.root / f"{item.dataset.lower()}_hesf")
        )
        if not (graph_dir / "schema.json").exists() and not args.dry_run:
            run_subprocess_with_log(
                [
                    args.python,
                    "-m",
                    "hesf_coarsen.cli.main",
                    "import-hgb",
                    "--name",
                    item.dataset,
                    "--root",
                    str(raw_root),
                    "--output",
                    str(graph_dir),
                ],
                cwd=root,
                log_path=args.output / "_imports" / f"{item.dataset}.log",
                stream_output=args.progress,
            )
        run_dir = args.output / item.run_name
        config = deepcopy(item.config)
        if args.progress:
            config.setdefault("progress", {})["enabled"] = True
        config.setdefault("progress", {})["backend"] = args.progress_backend
        if args.progress_interval is not None:
            config.setdefault("progress", {})["min_interval_seconds"] = args.progress_interval
        config["output"] = {"dir": str(run_dir)}
        write_config_snapshot(run_dir / "config.yaml", config)
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            sketch_method=item.sketch_method,
            matching_method=item.matching_method,
            max_cluster_size=item.max_cluster_size,
            relation_weighting_method=item.relation_weighting_method,
            metapath_preset=item.metapath_preset,
            metapath_operator_weight_total=item.metapath_operator_weight_total,
            experiment_block=item.experiment_block,
            unique_run_key=item.unique_run_key,
            status="created",
        )
        if args.dry_run:
            continue
        command = [
            args.python,
            "-m",
            "hesf_coarsen.cli.main",
            "coarsen",
            "--config",
            str(run_dir / "config.yaml"),
            "--input",
            str(graph_dir),
            "--output",
            str(run_dir),
        ]
        if args.progress:
            command.extend(["--progress", "--progress-backend", args.progress_backend])
            if args.progress_interval is not None:
                command.extend(["--progress-interval", str(args.progress_interval)])
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            sketch_method=item.sketch_method,
            matching_method=item.matching_method,
            max_cluster_size=item.max_cluster_size,
            relation_weighting_method=item.relation_weighting_method,
            metapath_preset=item.metapath_preset,
            metapath_operator_weight_total=item.metapath_operator_weight_total,
            experiment_block=item.experiment_block,
            unique_run_key=item.unique_run_key,
            command=command,
            status="running",
        )
        completed = run_subprocess_with_log(
            command,
            cwd=root,
            log_path=run_dir / "run.log",
            stream_output=args.progress,
        )
        status = "success" if completed.returncode == 0 else "failed"
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            sketch_method=item.sketch_method,
            matching_method=item.matching_method,
            max_cluster_size=item.max_cluster_size,
            relation_weighting_method=item.relation_weighting_method,
            metapath_preset=item.metapath_preset,
            metapath_operator_weight_total=item.metapath_operator_weight_total,
            experiment_block=item.experiment_block,
            unique_run_key=item.unique_run_key,
            command=command,
            status=status,
            returncode=completed.returncode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
