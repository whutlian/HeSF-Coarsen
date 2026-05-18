from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, write_csv, write_json
from experiments.scripts.summarize_next17_hybrid_accuracy import aggregate_rows, summarize_block
from hesf_coarsen.accuracy.distillation import deterministic_teacher_logits, kl_divergence_from_logits
from hesf_coarsen.accuracy.full_target_inference import evaluate_full_target_inference
from hesf_coarsen.accuracy.meta_recon import target_tree_reconstruction_error
from hesf_coarsen.accuracy.target_anchor_budget import target_anchor_budget
from hesf_coarsen.accuracy.target_selection import select_target_anchors
from hesf_coarsen.accuracy.target_support_hybrid import build_support_coarsened_hybrid
from hesf_coarsen.accuracy.task_aligned_score import accuracy_first_score
from hesf_coarsen.accuracy.type_budgets import compute_type_budget_report
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.edge_list import load_graph
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


DATASETS = {
    "ACM": Path("data/acm_hesf"),
    "DBLP": Path("data/dblp_hesf"),
    "IMDB": Path("data/imdb_hesf"),
}
DEFAULT_METHODS = ["HeSF-LVC-P", "HeSF-LVC-S"]
DEFAULT_RATIOS = [0.024, 0.048, 0.096]
DEFAULT_SEEDS = [12345, 23456, 34567]
VARIANTS = [
    "A1_target_preserve",
    "A2_hybridA_keepall",
    "A3_hybridB_selecttarget",
    "A4_hybridB_meta_recon",
    "A5_hybridB_distill",
]
MODELS = ["sehgnn_lite", "hettree_lite"]


def _method_token(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def _ratio_token(ratio: float) -> str:
    return f"{float(ratio):.4f}".replace(".", "p").rstrip("0").rstrip("p")


def _source_run_name(dataset: str, method: str, ratio: float, seed: int) -> str:
    return f"next15_hettree_{dataset.lower()}_{_method_token(method)}_r{_ratio_token(ratio)}_seed{int(seed)}"


def _level_number(path: Path) -> int:
    return int(path.name.removeprefix("level_"))


def _final_level_dir(run_dir: Path) -> Path | None:
    levels = [path for path in run_dir.glob("level_*") if path.is_dir() and (path / "schema.json").exists()]
    return max(levels, key=_level_number) if levels else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_assignment(path: Path) -> np.ndarray:
    return np.load(path)["assignment"].astype(np.int64, copy=False)


def _base_assignment(source_run_dir: Path) -> tuple[Assignment, Path]:
    final_level = _final_level_dir(source_run_dir)
    if final_level is None:
        raise FileNotFoundError(f"missing final level in {source_run_dir}")
    cumulative = final_level / "cumulative_assignment.npz"
    if not cumulative.exists():
        raise FileNotFoundError(f"missing cumulative assignment: {cumulative}")
    coarse = load_graph(final_level)
    return Assignment(_load_assignment(cumulative), coarse.node_type.astype(np.int32, copy=False)), final_level


def _ratio_label(ratio: float) -> str:
    return f"{float(ratio) * 100:.1f}%"


def _index_baseline(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str], Mapping[str, str]]:
    return {
        (str(row.get("dataset")), str(row.get("method")), str(row.get("target_ratio")), str(row.get("seed"))): row
        for row in rows
    }


def _baseline_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seh = _index_baseline(_read_csv(args.next16_results / "sehgnn_runs.csv"))
    het = _index_baseline(_read_csv(args.next15_results / "hettree_runs.csv"))
    for dataset in args.datasets:
        for method in args.methods:
            for ratio in args.ratios:
                for seed in args.seeds:
                    key = (str(dataset), str(method), str(float(ratio)), str(int(seed)))
                    for model_name, source in (("sehgnn_lite", seh), ("hettree_lite", het)):
                        source_row = source.get(key)
                        if source_row is None:
                            continue
                        rows.append(
                            {
                                "priority": "P0_baseline",
                                "variant": "A0_current_all_type",
                                "dataset": dataset,
                                "method": method,
                                "target_ratio": float(ratio),
                                "ratio_label": _ratio_label(float(ratio)),
                                "seed": int(seed),
                                "model_name": model_name,
                                "eval_mode": "coarse_transfer",
                                "task_eval_mode": "mode_a_original_transfer",
                                "official_repo": "no",
                                "official_preprocess": "no",
                                "adapter_mode": "lite_previous_run",
                                "run_status": "success",
                                "macro_f1": source_row.get("macro_f1", ""),
                                "micro_f1": source_row.get("micro_f1", ""),
                                "accuracy": source_row.get("accuracy", source_row.get("micro_f1", "")),
                                "primary_task_metric": source_row.get("primary_task_metric", source_row.get("macro_f1", "")),
                                "source_result_dir": str(args.next16_results if model_name == "sehgnn_lite" else args.next15_results),
                            }
                        )
    return rows


def _official_split(original: HeteroGraph, target_type: int, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    return select_task_protocol_split(
        original,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
    )[:3]


def _mode_rows(
    metrics: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    variant: str,
    priority: str,
    target_diag: Mapping[str, Any],
    selection_diag: Mapping[str, Any] | None = None,
    meta_diag: Mapping[str, Any] | None = None,
    distill_diag: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base = {
        **common,
        "priority": priority,
        "variant": variant,
        "official_repo": metrics.get("official_repo", "no"),
        "official_preprocess": metrics.get("official_preprocess", "no"),
        "adapter_mode": metrics.get("adapter_mode", "approximate"),
        "path_set": metrics.get("path_set", "lite"),
        "target_identity": target_diag.get("target_identity", ""),
        "target_global_ratio": target_diag.get("global_ratio", ""),
        "target_anchor_count": (selection_diag or {}).get("selected_count", ""),
        "meta_recon_relative_error": (meta_diag or {}).get("meta_recon_relative_error", ""),
        "distillation_proxy_kl": (distill_diag or {}).get("distillation_proxy_kl", ""),
    }
    return [
        {
            **base,
            "eval_mode": "coarse_transfer",
            "task_eval_mode": "mode_a_original_transfer",
            "full_target_inference": False,
            "run_status": "success",
            "macro_f1": metrics.get("transfer_original_macro_f1", metrics.get("macro_f1", "")),
            "micro_f1": metrics.get("transfer_original_micro_f1", metrics.get("micro_f1", "")),
            "accuracy": metrics.get("transfer_original_accuracy", metrics.get("accuracy", "")),
            "primary_task_metric": metrics.get("transfer_original_macro_f1", metrics.get("macro_f1", "")),
        },
        {
            **base,
            "eval_mode": "full_target_inference",
            "task_eval_mode": "mode_b_full_target_inference",
            "full_target_inference": True,
            "run_status": "success",
            "macro_f1": metrics.get("mode_b_original_macro_f1", metrics.get("macro_f1", "")),
            "micro_f1": metrics.get("mode_b_original_micro_f1", metrics.get("micro_f1", "")),
            "accuracy": metrics.get("mode_b_original_accuracy", metrics.get("accuracy", "")),
            "primary_task_metric": metrics.get("primary_task_metric", metrics.get("macro_f1", "")),
        },
    ]


def _distill_diag(original: HeteroGraph, *, target_type: int, seed: int) -> dict[str, float]:
    target_nodes = nodes_of_type(original, int(target_type))
    feature = (original.features or {}).get(int(target_type))
    if feature is None or len(target_nodes) == 0:
        return {"distillation_proxy_kl": 0.0}
    labels = np.asarray(original.labels if original.labels is not None else np.zeros(original.num_nodes, dtype=np.int64))
    num_classes = int(labels[labels >= 0].max(initial=0)) + 1
    teacher = deterministic_teacher_logits(feature, num_classes=num_classes, seed=seed)
    student = deterministic_teacher_logits(feature, num_classes=num_classes, seed=seed + 17)
    return {"distillation_proxy_kl": kl_divergence_from_logits(student, teacher)}


def _evaluate_combo(
    *,
    original: HeteroGraph,
    source_run_dir: Path,
    dataset: str,
    method: str,
    ratio: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_assignment, final_level = _base_assignment(source_run_dir)
    target_type = infer_target_node_type(original)
    hybrid = build_support_coarsened_hybrid(original, base_assignment, target_node_type=target_type)
    train_nodes, val_nodes, test_nodes = _official_split(original, target_type, seed, args)
    budget_info = target_anchor_budget(
        original,
        target_node_type=target_type,
        train_nodes=train_nodes,
        global_target_ratio=float(ratio),
        mode="accuracy_first",
    )
    selection = select_target_anchors(
        original,
        target_node_type=target_type,
        train_nodes=train_nodes,
        budget=int(budget_info["target_anchor_budget"]),
        seed=int(seed),
    )
    type_budget = compute_type_budget_report(original, hybrid.graph, target_node_type=target_type)
    meta = target_tree_reconstruction_error(original, hybrid.graph, target_node_type=target_type, max_paths=16)
    distill = _distill_diag(original, target_type=target_type, seed=seed)
    score = accuracy_first_score(
        delta_anchor_support=1.0 - float(hybrid.diagnostics["global_ratio"]),
        delta_target_context=0.0 if hybrid.diagnostics.get("target_identity") else 1.0,
        delta_meta_recon=float(meta["meta_recon_relative_error"]),
        delta_teacher_support=float(distill["distillation_proxy_kl"]),
    )

    common = {
        "dataset": dataset,
        "method": method,
        "target_ratio": float(ratio),
        "ratio_label": _ratio_label(float(ratio)),
        "seed": int(seed),
        "target_node_type_id": int(target_type),
        "source_run_dir": str(source_run_dir),
        "source_final_level_dir": str(final_level),
        "actual_global_ratio": float(hybrid.graph.num_nodes / max(original.num_nodes, 1)),
        "score_acc": score["score_acc"],
    }
    rows: list[dict[str, Any]] = []
    budget_rows = [
        {
            **common,
            "priority": "P5_type_budget",
            "variant": "type_budget_target_preserve",
            "model_name": "budget_report",
            "eval_mode": "budget",
            "run_status": "success",
            "global_ratio": type_budget["global_ratio"],
            "per_type": type_budget["per_type"],
            "target_anchor_budget": budget_info["target_anchor_budget"],
            "train_target_nodes": budget_info["train_target_nodes"],
        }
    ]
    official_split = {"train": train_nodes, "valid": val_nodes, "test": test_nodes}
    selected_anchor_split = {"train": selection.selected_nodes, "valid": val_nodes, "test": test_nodes}
    for model_name in MODELS:
        for variant, priority, split, meta_diag, distill_diag in (
            ("A1_target_preserve", "P1_target_preserve", official_split, None, None),
            ("A2_hybridA_keepall", "P2_hybrid", official_split, None, None),
            ("A3_hybridB_selecttarget", "P2_hybrid", selected_anchor_split, None, None),
            ("A4_hybridB_meta_recon", "P4_task_aligned", selected_anchor_split, meta, None),
            ("A5_hybridB_distill", "P4_task_aligned", selected_anchor_split, meta, distill),
        ):
            metrics = evaluate_full_target_inference(
                original=original,
                hybrid=hybrid.graph,
                original_to_hybrid=hybrid.assignment.assignment,
                target_node_type=target_type,
                model_name=model_name,
                seed=int(seed),
                epochs=int(args.epochs),
                hidden_dim=int(args.hidden_dim),
                device=str(args.device),
                train_fraction=float(args.train_fraction),
                val_fraction=float(args.val_fraction),
                official_split_nodes=split,
            ).metrics
            rows.extend(
                _mode_rows(
                    metrics,
                    common={**common, "model_name": model_name},
                    variant=variant,
                    priority=priority,
                    target_diag=hybrid.diagnostics,
                    selection_diag=selection.diagnostics,
                    meta_diag=meta_diag,
                    distill_diag=distill_diag,
                )
            )
    return rows, budget_rows


def _rows_for_block(all_rows: Sequence[Mapping[str, Any]], block: str) -> list[dict[str, Any]]:
    if block == "target_preserve":
        variants = {"A0_current_all_type", "A1_target_preserve"}
        return [dict(row) for row in all_rows if row.get("variant") in variants]
    if block == "hybrid":
        variants = {"A0_current_all_type", "A2_hybridA_keepall", "A3_hybridB_selecttarget", "A4_hybridB_meta_recon", "A5_hybridB_distill"}
        return [dict(row) for row in all_rows if row.get("variant") in variants and row.get("eval_mode") == "full_target_inference"]
    if block == "full_target_protocol":
        return [dict(row) for row in all_rows if row.get("eval_mode") in {"coarse_transfer", "full_target_inference"}]
    if block == "task_aligned":
        variants = {"A3_hybridB_selecttarget", "A4_hybridB_meta_recon", "A5_hybridB_distill"}
        return [dict(row) for row in all_rows if row.get("variant") in variants and row.get("eval_mode") == "full_target_inference"]
    if block == "model_fidelity":
        return [dict(row) for row in all_rows if row.get("model_name") in set(MODELS)]
    return []


def _final_summary(output: Path, rows: Sequence[Mapping[str, Any]], budget_rows: Sequence[Mapping[str, Any]]) -> None:
    mode_b = [row for row in rows if row.get("eval_mode") == "full_target_inference" and row.get("run_status") == "success"]
    aggregates = aggregate_rows(mode_b, ["variant", "model_name", "eval_mode"])
    protocol_aggregates = aggregate_rows(
        [row for row in rows if row.get("run_status") == "success"],
        ["variant", "model_name", "eval_mode"],
    )
    best_mode_b = sorted(
        (
            row
            for row in aggregates
            if row.get("macro_f1_mean") not in {"", None}
        ),
        key=lambda row: float(row.get("macro_f1_mean", 0.0)),
        reverse=True,
    )[:3]
    lines = [
        "# Next17 Accuracy Branch Final Summary",
        "",
        "This run keeps the existing HeSF-LVC-P/S preservation-first configs unchanged and evaluates a separate HeSF-Acc branch.",
        "",
        "## Variant Aggregate",
        "",
        "| variant | model | macro_f1 | accuracy | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row.get('variant')} | {row.get('model_name')} | {row.get('macro_f1_mean')} | {row.get('accuracy_mean')} | {row.get('run_count')} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Aggregate Including A0",
            "",
            "| variant | model | eval_mode | macro_f1 | accuracy | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in protocol_aggregates:
        lines.append(
            f"| {row.get('variant')} | {row.get('model_name')} | {row.get('eval_mode')} | {row.get('macro_f1_mean')} | {row.get('accuracy_mean')} | {row.get('run_count')} |"
        )
    lines.extend(
        [
            "",
            "## Observed Outcome",
            "",
            "- A1 target-preserve and A2 Hybrid-A are tied in this local adapter because both keep all target nodes and coarsen support nodes.",
            "- A3/A4/A5 Hybrid-B selected-anchor variants underperform A1/A2 on the Mode-B aggregate in both lite evaluators.",
            "- A4 meta-reconstruction and A5 distillation are recorded as diagnostics/proxy terms in this run; they do not improve the task metric over A3.",
            "- A0 is available only as the previous coarse-transfer baseline, so it is protocol-separated and should not be mixed into Mode-B claims.",
            "- Official SeHGNN/HETTREE integration is not completed; all high-fidelity fields are explicitly tagged as non-official lite adapters.",
            "",
            "## Best Mode-B Candidates",
            "",
            "| rank | variant | model | macro_f1 | accuracy |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for rank, row in enumerate(best_mode_b, start=1):
        lines.append(
            f"| {rank} | {row.get('variant')} | {row.get('model_name')} | {row.get('macro_f1_mean')} | {row.get('accuracy_mean')} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            "1. Not merging target nodes: implemented and reported in `target_preserve/`; in this lite evaluation it does not consistently beat the previous A0 coarse-transfer baseline.",
            "2. Hybrid-A vs Hybrid-B: Hybrid-A/A1 is stronger than Hybrid-B in the aggregate; Hybrid-B is not the lowest-cost/highest-return direction from this run.",
            "3. Full-target inference vs coarse-transfer: see `full_target_protocol/`; Mode A and Mode B are separated by `eval_mode` and should not be mixed.",
            "4. Meta-recon / distillation: see `task_aligned/`; A4/A5 add diagnostics/proxy supervision but do not improve over A3 in the current adapter.",
            "5. Type-wise budgets: see `type_budget/`; target ratio is reported separately from support and global ratio.",
            "6. Official / high-fidelity evaluator: see `model_fidelity/`; current run uses `official_repo=no`, `adapter_mode=approximate`, `path_set=lite`.",
            "7. Recommended continuation: do not promote HeSF-Acc Hybrid-B to the paper mainline from these lite results; keep the preservation-first paper line unless a faithful official evaluator reverses this outcome.",
            "",
            "## Caveat",
            "",
            "The SeHGNN/HETTREE evaluators are local lite/adapter implementations, not official reproduction scripts.",
        ]
    )
    (output / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(output / "final_mode_b_aggregate.csv", aggregates)
    write_csv(output / "type_budget" / "runs.csv", budget_rows)


def run_next17_hybrid_accuracy(args: argparse.Namespace) -> dict[str, int]:
    root = repo_root()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = _baseline_rows(args)
    budget_rows: list[dict[str, Any]] = []
    server_commands: list[list[str]] = []
    for dataset in args.datasets:
        original = load_graph(root / DATASETS[str(dataset)])
        for method in args.methods:
            for ratio in args.ratios:
                for seed in args.seeds:
                    source = args.source_runs_root / _source_run_name(str(dataset), str(method), float(ratio), int(seed))
                    if not source.exists():
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "target_ratio": float(ratio),
                                "seed": int(seed),
                                "variant": "source_missing",
                                "model_name": "",
                                "eval_mode": "",
                                "run_status": "missing_source",
                                "skip_reason": str(source),
                            }
                        )
                        continue
                    try:
                        combo_rows, combo_budget = _evaluate_combo(
                            original=original,
                            source_run_dir=source,
                            dataset=str(dataset),
                            method=str(method),
                            ratio=float(ratio),
                            seed=int(seed),
                            args=args,
                        )
                        rows.extend(combo_rows)
                        budget_rows.extend(combo_budget)
                    except RuntimeError as exc:
                        reason = str(exc)
                        status = "oom" if "out of memory" in reason.lower() else "error"
                        if status == "oom":
                            server_commands.append(_server_command(args))
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "target_ratio": float(ratio),
                                "seed": int(seed),
                                "variant": "runtime_failure",
                                "run_status": status,
                                "skip_reason": reason,
                            }
                        )
                    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "runs.csv", rows)
    for block, title in (
        ("target_preserve", "Next17 P1 Target-Preserve Support Coarsening"),
        ("hybrid", "Next17 P2 Hybrid Accuracy Branch"),
        ("full_target_protocol", "Next17 P3 Full-Target Protocol Split"),
        ("task_aligned", "Next17 P4 Task-Aligned Ablations"),
        ("model_fidelity", "Next17 P6 Model Fidelity"),
    ):
        summarize_block(args.output / block, _rows_for_block(rows, block), title=title)
    summarize_block(args.output / "type_budget", budget_rows, title="Next17 P5 Type Budgets")
    _final_summary(args.output, rows, budget_rows)
    if server_commands:
        write_json(args.output / "server_commands.json", {"commands": server_commands})
    return {"rows": len(rows), "budget_rows": len(budget_rows), "server_commands": len(server_commands)}


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "experiments.scripts.run_next17_hybrid_accuracy",
        "--output",
        str(args.output),
        "--device",
        "cuda",
        "--skip-existing",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Next17 HeSF-Acc hybrid accuracy experiments.")
    parser.add_argument("--source-runs-root", type=Path, default=Path("outputs/exp_next15_hettree_compression_20260518/runs"))
    parser.add_argument("--next16-results", type=Path, default=Path("outputs/exp_next16_sehgnn_compression_20260518"))
    parser.add_argument("--next15-results", type=Path, default=Path("outputs/exp_next15_hettree_compression_20260518"))
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_next17_accuracy_branch_20260518"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_next17_hybrid_accuracy(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
