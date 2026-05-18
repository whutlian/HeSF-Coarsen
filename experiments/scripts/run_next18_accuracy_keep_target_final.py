from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, repo_root, write_csv, write_json
from experiments.scripts.run_next17_hybrid_accuracy import (
    DATASETS,
    DEFAULT_METHODS,
    DEFAULT_RATIOS,
    DEFAULT_SEEDS,
    _base_assignment,
    _official_split,
    _source_run_name,
)
from experiments.scripts.summarize_next17_hybrid_accuracy import aggregate_rows
from hesf_coarsen.accuracy.full_target_inference import evaluate_full_target_inference
from hesf_coarsen.accuracy.full_target_protocol import make_protocol_row
from hesf_coarsen.accuracy.model_fidelity_registry import fidelity_record
from hesf_coarsen.accuracy.target_support_hybrid import build_support_coarsened_hybrid
from hesf_coarsen.accuracy.type_budgets import compute_type_budget_report
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.io.edge_list import load_graph


VARIANTS = ("A1_target_preserve", "A2_hybridA_keepall")
MODELS = ("sehgnn_lite", "hettree_lite")
COMPARATORS = ("flatten-sum_keep_target", "H6_keep_target", "TypedHash-ChebHeat_keep_target")


def _ratio_label(ratio: float) -> str:
    return f"{float(ratio) * 100:.1f}%"


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "experiments.scripts.run_next18_accuracy_keep_target_final",
        "--output",
        str(args.output),
        "--device",
        "cuda",
    ]


def _protocol_rows(
    metrics: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    model_name: str,
) -> list[dict[str, Any]]:
    fidelity = fidelity_record(model_name)
    rows = []
    for eval_mode in ("coarse_transfer", "real_full_target_inference"):
        row = make_protocol_row(
            metrics,
            eval_mode=eval_mode,
            model_name=model_name,
            model_fidelity=fidelity["model_fidelity"],
            official_repo=fidelity["official_repo"],
            official_preprocess=fidelity["official_preprocess"],
            adapter_mode="target_preserve_direct" if eval_mode == "real_full_target_inference" else fidelity["adapter_mode"],
            path_set=fidelity["path_set"],
            split_policy=str(metrics.get("task_split_policy", fidelity["split_policy"])),
            max_hops=metrics.get("max_hops", fidelity["max_hops"]),
            extra={
                **common,
                "run_status": "success",
                "runtime": metrics.get("total_time", ""),
                "train_time": metrics.get("train_time", ""),
                "peak_vram_allocated_mb": metrics.get("peak_vram_allocated_mb", ""),
                "micro_f1_secondary": metrics.get("micro_f1", ""),
            },
        )
        rows.append(row)
    return rows


def _comparator_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for dataset in args.datasets:
        for comparator in COMPARATORS:
            for model_name in MODELS:
                fidelity = fidelity_record(model_name)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": comparator,
                        "variant": comparator,
                        "target_ratio": "",
                        "ratio_label": "",
                        "seed": "",
                        "model_name": model_name,
                        "model_fidelity": fidelity["model_fidelity"],
                        "eval_mode": "real_full_target_inference",
                        "official_repo": fidelity["official_repo"],
                        "official_preprocess": fidelity["official_preprocess"],
                        "adapter_mode": fidelity["adapter_mode"],
                        "split_policy": fidelity["split_policy"],
                        "path_set": fidelity["path_set"],
                        "max_hops": fidelity["max_hops"],
                        "target_domain": "original_target_nodes",
                        "support_domain": "compressed_support_nodes",
                        "inference_domain": "full_original_target_set",
                        "run_status": "skipped_no_keep_target_assignment",
                        "skip_reason": "Protocol-matched keep-target comparator assignments are not available locally.",
                    }
                )
    return rows


def _write_summaries(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    successful = [row for row in rows if row.get("run_status") == "success"]
    by_dataset = aggregate_rows(successful, ["dataset", "variant", "model_name", "eval_mode"])
    by_seed = aggregate_rows(successful, ["dataset", "variant", "model_name", "eval_mode", "seed"])
    protocol = aggregate_rows(successful, ["variant", "model_name", "eval_mode"])
    write_csv(output / "variant_main_table.csv", protocol)
    write_csv(output / "by_dataset.csv", by_dataset)
    write_csv(output / "by_seed.csv", by_seed)
    write_csv(output / "protocol_separated_tables.csv", protocol)
    skipped = [row for row in rows if row.get("run_status") != "success"]
    lines = [
        "# Next18 Keep-Target Accuracy Final Validation",
        "",
        "This run evaluates only A1/A2 as serious local diagnostics. Official/high-fidelity rows are unavailable, so these results do not support paper-facing task claims.",
        "",
        "## Protocol-Separated Local Adapter Results",
        "",
        markdown_table(protocol, ["variant", "model_name", "eval_mode", "macro_f1_mean", "accuracy_mean", "run_count", "failed_count"]),
        "",
        "## Required Questions",
        "",
        "1. Keeping all target nodes is implemented and audited, but current rows are lite-adapter diagnostics rather than faithful evaluator evidence.",
        "2. Real full-target inference is separated from coarse transfer and uses explicit `hybrid_target_original_*` metrics. For A1/A2 the values can match coarse-transfer numerically because every target node is singleton, but the metric provenance is no longer the old approximate wrapper.",
        "3. A1/A2 cannot be honestly declared competitive with keep-target flatten-sum/H6/TypedHash because protocol-matched comparator assignments are unavailable locally.",
        "4. Stability across ACM/DBLP/IMDB is reported in `by_dataset.csv`, but final decision does not rely on lite-only evidence.",
        "",
        "## Skipped Comparators",
        "",
        markdown_table(skipped[:20], ["dataset", "variant", "model_name", "run_status", "skip_reason"]),
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_next18_accuracy_keep_target_final(args: argparse.Namespace) -> dict[str, int]:
    root = repo_root()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    server_commands: list[list[str]] = []
    for dataset in args.datasets:
        original = load_graph(root / DATASETS[str(dataset)])
        target_type = infer_target_node_type(original)
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
                                "run_status": "missing_source",
                                "skip_reason": str(source),
                            }
                        )
                        continue
                    base_assignment, final_level = _base_assignment(source)
                    hybrid = build_support_coarsened_hybrid(original, base_assignment, target_node_type=target_type)
                    train_nodes, val_nodes, test_nodes = _official_split(original, target_type, int(seed), args)
                    type_budget = compute_type_budget_report(original, hybrid.graph, target_node_type=target_type)
                    common = {
                        "dataset": dataset,
                        "method": method,
                        "target_ratio": float(ratio),
                        "ratio_label": _ratio_label(float(ratio)),
                        "seed": int(seed),
                        "source_run_dir": str(source),
                        "source_final_level_dir": str(final_level),
                        "target_identity": hybrid.diagnostics.get("target_identity", ""),
                        "target_cluster_size_max": hybrid.diagnostics.get("target_cluster_size_max", ""),
                        "support_compression_ratio": type_budget["global_ratio"],
                        "global_ratio": type_budget["global_ratio"],
                        "target_node_type_id": int(target_type),
                    }
                    for model_name in MODELS:
                        try:
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
                                official_split_nodes={"train": train_nodes, "valid": val_nodes, "test": test_nodes},
                            ).metrics
                            for variant in VARIANTS:
                                rows.extend(
                                    _protocol_rows(
                                        metrics,
                                        common={**common, "variant": variant},
                                        model_name=model_name,
                                    )
                                )
                        except RuntimeError as exc:
                            reason = str(exc)
                            status = "oom" if "out of memory" in reason.lower() else "error"
                            if status == "oom":
                                server_commands.append(_server_command(args))
                            rows.append({**common, "model_name": model_name, "run_status": status, "skip_reason": reason})
                    write_csv(args.output / "runs.csv", rows)
    rows.extend(_comparator_rows(args))
    write_csv(args.output / "runs.csv", rows)
    _write_summaries(args.output, rows)
    if server_commands:
        write_json(args.output / "server_commands.json", {"commands": server_commands})
    return {"rows": len(rows), "server_commands": len(server_commands)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-runs-root", type=Path, default=Path("outputs/exp_next15_hettree_compression_20260518/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_next18_accuracy_keep_target_final"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS[:3])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_next18_accuracy_keep_target_final(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
