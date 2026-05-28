from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import add_gate21_7_common_args, ensure_layout, training_seeds
from experiments.scripts.run_gate21_7_adapter_package_repaired import run as run_adapter
from experiments.scripts.run_gate21_7_apv16_graph_seed_stability import run as run_apv
from experiments.scripts.run_gate21_7_cross_dataset_task_results import run as run_cross
from experiments.scripts.run_gate21_7_external_tp_task_results import run as run_external
from experiments.scripts.run_gate21_7_feature_ablation_repaired import run as run_feature
from experiments.scripts.run_gate21_7_freehgc_protocols import run as run_freehgc
from experiments.scripts.run_gate21_7_semantic_audit_repair import run as run_semantic
from experiments.scripts.run_gate21_7_storage_system_costs import run as run_storage
from experiments.scripts.summarize_gate21_7_icde_ready import summarize
from hesf_coarsen.eval.official.runner_utils import write_json


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = ensure_layout(Path(args.output_root))
    if args.dry_run:
        plan = {
            "output_root": str(args.output_root),
            "dataset": args.dataset,
            "seeds": list(args.seeds),
            "graph_seeds": list(args.graph_seeds),
            "training_seeds": training_seeds(args),
            "quick": bool(args.quick),
            "strict": bool(args.strict),
            "subdirs": sorted(k for k in paths if k != "root"),
        }
        write_json(Path(args.output_root) / "gate21_7_dry_run_plan.json", plan)
        return plan

    common = dict(
        output_root=Path(args.output_root),
        dataset=args.dataset,
        datasets=None,
        seeds=list(args.seeds),
        graph_seeds=list(args.graph_seeds),
        training_seeds=training_seeds(args),
        dry_run=False,
        quick=bool(args.quick),
        force_reprocess=bool(args.force_reprocess),
        continue_on_failure=bool(args.continue_on_failure),
        strict=bool(args.strict),
        device=args.device,
        freehgc_root=Path(args.freehgc_root),
        official_sehgnn_root=Path(args.official_sehgnn_root),
        run_task_metrics=bool(args.run_task_metrics),
        task_epochs=int(args.task_epochs),
    )
    results = {}
    results["apv16_stability"] = _call(run_apv, argparse.Namespace(**common, methods=[
        "HeSF-RCS-APV12",
        "HeSF-RCS-APV16",
        "AP100-PA50-PV100-VP00-PTTP00",
        "AP100-PA00-PV100-VP50-PTTP00",
        "AP100-PA25-PV100-VP25-PTTP00",
        "AP100-PA75-PV100-VP75-PTTP00",
    ], gate21_6_dir=Path(args.gate21_6_dir)), args)
    results["external_tp"] = _call(run_external, argparse.Namespace(**common, methods=[
        "Random-HG-TP",
        "Herding-HG-TP",
        "KCenter-HG-TP",
        "Coarsening-HG-TP",
        "GraphSparsify-TP",
        "FreeHGC-TP",
    ], support_node_ratios=[0.10, 0.20, 0.30, 0.50], structural_budgets=[0.12, 0.16, 0.20, 0.30], task_metrics_csv=None), args)
    results["freehgc_protocols"] = _call(run_freehgc, argparse.Namespace(**common, protocol="both", ratios=[0.012, 0.024, 0.048, 0.096, 0.12], timeout_seconds=args.freehgc_timeout_seconds), args)
    results["semantic_audit"] = _call(run_semantic, argparse.Namespace(**common, methods=[
        "export-full-SeHGNN",
        "H6-node30",
        "H6-APV-skeleton",
        "HeSF-RCS-APV12",
        "HeSF-RCS-APV16",
        "APV12-PTTP10",
        "APV12-PV75",
    ], gate21_6_dir=Path(args.gate21_6_dir), gate21_5_dir=Path(args.gate21_5_dir)), args)
    results["feature_ablation_repaired"] = _call(run_feature, argparse.Namespace(**common, methods=["full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"], gate21_6_dir=Path(args.gate21_6_dir), gate21_5_dir=Path(args.gate21_5_dir)), args)
    results["adapter_package_repaired"] = _call(run_adapter, argparse.Namespace(**common, adapters=["random_projection_dim64", "random_projection_dim128", "pca_svd_dim64", "pca_svd_dim128", "int8_per_feature", "fp16_node_features"], gate21_6_dir=Path(args.gate21_6_dir)), args)
    results["storage_system_costs"] = _call(run_storage, argparse.Namespace(**common, gate21_5_dir=Path(args.gate21_5_dir)), args)
    cross_common = dict(common)
    cross_common["datasets"] = ["DBLP", "ACM", "IMDB"]
    results["cross_dataset"] = _call(run_cross, argparse.Namespace(**cross_common, gate21_6_dir=Path(args.gate21_6_dir)), args)
    decision = summarize(Path(args.output_root), Path(args.output_root) / "summaries", strict=bool(args.strict))
    results["decision_flags"] = decision.get("flags", {})
    write_json(Path(args.output_root) / "gate21_7_run_summary.json", results)
    return results


def _call(fn, ns: argparse.Namespace, parent: argparse.Namespace):
    try:
        return fn(ns)
    except Exception as exc:
        if not parent.continue_on_failure:
            raise
        return {"failure_type": "runner_exception", "failure_message": str(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--gate21-6-dir", type=Path, default=Path("results/gate21_6_icde_ready"))
    parser.add_argument("--gate21-5-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    parser.add_argument("--freehgc-timeout-seconds", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
