from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import add_gate21_7_common_args, datasets, ensure_layout, training_seeds, write_plan
from hesf_coarsen.eval.official.external_tp_hgb_runner import ensure_external_tp_task_metrics
from hesf_coarsen.eval.official.external_tp_task_runner import (
    artifact_audit_rows,
    build_external_tp_task_rows,
    failure_log_rows,
    summarize_external_tp_by_method,
)
from hesf_coarsen.eval.official.runner_utils import write_csv


DEFAULT_METHODS = ["Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "Coarsening-HG-TP", "GraphSparsify-TP", "FreeHGC-TP"]


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["external_tp"]
    task_metrics_csv = args.task_metrics_csv
    if task_metrics_csv is None and bool(args.run_task_metrics) and not bool(args.dry_run):
        task_metrics_csv = ensure_external_tp_task_metrics(
            dataset=str(args.dataset).upper(),
            methods=list(args.methods),
            source_data_root=Path(args.official_sehgnn_root) / "data",
            sehgnn_repo=Path(args.official_sehgnn_root),
            output_dir=out,
            support_node_ratio=0.50,
            graph_seed=int(args.graph_seeds[0]),
            training_seed=int(training_seeds(args)[0]),
            device=str(args.device),
            task_epochs=int(args.task_epochs),
            force_reprocess=False,
        )
    rows = []
    for dataset in datasets(args):
        rows.extend(
            build_external_tp_task_rows(
                dataset=dataset,
                methods=args.methods,
                support_node_ratios=args.support_node_ratios,
                structural_budgets=args.structural_budgets,
                graph_seeds=args.graph_seeds,
                training_seeds=training_seeds(args),
                freehgc_root=args.freehgc_root,
                native_hgb_root=Path(args.official_sehgnn_root) / "data",
                task_metrics_csv=task_metrics_csv,
                quick=bool(args.quick),
            )
        )
    by_method = summarize_external_tp_by_method(rows)
    audit = artifact_audit_rows(rows)
    failures = failure_log_rows(rows)
    write_csv(out / "gate21_7_external_tp_by_run.csv", rows)
    write_csv(out / "gate21_7_external_tp_by_method.csv", by_method)
    write_csv(out / "gate21_7_external_tp_artifact_audit.csv", audit)
    write_csv(out / "gate21_7_external_tp_failure_log.csv", failures)
    write_plan(
        out / "gate21_7_external_tp_plan.json",
        {
            "methods": list(args.methods),
            "support_node_ratios": list(args.support_node_ratios),
            "structural_budgets": list(args.structural_budgets),
            "graph_seeds": list(args.graph_seeds),
            "training_seeds": training_seeds(args),
            "freehgc_root": str(args.freehgc_root),
            "task_metrics_csv": "" if task_metrics_csv is None else str(task_metrics_csv),
            "strict_ready_rule": "official_hgb_exported && training_executed && success_count>0 && finite test metrics",
            "run_task_metrics": bool(args.run_task_metrics),
            "task_epochs": int(args.task_epochs),
        },
    )
    (out / "gate21_7_external_tp_decision.md").write_text(
        "# Gate21.7 External TP Decision\n\n"
        f"- run rows: {len(rows)}\n"
        f"- ready method rows: {sum(1 for row in by_method if str(row.get('task_result_ready')).lower() == 'true')}\n"
        "- Rows without task metrics remain explicit failure/not-ready rows.\n",
        encoding="utf-8",
    )
    return {"run_rows": len(rows), "method_rows": len(by_method), "failure_rows": len(failures)}


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--support-node-ratios", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--structural-budgets", nargs="+", type=float, default=[0.12, 0.16, 0.20, 0.30])
    parser.add_argument("--task-metrics-csv", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
