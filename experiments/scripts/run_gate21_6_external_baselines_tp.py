from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.external_baselines_tp import plan_external_tp_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def _by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("baseline_name", "")), float(row.get("budget_value") or 0.0)), []).append(row)
    out = []
    for (name, budget), group in sorted(groups.items()):
        success = [row for row in group if str(row.get("success", "")).lower() == "true"]
        micros = [float(row.get("test_micro_f1") or 0.0) for row in success if row.get("test_micro_f1") not in {"", None}]
        first = group[0]
        out.append(
            {
                **{key: first.get(key, "") for key in first},
                "baseline_name": name,
                "budget_value": budget,
                "runs": len(group),
                "success_count": len(success),
                "test_micro_f1": float(mean(micros)) if micros else "",
                "failure_type": "" if success else first.get("failure_type", "not_run"),
                "failure_message": "" if success else first.get("failure_message", ""),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        rows.extend(
            plan_external_tp_rows(
                dataset=str(dataset).upper(),
                methods=args.methods,
                budgets=args.budgets,
                graph_seeds=args.graph_seeds,
                training_seeds=args.training_seeds,
                freehgc_root=args.freehgc_root,
                native_hgb_root=Path(args.official_sehgnn_root) / "data",
            )
        )
    by_method = _by_method(rows)
    artifact_audit = [
        {
            "dataset": row.get("dataset", ""),
            "baseline_name": row.get("baseline_name", ""),
            "graph_seed": row.get("graph_seed", ""),
            "training_seed": row.get("training_seed", ""),
            "schema_compatible": row.get("schema_compatible", ""),
            "keeps_all_target_nodes": row.get("keeps_all_target_nodes", ""),
            "official_hgb_exported": False,
            "artifact_audit_pass": str(row.get("success", "")).lower() == "true",
            "artifact_manifest_path": row.get("artifact_manifest_path", ""),
            "construction_status": row.get("construction_status", ""),
            "selection_signal": row.get("selection_signal", ""),
            "export_hash": row.get("export_hash", ""),
            "failure_type": row.get("failure_type", ""),
        }
        for row in rows
    ]
    failures = [row for row in rows if str(row.get("success", "")).lower() != "true"]
    write_csv(out / "gate21_6_external_tp_by_run.csv", rows)
    write_csv(out / "gate21_6_external_tp_by_method.csv", by_method)
    write_csv(out / "gate21_6_external_tp_artifact_audit.csv", artifact_audit)
    write_csv(out / "gate21_6_external_tp_failure_log.csv", failures)
    write_json(
        out / "gate21_6_external_tp_plan.json",
        {
            "methods": list(args.methods),
            "budgets": list(args.budgets),
            "freehgc_root": "" if args.freehgc_root is None else str(args.freehgc_root),
            "official_sehgnn_root": str(args.official_sehgnn_root),
            "construction_scope": "manifest-level persistent TP construction estimates; official training not executed in Gate21.6 budget run",
        },
    )
    return {"external_tp_run_rows": len(rows), "external_tp_method_rows": len(by_method), "failure_rows": len(failures)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--methods", nargs="+", default=["Random-HG-TP", "Herding-HG-TP", "KCenter-HG-TP", "Coarsening-HG-TP", "GraphSparsify-TP", "FreeHGC-TP"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.50, 0.30, 0.20, 0.10])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate21_6_icde_ready"))
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--freehgc-root", type=Path, default=None)
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.methods = ["Random-HG-TP", "FreeHGC-TP"]
        args.budgets = [args.budgets[0]]
        args.graph_seeds = [args.graph_seeds[0]]
        args.training_seeds = [args.training_seeds[0]]
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
