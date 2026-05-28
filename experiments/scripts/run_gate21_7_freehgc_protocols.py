from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import add_gate21_7_common_args, datasets, ensure_layout
from hesf_coarsen.eval.official.freehgc_env_bridge import freehgc_preflight
from hesf_coarsen.eval.official.freehgc_protocol_runner import run_freehgc_protocol_rows
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


STANDARD_METHODS = ["FreeHGC", "HGCond", "GCond-HG", "Random-HG", "Herding-HG", "KCenter-HG", "Coarsening-HG"]


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["standard_condensation"]
    rows = []
    failures = []
    preflight = freehgc_preflight(freehgc_root=args.freehgc_root)
    write_json(out / "gate21_7_freehgc_preflight.json", preflight)
    if args.dry_run:
        for dataset in datasets(args):
            for method in STANDARD_METHODS:
                rows.append(_not_configured(dataset, method, "dry_run"))
    else:
        for dataset in datasets(args):
            if args.protocol in {"standard", "both"}:
                protocol_rows, protocol_failures = run_freehgc_protocol_rows(
                    dataset=dataset,
                    ratios=args.ratios,
                    freehgc_root=args.freehgc_root,
                    data_root=Path(args.official_sehgnn_root) / "data",
                    output_dir=out,
                    device=args.device,
                    quick=bool(args.quick),
                    strict=bool(args.strict),
                    run_upstream=True,
                    timeout_seconds=args.timeout_seconds,
                )
                rows.extend([row for row in protocol_rows if row.get("method") == "FreeHGC"])
                failures.extend(protocol_failures)
            if args.protocol in {"tp", "both"}:
                protocol_rows, protocol_failures = run_freehgc_protocol_rows(
                    dataset=dataset,
                    ratios=args.ratios,
                    freehgc_root=args.freehgc_root,
                    data_root=Path(args.official_sehgnn_root) / "data",
                    output_dir=out,
                    device=args.device,
                    quick=bool(args.quick),
                    strict=bool(args.strict),
                    run_upstream=False,
                    timeout_seconds=args.timeout_seconds,
                )
                rows.extend([row for row in protocol_rows if row.get("method") == "FreeHGC-TP"])
                failures.extend(protocol_failures)
            for method in [m for m in STANDARD_METHODS if m != "FreeHGC"]:
                rows.append(_not_configured(dataset, method, "external_repository_not_integrated"))
    write_csv(out / "gate21_7_standard_condensation_by_run.csv", rows)
    write_csv(out / "gate21_7_standard_condensation_by_method.csv", rows)
    write_csv(out / "gate21_7_standard_condensation_failure_log.csv", [row for row in rows if not _bool(row.get("success"))] + failures)
    write_json(out / "gate21_7_standard_condensation_plan.json", {"methods": STANDARD_METHODS, "ratios": list(args.ratios), "protocol": args.protocol})
    return {"rows": len(rows), "failure_rows": len([row for row in rows if not _bool(row.get("success"))])}


def _not_configured(dataset: str, method: str, reason: str) -> dict[str, object]:
    return {
        "dataset": str(dataset).upper(),
        "method": method,
        "baseline_name": method,
        "protocol": "standard_condensation",
        "method_family": "standard_condensation_baseline",
        "external_baseline": True,
        "success": False,
        "success_count": 0,
        "training_executed": False,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "eligible_for_standard_condensation_table": False,
        "failure_type": "not_configured",
        "failure_message": reason,
        "test_micro_f1": "",
        "test_macro_f1": "",
    }


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--protocol", choices=["standard", "tp", "both"], default="both")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.012, 0.024, 0.048, 0.096, 0.12])
    parser.add_argument("--timeout-seconds", type=int, default=120 if sys.platform.startswith("win") else 300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
