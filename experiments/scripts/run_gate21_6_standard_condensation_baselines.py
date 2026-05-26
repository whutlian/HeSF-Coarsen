from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.icde_protocol import PROTOCOL_STANDARD_CONDENSATION, build_protocol_row
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def run(args: argparse.Namespace) -> dict[str, int]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            rows.append(
                {
                    **build_protocol_row(
                        baseline_name=str(method),
                        protocol=PROTOCOL_STANDARD_CONDENSATION,
                        method_family="standard_condensation_baseline",
                        schema_compatible=False,
                        official_sehgnn_unmodified=False,
                        keeps_all_target_nodes=False,
                    ),
                    "dataset": str(dataset).upper(),
                    "method": str(method),
                    "success": False,
                    "failure_type": "missing_external_dependency_or_not_configured",
                    "failure_message": "Standard condensation baseline hook is separated from the schema-preserving official table.",
                }
            )
    write_csv(out / "gate21_6_standard_condensation_by_run.csv", rows)
    write_csv(out / "gate21_6_standard_condensation_by_method.csv", rows)
    write_json(out / "gate21_6_standard_condensation_plan.json", {"methods": list(args.methods)})
    return {"standard_condensation_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--methods", nargs="+", default=["FreeHGC", "HGCond", "GCond-HG", "Random-HG", "Herding-HG", "KCenter-HG", "Coarsening-HG"])
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
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
