from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import GATE21_6_SOURCE, add_gate21_7_common_args, datasets, ensure_layout, read_csv
from hesf_coarsen.eval.official.auto_relation_channel_selector import build_relation_channel_keep_plan, plan_to_jsonl_row
from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS


METHODS = [
    "full-native",
    "export-full",
    "H6-node30",
    "random-edge",
    "HeSF-RCS-auto-coverage-structural30",
    "HeSF-RCS-auto-probe-structural30",
    "HeSF-RCS-auto-coverage-structural20",
    "HeSF-RCS-auto-probe-structural20",
    "best-external-TP",
]


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["cross_dataset"]
    plan_lines: list[str] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in datasets(args):
        counts, error = _relation_counts(dataset, Path(args.official_sehgnn_root))
        if error:
            failures.append({"dataset": dataset, "failure_type": "dataset_not_available_locally", "failure_message": error})
            continue
        for mode in ["coverage_greedy", "validation_probe_greedy"]:
            plan = build_relation_channel_keep_plan(dataset=dataset, target_type=str(SEHGNN_HGB_SCHEMAS[dataset]["target_type"]), relation_edge_counts=counts, mode=mode)
            plan_lines.append(plan_to_jsonl_row(plan))
        for method in METHODS:
            training_required = dataset in {"ACM", "IMDB"} or method in {"full-native", "export-full"}
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "plan_ready": True,
                    "auto_channel_plan_ready": method.startswith("HeSF-RCS-auto"),
                    "relation_count_source": str(Path(args.official_sehgnn_root) / "data" / dataset / "link.dat"),
                    "training_executed": False,
                    "success": False,
                    "success_count": 0,
                    "test_micro_f1": "",
                    "test_macro_f1": "",
                    "failure_type": "not_executed_local_gate21_7_quick" if training_required else "diagnostic_plan_only",
                    "failure_message": "Cross-dataset task training is not claimed without a local official SeHGNN run.",
                    "used_test_data": False,
                }
            )
    (out / "gate21_7_cross_dataset_auto_channel_plans.jsonl").write_text("\n".join(plan_lines) + ("\n" if plan_lines else ""), encoding="utf-8")
    write_csv(out / "gate21_7_cross_dataset_by_run.csv", rows)
    write_csv(out / "gate21_7_cross_dataset_by_method.csv", rows)
    write_csv(out / "gate21_7_cross_dataset_failure_log.csv", failures + [row for row in rows if row.get("failure_type")])
    return {"rows": len(rows), "failure_rows": len(failures)}


def _relation_counts(dataset: str, sehgnn_root: Path) -> tuple[dict[str, int], str]:
    schema = SEHGNN_HGB_SCHEMAS.get(dataset)
    if schema is None:
        return {}, "unsupported_dataset"
    link_path = sehgnn_root / "data" / dataset / "link.dat"
    if not link_path.exists():
        return {}, f"missing_link_dat:{link_path}"
    id_to_name = {int(value): str(name) for name, value in dict(schema["relation_id_order"]).items()}
    counts = {name: 0 for name in id_to_name.values()}
    with link_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = id_to_name.get(int(parts[2]), str(parts[2]))
                counts[name] = counts.get(name, 0) + 1
    return counts, ""


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(datasets=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--gate21-6-dir", type=Path, default=GATE21_6_SOURCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
