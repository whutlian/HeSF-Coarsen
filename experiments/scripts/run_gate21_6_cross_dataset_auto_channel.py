from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.auto_relation_channel_selector import build_relation_channel_keep_plan, plan_to_jsonl_row
from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS
from hesf_coarsen.eval.official.runner_utils import write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gate21_5_dblp_counts() -> dict[str, int]:
    rows = _read_csv(Path("results/gate21_5_directed_apv_feature_adapter/gate21_5_relation_edge_retention.csv"))
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("method") == "H6-APV-skeleton" and row.get("training_seed") == "1":
            counts[str(row.get("official_relation_name", ""))] = int(float(row.get("original_full_edge_count") or 0))
    return counts or {"AP": 19645, "PA": 19645, "PT": 85810, "TP": 85810, "PV": 14328, "VP": 14328}


def _official_relation_counts(dataset: str, sehgnn_root: Path) -> tuple[dict[str, int], str]:
    ds = str(dataset).upper()
    schema = SEHGNN_HGB_SCHEMAS.get(ds)
    if schema is None:
        return {}, "unsupported_dataset"
    dataset_dir = Path(sehgnn_root) / "data" / ds
    link_path = dataset_dir / "link.dat"
    if not link_path.exists():
        return {}, f"missing_link_dat:{link_path}"
    id_to_name = {int(value): str(name) for name, value in dict(schema["relation_id_order"]).items()}
    counts = {name: 0 for name in id_to_name.values()}
    with link_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                relation_id = int(parts[2])
                counts[id_to_name.get(relation_id, str(relation_id))] = counts.get(id_to_name.get(relation_id, str(relation_id)), 0) + 1
    return counts, ""


def run(args: argparse.Namespace) -> dict[str, int]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    jsonl = []
    for dataset in args.datasets:
        ds = str(dataset).upper()
        if ds == "DBLP":
            counts = _gate21_5_dblp_counts()
            source = "gate21_5_relation_edge_retention"
        else:
            counts, error = _official_relation_counts(ds, Path(args.official_sehgnn_root))
            source = str(Path(args.official_sehgnn_root) / "data" / ds / "link.dat")
            if error:
                failures.append({"dataset": ds, "failure_type": "dataset_not_available_locally", "failure_message": error})
                continue
        if not counts:
            failures.append({"dataset": ds, "failure_type": "empty_relation_counts", "failure_message": "No relation channels were available for auto-channel planning."})
            continue
        for selector in args.selector:
            target_type = str(SEHGNN_HGB_SCHEMAS[ds]["target_type"])
            plan = build_relation_channel_keep_plan(dataset=ds, target_type=target_type, relation_edge_counts=counts, mode=str(selector))
            jsonl.append(plan_to_jsonl_row(plan))
            rows.append(
                {
                    "dataset": ds,
                    "selector": selector,
                    "method": f"auto-{selector}",
                    "success": True,
                    "used_test_data": False,
                    "relation_count_source": source,
                    "relation_channel_keep_plan_path": "gate21_6_auto_channel_plans.jsonl",
                }
            )
    (out / "gate21_6_auto_channel_plans.jsonl").write_text("\n".join(jsonl) + ("\n" if jsonl else ""), encoding="utf-8")
    write_csv(out / "gate21_6_cross_dataset_auto_channel_by_method.csv", rows)
    write_csv(out / "gate21_6_cross_dataset_failure_log.csv", failures)
    return {"auto_channel_rows": len(rows), "failure_rows": len(failures)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--selector", nargs="+", default=["coverage_greedy", "validation_probe_greedy"])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
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
        args.datasets = ["DBLP"]
        args.selector = [args.selector[0]]
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
