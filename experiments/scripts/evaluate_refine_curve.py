from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.run_hgb_task_eval import evaluate_run


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return ""


def _derived_rows(summary_row: Mapping[str, Any], refine_epochs: list[int]) -> list[dict[str, Any]]:
    rows = []
    for epoch in refine_epochs:
        rows.append(
            {
                "run_name": summary_row.get("run_name", ""),
                "run_dir": summary_row.get("run_dir", ""),
                "dataset": summary_row.get("dataset", ""),
                "variant": summary_row.get("variant", ""),
                "refine_epochs": int(epoch),
                "projected_original_macro_f1": _first(
                    summary_row,
                    f"task_projected_macro_f1@{epoch}",
                    f"task_projected_original_macro_f1@{epoch}",
                    "task_projected_macro_f1",
                    "task.projected_original_macro_f1",
                ),
                "refined_original_macro_f1": _first(
                    summary_row,
                    f"task_refined_macro_f1@{epoch}",
                    f"task_refined_original_macro_f1@{epoch}",
                    "task_refined_macro_f1",
                    "task.refined_original_macro_f1",
                ),
                "source": "summary",
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate or collect task refine curves for summarized runs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--refine-epochs", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-graph-rgcn-lite", action="store_true")
    return parser


def _run_dir_for(row: Mapping[str, Any], runs_root: Path | None) -> Path | None:
    run_dir = str(row.get("run_dir", "") or "")
    if run_dir:
        path = Path(run_dir)
        if path.exists():
            return path
    if runs_root is not None:
        name = str(row.get("run_name", "") or "")
        if name and (runs_root / name).exists():
            return runs_root / name
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wanted_variants = None if args.variants is None else {str(value) for value in args.variants}
    rows: list[dict[str, Any]] = []
    for summary_row in _read_csv(args.summary):
        if wanted_variants is not None and str(summary_row.get("variant", "")) not in wanted_variants:
            continue
        run_dir = _run_dir_for(summary_row, args.runs_root)
        if run_dir is None:
            rows.extend(_derived_rows(summary_row, args.refine_epochs))
            continue
        try:
            result = evaluate_run(
                run_dir,
                graph_root=args.graph_root,
                seed=args.seed,
                epochs=args.epochs,
                refine_epochs=max(args.refine_epochs),
                refine_epochs_list=args.refine_epochs,
                hidden_dim=args.hidden_dim,
                device=args.device,
                full_graph_rgcn_lite=args.full_graph_rgcn_lite,
            )
        except Exception as exc:
            derived = _derived_rows(summary_row, args.refine_epochs)
            for row in derived:
                row["status"] = "failed"
                row["failure_reason"] = str(exc)
            rows.extend(derived)
            continue
        for epoch in args.refine_epochs:
            rows.append(
                {
                    "run_name": result.get("run_name", summary_row.get("run_name", "")),
                    "run_dir": str(run_dir),
                    "dataset": result.get("dataset", summary_row.get("dataset", "")),
                    "variant": result.get("variant", summary_row.get("variant", "")),
                    "refine_epochs": int(epoch),
                    "projected_original_macro_f1": result.get("projected_original_macro_f1", ""),
                    "refined_original_macro_f1": result.get(f"refined_original_macro_f1@{epoch}", result.get("refined_original_macro_f1", "")),
                    "source": "evaluate_run",
                    "status": "success",
                }
            )
    write_csv(args.output / "task_refine_curve.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
