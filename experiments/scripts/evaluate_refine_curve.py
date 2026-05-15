from __future__ import annotations

import argparse
import csv
import json
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


def _cached_task_eval(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "task_eval.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _rows_from_task_payload(
    summary_row: Mapping[str, Any],
    run_dir: Path,
    payload: Mapping[str, Any],
    refine_epochs: list[int],
    *,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in refine_epochs:
        rows.append(
            {
                "run_name": payload.get("run_name", summary_row.get("run_name", "")),
                "run_dir": str(run_dir),
                "dataset": payload.get("dataset", summary_row.get("dataset", "")),
                "variant": payload.get("variant", summary_row.get("variant", "")),
                "refine_epochs": int(epoch),
                "projected_original_macro_f1": payload.get(
                    "projected_original_macro_f1",
                    summary_row.get("task_projected_macro_f1", ""),
                ),
                "refined_original_macro_f1": payload.get(
                    f"refined_original_macro_f1@{epoch}",
                    payload.get("refined_original_macro_f1", ""),
                ),
                "best_refined_macro_f1": payload.get("best_refined_macro_f1", ""),
                "best_refined_epoch": payload.get("best_refined_epoch", ""),
                "refine_auc_macro_f1": payload.get("refine_auc_macro_f1", ""),
                "full_graph_macro_f1": payload.get("full_graph_rgcn_lite_macro_f1", ""),
                "source": source,
                "status": "success",
            }
        )
    return rows


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
        cached = _cached_task_eval(run_dir)
        if cached is not None:
            rows.extend(
                _rows_from_task_payload(
                    summary_row,
                    run_dir,
                    cached,
                    args.refine_epochs,
                    source="task_eval_cache",
                )
            )
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
            rows.extend(
                _rows_from_task_payload(
                    summary_row,
                    run_dir,
                    result,
                    [int(epoch)],
                    source="evaluate_run",
                )
            )
    write_csv(args.output / "task_refine_curve.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
