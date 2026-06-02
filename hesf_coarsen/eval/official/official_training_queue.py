from __future__ import annotations

import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command, run_native_command
from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


REQUIRED_HGB_FILES = ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat")


def verify_hgb_export_dir(export_dir: str | Path | None) -> dict[str, Any]:
    path = Path(str(export_dir or ""))
    row: dict[str, Any] = {
        "export_dir": str(path) if str(export_dir or "") else "",
        "export_dir_exists": path.exists() if str(export_dir or "") else False,
    }
    missing: list[str] = []
    for filename in REQUIRED_HGB_FILES:
        exists = path.exists() and (path / filename).exists()
        row[f"{filename.replace('.', '_')}_exists"] = bool(exists)
        if not exists:
            missing.append(filename)
    row["missing_required_files"] = ";".join(missing)
    row["export_dir_ready"] = not missing and bool(row["export_dir_exists"])
    row["failure_type"] = "" if row["export_dir_ready"] else "export_schema_failure"
    row["failure_reason"] = "" if row["export_dir_ready"] else f"Missing required HGB files in export_dir: {row['missing_required_files'] or 'export_dir'}"
    return row


def build_training_queue(
    rows: Iterable[Mapping[str, Any]],
    *,
    graph_seeds: Sequence[int] = (1,),
    training_seeds: Sequence[int] = (1,),
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for source_row_id, row in enumerate(rows):
        if not _eligible_for_queue(row):
            continue
        for graph_seed in graph_seeds:
            for training_seed in training_seeds:
                queue.append(
                    {
                        "dataset": normalize_dataset(row.get("dataset")),
                        "method": row.get("method", ""),
                        "requested_budget_type": row.get("requested_budget_type", ""),
                        "requested_budget": row.get("requested_budget", ""),
                        "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
                        "export_dir": row.get("export_dir", ""),
                        "selected_edge_hash": row.get("selected_edge_hash", ""),
                        "planner_config_hash": row.get("planner_config_hash", ""),
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "source_row_id": int(source_row_id),
                    }
                )
    return queue


def execute_training_queue(
    queue: Iterable[Mapping[str, Any]],
    *,
    sehgnn_repo: str | Path,
    device: str,
    out_dir: str | Path,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    out_path = Path(out_dir)
    for item in queue:
        run_row = dict(item)
        export_dir = Path(str(item.get("export_dir", "")))
        audit = verify_hgb_export_dir(export_dir)
        run_row.update({f"export_{key}": value for key, value in audit.items() if key != "export_dir"})
        if not audit["export_dir_ready"]:
            run_row.update(
                {
                    "status": "failed_export_schema",
                    "training_executed": False,
                    "success": False,
                    "failure_type": "export_schema_failure",
                    "failure_reason": audit["failure_reason"],
                    "stdout_path": "",
                    "stderr_path": "",
                }
            )
            run_rows.append(run_row)
            failure_rows.append(run_row)
            continue
        if dry_run:
            run_row.update(
                {
                    "status": "dry_run_export_verified",
                    "training_executed": False,
                    "success": False,
                    "failure_type": "official_training_not_requested",
                    "failure_reason": "Preflight mode verified export_dir but did not run official SeHGNN training.",
                    "stdout_path": "",
                    "stderr_path": "",
                }
            )
            run_rows.append(run_row)
            failure_rows.append(run_row)
            continue
        stdout_path = out_path / "training_logs" / "stdout" / _log_name(item, ".log")
        stderr_path = out_path / "training_logs" / "stderr" / _log_name(item, ".stderr")
        try:
            command = build_official_hgb_command(
                dataset=normalize_dataset(item.get("dataset")),
                seed=int(item.get("training_seed", 1)),
                repo_dir=Path(sehgnn_repo),
                data_root=export_dir.parent,
                device=device,
                python_executable=python_executable or None or __import__("sys").executable,
            )
            metrics = run_native_command(command, stdout_path=stdout_path, stderr_path=stderr_path)
            run_row.update(metrics)
            status = str(metrics.get("status", ""))
            run_row["training_executed"] = status == "success"
            run_row["success"] = status == "success"
            run_row["failure_type"] = "" if status == "success" else _failure_type_from_status(status)
            run_row["failure_reason"] = "" if status == "success" else str(metrics.get("error_message", ""))
        except Exception as exc:  # pragma: no cover - runtime/environment dependent.
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
            run_row.update(
                {
                    "status": "failed_exception",
                    "training_executed": False,
                    "success": False,
                    "failure_type": "official_training_runtime_error",
                    "failure_reason": str(exc),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
        run_rows.append(run_row)
        if not bool_value(run_row.get("success")):
            failure_rows.append(run_row)
    return run_rows, failure_rows


def aggregate_training_runs(run_rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in run_rows:
        source_id = int(row.get("source_row_id", -1))
        grouped.setdefault(source_id, []).append(dict(row))
    out: dict[int, dict[str, Any]] = {}
    for source_id, rows in grouped.items():
        successes = [row for row in rows if str(row.get("status", "")) == "success" or bool_value(row.get("success"))]
        if successes:
            out[source_id] = {
                "training_executed": True,
                "success": True,
                "test_micro_f1_mean": _mean(successes, "test_micro_f1"),
                "test_micro_f1_std": _std(successes, "test_micro_f1"),
                "test_macro_f1_mean": _mean(successes, "test_macro_f1"),
                "test_macro_f1_std": _std(successes, "test_macro_f1"),
                "validation_micro_f1_mean": _mean(successes, "validation_micro_f1"),
                "validation_macro_f1_mean": _mean(successes, "validation_macro_f1"),
                "training_seed_count": len(successes),
                "failure_type": "",
                "failure_reason": "",
                "stdout_path": ";".join(str(row.get("stdout_path", "")) for row in successes if row.get("stdout_path")),
                "stderr_path": ";".join(str(row.get("stderr_path", "")) for row in successes if row.get("stderr_path")),
            }
        else:
            first = rows[0] if rows else {}
            out[source_id] = {
                "training_executed": False,
                "success": False,
                "failure_type": first.get("failure_type", "official_training_runtime_error"),
                "failure_reason": first.get("failure_reason", first.get("error_message", "")),
                "stdout_path": first.get("stdout_path", ""),
                "stderr_path": first.get("stderr_path", ""),
            }
    return out


def _eligible_for_queue(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("schema_compatible"))
        and bool_value(row.get("target_preserving"))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and not bool_value(row.get("training_executed"))
        and str(row.get("failure_type", "")) == "implemented_pending_official_training"
    )


def _log_name(item: Mapping[str, Any], suffix: str) -> str:
    method = str(item.get("method", "")).replace(" ", "_").replace("/", "_").replace("\\", "_")
    budget = str(item.get("requested_budget", "")).replace(".", "p")
    return f"{normalize_dataset(item.get('dataset'))}_{method}_{item.get('requested_budget_type', '')}_{budget}_g{item.get('graph_seed', 1)}_t{item.get('training_seed', 1)}{suffix}"


def _failure_type_from_status(status: str) -> str:
    if status in {"failed_dependency", "failed_metric_parse"}:
        return "official_training_runtime_error"
    if status == "failed_oom":
        return "official_training_oom"
    return "official_training_runtime_error"


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else ""


def _std(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [float_value(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return ""
    return pstdev(finite) if len(finite) > 1 else 0.0
