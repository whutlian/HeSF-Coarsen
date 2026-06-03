from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


CRITICAL_METHODS: tuple[tuple[str, str], ...] = (
    ("DBLP", "HeSF-RCS-auto structural12"),
    ("DBLP", "HeSF-RCS-auto structural16"),
    ("DBLP", "Random-edge-relwise"),
    ("DBLP", "Proportional-relation-budget"),
    ("DBLP", "HGCond-score-TP-local"),
    ("DBLP", "GCond-score-TP-local"),
    ("DBLP", "FreeHGC-score-TP-local"),
    ("DBLP", "FreeHGC-score-as-selector structural20"),
    ("ACM", "ACM-HeSF-RCS-auto-field20"),
    ("ACM", "ACM-Degree-field20"),
    ("ACM", "ACM-ValidationGreedy-field20"),
    ("ACM", "ACM-Random-field20"),
    ("IMDB", "IMDB-HeSF-RCS-channel50"),
    ("IMDB", "IMDB-ValidationGreedy-channel50"),
    ("IMDB", "IMDB-MDfull-MA50-MK50"),
    ("IMDB", "IMDB-Degree-channel20"),
    ("IMDB", "IMDB-HeSF-RCS-auto structural30"),
)

ROBUSTNESS_FIELDS = (
    "dataset",
    "method",
    "critical_row",
    "training_executed_count",
    "training_seed_count",
    "graph_seed_count",
    "failure_count",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "deterministic_export_hash_count",
    "deterministic_export_proof",
    "robustness_mode",
    "robustness_ready",
    "failure_type",
    "failure_reason",
)


def build_critical_robustness_rows(
    main_rows: Iterable[Mapping[str, Any]],
    training_runs: Iterable[Mapping[str, Any]] = (),
    *,
    critical_methods: Sequence[tuple[str, str]] = CRITICAL_METHODS,
) -> list[dict[str, Any]]:
    main = [dict(row) for row in main_rows]
    runs = [dict(row) for row in training_runs]
    out: list[dict[str, Any]] = []
    for dataset, method in critical_methods:
        row = _find_row(main, dataset, method)
        grouped_runs = [
            run
            for run in runs
            if normalize_dataset(run.get("dataset")) == dataset and str(run.get("method", "")) == method
        ]
        out.append(_robustness_row(dataset=dataset, method=method, main_row=row, run_rows=grouped_runs, critical=True))
    return out


def aggregate_robustness_by_method(
    training_runs: Iterable[Mapping[str, Any]],
    *,
    main_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    main = [dict(row) for row in main_rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in training_runs:
        grouped.setdefault((normalize_dataset(run.get("dataset")), str(run.get("method", ""))), []).append(dict(run))
    out: list[dict[str, Any]] = []
    for (dataset, method), runs in sorted(grouped.items()):
        out.append(_robustness_row(dataset=dataset, method=method, main_row=_find_row(main, dataset, method), run_rows=runs, critical=False))
    return out


def build_missing_robustness_queue_rows(
    main_rows: Iterable[Mapping[str, Any]],
    robustness_rows: Iterable[Mapping[str, Any]],
    *,
    target_training_seeds: int = 3,
) -> list[dict[str, Any]]:
    by_key = {(normalize_dataset(row.get("dataset")), str(row.get("method", ""))): row for row in robustness_rows}
    out: list[dict[str, Any]] = []
    for row in main_rows:
        dataset = normalize_dataset(row.get("dataset"))
        method = str(row.get("method", ""))
        if (dataset, method) not in set(CRITICAL_METHODS):
            continue
        if not bool_value(row.get("official_hgb_exported", True)) or not row.get("export_dir"):
            continue
        current = by_key.get((dataset, method), {})
        existing = int(float_value(current.get("training_seed_count")) or float_value(row.get("training_seed_count")) or 0)
        for seed in range(existing + 1, int(target_training_seeds) + 1):
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "requested_budget_type": row.get("requested_budget_type", ""),
                    "requested_budget": row.get("requested_budget", ""),
                    "export_dir": row.get("export_dir", ""),
                    "selected_edge_hash": row.get("selected_edge_hash", ""),
                    "planner_config_hash": row.get("planner_config_hash", ""),
                    "graph_seed": 1,
                    "training_seed": seed,
                    "source_row_id": "",
                    "robustness_queue_reason": "missing_training_seed_for_deterministic_export_plus_3_training_seeds",
                }
            )
    return out


def _robustness_row(
    *,
    dataset: str,
    method: str,
    main_row: Mapping[str, Any] | None,
    run_rows: Sequence[Mapping[str, Any]],
    critical: bool,
) -> dict[str, Any]:
    successes = [row for row in run_rows if _run_success(row)]
    failures = [row for row in run_rows if not _run_success(row) and (row.get("status") or row.get("failure_type"))]
    seed_values = {int(float_value(row.get("training_seed", row.get("seed"))) or 0) for row in successes if float_value(row.get("training_seed", row.get("seed"))) is not None}
    graph_values = {int(float_value(row.get("graph_seed")) or 0) for row in successes if float_value(row.get("graph_seed")) is not None}
    hash_values = {_artifact_key(row) for row in successes if _artifact_key(row)}
    if main_row:
        main_seed_count = int(float_value(main_row.get("training_seed_count")) or 0)
        main_graph_count = int(float_value(main_row.get("graph_seed_count")) or 0)
        if main_seed_count:
            seed_values.update(range(1, main_seed_count + 1))
        if main_graph_count:
            graph_values.update(range(1, main_graph_count + 1))
        if not hash_values and _artifact_key(main_row):
            hash_values = {_artifact_key(main_row)}
    executed_count = max(len(successes), len({seed for seed in seed_values if seed > 0}))
    training_seed_count = len({seed for seed in seed_values if seed > 0})
    graph_seed_count = len({seed for seed in graph_values if seed > 0})
    hash_count = len(hash_values)
    deterministic_proof = bool(hash_count == 1 and training_seed_count >= 3)
    executed_3x3 = bool(graph_seed_count >= 3 and training_seed_count >= 3 and executed_count >= 9)
    ready = bool(executed_3x3 or deterministic_proof)
    mode = "executed_3x3" if executed_3x3 else "deterministic_export_plus_3_training_seeds" if deterministic_proof else "incomplete"
    metric_source = successes if successes else [main_row] if main_row else []
    return {
        "dataset": dataset,
        "method": method,
        "critical_row": bool(critical),
        "training_executed_count": executed_count,
        "training_seed_count": training_seed_count,
        "graph_seed_count": graph_seed_count,
        "failure_count": len(failures),
        "test_micro_f1_mean": _mean(metric_source, "test_micro_f1", "test_micro_f1_mean"),
        "test_micro_f1_std": _std(metric_source, "test_micro_f1", "test_micro_f1_mean"),
        "test_macro_f1_mean": _mean(metric_source, "test_macro_f1", "test_macro_f1_mean"),
        "test_macro_f1_std": _std(metric_source, "test_macro_f1", "test_macro_f1_mean"),
        "validation_micro_f1_mean": _mean(metric_source, "validation_micro_f1", "validation_micro_f1_mean"),
        "validation_macro_f1_mean": _mean(metric_source, "validation_macro_f1", "validation_macro_f1_mean"),
        "deterministic_export_hash_count": hash_count,
        "deterministic_export_proof": deterministic_proof,
        "robustness_mode": mode,
        "robustness_ready": ready,
        "failure_type": "" if ready else "robustness_incomplete",
        "failure_reason": "" if ready else "Need either 3 graph seeds x 3 training seeds or one deterministic export hash with at least 3 successful training seeds.",
    }


def _find_row(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method:
            return row
    return None


def _run_success(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status", ""))
    return bool(status == "success" or bool_value(row.get("success")) or bool_value(row.get("training_executed")) and float_value(row.get("test_micro_f1")) is not None)


def _artifact_key(row: Mapping[str, Any]) -> str:
    selected = str(row.get("selected_edge_hash", "")).strip()
    if selected:
        return f"selected_edge_hash:{selected}"
    export_dir = str(row.get("export_dir", "")).strip()
    return f"export_dir:{export_dir}" if export_dir else ""


def _mean(rows: Sequence[Mapping[str, Any] | None], *fields: str) -> float | str:
    values = [_first_float(row, *fields) for row in rows if row]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else ""


def _std(rows: Sequence[Mapping[str, Any] | None], *fields: str) -> float | str:
    values = [_first_float(row, *fields) for row in rows if row]
    finite = [value for value in values if value is not None]
    if not finite:
        return ""
    return pstdev(finite) if len(finite) > 1 else 0.0


def _first_float(row: Mapping[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = float_value(row.get(field))
        if value is not None:
            return value
    return None
