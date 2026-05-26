from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


PATHAWARE_FIELDS = [
    "dataset",
    "method",
    "relation_channel_spec",
    "edge_score_strategy",
    "graph_seed",
    "training_seed",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "support_edge_ratio",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "val_test_micro_gap",
    "coverage_score_mean",
    "coverage_score_std",
    "hub_penalty_mean",
    "num_isolated_target_nodes_after_pruning",
    "edge_score_diagnostics_path",
    "coverage_diagnostics_path",
    "success",
    "status",
    "failed_reason",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float | str:
    return float(sum(values) / len(values)) if values else ""


def _std(values: Sequence[float]) -> float | str:
    return float(statistics.pstdev(values)) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def _strategy(method: str, fallback: str) -> str:
    if method.endswith("-random"):
        return "random_edge_within_relation"
    if method.endswith("-degree"):
        return "degree_within_relation"
    if method.endswith("-pathaware-v2-topk-diagnostic"):
        return "pathaware_v2_topk_diagnostic"
    if method.endswith("-pathaware-v2-stratified"):
        return "pathaware_v2_stratified"
    return fallback


def _coverage_stats(rows: Sequence[Mapping[str, Any]], method: str, graph_seed: str) -> tuple[float | str, float | str]:
    values: list[float] = []
    for row in rows:
        if row.get("method") != method or str(row.get("graph_seed", "")) != str(graph_seed):
            continue
        for field in ("paper_coverage_ratio", "venue_coverage_ratio", "term_coverage_ratio", "target_author_reachability_after"):
            value = _float(row.get(field))
            if value is not None:
                values.append(value)
    return _mean(values), _std(values)


def _hub_penalty(rows: Sequence[Mapping[str, Any]], method: str, graph_seed: str) -> float | str:
    values = [
        value
        for row in rows
        if row.get("method") == method and str(row.get("graph_seed", "")) == str(graph_seed)
        if (value := _float(row.get("score_component_hub_penalty_mean"))) is not None
    ]
    return _mean(values)


def _decision(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    success = [row for row in rows if row.get("status") == "success" and _float(row.get("test_micro_f1")) is not None]
    by_strategy: dict[str, list[Mapping[str, Any]]] = {}
    for row in success:
        by_strategy.setdefault(str(row.get("edge_score_strategy", "")), []).append(row)
    random_rows = by_strategy.get("random_edge_within_relation", [])
    strat_rows = by_strategy.get("pathaware_v2_stratified", [])
    random_micro = _mean([float(row["test_micro_f1"]) for row in random_rows])
    strat_micro = _mean([float(row["test_micro_f1"]) for row in strat_rows])
    random_gap = _mean([float(row["val_test_micro_gap"]) for row in random_rows if _float(row.get("val_test_micro_gap")) is not None])
    strat_gap = _mean([float(row["val_test_micro_gap"]) for row in strat_rows if _float(row.get("val_test_micro_gap")) is not None])
    random_struct = _mean([float(row["semantic_structural_storage_ratio"]) for row in random_rows if _float(row.get("semantic_structural_storage_ratio")) is not None])
    strat_struct = _mean([float(row["semantic_structural_storage_ratio"]) for row in strat_rows if _float(row.get("semantic_structural_storage_ratio")) is not None])
    meta = {
        "random_success_count": len(random_rows),
        "stratified_success_count": len(strat_rows),
        "random_mean_micro": random_micro,
        "stratified_mean_micro": strat_micro,
        "random_mean_val_test_gap": random_gap,
        "stratified_mean_val_test_gap": strat_gap,
        "random_mean_structural_ratio": random_struct,
        "stratified_mean_structural_ratio": strat_struct,
    }
    if len(strat_rows) < 9 or len(random_rows) < 9:
        return "PATHAWARE_V2_GAIN_NOT_VALIDATED", meta
    same_budget = isinstance(random_struct, float) and isinstance(strat_struct, float) and abs(random_struct - strat_struct) <= 0.001
    no_gap_penalty = isinstance(random_gap, float) and isinstance(strat_gap, float) and strat_gap <= random_gap + 0.005
    gain = isinstance(random_micro, float) and isinstance(strat_micro, float) and strat_micro >= random_micro + 0.002
    if same_budget and no_gap_penalty and gain:
        return "PATHAWARE_V2_GAIN_PASS", meta
    return "PATHAWARE_V2_GAIN_FAIL", meta


def export_pathaware(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _read_csv(input_dir / "gate21_3_raw_rows.csv")
    edge_rows = _read_csv(input_dir / "gate21_3_edge_score_diagnostics.csv")
    coverage_rows = _read_csv(input_dir / "gate21_3_coverage_diagnostics.csv")
    path_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        method = str(row.get("method", ""))
        if not method.startswith("H6-struct40-relgrid-best-"):
            continue
        graph_seed = row.get("graph_seed", "")
        coverage_mean, coverage_std = _coverage_stats(coverage_rows, method, str(graph_seed))
        val_micro = _float(row.get("validation_micro_f1"))
        test_micro = _float(row.get("test_micro_f1"))
        path_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "method": method,
                "relation_channel_spec": row.get("relation_channel_spec", ""),
                "edge_score_strategy": _strategy(method, row.get("edge_score_strategy", "")),
                "graph_seed": graph_seed,
                "training_seed": row.get("training_seed", ""),
                "semantic_structural_storage_ratio": row.get("semantic_structural_storage_ratio", ""),
                "hgb_raw_file_byte_ratio": row.get("hgb_raw_file_byte_ratio", ""),
                "support_edge_ratio": row.get("support_edge_ratio", ""),
                "test_micro_f1": row.get("test_micro_f1", ""),
                "test_macro_f1": row.get("test_macro_f1", ""),
                "validation_micro_f1": row.get("validation_micro_f1", ""),
                "validation_macro_f1": row.get("validation_macro_f1", ""),
                "val_test_micro_gap": "" if val_micro is None or test_micro is None else abs(float(val_micro) - float(test_micro)),
                "coverage_score_mean": coverage_mean,
                "coverage_score_std": coverage_std,
                "hub_penalty_mean": _hub_penalty(edge_rows, method, str(graph_seed)),
                "num_isolated_target_nodes_after_pruning": "",
                "edge_score_diagnostics_path": str(output_dir / "gate21_4_edge_score_diagnostics.csv"),
                "coverage_diagnostics_path": str(output_dir / "gate21_4_coverage_diagnostics.csv"),
                "success": row.get("success", ""),
                "status": row.get("status", ""),
                "failed_reason": row.get("failed_reason", ""),
            }
        )
    flag, meta = _decision(path_rows)
    write_csv(output_dir / "gate21_4_pathaware_v2_validation.csv", path_rows, fieldnames=PATHAWARE_FIELDS)
    write_csv(output_dir / "gate21_4_edge_score_diagnostics.csv", edge_rows)
    write_csv(output_dir / "gate21_4_coverage_diagnostics.csv", coverage_rows)
    decision = {"decisions": [flag], "pathaware_rows": len(path_rows), **meta}
    write_json(output_dir / "gate21_4_decision.json", decision)
    lines = [
        "# Gate21.4 Pathaware v2 Decision",
        "",
        f"- `{flag}`",
        f"- random_success_count: `{meta.get('random_success_count', '')}`",
        f"- stratified_success_count: `{meta.get('stratified_success_count', '')}`",
        f"- random_mean_micro: `{meta.get('random_mean_micro', '')}`",
        f"- stratified_mean_micro: `{meta.get('stratified_mean_micro', '')}`",
    ]
    (output_dir / "gate21_4_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_pathaware(args.input_dir, args.output_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
