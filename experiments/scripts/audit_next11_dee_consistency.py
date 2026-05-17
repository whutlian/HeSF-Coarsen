from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, fmt, read_csv, read_json, sha256_array_payload, sha256_file


DEE_FIELD = "dirichlet_energy_relative_error"
METHODS_WITH_RUN_DIR = {"HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H0-mutual-best", "H6-no-spec"}


def _method_run_token(method: str) -> str:
    return method.replace(" ", "_").replace("-", "_")


def _final_level(run_dir: Path) -> Path | None:
    levels = []
    for path in run_dir.glob("level_*"):
        if path.is_dir() and (path / "diagnostics.json").exists():
            try:
                levels.append((int(path.name.removeprefix("level_")), path))
            except ValueError:
                pass
    return max(levels)[1] if levels else None


def _find_resource_run(resource_logged: Path, method: str, dataset: str, seed: str) -> Path | None:
    token = _method_run_token(method)
    candidates = [
        resource_logged / "runs" / f"next10_resource_{dataset}_{token}_seed{seed}",
        resource_logged / "runs" / f"next10_guard_{dataset}_{token}_seed{seed}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list((resource_logged / "runs").glob(f"*{dataset}*{token}*seed{seed}*"))
    return matches[0] if matches else None


def _diagnostic_fields(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    level = _final_level(run_dir)
    if level is None:
        return {}
    diagnostics = read_json(level / "diagnostics.json")
    spectral = diagnostics.get("spectral", {}) if isinstance(diagnostics.get("spectral"), Mapping) else {}
    cumulative = diagnostics.get("cumulative_spectral", {}) if isinstance(diagnostics.get("cumulative_spectral"), Mapping) else {}
    return {
        "raw_dee": spectral.get(DEE_FIELD, ""),
        "normalized_dee": cumulative.get(DEE_FIELD, ""),
        "cumulative_dee": cumulative.get(DEE_FIELD, ""),
        "final_level_dee": spectral.get(DEE_FIELD, ""),
        "diagnostics_path": str(level / "diagnostics.json"),
        "config_hash": sha256_file(run_dir / "config.yaml"),
        "assignment_hash": sha256_array_payload(level / "cumulative_assignment.npz") or sha256_array_payload(level / "assignment.npz"),
        "num_levels": int(level.name.removeprefix("level_")),
    }


def _status(method: str, paper_dee: Any, resource_dee: Any, diag: Mapping[str, Any]) -> tuple[str, str]:
    paper = as_float(paper_dee, None)
    resource = as_float(resource_dee, None)
    if paper is None and resource is None and method.lower().startswith("full rgcn"):
        return "spectral_not_applicable", "full graph baseline has no coarse spectral metric"
    if paper is None or resource is None:
        return "ambiguous_missing_dee", "one side has missing or ambiguous DEE"
    if abs(paper - resource) <= 1.0e-8:
        return "same_metric", "paper and resource DEE are numerically equal"
    raw = as_float(diag.get("raw_dee"), None)
    cumulative = as_float(diag.get("cumulative_dee"), None)
    if raw is not None and abs(paper - raw) <= max(1.0e-8, abs(raw) * 1.0e-6) and cumulative is not None:
        return "field_mismatch_fixed", "paper DEE matches final-level/raw diagnostic while resource DEE uses cumulative diagnostic"
    return "different_metric_renamed", "same key uses different graph/config/metric provenance; keep names explicit"


def audit_next11_dee_consistency(
    *,
    paper_final: str | Path,
    resource_logged: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    paper_final = Path(paper_final)
    resource_logged = Path(resource_logged)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paper_rows = read_csv(paper_final / "final_main_table_by_seed.csv")
    resource_rows = read_csv(resource_logged / "hgb_resource_logged_runs.csv")
    resource_index = {
        (row.get("method", ""), row.get("dataset", ""), str(row.get("seed", ""))): row
        for row in resource_rows
    }
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for paper in paper_rows:
        method = str(paper.get("method", ""))
        dataset = str(paper.get("dataset", ""))
        seed = str(paper.get("seed", ""))
        resource = resource_index.get((method, dataset, seed), {})
        run_dir = _find_resource_run(resource_logged, method, dataset, seed) if method in METHODS_WITH_RUN_DIR else None
        diag = _diagnostic_fields(run_dir)
        status, explanation = _status(method, paper.get("DEE", ""), resource.get("DEE", ""), diag)
        paper_dee = as_float(paper.get("DEE"), None)
        resource_dee = as_float(resource.get("DEE"), None)
        ratio = resource_dee / paper_dee if paper_dee not in {None, 0.0} and resource_dee is not None else ""
        row = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "paper_final_dee": paper.get("DEE", ""),
            "paper_final_cumulative_dee": "",
            "paper_final_final_level_dee": paper.get("DEE", ""),
            "resource_logged_dee": resource.get("DEE", ""),
            "resource_logged_cumulative_dee": diag.get("cumulative_dee", resource.get("DEE", "")),
            "resource_logged_final_level_dee": diag.get("final_level_dee", ""),
            "raw_dee": diag.get("raw_dee", ""),
            "normalized_dee": diag.get("normalized_dee", ""),
            "field_used_by_paper_final_summary": "final_main_table_by_seed.DEE",
            "field_used_by_resource_summary": "cumulative_spectral.dirichlet_energy_relative_error" if diag else "hgb_resource_logged_runs.DEE",
            "config_hash": diag.get("config_hash", ""),
            "assignment_hash": diag.get("assignment_hash", ""),
            "coarse_nodes": resource.get("coarse_nodes", paper.get("coarse_nodes", "")),
            "coarse_edges": resource.get("coarse_edges", paper.get("coarse_edges", "")),
            "edge_compression_ratio": resource.get("edge_compression_ratio", paper.get("coarse_graph_ratio", "")),
            "target_ratio": resource.get("target_ratio", paper.get("target_ratio", "")),
            "final_ratio": resource.get("final_ratio", resource.get("target_ratio", "")),
            "num_levels": diag.get("num_levels", ""),
            "metric_scale_ratio_resource_to_paper": ratio,
            "status": status,
            "explanation": explanation,
        }
        rows.append(row)
        for field_name in ("DEE", "FSE", "REEmax", "SIPE"):
            if field_name in paper:
                inventory.append({"source": "paper_final", "field": field_name, "method": method, "dataset": dataset, "seed": seed})
            if field_name in resource:
                inventory.append({"source": "resource_logged", "field": field_name, "method": method, "dataset": dataset, "seed": seed})
        config_rows.append({"method": method, "dataset": dataset, "seed": seed, "resource_config_hash": diag.get("config_hash", ""), "comparable": bool(diag.get("config_hash"))})
        assignment_rows.append({"method": method, "dataset": dataset, "seed": seed, "resource_assignment_hash": diag.get("assignment_hash", ""), "comparable": bool(diag.get("assignment_hash"))})

    statuses = {str(row["status"]) for row in rows if row.get("status") != "spectral_not_applicable"}
    if not statuses or statuses == {"same_metric"}:
        conclusion = "same_metric"
    elif "field_mismatch_fixed" in statuses and statuses <= {"field_mismatch_fixed", "same_metric"}:
        conclusion = "field_mismatch_fixed"
    elif "different_metric_renamed" in statuses or "ambiguous_missing_dee" in statuses:
        conclusion = "different_metric_renamed"
    else:
        conclusion = "requires_rerun"

    corrected = [dict(row, paper_cumulative_dee=row["paper_final_dee"], resource_logged_raw_dee=row["raw_dee"]) for row in rows]
    write_csv(output / "dee_consistency_by_run.csv", rows)
    write_csv(output / "dee_consistency_by_method_dataset.csv", _aggregate_status(rows))
    write_csv(output / "dee_field_inventory.csv", inventory)
    write_csv(output / "config_hash_comparison.csv", config_rows)
    write_csv(output / "assignment_hash_comparison.csv", assignment_rows)
    write_csv(output / "diagnostics_field_map.csv", [{"logical_metric": "DEE", "paper_field": "DEE", "resource_field": "cumulative_spectral.dirichlet_energy_relative_error"}])
    write_csv(output / "corrected_dee_named_metrics.csv", corrected)
    lines = [
        "# Next11 DEE Consistency Audit",
        "",
        f"Conclusion: `{conclusion}`.",
        "",
        "The audit treats a bare `DEE` as ambiguous when the source diagnostics cannot be traced.",
        "Next11 paper-facing tables should use explicit names such as `paper_final_dee`, `resource_logged_cumulative_dee`, and `resource_logged_final_level_dee`.",
        "",
        markdown_table(rows[:10], ["method", "dataset", "seed", "paper_final_dee", "resource_logged_dee", "metric_scale_ratio_resource_to_paper", "status"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"conclusion": conclusion, "rows": rows}


def _aggregate_status(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("method", "")), str(row.get("dataset", ""))), []).append(row)
    out = []
    for (method, dataset), group in sorted(groups.items()):
        ratios = [as_float(row.get("metric_scale_ratio_resource_to_paper"), None) for row in group]
        ratios = [value for value in ratios if value is not None]
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "run_count": len(group),
                "scale_ratio_mean": sum(ratios) / len(ratios) if ratios else "",
                "statuses": ";".join(sorted({str(row.get("status", "")) for row in group})),
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-final", type=Path, required=True)
    parser.add_argument("--resource-logged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_next11_dee_consistency(paper_final=args.paper_final, resource_logged=args.resource_logged, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

