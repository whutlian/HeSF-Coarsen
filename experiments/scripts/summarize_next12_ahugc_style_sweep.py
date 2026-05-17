from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv


KEYS = ["hash_bits", "bucket_topk", "assignment_source"]


def _config_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(key, "")) for key in KEYS)


def summarize_next12_ahugc_style_sweep(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "ahugc_style_sweep_runs.csv")
    available = [row for row in rows if str(row.get("run_status", "")) == "available"]
    metrics = ["best_macro_f1", "projected_macro_f1", "refined@0", "refined@1", "refined@3", "refined@5", "AUC", "resource_logged_cumulative_dee", "relation_energy_error", "coarsening_wall_clock_sec", "peak_rss_gb", "coarse_nodes", "coarse_edges"]
    by_config = aggregate(available, KEYS, metrics)
    target_hits: dict[tuple[str, str, str], list[bool]] = {}
    for row in available:
        target_hits.setdefault(_config_key(row), []).append(str(row.get("target_hit", "")).lower() in {"true", "1", "1.0"})
    for row in by_config:
        hits = target_hits.get(_config_key(row), [])
        row["target_hit_all"] = bool(hits) and all(hits)
        row["target_hit_rate"] = sum(hits) / max(len(hits), 1)
    by_dataset = []
    for dataset in sorted({row.get("dataset", "") for row in available}):
        candidates = [row for row in available if row.get("dataset", "") == dataset]
        best = max(candidates, key=lambda row: as_float(row.get("best_macro_f1"), -1.0) or -1.0, default=None)
        if best:
            by_dataset.append({key: best.get(key, "") for key in ["dataset", *KEYS, "best_macro_f1", "target_hit", "resource_logged_cumulative_dee", "relation_energy_error"]})
    best_overall = max(by_config, key=lambda row: as_float(row.get("best_macro_f1_mean"), -1.0) or -1.0, default={})
    write_csv(output / "ahugc_style_sweep_runs.csv", rows)
    write_csv(output / "ahugc_style_sweep_by_config.csv", by_config)
    write_csv(output / "ahugc_style_best_config_by_dataset.csv", by_dataset)
    write_csv(output / "ahugc_style_best_overall.csv", [best_overall] if best_overall else [])
    accepted = any(bool(row.get("target_hit_all")) for row in by_config)
    lines = [
        "# Next12 AH-UGC-Style Sweep",
        "",
        "This is a protocol-matched type-isolated hash/LSH baseline, not an official AH-UGC reproduction.",
        (
            "Acceptance: at least one tuned config hits the target on all dataset/seed runs."
            if accepted
            else "Acceptance not met: no tuned config hit the target on all dataset/seed runs."
        ),
        "",
        markdown_table([best_overall] if best_overall else [], [*KEYS, "best_macro_f1_mean", "target_hit_all", "target_hit_rate"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"accepted": accepted}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next12_ahugc_style_sweep(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
