from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, as_float, read_csv, write_png


def _delta(rows: Sequence[Mapping[str, Any]], baseline: str, prefix: str) -> list[dict[str, Any]]:
    base = {(row.get("dataset", ""), row.get("seed", "")): row for row in rows if row.get("variant") == baseline}
    out = []
    for row in rows:
        variant = str(row.get("variant", ""))
        if not variant.startswith(prefix) or variant == baseline:
            continue
        ref = base.get((row.get("dataset", ""), row.get("seed", "")))
        if not ref:
            continue
        out.append(
            {
                "variant": variant,
                "baseline": baseline,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "dee_delta": as_float(row.get("DEE"), 0.0) - as_float(ref.get("DEE"), 0.0),
                "best_macro_f1_drop": as_float(ref.get("best_macro_f1"), 0.0) - as_float(row.get("best_macro_f1"), 0.0),
                "onehop_high_delta_selected_share_delta": as_float(row.get("onehop_high_delta_selected_share"), 0.0) - as_float(ref.get("onehop_high_delta_selected_share"), 0.0),
                "target_hit": row.get("target_hit", ""),
            }
        )
    return out


def summarize_next11_guard_appendix(*, guard: str | Path, output: str | Path) -> dict[str, Any]:
    guard = Path(guard)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(guard / "summary" / "guard_ablation_main_table.csv")
    p_delta = _delta(rows, "P_baseline", "P_")
    s_delta = _delta(rows, "S_baseline", "S_")
    acceptance = []
    for name, delta_rows in (("P", p_delta), ("S", s_delta)):
        grouped = aggregate(delta_rows, ["variant"], ["dee_delta", "best_macro_f1_drop", "onehop_high_delta_selected_share_delta"])
        for row in grouped:
            failures = [item for item in delta_rows if item.get("variant") == row.get("variant") and str(item.get("target_hit", "")).lower() != "true"]
            row["target_hit_failures"] = len(failures)
            row["acceptance_threshold"] = "best/refined task drop <= 0.005"
            row["accepted"] = bool(as_float(row.get("dee_delta_mean"), 1.0) < 0.0 and as_float(row.get("best_macro_f1_drop_mean"), 1.0) <= 0.005 and not failures)
            acceptance.append(row)
    source = aggregate(rows, ["variant"], ["onehop_high_delta_selected_share"])
    target_failures = [row for row in rows if str(row.get("target_hit", "")).lower() != "true"]
    write_csv(output / "guard_appendix_table.csv", rows)
    write_csv(output / "guard_delta_vs_baseline_p.csv", p_delta)
    write_csv(output / "guard_delta_vs_baseline_s.csv", s_delta)
    write_csv(output / "guard_acceptance_summary.csv", acceptance)
    write_csv(output / "guard_source_distribution_summary.csv", source)
    write_csv(output / "guard_target_hit_failures.csv", target_failures)
    write_png(output / "figures" / "guard_delta_dee.png", p_delta + s_delta, "dee_delta", "best_macro_f1_drop")
    write_png(output / "figures" / "guard_delta_task.png", p_delta + s_delta, "best_macro_f1_drop", "dee_delta")
    write_png(output / "figures" / "guard_source_shift.png", p_delta + s_delta, "onehop_high_delta_selected_share_delta", "dee_delta")
    lines = [
        "# Next11 Guard Appendix",
        "",
        "Acceptance threshold: best/refined task drop must be <= 0.005 with target_hit=true.",
        "P_spectral_guard and P_spectral_guard_plus_source_aware_auto can improve preservation with task drop under the specified threshold when their acceptance rows pass.",
        "S-side guard violates the task-drop threshold in Next10 evidence.",
        "source-aware-auto alone reduces onehop high-delta selected share but does not improve DEE/REEmax/SIPE.",
        "Therefore guard remains appendix/future safeguard and is not the main method.",
        "",
        markdown_table(acceptance, ["variant", "dee_delta_mean", "best_macro_f1_drop_mean", "target_hit_failures", "accepted"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"acceptance": acceptance}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_guard_appendix(guard=args.guard, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

