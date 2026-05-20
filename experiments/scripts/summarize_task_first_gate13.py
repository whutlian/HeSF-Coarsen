from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, markdown_table, write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _mean(rows: list[dict], method: str, ratio_filter=(0.048, 0.096)) -> tuple[float, float, int]:
    vals_m = []
    vals_a = []
    for row in rows:
        if row.get("method") != method:
            continue
        try:
            ratio = float(row.get("ratio", 0))
        except ValueError:
            continue
        if ratio not in ratio_filter:
            continue
        for key, out in (("macro_f1_mean", vals_m), ("task.macro_f1_mean", vals_m)):
            if row.get(key) not in {None, ""}:
                out.append(float(row[key]))
                break
        for key, out in (("accuracy_mean", vals_a), ("task.accuracy_mean", vals_a)):
            if row.get(key) not in {None, ""}:
                out.append(float(row[key]))
                break
    return (float(np.mean(vals_m)) if vals_m else 0.0, float(np.mean(vals_a)) if vals_a else 0.0, len(vals_m))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate13 TaskFirst outputs.")
    parser.add_argument("--gate12-summary", type=Path)
    parser.add_argument("--full-graph-ceiling", type=Path, required=True)
    parser.add_argument("--candidate-ablation", type=Path, required=True)
    parser.add_argument("--pair-delta-ablation", type=Path, required=True)
    parser.add_argument("--coverage-purity-ablation", type=Path, required=True)
    parser.add_argument("--support-baselines", type=Path, required=True)
    parser.add_argument("--relation-response", type=Path)
    parser.add_argument("--ratio-budget", type=Path)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(args.full_graph_ceiling / "full_graph_lite_ceiling_summary.md", args.output / "full_graph_lite_ceiling_summary.md")
    _copy_if_exists(args.candidate_ablation / "candidate_source_summary.md", args.output / "candidate_source_summary.md")
    _copy_if_exists(args.pair_delta_ablation / "pair_delta_summary.md", args.output / "pair_delta_summary.md")
    _copy_if_exists(args.coverage_purity_ablation / "coverage_summary.md", args.output / "coverage_summary.md")
    _copy_if_exists(args.coverage_purity_ablation / "purity_summary.md", args.output / "purity_summary.md")
    if args.relation_response:
        _copy_if_exists(args.relation_response / "relation_response_summary.md", args.output / "relation_response_summary.md")
    if args.ratio_budget:
        _copy_if_exists(args.ratio_budget / "ratio_budget_summary.md", args.output / "ratio_budget_summary.md")
    _copy_if_exists(args.support_baselines / "support_only_baseline_summary.md", args.output / "support_only_baseline_summary.md")
    for name in [
        "gate13_final_by_method.csv",
        "gate13_final_gap_vs_baselines.csv",
        "gate13_final_recovery_vs_ceiling.csv",
        "gate13_final_by_dataset.csv",
        "gate13_final_win_rate.csv",
        "gate13_final_selected_merge_diagnostics.csv",
    ]:
        _copy_if_exists(args.final / name, args.output / name)

    by_method = _read_csv(args.final / "gate13_final_by_method.csv")
    gaps = _read_csv(args.final / "gate13_final_gap_vs_baselines.csv")
    recovery = _read_csv(args.final / "gate13_final_recovery_vs_ceiling.csv")
    hesf_methods = ["HeSF-TC-P-response", "HeSF-TC-S-response"]
    baseline_methods = ["flatten-sum-support-only", "H6-no-spec-support-only", "TypedHash-ChebHeat-support-only"]
    evidence = []
    for method in hesf_methods + baseline_methods:
        macro, acc, n = _mean(by_method, method)
        evidence.append({"method": method, "primary_ratio_runs": n, "macro_f1_mean": round(macro, 6), "accuracy_mean": round(acc, 6)})
    best_hesf = max(evidence[:2], key=lambda row: float(row["macro_f1_mean"])) if evidence[:2] else {"macro_f1_mean": 0}
    baseline_lookup = {row["method"]: row for row in evidence[2:]}
    flatten_gap = float(best_hesf["macro_f1_mean"]) - float(baseline_lookup.get("flatten-sum-support-only", {}).get("macro_f1_mean", 0))
    h6_gap = float(best_hesf["macro_f1_mean"]) - float(baseline_lookup.get("H6-no-spec-support-only", {}).get("macro_f1_mean", 0))
    typed_gap = float(best_hesf["macro_f1_mean"]) - float(baseline_lookup.get("TypedHash-ChebHeat-support-only", {}).get("macro_f1_mean", 0))
    rec_values = []
    for row in recovery:
        if row.get("method") in hesf_methods:
            try:
                rec_values.append(float(row.get("recovery_vs_full_lite_macro", "")))
            except ValueError:
                pass
    mean_recovery = float(np.mean(rec_values)) if rec_values else 0.0
    continue_tc = flatten_gap >= 0.02 and h6_gap >= 0.01 and typed_gap >= 0.02 and mean_recovery >= 0.80
    decision = "CONTINUE_HESF_TC" if continue_tc else "DROP_HESF_TC_RETURN_TO_PRESERVATION_MAINLINE"
    write_csv(args.output / "gate13_decision_evidence.csv", evidence)
    decision_text = f"""# Gate13 Decision

Decision: `{decision}`

Git commit: `{git_commit_hash()}`

## Evidence

{markdown_table(evidence, ["method", "primary_ratio_runs", "macro_f1_mean", "accuracy_mean"])}

## Acceptance Deltas

- Best HeSF-TC vs flatten-sum macro-F1 delta: `{flatten_gap:.6f}`
- Best HeSF-TC vs H6-no-spec macro-F1 delta: `{h6_gap:.6f}`
- Best HeSF-TC vs TypedHash-ChebHeat macro-F1 delta: `{typed_gap:.6f}`
- Mean macro recovery vs full graph lite ceiling: `{mean_recovery:.6f}`

Lite evaluator numbers are diagnostic only.
"""
    (args.output / "gate13_decision.md").write_text(decision_text, encoding="utf-8")
    (args.output / "gate13_final_code_issue_report.md").write_text(
        "# Gate13 Code Issue Report\n\n"
        "- `stateful_approx` remains explicitly not implemented.\n"
        "- `score_rank_spearman_proxy` uses selected-pair overlap as a practical rank-shift proxy for this local Gate13 run.\n"
        "- All HETTREE outputs remain lite diagnostic, not official reproduction.\n",
        encoding="utf-8",
    )
    final_report = f"""# Gate13 Final Report

## 1. Environment and git commit

- Git commit: `{git_commit_hash()}`
- Local conda environment requested: `pytorch`
- Evaluator: `hettree_lite` diagnostic

## 2. Gate12 reproduction summary

Gate12 input summary: `{args.gate12_summary or ""}`.

## 3. Code audit findings

See `code_audit.md`.

## 4. Full graph lite ceiling

See `full_graph_lite_ceiling_summary.md`.

## 5. Candidate-source ablation

See `candidate_source_summary.md`.

## 6. Pair-delta mode ablation

See `pair_delta_summary.md`.

## 7. Coverage and purity fixes

See `coverage_summary.md` and `purity_summary.md`.

## 8. Relation-response activation sanity

See `relation_response_summary.md`.

## 9. Ratio-budget sanity

See `ratio_budget_summary.md`.

## 10. Support-only baseline comparison

See `support_only_baseline_summary.md`.

## 11. Final Gate13 HGB table

See `gate13_final_by_method.csv`.

## 12. Go/no-go decision

{decision}

## 13. Remaining limitations

- Official SeHGNN/HETTREE/FreeHGC comparison is still not integrated.
- `hettree_lite` remains a local diagnostic evaluator.
- `stateful_approx` is reported as not implemented.

## 14. Exact commands

Commands are recorded in the shell history and stage result files under this output directory.

## 15. Output index

- `gate13_decision.md`
- `gate13_final_by_method.csv`
- `gate13_final_gap_vs_baselines.csv`
- `gate13_final_recovery_vs_ceiling.csv`
- Stage summaries copied into this directory.
"""
    (args.output / "final_report.md").write_text(final_report, encoding="utf-8")
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "task_first_gate13_claim_boundary.md").write_text(
        "# HeSF-TC Gate13 Claim Boundary\n\n"
        "- HeSF-TC is under validation and is not paper mainline unless it passes Gate13.\n"
        "- `hettree_lite` numbers are diagnostic only.\n"
        "- Official SeHGNN/HETTREE/FreeHGC comparisons remain non-comparable unless official/faithful integration is completed.\n"
        "- Preservation-first HeSF-LVC-P/S remains separate and unchanged.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
