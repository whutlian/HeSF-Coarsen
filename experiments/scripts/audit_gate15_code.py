from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, markdown_table, write_csv


GATE14_EXPECTED_COMMIT = "fa0fa465b8b021d42e4bb3f6da10456c77061a7f"


METHOD_ROWS = [
    ("HeSF-TC-P-response-static", "experiments/scripts/run_task_first_gate14_final.py; experiments/scripts/gate13_task_first_common.py; hesf_coarsen/task_first/pipeline.py", "reproducible_from_local_main", "response_signature + combined coverage + unknown_blocks_known purity"),
    ("HeSF-TC-S-response-static", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "S weights in task_first_config"),
    ("HeSF-TC-no-rel-response", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "lambda_rel_response=0"),
    ("HeSF-TC-no-target-spec", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "lambda_target_spec=0"),
    ("HeSF-TC-coverage-v2", "hesf_coarsen/task_first/support_coverage.py", "reproducible_from_local_main", "method code, not only summarizer"),
    ("HeSF-TC-purity-v2", "hesf_coarsen/task_first/support_purity.py", "reproducible_from_local_main", "method code, hybrid propagated footprint"),
    ("HeSF-TC-coverage-v2-purity-v2", "hesf_coarsen/task_first/support_coverage.py; hesf_coarsen/task_first/support_purity.py", "reproducible_from_local_main", "combined method code"),
    ("HeSF-TC-stateful-v1", "hesf_coarsen/task_first/stateful_matching.py; hesf_coarsen/task_first/pipeline.py", "reproducible_from_local_main", "stateful_signature branch"),
    ("HeSF-TC-stateful-v1-coverage-v2", "hesf_coarsen/task_first/stateful_matching.py; hesf_coarsen/task_first/support_coverage.py", "reproducible_from_local_main", "stateful signature + coverage_v2"),
    ("HeSF-TC-stateful-v1-purity-v2", "hesf_coarsen/task_first/stateful_matching.py; hesf_coarsen/task_first/support_purity.py", "reproducible_from_local_main", "stateful signature + purity_v2"),
    ("HeSF-TC-stateful-v1-coverage-v2-purity-v2", "hesf_coarsen/task_first/stateful_matching.py", "reproducible_from_local_main", "stateful signature + coverage/purity v2"),
    ("flatten-sum-support-only", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "support-only baseline"),
    ("H6-no-spec-support-only", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "support relation footprint baseline"),
    ("TypedHash-ChebHeat-support-only", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "typed hash ChebHeat baseline"),
    ("random-support-only", "experiments/scripts/gate13_task_first_common.py", "reproducible_from_local_main", "random support-only baseline"),
    ("A0-current-all-type-coarse-transfer-reference", "experiments/scripts/run_task_first_gate14_final.py", "reproducible_from_local_main", "all-type LSH reference"),
    ("full-graph-hettree-lite-tuned", "experiments/scripts/run_task_first_gate14_full_graph_lite_tuning.py; hesf_coarsen/eval/hettree_task.py", "reproducible_from_local_main", "diagnostic lite full graph ceiling"),
]


def _git_is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def write_audit(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    current = git_commit_hash() or "unknown"
    method_rows = [
        {
            "method": method,
            "code_path": path,
            "reproducibility_status": status,
            "notes": notes,
        }
        for method, path, status, notes in METHOD_ROWS
    ]
    write_csv(output / "method_to_code_path.csv", method_rows)
    checks: list[dict[str, Any]] = [
        {
            "check": "current_main_commit",
            "status": current,
            "evidence": "git rev-parse HEAD",
        },
        {
            "check": "gate14_expected_commit",
            "status": GATE14_EXPECTED_COMMIT,
            "evidence": "prompt baseline commit",
        },
        {
            "check": "current_differs_from_gate14_prompt_commit",
            "status": str(current != GATE14_EXPECTED_COMMIT),
            "evidence": "Gate15 is continuing from the current committed local main, not the older Gate14 prompt hash",
        },
        {
            "check": "gate14_prompt_commit_is_ancestor_of_head",
            "status": str(_git_is_ancestor(GATE14_EXPECTED_COMMIT)),
            "evidence": "git merge-base --is-ancestor",
        },
        {
            "check": "coverage_v1_deprecated",
            "status": "risk_confirmed",
            "evidence": "coverage_v1_legacy returns zero when support nodes share no common anchor",
        },
        {
            "check": "purity_zero_footprint_risk",
            "status": "risk_confirmed",
            "evidence": "zero-footprint nodes are explicitly counted; legacy zero_as_no_conflict is not Gate15 main criterion",
        },
        {
            "check": "local_surrogate_is_not_true_task_delta",
            "status": "risk_confirmed",
            "evidence": "scoring.py local_surrogate uses row distances over footprints/signatures",
        },
        {
            "check": "exact_pair_delta_is_pair_isolated",
            "status": "risk_confirmed",
            "evidence": "scoring.py exact builds a pair-only assignment with other support nodes singleton",
        },
        {
            "check": "non_stateful_pipeline_static_per_layer",
            "status": "risk_confirmed",
            "evidence": "pipeline.py computes all deltas once per layer then runs greedy matching",
        },
        {
            "check": "evaluator_status",
            "status": "diagnostic_lite_only",
            "evidence": "official SeHGNN/HETTREE/FreeHGC evaluator is not integrated",
        },
    ]
    report = "# Gate15 Code Sync Report\n\n"
    report += f"Current local `main` commit: `{current}`\n\n"
    report += f"Gate14 prompt commit: `{GATE14_EXPECTED_COMMIT}`\n\n"
    report += "Gate15 uses the current committed local `main` as the reproducibility baseline. This local branch is ahead of `origin/main`, so remote reproducibility requires pushing those commits.\n\n"
    report += "## Method Mapping\n\n"
    report += markdown_table(method_rows, ["method", "reproducibility_status", "code_path", "notes"])
    report += "\n\n## Required Checks\n\n"
    report += markdown_table(checks, ["check", "status", "evidence"])
    report += "\n\n## Gate15 Boundary\n\n"
    report += "Current Gate14 handcrafted HeSF-TC variants are retained only as references. Gate15 main rows must use `primary_method_family=supervised_support_selection` and must not promote static pairwise coarsening as the primary method.\n"
    (output / "code_sync_report.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Gate15 code sync audit outputs.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gate15_code_audit"))
    args = parser.parse_args(argv)
    write_audit(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
