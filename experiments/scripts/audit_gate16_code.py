from __future__ import annotations

import argparse
import csv
import glob
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REQUIRED_PATHS = (
    "hesf_coarsen/eval/hettree_task.py",
    "hesf_coarsen/eval/sehgnn_task.py",
    "hesf_coarsen/task_first/selection/config.py",
    "hesf_coarsen/task_first/selection/teacher.py",
    "hesf_coarsen/task_first/selection/support_features.py",
    "hesf_coarsen/task_first/selection/contribution.py",
    "hesf_coarsen/task_first/selection/selector.py",
    "hesf_coarsen/task_first/selection/condensation.py",
    "hesf_coarsen/task_first/selection/pipeline.py",
    "experiments/scripts/run_task_first_gate*.py",
    "experiments/scripts/run_gate15*.py",
    "experiments/scripts/summarize_gate15*.py",
)


METHOD_ROWS = (
    ("full-graph-hettree-lite-tuned", "full_graph_teacher", "hesf_coarsen/eval/hettree_task.py", "evaluate_hettree_task", "implemented", "Gate16 primary_eval_mode=compressed_projected"),
    ("H6-no-spec-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "exact budget diagnostics added in Gate16 runner"),
    ("flatten-sum-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "support-only baseline"),
    ("TypedHash-ChebHeat-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "support-only baseline"),
    ("random-support-only", "weak_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "sanity baseline"),
    ("HeSF-SS-teacher-topk", "proxy_selector_baseline", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "implemented", "not a primary Gate16 method"),
    ("HeSF-SS-teacher-diverse-topk", "proxy_selector_baseline", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "implemented", "not a primary Gate16 method"),
    ("HeSF-SS-validation-greedy", "legacy_alias", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "renamed", "legacy proxy alias; reports as validation_proxy_diverse"),
    ("HeSF-SS-validation-proxy-diverse", "proxy_selector_baseline", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "implemented", "selector_uses_true_validation_feedback=False"),
    ("HeSF-SS-hybrid-teacher-response", "proxy_selector_baseline", "hesf_coarsen/task_first/selection/contribution.py", "compute_support_importance", "implemented", "teacher/response hybrid"),
    ("HeSF-SS-selection-background-condense", "prototype_reference", "hesf_coarsen/task_first/selection/condensation.py", "build_selected_support_graph", "implemented", "Gate16 should prefer prototype residual strategy"),
    ("HeSF-SS-true-validation-block-greedy", "validation_selector", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "implemented_lite", "diagnostic block greedy flag; local first implementation uses block sensitivity order"),
    ("HeSF-SS-sensitivity-block-selector", "validation_sensitivity_selector", "hesf_coarsen/task_first/selection/selector.py", "select_support_nodes", "implemented", "block sensitivity selector"),
    ("HeSF-SS-prototype-residual-condense", "prototype_condensation", "hesf_coarsen/task_first/selection/condensation.py", "build_selected_support_graph", "implemented", "class_anchor_relation_prototype background"),
    ("HeSF-SS-sensitivity-plus-prototype", "selection_plus_prototype", "hesf_coarsen/task_first/selection/pipeline.py", "run_supervised_support_selection_pipeline", "implemented", "sensitivity selector with prototype residual background"),
)


def _run_git(args: list[str], root: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(output: str | Path) -> None:
    root = Path(__file__).resolve().parents[2]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report = "# Gate16 Code Sync Report\n\n"
    report += "## Git\n\n"
    report += f"- `git rev-parse HEAD`: `{_run_git(['rev-parse', 'HEAD'], root)}`\n"
    report += f"- `git branch --show-current`: `{_run_git(['branch', '--show-current'], root)}`\n"
    report += "\n### git status --short\n\n```text\n"
    report += _run_git(["status", "--short"], root) + "\n```\n\n"
    report += "### git log --oneline --decorate -n 20\n\n```text\n"
    report += _run_git(["log", "--oneline", "--decorate", "-n", "20"], root) + "\n```\n\n"
    report += "## Required Paths\n\n"
    for item in REQUIRED_PATHS:
        matches = sorted(glob.glob(str(root / item)))
        exists = bool(matches)
        report += f"- `{item}`: `{exists}`"
        if "*" in item:
            report += f" ({len(matches)} matches)"
        report += "\n"
    (out / "code_sync_report.md").write_text(report, encoding="utf-8")
    rows = [
        {
            "method": method,
            "family": family,
            "code_path": code_path,
            "function_or_class": function,
            "status": status,
            "notes": notes,
        }
        for method, family, code_path, function, status, notes in METHOD_ROWS
    ]
    _write_csv(out / "method_to_code_path.csv", rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Gate16 code audit outputs.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gate16_code_audit"))
    args = parser.parse_args(argv)
    write_audit(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
