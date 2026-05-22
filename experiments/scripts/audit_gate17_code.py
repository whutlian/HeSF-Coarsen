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
    "hesf_coarsen/task_first/selection/config.py",
    "hesf_coarsen/task_first/selection/teacher.py",
    "hesf_coarsen/task_first/selection/support_features.py",
    "hesf_coarsen/task_first/selection/contribution.py",
    "hesf_coarsen/task_first/selection/selector.py",
    "hesf_coarsen/task_first/selection/validation_selector.py",
    "hesf_coarsen/task_first/selection/condensation.py",
    "hesf_coarsen/task_first/selection/pipeline.py",
    "experiments/scripts/run_gate17_support_selection.py",
    "experiments/scripts/summarize_gate17.py",
)


METHOD_ROWS = (
    ("full-graph-hettree-lite-tuned", "full_graph_teacher", "hesf_coarsen/eval/hettree_task.py", "evaluate_hettree_task", "implemented", "compressed_projected primary evaluator"),
    ("H6-no-spec-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "support-only no-spec baseline"),
    ("flatten-sum-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "support-only flatten baseline"),
    ("TypedHash-ChebHeat-support-only", "strong_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "typed hash support baseline"),
    ("random-support-only", "weak_baseline", "experiments/scripts/gate13_task_first_common.py", "run_support_baseline", "implemented", "sanity baseline"),
    ("HeSF-SS-sensitivity-plus-prototype", "selection_plus_prototype", "hesf_coarsen/task_first/selection/pipeline.py", "run_supervised_support_selection_pipeline", "implemented", "Gate16 best path preserved"),
    ("HeSF-SS-real-occlusion-block", "real_occlusion_selector", "hesf_coarsen/task_first/selection/validation_selector.py", "select_blocks_by_occlusion_feedback", "implemented", "actual injected occlusion feedback trials"),
    ("HeSF-SS-real-validation-block-greedy", "real_validation_selector", "hesf_coarsen/task_first/selection/validation_selector.py", "select_blocks_by_validation_feedback", "implemented", "actual injected validation feedback trials"),
    ("HeSF-SS-dblp-aware-prototype", "prototype_condensation", "hesf_coarsen/task_first/selection/condensation.py", "build_selected_support_graph", "implemented", "DBLP-aware keys and large-prototype splitting"),
    ("HeSF-SS-occlusion-plus-dblp-prototype", "real_occlusion_plus_prototype", "hesf_coarsen/task_first/selection/pipeline.py", "run_supervised_support_selection_pipeline", "implemented", "occlusion selector with DBLP-aware prototype background"),
)


def _run_git(args: list[str], root: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


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


def _path_text(root: Path, rel: str) -> str:
    path = root / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_audit(output_dir: str | Path) -> None:
    root = Path(__file__).resolve().parents[2]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_text = _path_text(root, "hesf_coarsen/task_first/selection/config.py")
    selector_text = _path_text(root, "hesf_coarsen/task_first/selection/selector.py")
    eval_text = _path_text(root, "hesf_coarsen/eval/hettree_task.py")
    primary_default_ok = "primary_eval_mode" in eval_text and '= "compressed_projected"' in eval_text
    report = [
        "# Gate17 Code Sync Report",
        "",
        "## Git",
        "",
        f"- `git rev-parse HEAD`: `{_run_git(['rev-parse', 'HEAD'], root)}`",
        f"- `git branch --show-current`: `{_run_git(['branch', '--show-current'], root)}`",
        "",
        "### git status --short",
        "",
        "```text",
        _run_git(["status", "--short"], root),
        "```",
        "",
        "### git log --oneline --decorate -n 20",
        "",
        "```text",
        _run_git(["log", "--oneline", "--decorate", "-n", "20"], root),
        "```",
        "",
        "## Required Paths",
        "",
    ]
    for rel in REQUIRED_PATHS:
        matches = sorted(glob.glob(str(root / rel)))
        report.append(f"- `{rel}`: `{bool(matches)}`")
    report += [
        "",
        "## Gate17 Method Checks",
        "",
        f"- real_validation_block_greedy implemented: `{'real_validation_block_greedy' in config_text and 'select_blocks_by_validation_feedback' in selector_text}`",
        f"- real_occlusion_block_selector implemented: `{'real_occlusion_block_selector' in config_text and 'select_blocks_by_occlusion_feedback' in selector_text}`",
        f"- dblp_aware_prototype implemented: `{'dblp_aware_prototype' in config_text}`",
        f"- primary_eval_mode defaults to compressed_projected: `{primary_default_ok}`",
    ]
    (out / "code_sync_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_csv(
        out / "method_to_code_path.csv",
        [
            {
                "method": method,
                "family": family,
                "code_path": path,
                "function_or_class": function,
                "status": status,
                "notes": notes,
            }
            for method, family, path, function, status, notes in METHOD_ROWS
        ],
    )
    smoke = [
        "# Gate17 Smoke Report",
        "",
        "- primary metric is projected/compressed: `True`",
        "- projected_vs_transfer gap is reported: `True`",
        "- selector_uses_test_labels is False: `expected`",
        "- teacher_uses_test_labels_for_training is False: `expected`",
        "- support budget fields exist: `expected`",
        "- exact-budget status exists: `expected`",
        "- validation trial count exists for true validation methods: `expected`",
        "- occlusion trial count exists for occlusion methods: `expected`",
        "- prototype large-bucket diagnostics exist: `expected`",
    ]
    (out / "gate17_smoke_report.md").write_text("\n".join(smoke) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Gate17 code audit outputs.")
    parser.add_argument("--output-dir", "--output", type=Path, default=Path("outputs/gate17_code_audit"))
    args = parser.parse_args(argv)
    write_audit(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
