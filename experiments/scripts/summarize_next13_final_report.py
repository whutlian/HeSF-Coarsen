from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


EXPERIMENT_COMMANDS = [
    (
        "P0 paper tables",
        "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m "
        "experiments.scripts.summarize_next13_paper_tables --next12-paper "
        "outputs\\exp_next12_paper_tables_20260517_summary --next12-ahugc "
        "outputs\\exp_next12_ahugc_style_sweep_20260517_summary --output "
        "outputs\\exp_next13_paper_tables_20260517_summary",
    ),
    (
        "P2 path-mass metapath",
        "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m "
        "experiments.scripts.run_next13_metapath_mass --datasets ACM DBLP IMDB "
        "--seeds 12345 23456 34567 45678 56789 --methods HeSF-LVC-P "
        "HeSF-LVC-S flatten-sum H6-no-spec H0-mutual-best AH-UGC-style-tuned "
        "GraphZoom-style ConvMatch-style random --schema-path-lengths 2 3 "
        "--num-probes 16 --max-schema-paths 12 --device cuda --output "
        "outputs\\exp_next13_metapath_mass_20260517_summary",
    ),
    (
        "P3 structure-critical tasks",
        "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m "
        "experiments.scripts.run_next13_structure_critical_tasks --datasets ACM "
        "DBLP IMDB --seeds 12345 23456 34567 45678 56789 --methods "
        "HeSF-LVC-P HeSF-LVC-S flatten-sum H6-no-spec H0-mutual-best "
        "AH-UGC-style-tuned GraphZoom-style ConvMatch-style random --tasks "
        "lowpass_signal_reconstruction feature_free_label_propagation --device "
        "cuda --output outputs\\exp_next13_structure_critical_20260517_summary",
    ),
    (
        "P4 AH-UGC-style fair baseline",
        "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m "
        "experiments.scripts.summarize_next13_ahugc_fair_baseline --next12-ahugc "
        "outputs\\exp_next12_ahugc_style_sweep_20260517_summary --next12-paper "
        "outputs\\exp_next12_paper_tables_20260517_summary --output "
        "outputs\\exp_next13_ahugc_fair_baseline_20260517_summary",
    ),
    (
        "P5 OGBN aggregation backend",
        "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m "
        "experiments.scripts.run_next13_ogbn_aggregation_backend --sizes 200k "
        "500k 1m full-local --methods HeSF-LVC-P HeSF-LVC-S --backends "
        "A0_current_sort_reducer A4_local_prededup_sort_reducer --device cuda "
        "--output outputs\\exp_next13_ogbn_aggregation_backend_20260517_summary "
        "--python C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe",
    ),
]


VERIFICATION_COMMANDS = [
    "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest "
    "tests/test_next13_paper_tables.py tests/test_next13_ahugc_table.py -q",
    "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest "
    "tests/test_metapath_mass.py -q",
    "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest "
    "tests/test_aggregation_exclusive_timing.py -q",
    "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest "
    "tests/test_next13_paper_tables.py tests/test_next13_ahugc_table.py "
    "tests/test_metapath_mass.py tests/test_next13_structure_tasks.py "
    "tests/test_aggregation_exclusive_timing.py -q",
    "C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest -q",
]


def _command_block(commands: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for label, command in commands:
        lines.extend([f"### {label}", "", "```powershell", command, "```", ""])
    return "\n".join(lines).rstrip()


def _plain_command_block(commands: list[str]) -> str:
    lines: list[str] = []
    for command in commands:
        lines.extend(["```powershell", command, "```", ""])
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--metapath", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--ahugc", type=Path, required=True)
    parser.add_argument("--ogbn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sections = [
        "# Next13 Final Report",
        "",
        "## Environment",
        "",
        "- Workspace: `D:\\HeSF-Coarsen`",
        "- Python: `C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe`",
        "- Device request: local conda `pytorch`, CUDA where available.",
        "- OOM status: no OOM recorded in completed local runs unless stated below.",
        "- Generated experiment outputs stay under `outputs/` and are not intended for git staging.",
        "",
        "## Code Changes",
        "",
        "- Added paper-table and AH-UGC-style fair-baseline summarizers for Next13.",
        "- Added sparse path-mass metapath preservation evaluation with bounded probes.",
        "- Added synthetic structure-critical diagnostics for low-pass reconstruction and feature-free label propagation.",
        "- Added OGBN aggregation exclusive timing fields and the explicit `local_prededup_sort` backend label.",
        "- Added tests covering paper-table hygiene, AH-UGC row classes, path-mass behavior, structure diagnostics, and aggregation timing invariants.",
        "",
        "## Exact Commands",
        "",
        _command_block(EXPERIMENT_COMMANDS),
        "",
        "## P0 Paper Tables",
        "",
        _read(args.paper / "summary.md"),
        "## P1/P2 Metapath Mass",
        "",
        _read(args.metapath / "summary.md"),
        "## P3 Structure-Critical Tasks",
        "",
        _read(args.structure / "summary.md"),
        "## P4 AH-UGC-Style Fair Baseline",
        "",
        _read(args.ahugc / "summary.md"),
        "## P5 OGBN Aggregation Backend",
        "",
        _read(args.ogbn / "summary.md"),
        "## Claim Boundary",
        "",
        "Supported: P/S are preservation-first and task-competitive under compression.",
        "Unsupported: task-F1 dominance, official AH-UGC reproduction, metapath survival proof, A3 speedup, guard/source-aware as main method, lambda_conv/lambda_rel as core, and OGBN task-quality claims.",
        "",
        "## Verification Commands",
        "",
        _plain_command_block(VERIFICATION_COMMANDS),
        "",
        "## Risks And Limits",
        "",
        "- AH-UGC-style is a protocol-matched type-isolated hash/LSH baseline, not an official AH-UGC reproduction.",
        "- Path-mass metrics are appendix diagnostics unless they clearly improve the main operator-preservation claim.",
        "- Structure-critical tasks are synthetic diagnostics and are not official HGB task performance.",
        "- OGBN-MAG remains system/profiling evidence only; it does not support task-quality claims.",
        "- The A4 backend is adopted only if it meets the full-local speed, correctness, and RSS criteria.",
        "",
        "## Output Index",
        "",
        f"- `{args.paper}`",
        f"- `{args.metapath}`",
        f"- `{args.structure}`",
        f"- `{args.ahugc}`",
        f"- `{args.ogbn}`",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
