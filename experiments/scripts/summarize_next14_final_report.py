from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


PYTHON = r"C:\Users\slian\anaconda3\envs\pytorch\python.exe"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


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


EXPERIMENT_COMMANDS = [
    (
        "P0 paper tables",
        PYTHON
        + " -m experiments.scripts.summarize_next14_paper_tables --next13-paper "
        "outputs\\exp_next13_paper_tables_20260517_summary --next13-ahugc "
        "outputs\\exp_next13_ahugc_fair_baseline_20260517_summary --next13-metapath "
        "outputs\\exp_next13_metapath_mass_20260517_summary --next13-structure "
        "outputs\\exp_next13_structure_critical_20260517_summary --next13-ogbn "
        "outputs\\exp_next13_ogbn_aggregation_backend_20260517_summary --next12-paper "
        "outputs\\exp_next12_paper_tables_20260517_summary --next10-rebuttal "
        "outputs\\exp_next10_hgb_rebuttal_tables_20260517_summary --next10-resource "
        "outputs\\exp_next10_hgb_resource_logged_20260517 --output "
        "outputs\\exp_next14_paper_tables_20260518_summary",
    ),
    (
        "P1 held-out fused operator probe",
        PYTHON
        + " -m experiments.scripts.run_next14_operator_holdout --datasets ACM DBLP IMDB "
        "--seeds 12345 23456 34567 45678 56789 --methods HeSF-LVC-P HeSF-LVC-S "
        "flatten-sum H6-no-spec H0-mutual-best TypedHash-ChebHeat GraphZoom-style "
        "ConvMatch-style random --probe-dim 32 --cheb-order 5 --device cuda --output "
        "outputs\\exp_next14_operator_holdout_20260518_summary",
    ),
    (
        "P2 metapath appendix",
        PYTHON
        + " -m experiments.scripts.summarize_next14_metapath_appendix --next13-metapath "
        "outputs\\exp_next13_metapath_mass_20260517_summary --output "
        "outputs\\exp_next14_metapath_appendix_20260518_summary",
    ),
    (
        "P3 TypedHash fair baseline",
        PYTHON
        + " -m experiments.scripts.summarize_next14_typedhash_baseline --next12-ahugc "
        "outputs\\exp_next12_ahugc_style_sweep_20260517_summary --next13-paper "
        "outputs\\exp_next13_paper_tables_20260517_summary --output "
        "outputs\\exp_next14_typedhash_fair_baseline_20260518_summary",
    ),
    (
        "P4 OGBN output/merge backend",
        PYTHON
        + " -m experiments.scripts.run_next14_ogbn_output_merge_backend --sizes 200k 500k 1m "
        "full-local --methods HeSF-LVC-P HeSF-LVC-S --backends A0_current_sort_reducer "
        "A6_direct_relation_writer A7_parallel_relation_output_writer A8_shard_count_chunk_sweep "
        "--device cuda --output outputs\\exp_next14_ogbn_output_merge_backend_20260518_summary "
        "--python "
        + PYTHON,
    ),
]


VERIFICATION_COMMANDS = [
    PYTHON + " -m pytest tests/test_next14_paper_tables.py -q",
    PYTHON + " -m pytest tests/test_holdout_operator_probes.py -q",
    PYTHON + " -m pytest tests/test_next14_metapath_position.py -q",
    PYTHON + " -m pytest tests/test_next14_typedhash_baseline.py -q",
    PYTHON
    + " -m pytest tests/test_aggregation_output_merge_backend.py "
    "tests/test_aggregation_exclusive_timing.py -q",
    PYTHON
    + " -m pytest tests/test_next14_paper_tables.py tests/test_holdout_operator_probes.py "
    "tests/test_next14_metapath_position.py tests/test_next14_typedhash_baseline.py "
    "tests/test_aggregation_output_merge_backend.py tests/test_aggregation_exclusive_timing.py -q",
    PYTHON + " -m pytest -q",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--metapath", type=Path, required=True)
    parser.add_argument("--typedhash", type=Path, required=True)
    parser.add_argument("--ogbn", type=Path, required=True)
    parser.add_argument("--verification-status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    verification_status = (
        _read(args.verification_status)
        if args.verification_status is not None
        else "Verification status not supplied to the report generator."
    )

    sections = [
        "# Next14 Final Report",
        "",
        "## Environment",
        "",
        "- Workspace: `D:\\HeSF-Coarsen`",
        f"- Python: `{PYTHON}`",
        "- Device request: local conda `pytorch`, CUDA where available.",
        "- OOM status: no OOM recorded in completed local runs unless stated below.",
        "- Generated experiment outputs stay under `outputs/` and are not intended for git staging.",
        "",
        "## Code Changes",
        "",
        "- Added Next14 paper-table cleanup and claim-safe summary generation.",
        "- Added held-out fused-operator probe evaluation using deterministic probes and sparse relation matvecs.",
        "- Added appendix-only metapath positioning and TypedHash fairness summaries.",
        "- Added OGBN output/merge backend labels and additional aggregation diagnostics while keeping A0 valid.",
        "- Added tests covering paper-table hygiene, operator probes, appendix boundaries, TypedHash naming, and aggregation backend gates.",
        "",
        "## Exact Commands",
        "",
        _command_block(EXPERIMENT_COMMANDS),
        "",
        "## Tests Run And Status",
        "",
        verification_status.strip(),
        "",
        "## Verification Commands",
        "",
        _plain_command_block(VERIFICATION_COMMANDS),
        "",
        "## P0 Paper Table Hygiene Results",
        "",
        _read(args.paper / "summary.md"),
        "## P1 Held-Out Operator Probe Results",
        "",
        _read(args.operator / "summary.md"),
        "## P2 Metapath Appendix Positioning",
        "",
        _read(args.metapath / "summary.md"),
        "## P3 TypedHash Baseline Fairness",
        "",
        _read(args.typedhash / "summary.md"),
        "## P4 OGBN Output/Merge Backend Results",
        "",
        _read(args.ogbn / "summary.md"),
        "## Claim Boundary",
        "",
        "Supported wording:",
        "",
        "- HeSF-LVC-P/S are preservation-first heterogeneous graph coarsening methods.",
        "- They strongly preserve typed fused-operator / relation-energy structure under HGB coarsening.",
        "- They maintain competitive task recovery under compression.",
        "- Flatten-sum and H6-no-spec can be task-competitive while damaging operator/relation preservation.",
        "- TypedHash-ChebHeat is a strong protocol-matched hash baseline; P/S improve preservation and task recovery at higher coarsening cost.",
        "- OGBN-MAG is used for scalability and profiling, not task-quality claims.",
        "",
        "Unsupported wording:",
        "",
        "- HeSF-LVC beats full tuned RGCN.",
        "- HeSF-LVC dominates flatten-sum or H6 on task F1.",
        "- HeSF-LVC preserves metapath/path-mass better than flatten-sum/H6.",
        "- TypedHash-ChebHeat is official AH-UGC.",
        "- A4/A6/A7/A8 improves aggregation unless adoption criteria are met.",
        "- `lambda_conv`, `lambda_rel`, guard, or source-aware filtering is core.",
        "- OGBN-MAG proves task quality.",
        "",
        "## Known Risks And Missing Rows",
        "",
        "- Held-out fused-operator probes did not give a clean P/S win over flatten-sum/H6, so this diagnostic must be bounded.",
        "- Metapath/path-mass diagnostics remain appendix-only.",
        "- TypedHash-ChebHeat is a protocol-matched baseline, not an official AH-UGC result.",
        "- OGBN-MAG remains system/profiling evidence only.",
        "- A6/A7/A8 are adopted only if the generated speedup summary marks them recommended.",
        f"- Missing row details, if any: `{args.paper / 'missing_rows.md'}`",
        "",
        "## Output Index",
        "",
        f"- `{args.paper}`",
        f"- `{args.operator}`",
        f"- `{args.metapath}`",
        f"- `{args.typedhash}`",
        f"- `{args.ogbn}`",
        f"- `{args.output}`",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
