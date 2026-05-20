from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write Gate13 TaskFirst code audit.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Gate13 TaskFirst Code Audit

Git commit: `{git_commit_hash()}`

## Findings

- Current Gate12 default `candidate_source`: script default is `random_support`; Gate13 random support has been corrected to emit support-node candidates only.
- Current library default `pair_delta_mode`: `exact`; Gate12 script default was `local_surrogate`. Gate13 tests `local_surrogate`, `exact_pair_isolated`, and `response_signature`.
- Target nodes can never be merged: `TaskFirstConfig.keep_all_target_nodes=True`, `support_only_coarsening=True`, target singleton template, hard merge constraints, and assignment validation all preserve target nodes.
- Candidate pairs are filtered to support-only before scoring by `allow_task_first_merge`; Gate13 target-aware candidate sources also avoid target nodes at emission time.
- Relation response affects score ranking through `lambda_rel_response * delta_rel_response`; Gate13 logs rank/selection shifts for lambda sweeps.
- Old support coverage only detects same-anchor overlap. Gate13 adds `cross_anchor_collision` and `class_context_collision`, plus `combined`.
- Old zero-footprint support nodes could behave as no-conflict. Gate13 adds footprint states and policies: `zero_as_no_conflict`, `unknown_blocks_known`, `unknown_propagated`, `unknown_only_merge`.
- `exact` is pair-isolated, not stateful. Gate13 exposes this as `exact_pair_isolated`; `stateful_approx_status=not_implemented` is reported explicitly.
- Evaluator outputs remain `hettree_lite` diagnostics: train on coarse train labels and evaluate original target transfer/projected predictions. They are not official HETTREE/SeHGNN/FreeHGC numbers.
- Ratio-mode and budget calculation can produce floors because target nodes are singleton and support candidate exhaustion or constraints can prevent further merges. Gate13 logs stop/floor reasons by level.

## Claim Boundary

HeSF-TC is under validation only. Preservation-first HeSF-LVC-P/S remains separate and unchanged.
"""
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
