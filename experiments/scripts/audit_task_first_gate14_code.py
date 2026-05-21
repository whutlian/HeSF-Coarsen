from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write Gate14 task-first code audit.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Gate14 Code Audit

Git commit: `{git_commit_hash()}`

## Stage 0 Findings

- `coverage_v1_legacy` is explicitly preserved for old common-anchor-only behavior.
- `coverage_v2` is implemented as anchor distribution collision + class context collision + receptive field diversity loss.
- `purity_v2` is implemented with explicit known / unknown structured / unknown weak diagnostics and hybrid propagated footprints.
- `stateful_signature` is implemented through bounded cluster-signature greedy matching.
- Candidate sources include random support, graph sketch, target-anchor co-support, class-footprint KNN, target-response-signature KNN, relation-response KNN, and hybrid task-aware union.
- Ratio matching is separated from requested-ratio aggregation through nearest realized-support-ratio matching.
- `hettree_lite` remains diagnostic; official SeHGNN/HETTREE/FreeHGC status is `not_integrated`.

## Non-Goals

- No dense adjacency materialization.
- No explicit A^2 / relation-product adjacency.
- No large eigendecomposition.
- No official literature claim from lite evaluator outputs.
"""
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
