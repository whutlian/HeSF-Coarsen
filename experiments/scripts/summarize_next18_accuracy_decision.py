from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.accuracy.accuracy_branch_decision import decide_accuracy_branch


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_next18_accuracy_decision(
    *,
    keep_target_dir: str | Path,
    literature_dir: str | Path,
    docs_dir: str | Path = "docs",
) -> dict[str, str]:
    keep_target_dir = Path(keep_target_dir)
    literature_dir = Path(literature_dir)
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(keep_target_dir / "runs.csv")
    verdict = decide_accuracy_branch(rows)
    decision_path = docs_dir / "next18_accuracy_branch_decision.md"
    lines = [
        "# Next18 Accuracy Branch Decision",
        "",
        f"Final verdict: `{verdict['decision']}`",
        "",
        "## Evidence Used",
        "",
        f"- Eligible official/faithful real full-target rows: {verdict.get('eligible_rows', 0)}.",
        f"- Wins vs internal keep-target comparator: {verdict.get('wins_vs_internal_comparator', 0)}.",
        f"- Reason: {verdict['reason']}",
        "- Stage 0 audit keeps only A1/A2 in serious scope and deprecates A3/A4/A5.",
        "- Stage 1 separates `coarse_transfer`, `approx_full_target_adapter`, and `real_full_target_inference`.",
        "- Stage 2 records that official SeHGNN/HETTREE/FreeHGC are not integrated locally.",
        "- Stage 3 local A1/A2 rows are lite-adapter diagnostics only.",
        "- Stage 4 literature alignment marks all direct comparisons as non-comparable.",
        "",
        "## Carry Forward",
        "",
        "- Carry forward: preservation-first HeSF-LVC-P/S mainline.",
        "- Carry forward only as experimental utility: target-preserve support-only coarsening helper.",
        "- Do not carry forward: Hybrid-B target selection, meta-reconstruction proxy, deterministic distillation proxy, or task-aligned score as method claims.",
    ]
    decision_path.write_text("\n".join(lines), encoding="utf-8")
    postmortem = docs_dir / "accuracy_branch_postmortem.md"
    postmortem.write_text(
        "\n".join(
            [
                "# Accuracy Branch Postmortem",
                "",
                "Next18 drops the accuracy-first branch as a paper direction because no official or faithful real full-target evaluator evidence is available locally.",
                "",
                "A1/A2 remain useful as experimental diagnostics for target-preserve support-only compression, but not as a mainline method.",
                "",
                "A3/A4/A5 are deprecated. Their modules now carry experimental/deprecated notes and should not be extended in future runs unless a new faithful evaluator motivates revisiting them.",
                "",
                f"Literature alignment notes: `{literature_dir / 'literature_alignment_notes.md'}`.",
                f"Keep-target summary: `{keep_target_dir / 'summary.md'}`.",
            ]
        ),
        encoding="utf-8",
    )
    return {"decision": str(verdict["decision"]), "decision_path": str(decision_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-target-dir", type=Path, default=Path("outputs/exp_next18_accuracy_keep_target_final"))
    parser.add_argument("--literature-dir", type=Path, default=Path("outputs/exp_next18_accuracy_literature_alignment"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)
    print(summarize_next18_accuracy_decision(keep_target_dir=args.keep_target_dir, literature_dir=args.literature_dir, docs_dir=args.docs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
