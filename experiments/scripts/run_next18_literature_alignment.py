from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv


def literature_rows() -> list[dict[str, object]]:
    return [
        {
            "method": "FreeHGC official",
            "our_protocol_or_paper_protocol": "paper_protocol_reference_only",
            "split_policy": "paper_defined",
            "train_domain": "condensed/selected graph",
            "inference_domain": "paper full-target task protocol",
            "model_fidelity": "not_integrated",
            "macro_f1": "",
            "accuracy": "",
            "comparable_directly": False,
            "comparison_note": "PKU-DAIR/FreeHGC is not integrated locally; ratio semantics are documented but results are not reproduced.",
        },
        {
            "method": "SeHGNN full-graph reference",
            "our_protocol_or_paper_protocol": "official_repo_available_not_integrated",
            "split_policy": "not_aligned",
            "train_domain": "full graph",
            "inference_domain": "full target",
            "model_fidelity": "official_not_integrated",
            "macro_f1": "",
            "accuracy": "",
            "comparable_directly": False,
            "comparison_note": "ICT-GIMLab/SeHGNN is reachable but official preprocessing is not wired to HeSF graph outputs.",
        },
        {
            "method": "HETTREE official",
            "our_protocol_or_paper_protocol": "official_repo_unavailable",
            "split_policy": "not_aligned",
            "train_domain": "full graph",
            "inference_domain": "full target",
            "model_fidelity": "unavailable",
            "macro_f1": "",
            "accuracy": "",
            "comparable_directly": False,
            "comparison_note": "The advertised microsoft/HetTree GitHub URL was not accessible from this environment.",
        },
        {
            "method": "Next18 A1/A2 local adapters",
            "our_protocol_or_paper_protocol": "local_diagnostic_protocol",
            "split_policy": "synthetic_stratified",
            "train_domain": "target-preserve hybrid graph",
            "inference_domain": "full original target set via target-preserved hybrid predictions",
            "model_fidelity": "lite_adapter",
            "macro_f1": "see keep_target_final",
            "accuracy": "see keep_target_final",
            "comparable_directly": False,
            "comparison_note": "Useful for claim boundary only; not a literature-facing task baseline.",
        },
    ]


def run_next18_literature_alignment(output: str | Path) -> dict[str, int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = literature_rows()
    write_csv(output / "literature_alignment_table.csv", rows)
    lines = [
        "# Next18 Literature Alignment Notes",
        "",
        markdown_table(rows, ["method", "model_fidelity", "split_policy", "train_domain", "inference_domain", "comparable_directly", "comparison_note"]),
        "",
        "No direct literature comparison is claimed because official preprocessing, split policy, train graph type, and model implementation fidelity are not aligned.",
    ]
    (output / "literature_alignment_notes.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_next18_accuracy_literature_alignment"))
    args = parser.parse_args(argv)
    print(run_next18_literature_alignment(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
