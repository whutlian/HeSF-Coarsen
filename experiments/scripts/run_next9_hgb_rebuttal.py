from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.summarize_next9_hgb_rebuttal import summarize_next9_hgb_rebuttal


DEFAULT_RUN_SUMMARIES = [
    "outputs/exp_next7_hgb_lvc_pst_5seed_20260516_summary",
    "outputs/exp_next7_hgb_flatten_sum_5seed_20260516_summary",
    "outputs/exp_next5_hgb_final_5seed_20260516_summary",
    "outputs/exp_next4_mainline_full_20260515_summary",
]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", nargs="*", default=["12345", "23456", "34567", "45678", "56789"])
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--next8-summary-dir", default="outputs/exp_next8_final_gap_20260517_summary")
    parser.add_argument("--run-summary-dirs", nargs="*", default=DEFAULT_RUN_SUMMARIES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    command = (
        "run_next9_hgb_rebuttal "
        f"--datasets {' '.join(args.datasets)} --seeds {' '.join(map(str, args.seeds))} "
        f"--methods {' '.join(args.methods)} --device {args.device}"
    )
    summarize_next9_hgb_rebuttal(
        next8_summary_dir=args.next8_summary_dir,
        run_summary_dirs=args.run_summary_dirs,
        output=args.output,
        command_lines=[command],
    )


if __name__ == "__main__":
    main()
