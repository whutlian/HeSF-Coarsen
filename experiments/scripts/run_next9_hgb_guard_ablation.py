from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.summarize_next9_hgb_guard_ablation import summarize_next9_hgb_guard_ablation


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", nargs="*", default=["12345", "23456", "34567", "45678", "56789"])
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hgb-summary", default="outputs/exp_next9_hgb_paper_final_20260517_summary")
    parser.add_argument("--sourceaware-summary", default="outputs/exp_next8_hgb_lvc_sourceaware_5seed_20260517_summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    command = (
        "run_next9_hgb_guard_ablation "
        f"--datasets {' '.join(args.datasets)} --seeds {' '.join(map(str, args.seeds))} "
        f"--device {args.device}"
    )
    summarize_next9_hgb_guard_ablation(
        hgb_summary=args.hgb_summary,
        sourceaware_summary=args.sourceaware_summary,
        output=args.output,
        command_lines=[command],
    )


if __name__ == "__main__":
    main()
