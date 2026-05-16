from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.summarize_next9_ogbn_aggregation import summarize_next9_ogbn_aggregation


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=["200k", "500k", "1m", "full-local"])
    parser.add_argument("--methods", nargs="*", default=["HeSF-LVC-P", "HeSF-LVC-S"])
    parser.add_argument("--candidate-mode", default="optimized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input", default="outputs/exp_next8_ogbn_system_scale_20260517_summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    command = (
        "run_next9_ogbn_aggregation_benchmark "
        f"--sizes {' '.join(args.sizes)} --methods {' '.join(args.methods)} "
        f"--candidate-mode {args.candidate_mode} --device {args.device}"
    )
    summarize_next9_ogbn_aggregation(
        input_summary=args.input,
        output=args.output,
        command_lines=[command],
    )


if __name__ == "__main__":
    main()
