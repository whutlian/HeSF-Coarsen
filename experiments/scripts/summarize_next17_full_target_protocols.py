from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from experiments.scripts.run_next17_hybrid_accuracy import _rows_for_block
from experiments.scripts.summarize_next17_hybrid_accuracy import summarize_block


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Next17 full-target protocol split results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or (args.input / "full_target_protocol")
    summarize_block(
        output,
        _rows_for_block(_read_csv(args.input / "runs.csv"), "full_target_protocol"),
        title="Next17 P3 Full-Target Protocol Split",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
