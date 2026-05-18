from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.scripts import run_next17_hybrid_accuracy as hybrid


def main(argv: Sequence[str] | None = None) -> int:
    parser = hybrid.build_parser()
    parser.description = "Run Next17 P1 target-preserve support-coarsening experiments."
    parser.set_defaults(output=Path("outputs/exp_next17_target_preserve_support_coarsen_20260518"))
    args = parser.parse_args(argv)
    result = hybrid.run_next17_hybrid_accuracy(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
