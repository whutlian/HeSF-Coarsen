from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.scripts import run_next17_hybrid_accuracy as hybrid


def main(argv: Sequence[str] | None = None) -> int:
    hybrid.MODELS[:] = ["sehgnn_lite"]
    parser = hybrid.build_parser()
    parser.description = "Run Next17 SeHGNN-style Mode A/Mode B protocol split."
    parser.set_defaults(output=Path("outputs/exp_next17_sehgnn_full_target_20260518"))
    args = parser.parse_args(argv)
    result = hybrid.run_next17_hybrid_accuracy(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
