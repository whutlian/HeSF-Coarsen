from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.scripts import run_next17_hybrid_accuracy as hybrid


def main(argv: Sequence[str] | None = None) -> int:
    hybrid.MODELS[:] = ["sehgnn_lite"]
    parser = hybrid.build_parser()
    parser.description = "Run the Next17 SeHGNN fidelity-labeled adapter. Official repo integration is not claimed."
    parser.set_defaults(output=Path("outputs/exp_next17_official_sehgnn_20260518"))
    args = parser.parse_args(argv)
    result = hybrid.run_next17_hybrid_accuracy(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
