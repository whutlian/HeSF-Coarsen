from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import aggregate, read_csv, write_png


def summarize_next11_bounded_metapath_sanity(*, input: str | Path, output: str | Path) -> dict:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = read_csv(input / "bounded_metapath_runs.csv")
    by_method = aggregate(rows, ["method", "dataset"], ["bounded_metapath_samples", "schema_path_survival_rate", "typed_path_count_drift", "metapath_connectivity_retention"])
    write_csv(output / "bounded_metapath_runs.csv", rows)
    write_csv(output / "bounded_metapath_by_method_dataset.csv", by_method)
    write_png(output / "figures" / "metapath_retention.png", by_method, "typed_path_count_drift_mean", "metapath_connectivity_retention_mean")
    complete = bool(rows) and all(row.get("sample_status") == "bounded_actual_graph" for row in rows)
    lines = [
        "# Next11 Bounded Metapath Sanity",
        "",
        "All samples are bounded and generated from actual graph structure." if complete else "Metapath sanity is incomplete; omit metapath claims.",
        "",
        markdown_table(by_method, ["method", "dataset", "run_count", "metapath_connectivity_retention_mean"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "by_method": by_method}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_bounded_metapath_sanity(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
