from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_json

APV_UNCOMPRESSED_MICRO = 0.9481692


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_gate21_4_cache_feature(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = _read_csv(input_dir / "gate21_4_feature_channel_ablation.csv")
    adapter_rows = _read_csv(input_dir / "gate21_4_feature_cache_compression_results.csv")
    metric_rows = [row for row in adapter_rows if row.get("status") == "success" and _float(row.get("test_micro_f1")) is not None]
    byte50 = _adapter_pass(metric_rows, ratio_threshold=0.50, accuracy_drop=0.010)
    byte30 = _adapter_pass(metric_rows, ratio_threshold=0.30, accuracy_drop=0.020)
    decisions = [
        "FEATURE_ADAPTER_BYTE50_PASS" if byte50 else "FEATURE_ADAPTER_BYTE50_FAIL",
        "FEATURE_ADAPTER_BYTE30_PASS" if byte30 else "FEATURE_ADAPTER_BYTE30_FAIL",
        _term_redundancy_flag(feature_rows),
    ]
    if not metric_rows:
        decisions.append("FEATURE_ADAPTER_ACCURACY_NOT_VALIDATED")
    decision = {
        "decisions": decisions,
        "feature_channel_rows": len(feature_rows),
        "feature_cache_adapter_rows": len(adapter_rows),
        "feature_cache_adapter_success_rows": len(metric_rows),
        "feature_adapter_byte50_pass": byte50,
        "feature_adapter_byte30_pass": byte30,
    }
    write_json(output_dir / "gate21_4_decision.json", decision)
    (output_dir / "gate21_4_decision.md").write_text(
        "# Gate21.4 Cache/Feature Decision\n\n" + "\n".join(f"- `{flag}`" for flag in decisions) + "\n",
        encoding="utf-8",
    )
    return decision


def _adapter_pass(rows: Sequence[Mapping[str, Any]], *, ratio_threshold: float, accuracy_drop: float) -> bool:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("feature_compression_method", "")), []).append(row)
    for group in grouped.values():
        if len(group) < 3:
            continue
        ratios = [_float(row.get("effective_total_byte_ratio")) for row in group]
        micros = [_float(row.get("test_micro_f1")) for row in group]
        ratios = [value for value in ratios if value is not None]
        micros = [value for value in micros if value is not None]
        if len(ratios) != len(group) or len(micros) != len(group):
            continue
        if sum(ratios) / len(ratios) <= ratio_threshold and sum(micros) / len(micros) >= APV_UNCOMPRESSED_MICRO - accuracy_drop:
            return True
    return False


def _term_redundancy_flag(rows: Sequence[Mapping[str, Any]]) -> str:
    successes = [row for row in rows if row.get("status") == "success"]
    if not successes:
        return "PAPER_FEATURE_TERM_REDUNDANCY_NOT_VALIDATED"
    by_key = {(row.get("paper_feature_transform"), row.get("term_channel_spec")): _float(row.get("test_micro_f1")) for row in successes}
    raw_00 = by_key.get(("raw", "PTTP00"))
    zero_00 = by_key.get(("zero-paper", "PTTP00"))
    if raw_00 is not None and zero_00 is not None and raw_00 - zero_00 >= 0.01:
        return "PAPER_FEATURE_TERM_REDUNDANCY_SUPPORTED"
    return "PAPER_FEATURE_TERM_REDUNDANCY_NOT_SUPPORTED"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_gate21_4_cache_feature(args.input_dir, args.output_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
