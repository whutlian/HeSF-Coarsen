from __future__ import annotations

import argparse

from hesf_coarsen.eval.official.stage_report_summarizer import summarize_existing_output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Gate21.15 decision/summary artifacts from table CSVs.")
    parser.add_argument("--input-dir", default="outputs/gate21_15_stage_report")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    decision = summarize_existing_output_dir(args.input_dir, args.output_dir)
    print(f"Gate21.15 STAGE_REPORT_TABLE_READY={decision['STAGE_REPORT_TABLE_READY']}")


if __name__ == "__main__":
    main()
