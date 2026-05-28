from __future__ import annotations

from experiments.scripts.run_gate21_14_full_execution_push import build_arg_parser, run


def test_cross_dataset_selector_semantics_do_not_hardcode_dblp_relation_names(tmp_path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "ACM",
            "IMDB",
            "--only",
            "cross_dataset",
            "--output-dir",
            str(tmp_path / "gate21_14_cross"),
        ]
    )

    run(args)
    text = (tmp_path / "gate21_14_cross" / "gate21_14_cross_dataset_runs.csv").read_text(encoding="utf-8")

    assert "target-support" in text
    assert "support-context" in text
    assert "AP_keep" not in text
    assert "PV_keep" not in text
