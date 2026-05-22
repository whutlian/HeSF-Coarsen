import csv

from experiments.scripts.summarize_gate17 import exact_only_paired_gaps, requested_ratio_paired_gaps, summarize


def _row(dataset, seed, method, *, ratio=0.3, count=3, exact=True, macro=0.5, acc=0.5):
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "requested_support_ratio": ratio,
        "requested_support_count": count,
        "realized_support_count": count if exact else count + 1,
        "support_budget_exact_match": exact,
        "status": "success",
        "macro_f1": macro,
        "accuracy": acc,
        "validation_macro_f1": macro,
        "validation_accuracy": acc,
        "primary_eval_mode": "compressed_projected",
    }


def test_exact_only_paired_gaps_exclude_non_exact_rows():
    rows = [
        _row("ACM", 1, "H6-no-spec-support-only", exact=True, macro=0.50, acc=0.50),
        _row("ACM", 1, "HeSF-SS-real-occlusion-block", exact=True, macro=0.55, acc=0.56),
        _row("DBLP", 1, "flatten-sum-support-only", exact=True, macro=0.60, acc=0.60),
        _row("DBLP", 1, "HeSF-SS-real-occlusion-block", exact=False, macro=0.70, acc=0.70),
    ]

    gaps = exact_only_paired_gaps(rows)

    assert len(gaps) == 1
    assert gaps[0]["dataset"] == "ACM"
    assert gaps[0]["delta_macro_f1"] == 0.05
    assert gaps[0]["comparison_scope"] == "exact_only"


def test_requested_ratio_paired_gaps_keep_non_exact_but_label_them():
    rows = [
        _row("DBLP", 1, "flatten-sum-support-only", exact=True, macro=0.60, acc=0.60),
        _row("DBLP", 1, "HeSF-SS-real-occlusion-block", exact=False, macro=0.70, acc=0.70),
    ]

    gaps = requested_ratio_paired_gaps(rows)

    assert len(gaps) == 1
    assert gaps[0]["dataset"] == "DBLP"
    assert gaps[0]["comparison_scope"] == "requested_ratio"
    assert gaps[0]["method_budget_exact"] is False
    assert gaps[0]["baseline_budget_exact"] is True
    assert gaps[0]["delta_macro_f1"] == 0.10


def test_gate17_summary_writes_both_gap_tables(tmp_path):
    rows = [
        _row("ACM", 1, "H6-no-spec-support-only", exact=True, macro=0.50, acc=0.50),
        _row("ACM", 1, "HeSF-SS-real-occlusion-block", exact=True, macro=0.55, acc=0.56),
        _row("DBLP", 1, "flatten-sum-support-only", exact=True, macro=0.60, acc=0.60),
        _row("DBLP", 1, "HeSF-SS-real-occlusion-block", exact=False, macro=0.70, acc=0.70),
    ]
    path = tmp_path / "gate17_raw_rows.csv"
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summarize(tmp_path, tmp_path / "tables")
    exact = list(csv.DictReader((tmp_path / "tables" / "gate17_exact_only_paired_gaps.csv").open(encoding="utf-8")))
    requested = list(csv.DictReader((tmp_path / "tables" / "gate17_requested_ratio_paired_gaps.csv").open(encoding="utf-8")))

    assert len(exact) == 1
    assert len(requested) == 2
