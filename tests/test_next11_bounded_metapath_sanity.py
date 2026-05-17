from pathlib import Path

from experiments.scripts.run_next11_bounded_metapath_sanity import bounded_metapath_rows
from experiments.scripts.summarize_next11_bounded_metapath_sanity import summarize_next11_bounded_metapath_sanity
from experiments.scripts._common import write_csv
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_bounded_metapath_sanity_uses_actual_graph_structure(tmp_path: Path):
    graph = generate_synthetic_graph(num_users=12, num_items=8, num_tags=4, seed=11)

    rows = bounded_metapath_rows(graph, method="HeSF-LVC-P", dataset="Tiny", seed=11, sample_limit=20)

    assert rows
    assert all(row["sample_status"] == "bounded_actual_graph" for row in rows)
    assert all(int(row["bounded_metapath_samples"]) <= 20 for row in rows)


def test_bounded_metapath_summary_outputs_nonempty_tables(tmp_path: Path):
    inp = tmp_path / "input"
    out = tmp_path / "summary"
    write_csv(
        inp / "bounded_metapath_runs.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 1, "bounded_metapath_samples": 10, "schema_path_survival_rate": 0.8, "typed_path_count_drift": 0.1, "metapath_connectivity_retention": 0.7, "sample_status": "bounded_actual_graph"}],
    )

    summarize_next11_bounded_metapath_sanity(input=inp, output=out)

    assert (out / "bounded_metapath_by_method_dataset.csv").exists()
    assert (out / "figures/metapath_retention.png").exists()
    assert "bounded" in (out / "summary.md").read_text(encoding="utf-8").lower()
