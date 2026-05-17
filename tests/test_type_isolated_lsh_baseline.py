import numpy as np

from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import validate_schema


def test_type_isolated_lsh_preserves_type_and_schema():
    graph = generate_synthetic_graph(num_users=20, num_items=10, num_tags=6, seed=7)

    coarse, assignment, diagnostics = coarsen_type_isolated_lsh(graph, target_ratio=0.5, seed=7, max_cluster_size=4)

    validate_schema(coarse)
    for node, supernode in enumerate(assignment.assignment):
        assert graph.node_type[node] == assignment.supernode_type[supernode]
    assert int(assignment.cluster_sizes().max()) <= 4
    assert set(coarse.relation_specs) == set(graph.relation_specs)
    assert diagnostics["baseline_name"] == "AH-UGC-style protocol-matched baseline"
    assert diagnostics["target_ratio"] == 0.5
    assert np.isfinite(diagnostics["final_ratio"])


def test_type_isolated_lsh_supports_assignment_sources_and_bucket_topk():
    graph = generate_synthetic_graph(num_users=20, num_items=10, num_tags=6, seed=8)

    coarse_raw, assignment_raw, raw_diag = coarsen_type_isolated_lsh(
        graph,
        target_ratio=0.5,
        seed=7,
        hash_bits=8,
        bucket_topk=2,
        assignment_source="raw_feature",
    )
    coarse_sketch, assignment_sketch, sketch_diag = coarsen_type_isolated_lsh(
        graph,
        target_ratio=0.5,
        seed=7,
        hash_bits=12,
        bucket_topk=3,
        assignment_source="feature_plus_sketch",
    )

    validate_schema(coarse_raw)
    validate_schema(coarse_sketch)
    assert raw_diag["assignment_source"] == "raw_feature"
    assert sketch_diag["assignment_source"] == "feature_plus_sketch"
    assert raw_diag["bucket_topk"] == 2
    assert sketch_diag["hash_bits"] == 12
    assert assignment_raw.assignment.shape == assignment_sketch.assignment.shape
