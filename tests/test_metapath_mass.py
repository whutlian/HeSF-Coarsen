import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _toy_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    relations = {
        0: RelationAdj(src=np.array([0, 1, 1]), dst=np.array([2, 2, 3]), weight=np.ones(3, dtype=np.float32), src_type=0, dst_type=1, relation_id=0),
        1: RelationAdj(src=np.array([2, 3, 3]), dst=np.array([4, 4, 5]), weight=np.ones(3, dtype=np.float32), src_type=1, dst_type=2, relation_id=1),
        2: RelationAdj(src=np.array([0, 1]), dst=np.array([4, 5]), weight=np.ones(2, dtype=np.float32), src_type=0, dst_type=2, relation_id=2),
    }
    specs = {rid: RelationSpec(rid, f"r{rid}", rel.src_type, rel.dst_type) for rid, rel in relations.items()}
    return HeteroGraph(num_nodes=6, node_type=node_type, relations=relations, relation_specs=specs)


def test_identity_assignment_has_near_zero_path_mass_error():
    from hesf_coarsen.eval.metapath_mass import evaluate_metapath_transition_mass

    graph = _toy_graph()
    assignment = Assignment(np.arange(graph.num_nodes), graph.node_type)
    coarse = coarsen_graph(graph, assignment)
    rows = evaluate_metapath_transition_mass(
        graph,
        coarse,
        assignment,
        [{"name": "two_hop", "steps": [0, 1], "start_type": 0, "end_type": 2}],
        num_probes=4,
        sample_seed=7,
    )
    assert rows[0]["metapath_mass_relative_error"] < 1e-6
    assert rows[0]["metapath_probe_cosine_similarity"] > 0.999


def test_collapsed_assignment_increases_path_mass_error():
    from hesf_coarsen.eval.metapath_mass import evaluate_metapath_transition_mass

    graph = _toy_graph()
    identity = Assignment(np.arange(graph.num_nodes), graph.node_type)
    collapsed = Assignment(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 2], dtype=np.int32))
    path = [{"name": "two_hop", "steps": [0, 1], "start_type": 0, "end_type": 2}]
    identity_rows = evaluate_metapath_transition_mass(graph, coarsen_graph(graph, identity), identity, path, num_probes=4, sample_seed=9)
    collapsed_rows = evaluate_metapath_transition_mass(graph, coarsen_graph(graph, collapsed), collapsed, path, num_probes=4, sample_seed=9)
    assert collapsed_rows[0]["metapath_mass_relative_error"] > identity_rows[0]["metapath_mass_relative_error"] + 1e-3


def test_typed_mass_error_detects_missing_relation_but_untyped_control_can_remain_lower():
    from hesf_coarsen.eval.metapath_mass import evaluate_metapath_transition_mass

    graph = _toy_graph()
    assignment = Assignment(np.arange(graph.num_nodes), graph.node_type)
    flattened = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type,
        relations={2: graph.relations[2]},
        relation_specs={2: graph.relation_specs[2]},
    )
    rows = evaluate_metapath_transition_mass(
        graph,
        flattened,
        assignment,
        [{"name": "two_hop", "steps": [0, 1], "start_type": 0, "end_type": 2}],
        num_probes=4,
        sample_seed=11,
        include_untyped_control=True,
    )
    assert rows[0]["metapath_mass_relative_error"] > 0.1
    assert rows[0]["untyped_metapath_mass_relative_error"] <= rows[0]["metapath_mass_relative_error"]


def test_sequential_metapath_probe_shapes_and_deterministic_probes():
    from hesf_coarsen.eval.metapath_mass import make_terminal_probes, sequential_metapath_probe

    graph = _toy_graph()
    omega_a = make_terminal_probes(graph, terminal_type=2, num_probes=3, seed=123)
    omega_b = make_terminal_probes(graph, terminal_type=2, num_probes=3, seed=123)
    assert np.array_equal(omega_a, omega_b)
    two = sequential_metapath_probe(graph, {"steps": [0, 1], "start_type": 0, "end_type": 2}, omega_a)
    three = sequential_metapath_probe(graph, {"steps": [0, 1, 2], "start_type": 0, "end_type": 2}, omega_a)
    assert two.shape == (graph.num_nodes, 3)
    assert three.shape == (graph.num_nodes, 3)


def test_survival_only_result_is_not_accepted_as_positive():
    from hesf_coarsen.eval.metapath_mass import classify_metapath_mass_evidence

    verdict = classify_metapath_mass_evidence(
        [
            {"method": "HeSF-LVC-P", "typed_exact_step_survival_rate": 1.0, "metapath_mass_relative_error": ""},
            {"method": "flatten-sum", "typed_exact_step_survival_rate": 1.0, "metapath_mass_relative_error": ""},
        ]
    )
    assert verdict["paper_location"] == "not_supported"
    assert "survival" in verdict["reason"]
