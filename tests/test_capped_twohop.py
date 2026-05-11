import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.candidates.capped_twohop import generate_capped_twohop_candidates
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, validate_schema


def make_hub_graph(num_users=80):
    item_id = num_users
    node_type = np.array([0] * num_users + [1], dtype=np.int32)
    users = np.arange(num_users, dtype=np.int64)
    item = np.full(num_users, item_id, dtype=np.int64)
    relations = {
        0: RelationAdj(
            src=users.copy(),
            dst=item.copy(),
            weight=np.ones(num_users, dtype=np.float32),
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=item.copy(),
            dst=users.copy(),
            weight=np.ones(num_users, dtype=np.float32),
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
    }
    specs = {
        0: RelationSpec(0, "user_to_item", 0, 1),
        1: RelationSpec(1, "item_to_user", 1, 0),
    }
    graph = HeteroGraph(
        num_nodes=num_users + 1,
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
    )
    validate_schema(graph)
    return graph


def test_capped_twohop_does_not_emit_quadratic_pairs_on_hub():
    graph = make_hub_graph(num_users=80)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=6,
        twohop_budget_K2=6,
        middle_degree_cap_policy="none",
        per_middle_pair_cap=9,
    )
    store = BoundedCandidateStore(graph.node_type, K=6)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    z = np.zeros((graph.num_nodes, 4), dtype=np.float32)

    generate_capped_twohop_candidates(graph, z, partition_id, config, store)

    assert store.source_counts().get("capped_twohop", 0) <= 9
    assert len(store.to_pairs()) <= 9
    assert store.counts().max(initial=0) <= 6
