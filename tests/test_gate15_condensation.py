import numpy as np

from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from tests.gate15_test_utils import make_gate15_graph


def test_condensation_preserves_targets_and_uses_typed_backgrounds():
    graph = make_gate15_graph()
    selected_support = np.array([4, 6], dtype=np.int64)

    coarse, assignment, diagnostics = build_selected_support_graph(
        graph,
        selected_support,
        SupportSelectorConfig(background_strategy="typed_background"),
        target_node_type=0,
    )

    target_nodes = np.flatnonzero(graph.node_type == 0)
    assert len(np.unique(assignment.assignment[target_nodes])) == len(target_nodes)
    assert diagnostics["selected_support_count"] == 2
    assert diagnostics["background_node_count"] == 2
    for node, supernode in enumerate(assignment.assignment):
        assert graph.node_type[node] == assignment.supernode_type[supernode]
    assert coarse.num_nodes == diagnostics["coarse_nodes"]
