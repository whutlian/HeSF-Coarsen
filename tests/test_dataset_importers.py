import numpy as np
import torch
from torch_geometric.data import HeteroData

from hesf_coarsen.io.dataset_importers import (
    heterodata_to_hesf_graph,
    ogb_mag_to_hesf_graph,
)
from hesf_coarsen.io.edge_list import load_graph, save_graph
from hesf_coarsen.io.schema import validate_schema


def test_pyhg_heterodata_converts_to_hesf_graph(tmp_path):
    data = HeteroData()
    data["paper"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    data["paper"].y = torch.tensor([2, 3])
    data["author"].num_nodes = 3
    data["paper", "written_by", "author"].edge_index = torch.tensor(
        [[0, 1, 1], [0, 1, 2]],
        dtype=torch.long,
    )
    data["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 1]],
        dtype=torch.long,
    )

    graph = heterodata_to_hesf_graph(data, dataset_name="toy")

    validate_schema(graph)
    assert graph.num_nodes == 5
    assert graph.node_type.tolist() == [0, 0, 1, 1, 1]
    assert graph.relation_specs[0].name == "paper__written_by__author"
    assert graph.labels.tolist()[:2] == [2, 3]
    assert set(graph.features) == {0}

    save_graph(graph, tmp_path)
    loaded = load_graph(tmp_path)
    validate_schema(loaded)
    assert loaded.num_nodes == graph.num_nodes


def test_pyhg_multilabel_targets_are_converted_to_scalar_labels():
    data = HeteroData()
    data["movie"].x = torch.ones((3, 2), dtype=torch.float32)
    data["movie"].y = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    data["actor"].num_nodes = 1
    data["movie", "to", "actor"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    graph = heterodata_to_hesf_graph(data, dataset_name="toy_multilabel")

    validate_schema(graph)
    assert graph.labels.tolist()[:3] == [1, 0, -1]


def test_ogb_mag_dict_converts_to_hesf_graph():
    graph_dict = {
        "num_nodes_dict": {
            "paper": 2,
            "author": 2,
            "institution": 1,
            "field_of_study": 2,
        },
        "node_feat_dict": {
            "paper": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        },
        "edge_index_dict": {
            ("author", "writes", "paper"): np.array([[0, 1], [0, 1]], dtype=np.int64),
            ("paper", "has_topic", "field_of_study"): np.array(
                [[0, 1], [0, 1]],
                dtype=np.int64,
            ),
        },
    }
    labels = np.array([[4], [5]], dtype=np.int64)

    graph = ogb_mag_to_hesf_graph(graph_dict, labels)

    validate_schema(graph)
    assert graph.num_nodes == 7
    assert set(graph.relations) == {0, 1}
    assert graph.labels.tolist()[:2] == [4, 5]
    assert set(graph.features) == {0}


def test_ogb_mag_accepts_label_dict():
    graph_dict = {
        "num_nodes_dict": {"author": 1, "paper": 2},
        "node_feat_dict": {"paper": np.ones((2, 2), dtype=np.float32)},
        "edge_index_dict": {
            ("author", "writes", "paper"): np.array([[0, 0], [0, 1]], dtype=np.int64)
        },
    }
    labels = {"paper": np.array([[7], [8]], dtype=np.int64)}

    graph = ogb_mag_to_hesf_graph(graph_dict, labels)

    validate_schema(graph)
    assert graph.labels.tolist()[-2:] == [7, 8]
