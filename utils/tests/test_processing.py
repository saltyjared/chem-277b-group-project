import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from utils import processing

def test_build_node_features_shape_and_type():
    conn_mat = np.eye(11)
    feats = processing.build_node_features(conn_mat)
    assert isinstance(feats, torch.Tensor)
    assert feats.shape == (11, 11)

def test_compute_edge_index_and_attr():
    conn_mat = np.zeros((11, 11))
    conn_mat[0, 1] = 1
    conn_mat[1, 0] = 1
    edge_index, edge_attr = processing.compute_edge_index_and_attr(conn_mat)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 2
    assert edge_attr.shape == (2, 1)
    assert torch.all(edge_attr >= 0)

def test_construct_graph_data_normalization():
    conn_mat = np.eye(11)
    C_mat = np.arange(36).reshape(6, 6)
    row = pd.Series({
        "connectivity_matrix": conn_mat,
        "compliance_matrix": C_mat,
        "rho": 0.5
    })
    data = processing.construct_graph_data(row, normalize=True)
    assert isinstance(data, Data)
    assert data.x.shape == (11, 11)
    assert data.y.shape == (36,)
    assert torch.all(data.y >= 0) and torch.all(data.y <= 1)
    assert data.rho.shape == (1,)

def test_construct_graph_data_no_normalization():
    conn_mat = np.eye(11)
    C_mat = np.ones((6, 6)) * 5
    row = pd.Series({
        "connectivity_matrix": conn_mat,
        "compliance_matrix": C_mat,
        "rho": 0.5
    })
    data = processing.construct_graph_data(row, normalize=False)
    assert torch.allclose(data.y, torch.full((36,), 5.0))
