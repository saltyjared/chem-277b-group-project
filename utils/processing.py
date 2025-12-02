import pandas as pd
import numpy as np
import torch
from typing import Tuple
from torch_geometric.data import Data

# Define constants
SQRT3 = np.sqrt(3.0)

# Subnode coordinates to provide additional context for node features
# Each entry corresponds to a group of subnodes for a specific node group (11 total)
x_subnodes = [
    [1.0],
    [1.0, 1.0, 0.5],
    [1.0, 1.0, 0.0],
    [0.5, 0.5, 1.0],
    [0.5],
    [0.5, 0.5, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0],
    [1.0, 0.5, 0.0],
    [1.0, 0.5, 0.0],
]
y_subnodes = [
    [1.0],
    [1.0, 0.5, 1.0],
    [1.0, 0.0, 1.0],
    [0.5, 1.0, 0.5],
    [0.5],
    [0.5, 0.0, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0],
    [0.5, 0.0, 1.0],
    [0.0, 1.0, 0.5],
]
z_subnodes = [
    [1.0],
    [0.5, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.5],
    [0.5],
    [0.0, 0.5, 0.5],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.5],
    [0.0],
    [0.0, 1.0, 0.5],
    [0.5, 0.0, 1.0],
]

# Precompute subnode coordinates and normalized versions
num_node_groups = len(x_subnodes)
subnode_coords, subnode_coords_norm = [], []
for i in range(num_node_groups):
    xs = np.array(x_subnodes[i], dtype=float)
    ys = np.array(y_subnodes[i], dtype=float)
    zs = np.array(z_subnodes[i], dtype=float)
    coords = np.stack([xs, ys, zs], axis=1) # (k_i, 3)
    subnode_coords.append(coords)
    coords_norm = 2.0 * (coords - 0.5) # normalized to [-1,1]
    subnode_coords_norm.append(coords_norm)


def build_node_features(conn_mat: np.ndarray) -> torch.tensor:
    """
    11 features per node: [
        subnode_count, 
        degree, 
        3*(x,y,z) of up to 3 subnodes (normalized, padded)
    ]
    """
    num_nodes = conn_mat.shape[0]
    degrees = conn_mat.sum(axis=1)
    feats = []
    for i in range(num_nodes):
        coords_norm = subnode_coords_norm[i]
        k_i = coords_norm.shape[0]
        if k_i >= 3:
            coords3 = coords_norm[:3]
        else:
            reps = 3 - k_i
            pad = np.repeat(coords_norm[0:1, :], reps, axis=0)
            coords3 = np.vstack([coords_norm, pad])
        coords_flat = coords3.reshape(-1) # 9 values
        feats.append([float(k_i), float(degrees[i])] + coords_flat.tolist())
    return torch.tensor(feats, dtype=torch.float32) # (N, 11)


def compute_edge_index_and_attr(conn_mat: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:
    """
    Edge index (2 x n_edges) and attribute (min distance between node groups)
    constructed from connectivity matrix.
    """
    edges = np.array(np.nonzero(conn_mat)).T
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    dists = []
    for (i, j) in edges:
        ci, cj = subnode_coords[i], subnode_coords[j]
        diff = ci[:, None, :] - cj[None, :, :]
        dmat = np.linalg.norm(diff, axis=2)
        d_norm = float(dmat.min()) / SQRT3
        dists.append([d_norm])
    edge_attr = torch.tensor(dists, dtype=torch.float32)
    return edge_index, edge_attr


def construct_graph_data(row: pd.Series, normalize: bool=True) -> Data:
    """
    Construct the graph Data object from a singular row, consisting of:
        x: node features [11, 11]
        edge_index: edge indices [2, n_edges]
        edge_attr: min distance between node groups [n_edges, 1]
        rho: scalar relative density [1]
        y: flattened and min-max normalized compliance matrix [36]
    """
    conn_mat = row['connectivity_matrix']
    C_mat = row['compliance_matrix']
    rho = float(row['rho'])

    # Min-max normalize the compliance matrix if specified
    if normalize:
        cmin, cmax = float(np.min(C_mat)), float(np.max(C_mat))
        C_mat = np.zeros_like(C_mat, dtype=float) if (cmax - cmin) == 0 else (C_mat - cmin) / (cmax - cmin)

    x = build_node_features(conn_mat)
    edge_index, edge_attr = compute_edge_index_and_attr(conn_mat)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(C_mat.flatten(), dtype=torch.float32),
        rho=torch.tensor([rho], dtype=torch.float32),
    )