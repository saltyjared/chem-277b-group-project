import os
import ast
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
import warnings
warnings.filterwarnings("ignore")

# Append root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Fixed subnode geometry for 11 node groups ----
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
    coords = np.stack([xs, ys, zs], axis=1)  # [k_i, 3]
    subnode_coords.append(coords)
    coords_norm = 2.0 * (coords - 0.5)      # [0,1] -> [-1,1]
    subnode_coords_norm.append(coords_norm)

SQRT3 = np.sqrt(3.0)

def build_node_features(conn_mat):
    """
    Features per node i: [subnode_count, degree, 3*(x,y,z) of up to 3 subnodes (normalized, padded)]
    """
    num_nodes = conn_mat.shape[0]
    degrees = conn_mat.sum(axis=1)
    feats = []
    for i in range(num_nodes):
        coords_norm = subnode_coords_norm[i]   # [k_i, 3]
        k_i = coords_norm.shape[0]
        if k_i >= 3:
            coords3 = coords_norm[:3]
        else:
            reps = 3 - k_i
            pad = np.repeat(coords_norm[0:1, :], reps, axis=0)
            coords3 = np.vstack([coords_norm, pad])
        coords_flat = coords3.reshape(-1)      # 9 values
        feats.append([float(k_i), float(degrees[i])] + coords_flat.tolist())
    return torch.tensor(feats, dtype=torch.float32)  # [N, 11]

def compute_edge_index_and_attr(conn_mat):
    """
    edge_index from nonzero entries (directed).
    edge_attr: normalized min subnode-subnode distance per edge.
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

class GNN_v1(torch.nn.Module):
    """Graph-level regressor: GCNConv + pooling + MLP, uses edge weights and rho."""
    def __init__(self, in_channels, embedding_size=128, out_dim=36, dropout=0.0,
                 mlp_layers=2, hidden_neurons=128, num_convs=1, conv_channels=None):
        super().__init__()
        torch.manual_seed(42)
        widths = list(map(int, conv_channels)) if conv_channels is not None else [int(embedding_size)] * max(int(num_convs), 1)
        convs, in_c = [], in_channels
        for w in widths:
            convs.append(GCNConv(in_c, w))
            in_c = w
        self.convs = torch.nn.ModuleList(convs)
        self.dropout = Dropout(dropout)
        last_embed = widths[-1]
        d_in = last_embed * 2 + 1  # + rho
        hdim = last_embed if hidden_neurons is None else int(hidden_neurons)
        layers = [Linear(d_in, hdim), torch.nn.ReLU(), Dropout(dropout)]
        for _ in range(mlp_layers - 1):
            layers += [Linear(hdim, hdim), torch.nn.ReLU(), Dropout(dropout)]
        layers += [Linear(hdim, out_dim)]
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr.view(-1) if hasattr(data, 'edge_attr') else None
        rho = data.rho.to(x.device)
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
        hg = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)
        if rho.dim() == 1:
            rho = rho.unsqueeze(1)
        hg = torch.cat([hg, rho], dim=1)
        hg = self.dropout(hg)
        return self.mlp(hg)

class MultiHeadGraphAttention(nn.Module):
    """Multi-head self-attention with adjacency mask and distance penalty."""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.dist_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, A, D=None):
        B, N, _ = X.shape
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        h, d_k = self.num_heads, self.d_k
        Q = Q.view(B, N, h, d_k).transpose(1, 2)
        K = K.view(B, N, h, d_k).transpose(1, 2)
        V = V.view(B, N, h, d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if D is not None:
            scores = scores - self.dist_scale * D.unsqueeze(1)  # [B,1,N,N]
        scores = scores.masked_fill(A.unsqueeze(1) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, h * d_k)
        return self.out_proj(out)

class GNN_v2_Attention(nn.Module):
    """Attention-enabled graph regressor using MultiHeadGraphAttention and rho."""
    def __init__(self, in_channels, hidden_dim=64, out_dim=36, num_heads=4,
                 num_layers=3, mlp_layers=2, hidden_neurons=128, dropout=0.1):
        super().__init__()
        torch.manual_seed(42)
        layers, dim = [], in_channels
        for _ in range(int(num_layers)):
            layers.append(MultiHeadGraphAttention(dim, hidden_dim, num_heads=num_heads, dropout=dropout))
            dim = hidden_dim
        self.att_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        d_in = hidden_dim + 1  # + rho
        hdim = hidden_dim if hidden_neurons is None else int(hidden_neurons)
        mlp = [nn.Linear(d_in, hdim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(int(mlp_layers) - 1):
            mlp += [nn.Linear(hdim, hdim), nn.ReLU(), nn.Dropout(dropout)]
        mlp += [nn.Linear(hdim, out_dim)]
        self.mlp = nn.Sequential(*mlp)

    def _dense_mats(self, x, edge_index, batch, edge_attr):
        device = x.device
        num_graphs = int(batch.max().item()) + 1
        src, dst = edge_index
        eattr = edge_attr.view(-1).to(device)
        A_list, D_list, X_list = [], [], []
        for g in range(num_graphs):
            mask = (batch == g)
            idx = mask.nonzero(as_tuple=False).view(-1)
            Xg = x[idx]; Ng = Xg.size(0)
            map_idx = -torch.ones(x.size(0), dtype=torch.long, device=device)
            map_idx[idx] = torch.arange(Ng, device=device)
            e_mask = mask[src] & mask[dst]
            s_loc, d_loc = map_idx[src[e_mask]], map_idx[dst[e_mask]]
            e_g = eattr[e_mask]
            A = torch.zeros(Ng, Ng, device=device)
            D = torch.zeros(Ng, Ng, device=device)
            A[s_loc, d_loc] = 1.0; D[s_loc, d_loc] = e_g
            A[d_loc, s_loc] = 1.0; D[d_loc, s_loc] = e_g
            A.fill_diagonal_(1.0)
            A_list.append(A); D_list.append(D); X_list.append(Xg)
        return A_list, D_list, X_list

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr, rho = data.edge_attr, data.rho.to(x.device)
        A_list, D_list, X_list = self._dense_mats(x, edge_index, batch, edge_attr)
        g_embs = []
        for A, D, Xg in zip(A_list, D_list, X_list):
            h = Xg.unsqueeze(0); A_ = A.unsqueeze(0); D_ = D.unsqueeze(0)
            for att in self.att_layers:
                h = att(h, A_, D_); h = self.activation(h); h = self.dropout(h)
            g_embs.append(h.mean(dim=1))  # [1, hidden_dim]
        g_emb = torch.cat(g_embs, dim=0)
        if rho.dim() == 1:
            rho = rho.unsqueeze(1)
        g_emb = torch.cat([g_emb, rho], dim=1)
        return self.mlp(g_emb)

def load_models():
    """Load the pre-trained GNN models."""
    # Set default model parameters
    gcn = GNN_v1(in_channels=11, embedding_size=128, out_dim=36, dropout=0.0,
                 mlp_layers=2, hidden_neurons=128, num_convs=1).to(DEVICE)
    gat = GNN_v2_Attention(in_channels=11, hidden_dim=64, out_dim=36, 
                           num_heads=4, num_layers=3, mlp_layers=2, 
                           hidden_neurons=128, dropout=0.1).to(DEVICE)

    # Define checkpoint paths
    gcn_checkpoint = os.path.join(ROOT_DIR, 'notebooks', 'saved_models', 'gnn_gcn_checkpoint.pth')
    gat_checkpoint = os.path.join(ROOT_DIR, 'notebooks', 'saved_models', 'gnn_attention_checkpoint.pth')

    # Load state dictionaries
    gcn.load_state_dict(torch.load(gcn_checkpoint, map_location=DEVICE)['model_state_dict'])
    gat.load_state_dict(torch.load(gat_checkpoint, map_location=DEVICE)['model_state_dict'])

    return gcn, gat

def preprocess_input(input_json):
    """Convert input JSON to suitable PyTorch Geometric Data object."""
    # Validate input type
    if not isinstance(input_json, dict):
        raise ValueError("Input must be a dictionary representing the graph.")
    
    # Process input edge list and convert to connectivity matrix
    input_edges = [edge for e in input_json['edges'] for edge in e.split('\n')]
    edge_list = [ast.literal_eval(edge) for edge in input_edges if edge.strip()]
    conn_mat = np.zeros((11, 11), dtype=int)
    for i, j in edge_list:
        conn_mat[i, j] = 1
        conn_mat[j, i] = 1
    
    # Build node features and edge index/attributes
    x = build_node_features(conn_mat)
    edge_index, edge_attr = compute_edge_index_and_attr(conn_mat)
    
    # Extract rho
    rho = torch.tensor([float(input_json.get('rho', 0.5))], dtype=torch.float32)

    # Create geometric Data object
    processed_input = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, rho=rho)

    # Extract max and min compliance of original data (not used in prediction)
    c_max = float(input_json.get('compliance_max'))
    c_min = float(input_json.get('compliance_min'))

    processed_dict = {
        "model_input": processed_input,
        "c_max": c_max,
        "c_min": c_min
    }
    return processed_dict

def generate_predictions(processed_dict):
    """Generate predictions using the loaded GNN models."""
    # Validate input type
    if not isinstance(processed_dict, dict) or 'model_input' not in processed_dict:
        raise ValueError("Input must be a dictionary with 'model_input' key.")

    # Load models and prepare input
    gcn, gat = load_models()
    processed_input = processed_dict["model_input"].to(DEVICE)
    c_max = processed_dict.get("c_max")
    c_min = processed_dict.get("c_min")

    # Wrap single input in a batch
    single_batch = DataLoader([processed_input], batch_size=1)
    gcn.eval()
    gat.eval()

    with torch.no_grad():
        for data in single_batch:
            data = data.to(DEVICE)
        # Generate 6x6 unnormalized predictions
        gcn_output = ((gcn(data)[0].reshape(6, 6) * c_max) - c_min).cpu().numpy()
        gat_output = ((gat(data)[0].reshape(6, 6) * c_max) - c_min).cpu().numpy()
        print(gcn_output)
        print(gat_output)

        # Extract Young's moduli from predictions
        gcn_youngs_modulus = np.mean([1 / gcn_output[0, 0], 1 / gcn_output[1, 1], 1 / gcn_output[2, 2]])
        gat_youngs_modulus = np.mean([1 / gat_output[0, 0], 1 / gat_output[1, 1], 1 / gat_output[2, 2]])

        # Extract shear moduli
        gcn_shear_modulus = np.mean([1 / gcn_output[3, 3], 1 / gcn_output[4, 4], 1 / gcn_output[5, 5]])
        gat_shear_modulus = np.mean([1 / gat_output[3, 3], 1 / gat_output[4, 4], 1 / gat_output[5, 5]])
    
    return {
        "gcn_youngs_prediction": str(f"{gcn_youngs_modulus:.6f}"),
        "gat_youngs_prediction": str(f"{gat_youngs_modulus:.6f}"),
        "gcn_shear_prediction": str(f"{gcn_shear_modulus:.6f}"),
        "gat_shear_prediction": str(f"{gat_shear_modulus:.6f}")
    }