import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp


# Class definition for the graph convolutional network model
class GCN(nn.Module):
    """Graph-level regressor: GCNConv + pooling + MLP, uses edge weights and rho."""
    def __init__(self, in_channels: int, embedding_size: int=128, out_dim: int=36, dropout: float=0.0,
                 mlp_layers: int=2, hidden_neurons: int=128, num_convs: int=1, conv_channels=None):
        super().__init__()
        torch.manual_seed(42)
        widths = list(map(int, conv_channels)) if conv_channels is not None else [int(embedding_size)] * max(int(num_convs), 1)
        convs, in_c = [], in_channels
        for w in widths:
            convs.append(GCNConv(in_c, w))
            in_c = w
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        last_embed = widths[-1]
        d_in = last_embed * 2 + 1  # 1+ for rho
        hdim = last_embed if hidden_neurons is None else int(hidden_neurons)
        layers = [nn.Linear(d_in, hdim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(hdim, hdim), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hdim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, data: Data) -> nn.Sequential:
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
    

# Class definition for multi-head graph attention mechanism
class MultiHeadGraphAttention(nn.Module):
    """Multi-head self-attention with adjacency mask and distance penalty."""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int=4, dropout: float=0.1):
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

    def forward(self, X, A, D) -> nn.Linear:
        B, N, _ = X.shape
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        h, d_k = self.num_heads, self.d_k
        Q = Q.view(B, N, h, d_k).transpose(1, 2)
        K = K.view(B, N, h, d_k).transpose(1, 2)
        V = V.view(B, N, h, d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if D is not None:
            scores = scores - self.dist_scale * D.unsqueeze(1)
        scores = scores.masked_fill(A.unsqueeze(1) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, h * d_k)
        return self.out_proj(out)


# Class definition for graph attention network model
class GAT(nn.Module):
    """Attention-enabled graph regressor using MultiHeadGraphAttention and rho."""
    def __init__(self, in_channels, hidden_dim: int=64, out_dim: int=36, num_heads: int=4,
                 num_layers: int=3, mlp_layers: int=2, hidden_neurons: int=128, dropout: float=0.1):
        super().__init__()
        torch.manual_seed(42)
        layers, dim = [], in_channels
        for _ in range(int(num_layers)):
            layers.append(MultiHeadGraphAttention(dim, hidden_dim, num_heads=num_heads, dropout=dropout))
            dim = hidden_dim
        self.att_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        d_in = hidden_dim + 1
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

    def forward(self, data: Data) -> nn.Sequential:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr, rho = data.edge_attr, data.rho.to(x.device)
        A_list, D_list, X_list = self._dense_mats(x, edge_index, batch, edge_attr)
        g_embs = []
        for A, D, Xg in zip(A_list, D_list, X_list):
            h = Xg.unsqueeze(0); A_ = A.unsqueeze(0); D_ = D.unsqueeze(0)
            for att in self.att_layers:
                h = att(h, A_, D_); h = self.activation(h); h = self.dropout(h)
            g_embs.append(h.mean(dim=1))  # (1, hidden_dim)
        g_emb = torch.cat(g_embs, dim=0)
        if rho.dim() == 1:
            rho = rho.unsqueeze(1)
        g_emb = torch.cat([g_emb, rho], dim=1)
        return self.mlp(g_emb)