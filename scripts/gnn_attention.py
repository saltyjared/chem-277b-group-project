
import os

# BLAS conflicts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import types
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# NumPy pickle compatibility
if "numpy._core.numeric" not in sys.modules:
    import numpy.core.numeric as _np_core_numeric
    np_core_numeric = types.ModuleType("numpy._core.numeric")
    np_core_numeric.__dict__.update(_np_core_numeric.__dict__)
    sys.modules["numpy._core.numeric"] = np_core_numeric


class LatticePlainDataset(Dataset):
    """
    Each item:
        A   : [N, N] adjacency / connectivity matrix (float32)
        X   : [N, F] node features (degree, rho)
        y   : [num_targets] target engineering constants
    """

    def __init__(self, df: pd.DataFrame, target_cols=None):
        if target_cols is None:
            target_cols = ["E1", "E2", "E3", "G12", "G23", "G13"]

        self.target_cols = target_cols

        first_A = np.asarray(df.iloc[0]["connectivity_matrix"], dtype=float)
        self.N = first_A.shape[0]

        self.As = []
        self.Xs = []
        self.ys = []

        for _, row in df.iterrows():
            A = np.asarray(row["connectivity_matrix"], dtype=float)
            if A.shape != (self.N, self.N):
                raise ValueError(f"Inconsistent connectivity matrix shape: {A.shape}, expected {(self.N, self.N)}")

            # adjacency with self-loops
            A_with_self = A.copy()
            np.fill_diagonal(A_with_self, 1.0)

            # node features: [degree, rho]
            deg = A.sum(axis=1) + A.sum(axis=0)
            rho = float(row["ρ"])
            X = np.stack([deg, np.full_like(deg, rho)], axis=-1)  # [N, 2]

            y = np.array([row[col] for col in target_cols], dtype=float)

            self.As.append(torch.tensor(A_with_self, dtype=torch.float32))
            self.Xs.append(torch.tensor(X, dtype=torch.float32))
            self.ys.append(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        return self.As[idx], self.Xs[idx], self.ys[idx]


# stack along batch dimension
def lattice_collate(batch):
    As, Xs, ys = zip(*batch)
    A = torch.stack(As, dim=0)  # [B, N, N]
    X = torch.stack(Xs, dim=0)  # [B, N, F]
    y = torch.stack(ys, dim=0)  # [B, num_targets]
    return A, X, y

# Graph attention

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head self-attention over nodes, masked by adjacency matrix.

    Input:
        X: [B, N, F_in]
        A: [B, N, N] (0/1; 1 means "can attend")
    Output:
        X_out: [B, N, F_out]
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        # X: [B, N, F_in], A: [B, N, N]
        B, N, _ = X.shape

        Q = self.W_q(X)  # [B, N, out_dim]
        K = self.W_k(X)
        V = self.W_v(X)

        # reshape for multi head: [B, h, N, d_k]
        B, N, _ = Q.shape
        h = self.num_heads
        d_k = self.d_k

        Q = Q.view(B, N, h, d_k).transpose(1, 2)  # [B, h, N, d_k]
        K = K.view(B, N, h, d_k).transpose(1, 2)  # [B, h, N, d_k]
        V = V.view(B, N, h, d_k).transpose(1, 2)  # [B, h, N, d_k]

        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # mask using adjacency: where A == 0, set score to large negative
        # expand A to [B, 1, N, N]
        A_expanded = A.unsqueeze(1)  # [B, 1, N, N]
        mask = (A_expanded <= 0)  # True where we should mask
        scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=-1)  # [B, h, N, N]
        attn = self.dropout(attn)

        # weighted sum of V: [B, h, N, d_k]
        out = torch.matmul(attn, V)

        # merge heads: [B, N, h * d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, h * d_k)

        # final linear
        out = self.out_proj(out)  # [B, N, out_dim]
        return out


# stack attention layers + graph-level pooling

class AttentionGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        out_dim=6,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        in_dim = node_feat_dim
        for i in range(num_layers):
            layers.append(
                MultiHeadGraphAttention(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # graph-level pooling: mean over nodes
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, A, X):
        """
        A: [B, N, N]
        X: [B, N, F]
        """
        h = X
        for layer in self.layers:
            h_new = layer(h, A)  # [B, N, hidden]
            h = self.activation(h_new)
            h = self.dropout(h)

        # simple mean pooling over nodes
        graph_emb = h.mean(dim=1)  # [B, hidden_dim]

        out = self.readout_mlp(graph_emb)  # [B, out_dim]
        return out


# Train / eval helpers

def split_dataset(dataset, train_frac=0.7, val_frac=0.15, seed=0):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    def _subset(indices):
        return torch.utils.data.Subset(dataset, indices)

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.set_grad_enabled(train):
        for A, X, y in loader:
            A = A.to(device)
            X = X.to(device)
            y = y.to(device)

            pred = model(A, X)
            loss = loss_fn(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)



def main():
    data_path = "/Users/riya/Desktop/chem277_final_project/chem-277b-group-project/data/connectivity_compliance_matrices.pkl"

    target_cols = ["E1", "E2", "E3", "G12", "G23", "G13"]
    batch_size = 32
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    seed = 0

    print(f"Loading data from: {data_path}")
    df = pd.read_pickle(data_path)
    print("DataFrame shape:", df.shape)

    dataset = LatticePlainDataset(df, target_cols=target_cols)
    print(f"Built {len(dataset)} lattice samples with N={dataset.N} nodes each.")

    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    print(
        f"Split sizes → train: {len(train_set)}, "
        f"val: {len(val_set)}, test: {len(test_set)}"
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=lattice_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=lattice_collate
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=lattice_collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_feat_dim = dataset.Xs[0].shape[1]  # 2 (degree, rho)

    model = AttentionGNN(
        node_feat_dim=node_feat_dim,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        out_dim=len(target_cols),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    print("Starting training on device:", device)

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, loss_fn, device, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, loss_fn, device, train=False
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss = {train_loss:.6f} | val_loss = {val_loss:.6f}"
            )

    # final test evaluation
    test_loss = run_epoch(
        model, test_loader, optimizer, loss_fn, device, train=False
    )
    print(f"Final test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
