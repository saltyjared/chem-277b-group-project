import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Batch

def evaluate_model(loader_dict, model, device, model_name) -> pd.DataFrame:
    """
    Evaluate a trained model, specifying the split and model type.
    """
    model.eval()
    ys, preds, splits = [], [], []
    with torch.no_grad():
        for split_name, split_loader in loader_dict.items():
            for batch in split_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y.view_as(pred)
                ys.append(target.detach().cpu().numpy().ravel())
                preds.append(pred.detach().cpu().numpy().ravel())
                splits.append(np.full_like(ys[-1], fill_value=split_name, dtype=object))
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    split_arr = np.concatenate(splits, axis=0)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "split": split_arr, "model": model_name})


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_nodes: int=6) -> dict:
    """
    Compute regression metrics: MAE, MSE, RMSE, R2
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    sst = float(np.sum((y_true - y_true.mean())**2))
    sse = float(np.sum((y_true - y_pred)**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float('nan')
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def visualize_compliance_matrices_dual(model_attn: torch.nn.Module, model_gcn: torch.nn.Module, dataset: list, 
                                       device, num_examples=3, fmt=".2f", num_nodes: int=6) -> None:
    model_attn.eval(); model_gcn.eval()
    if len(dataset) == 0:
        print("Dataset is empty; nothing to visualize."); return
    n = min(num_examples, len(dataset))
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    samples = [dataset[i] for i in idxs]
    batch = Batch.from_data_list(samples).to(device)
    with torch.no_grad():
        pred_attn = model_attn(batch).cpu().numpy()
        pred_gcn  = model_gcn(batch).cpu().numpy()
    true_mats, attn_mats, gcn_mats = [], [], []
    for i, d in enumerate(samples):
        true_mats.append(d.y.view(num_nodes, num_nodes).cpu().numpy())
        attn_mats.append(pred_attn[i].reshape(num_nodes, num_nodes))
        gcn_mats.append(pred_gcn[i].reshape(num_nodes, num_nodes))
    all_vals = np.concatenate([m.ravel() for m in (true_mats + attn_mats + gcn_mats)])
    vmin, vmax = all_vals.min(), all_vals.max()
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 6.5), constrained_layout=True)
    if n == 1: axes = np.array(axes).reshape(3, 1)
    im_for_cbar = None; annot_kws = {"fontsize": 6}
    for i, (t, a, g) in enumerate(zip(true_mats, attn_mats, gcn_mats)):
        ax_t, ax_a, ax_g = axes[0, i], axes[1, i], axes[2, i]
        im_t = sns.heatmap(t, ax=ax_t, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=fmt, annot_kws=annot_kws, square=True, cbar=False)
        ax_t.set_title(f"True #{i+1}", fontsize=9); ax_t.set_xticks([]); ax_t.set_yticks([])
        im_a = sns.heatmap(a, ax=ax_a, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=fmt, annot_kws=annot_kws, square=True, cbar=False)
        ax_a.set_title("Attention", fontsize=8); ax_a.set_xticks([]); ax_a.set_yticks([])
        im_g = sns.heatmap(g, ax=ax_g, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=fmt, annot_kws=annot_kws, square=True, cbar=False)
        ax_g.set_title("GCN", fontsize=8); ax_g.set_xticks([]); ax_g.set_yticks([])
        if im_for_cbar is None: im_for_cbar = im_t.collections[0]
    cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Normalized compliance", fontsize=9); cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Compliance Matrices: True vs Attention vs GCN", fontsize=12, weight="bold")
    plt.show()