import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn as nn

from utils import evaluation

class DummyModel(nn.Module):
    def forward(self, batch):
        return batch.x.sum(dim=1, keepdim=True)

def make_loader():
    data_list = []
    for _ in range(2):
        x = torch.ones((6, 1))
        y = torch.ones((6, 1))
        data = Data(x=x, y=y)
        data_list.append(data)
    return [data_list]

def test_evaluate_model_returns_dataframe():
    model = DummyModel()
    loader_dict = {"train": make_loader()[0]}
    device = "cpu"
    df = evaluation.evaluate_model(loader_dict, model, device, "dummy")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"y_true", "y_pred", "split", "model"}
    assert (df["model"] == "dummy").all()

def test_regression_metrics_output():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    metrics = evaluation.regression_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"MAE", "MSE", "RMSE", "R2"}
    assert metrics["MAE"] >= 0
    assert metrics["MSE"] >= 0
    assert metrics["RMSE"] >= 0
