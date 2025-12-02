import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn

from utils import training

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a dummy parameter so optimizer does not get an empty parameter list
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        # Output depends on self.dummy so backward works
        return batch.x.sum(dim=1, keepdim=True) + self.dummy

def make_loader():
    data_list = []
    for _ in range(3):
        x = torch.ones((11, 11))
        y = torch.ones((11, 1))
        data = Data(x=x, y=y)
        data_list.append(data)
    return DataLoader(data_list, batch_size=2)

def test_train_epoch_runs():
    model = DummyModel()
    loader = make_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    device = "cpu"
    # Model has no parameters, but should still run
    loss = training.train_epoch(model, loader, optimizer, loss_fn, device)
    assert isinstance(loss, float)

def test_mse_over_loader_runs():
    model = DummyModel()
    loader = make_loader()
    device = "cpu"
    mse = training.mse_over_loader(model, loader, device)
    assert isinstance(mse, float)
