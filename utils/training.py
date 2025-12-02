import torch
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """
    Trains a model for one epoch using the provided data loader, optimizer, and loss function.
    """
    model.train()
    total_squared_error, total_elements = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y.view_as(pred).to(device)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_squared_error += F.mse_loss(pred, target, reduction='sum').item()
        total_elements += pred.numel()
    return total_squared_error / total_elements if total_elements > 0 else float('nan')


@torch.no_grad()
def mse_over_loader(model, loader, device) -> float:
    """
    Evaluates the model over the data loader and computes the mean squared error.
    """
    model.eval()
    total_se, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        tgt  = batch.y.view_as(pred)
        total_se += F.mse_loss(pred, tgt, reduction='sum').item()
        total_n  += pred.numel()
    return total_se / total_n if total_n > 0 else float('nan')