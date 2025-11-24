from torch_geometric.data import Data, DataLoader
import torch_geometric
import torch
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Callable, Optional

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor using min-max normalization."""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def unnormalize_tensor(normalized_tensor: torch.Tensor, original_tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalize a min-max normalized tensor."""
    min_val = original_tensor.min()
    max_val = original_tensor.max()
    unnormalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return unnormalized_tensor

def extract_youngs_moduli(tensor: torch.Tensor) -> float:
    """
    Extract the average Young's moduli from a given tensor.
    This method assumes that the tensor is a flattened representation of a
    compliance matrix, containing only the first 3x3 components and the
    remaining diagonal elements.
    """
    s11 = tensor[0]
    s22 = tensor[4]
    s33 = tensor[8]

    e1 = 1 / s11
    e2 = 1 / s22
    e3 = 1 / s33
    return np.mean([e1, e2, e3])

def evaluate_prediction(data: Data, prediction: torch.Tensor):
    """
    Evaluate a prediction using extracted engineering constants.
    Young's moduli (E1, E2, E3) are calculated as follows:
        E1 = 1 / S11
        E2 = 1 / S22
        E3 = 1 / S33
    To evaluate a prediction, the tensor (T') is first unnormalized using the
    original data tensor (T), followed by the extraction of Young's moduli in
    each direction. The average of Young's moduli is then computed for the 
    prediction, and the absolute error between this value and the original
    data's average Young's moduli is returned.
    """

    # 1. Unnormalize the prediction
    unnorm = unnormalize_tensor(prediction, data.y)

    # 2. Extract engineering constants from original data and prediction
    e = data.mean_E.item() if isinstance(data.mean_E, torch.Tensor) else data.mean_E
    e_pred = extract_youngs_moduli(unnorm)

    # 3. Compute absolute error
    abs_error = abs(e - e_pred)
    return abs_error

def ensure_batch_attribute(data_or_batch: Data) -> Data:
    """
    Ensure a Data or Batch object exposes a valid batch attribute.
    """
    batch_attr = getattr(data_or_batch, 'batch', None)
    if batch_attr is not None and batch_attr.numel() == data_or_batch.x.size(0):
        return data_or_batch

    device = data_or_batch.x.device
    data_or_batch.batch = torch.zeros(
        data_or_batch.x.size(0),
        dtype=torch.long,
        device=device
    )
    return data_or_batch


def train_model_default(
    model,
    train_data: List[Data],
    val_data: List[Data],
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = False
):

    device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = ensure_batch_attribute(batch.to(device))
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y.view_as(pred)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        if verbose and epoch % 10 == 0:
            avg_train_loss = total_train_loss / max(train_steps, 1)
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            val_steps = 0
            for batch in val_loader:
                batch = ensure_batch_attribute(batch.to(device))
                pred = model(batch)
                target = batch.y.view_as(pred)
                loss = criterion(pred, target)
                total_val_loss += loss.item()
                val_steps += 1

        if verbose and epoch % 10 == 0:
            avg_val_loss = total_val_loss / max(val_steps, 1)
            print(f"Epoch {epoch}: val_loss={avg_val_loss:.6f}")

    return model, {
        "final_train_loss": total_train_loss / max(train_steps, 1) if train_steps else 0.0,
        "final_val_loss": total_val_loss / max(val_steps, 1) if val_steps else 0.0,
    }


def evaluate_model(model, dataset: torch.utils.data.Dataset):
    """
    Evaluate a model on a given dataset by computing the average absolute error
    in Young's moduli predictions across all samples.
    """
    model.eval()
    total_error = 0.0
    num_samples = len(dataset)

    with torch.no_grad():
        for data in dataset:
            data = ensure_batch_attribute(data)
            prediction = model(data)
            error = evaluate_prediction(data, prediction)
            total_error += error

    average_error = total_error / num_samples
    return average_error


def kfold_cross_validation(
    graph_data_list: List[Data],
    model_class: Callable,
    model_params: Dict,
    train_fn: Optional[Callable] = None,
    train_params: Optional[Dict] = None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Perform k-fold cross-validation on graph data.
    
    """

    default_train_params = {
        "epochs": 50,
        "lr": 1e-4,
        "batch_size": 32,
        "device": None,
        "verbose": False,
    }
    if train_params:
        default_train_params.update(train_params)

    effective_train_fn = train_fn
    if effective_train_fn is None:
        effective_train_fn = lambda model, train, val: train_model_default(
            model,
            train,
            val,
            **default_train_params,
        )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    fold_models = []
    
    indices = np.arange(len(graph_data_list))
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'='*50}")
        
        train_data = [graph_data_list[i] for i in train_indices]
        val_data = [graph_data_list[i] for i in val_indices]
        
        if verbose:
            print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
        
        # Initialize model for this fold
        model = model_class(**model_params)
        
        trained_model, metrics = effective_train_fn(model, train_data, val_data)
        fold_models.append(trained_model)
        
        val_score = evaluate_model(trained_model, val_data)
        fold_scores.append(val_score)
        
        if verbose:
            print(f"Validation Score (MAE): {val_score:.6f}")
    
    # Calculate statistics
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results")
        print(f"{'='*50}")
        print(f"Mean Score: {mean_score:.6f}")
        print(f"Std Score: {std_score:.6f}")
        print(f"All Fold Scores: {[f'{s:.6f}' for s in fold_scores]}")
    
    return {
        'fold_scores': fold_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'fold_models': fold_models
    }