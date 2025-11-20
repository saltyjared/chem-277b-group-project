from torch_geometric import Data
import torch_geometric
import torch
import numpy as np

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
    e = data.mean_E
    e_pred = extract_youngs_moduli(unnorm)

    # 3. Compute absolute error
    abs_error = abs(e - e_pred)
    return abs_error

def evaluate_model(model, dataset: torch_geometric.data.Dataset):
    """
    Evaluate a model on a given dataset by computing the average absolute error
    in Young's moduli predictions across all samples.
    """
    model.eval()
    total_error = 0.0
    num_samples = len(dataset)

    with torch.no_grad():
        for data in dataset:
            prediction = model(data)
            error = evaluate_prediction(data, prediction)
            total_error += error

    average_error = total_error / num_samples
    return average_error