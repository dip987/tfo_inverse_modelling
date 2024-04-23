"""
Functions to generate data distributions, useful for visualizing model predictions and errors
"""
from typing import List, Tuple, Callable
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from inverse_modelling_tfo.data.datasets import DATA_LOADER_INPUT_INDEX, DATA_LOADER_LABEL_INDEX

default_error_func = lambda x, y: np.abs(x - y)

def generate_model_error_and_prediction(
    model: Module,
    data_loader: DataLoader,
    labels: List[str],
    labels_scaler,
    error_func: Callable = default_error_func,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a Model and a corresponding DataLoader, generates 2 dataframes: one containing the absolute errors and one
    containing the model's predicted values

    Args:
        model (Module): Model
        data_loader (DataLoader): Dataloader to the data, this is used to create the distribution
        labels (List[str]): Name of the labels
        labels_scaler: Scaler used to scale the labels
        error_func (Callable[[np.ndarray], np.ndarray]): The function used to calculate error, defaults to np.abs()

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 2 DataFrames containing per datapoint Errors, Predictions
    """
    # Set model to eval
    model = model.eval()
    model_cpu = model.cpu()  # All calculations need to be on the CPU for numpy to work
    row_count, _ = data_loader.dataset[:][1].shape  # Size of the entire y-labels Tensors
    batch_size = data_loader.batch_size if isinstance(data_loader.batch_size, int) else -1

    if batch_size == -1:
        raise ValueError("The data_loader's batch_size has not been defined")

    error_df = np.zeros((row_count, len(labels)))
    prediction_df = np.zeros((row_count, len(labels)))

    # Go through the data one batch at a time (This prevents loading the entire data at once and possibly running
    # into memory issues)
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x = data[DATA_LOADER_INPUT_INDEX].cpu()     # Numpy operations require the data to be on the CPU
            y = data[DATA_LOADER_LABEL_INDEX].cpu()    # Numpy operations require the data to be on the CPU
            predictions = labels_scaler.inverse_transform(model_cpu(x))
            ground_truth = labels_scaler.inverse_transform(y)
            error = error_func(predictions, ground_truth)
            # Store both the errors and predictions (in that order)
            left_pointer = index * batch_size
            right_pointer = left_pointer + len(x)
            error_df[left_pointer:right_pointer, :] = error
            prediction_df[left_pointer:right_pointer, :] = predictions
    error_names = [label + " Error" for label in labels]
    prediction_names = ["Predicted " + label for label in labels]
    error_df = pd.DataFrame(data=error_df, columns=error_names)
    prediction_df = pd.DataFrame(data=prediction_df, columns=prediction_names)
    return error_df, prediction_df
