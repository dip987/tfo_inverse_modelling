"""
A bunch of functions dedicated to the creation of tables for the visualization of the results.
"""

from typing import Callable, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rich.table import Table
from rich.console import Console
import torch
from torch.utils.data import DataLoader
import numpy as np
from model_trainer import ValidationMethod
from torchmetrics import MeanAbsoluteError, R2Score, Metric
from inverse_modelling_tfo.visualization.visualize import _generate_predictions, PerformanceMetric


def _torch_unscale(y: torch.Tensor, y_scaler: Optional[StandardScaler]) -> torch.Tensor:
    """
    Unscale the output features using the y_scaler object. If the y_scaler object is None, the function will return the
    input tensor as it is. Works with both cpu and gpu tensors
    Args:
        y (torch.Tensor): Tensor containing the output features
        y_scaler (StandardScaler): Scaler object used to scale the output features
    Returns:
        torch.Tensor: Tensor containing the unscaled output features
    """
    ## No Scaler Path
    if y_scaler is None:
        return y

    ## Scaler Path
    scale = torch.tensor(y_scaler.scale_, device=y.device, dtype=y.dtype)
    mean = torch.tensor(y_scaler.mean_, device=y.device, dtype=y.dtype)
    return y * scale + mean


def _calculate_performance(
    loader: DataLoader,
    model: torch.nn.Module,
    metric: Metric,
    y_scaler: Optional[StandardScaler],
    y_index: Optional[int] = None,
) -> float:
    model = model.eval()
    metric.reset()
    for x, y in loader:
        y_pred = _torch_unscale(model(x), y_scaler)
        y = _torch_unscale(y, y_scaler)

        # In case an index is passed, only use that index for the metric calculation - otherwise, use all columns
        if y_index is not None:
            y_pred = y_pred[:, y_index]
            y = y[:, y_index]
        metric(y_pred, y)
    return metric.compute()


def print_performance_metrics(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: Optional[DataLoader] = None,
    y_scaler: Optional[StandardScaler] = None,
    metrics: Optional[List[Metric]] = None,
    filtered_per_output: bool = False,
) -> Console:
    """
    Pretty print the performance metrics for the model on the training and validation sets. Default metrics are MAE and
    R2 Score. Implementaiton note: everything is transfered to the same device as the dataloaders. 

    Args:
        train_loader (DataLoader): DataLoader object for the training set
        validation_loader (DataLoader): DataLoader object for the validation set
        model (torch.nn.Module): PyTorch model object
        y_scaler (Optional[StandardScaler]): Scaler object used to scale the output features. Set to None if the output
                                                features are not scaled
        metrics (Optional[List[Metric]]): List of TorchMetrics objects to be used for calculating the performance.
                                            Default: [MeanAbsoluteError(), R2Score()]
        filtered_per_output (bool): If True, the performance metrics will be calculated for each output column
                                    separately. Default: False
    Returns:
        Console: Rich Console object containing the performance metrics (As well as prininting them)
    """
    if metrics is None:
        metrics = [MeanAbsoluteError(), R2Score()]

    ## Get output shape by loading a single batch
    sample_y = next(iter(train_loader))[1]
    output_column_count = sample_y.shape[1]
    y_indices = range(output_column_count) if filtered_per_output else [None]  # Indices for the output columns

    # Preparing model for evaluation
    model = model.eval()
    model = model.to(sample_y.device)

    ## Convert metrics to the same device as the model - Required for the calculation
    metrics = [metric.to(sample_y.device) for metric in metrics]

    ## Create the column names - One column for each metric for training and validation
    train_columns = [f"Train {metric.__class__.__name__}" for metric in metrics]
    validation_columns = [f"Val {metric.__class__.__name__}" for metric in metrics]

    ## Create the table
    table = Table(title="Performance Metrics")
    for column in train_columns:
        table.add_column(column, style="cyan", justify="right")
    for column in validation_columns:
        table.add_column(column, style="magenta", justify="right")

    ## Calculate the performance metrics for each output column
    for column in y_indices:
        table_row = []
        for train_metric in metrics:
            table_row.append(_calculate_performance(train_loader, model, train_metric, y_scaler))
        if validation_loader is not None:
            for val_metric in metrics:
                table_row.append(_calculate_performance(validation_loader, model, val_metric, y_scaler))
        else:
            table_row += ["N/A"] * len(metrics)  # Padding when no validation set is present
        table.add_row(*[f"{performance:.4f}" for performance in table_row])

    console = Console(record=True)
    console.print(table)
    return console


def _calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    # Convert continuous values to binary using the threshold
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision


def _calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    # Convert continuous values to binary using the threshold
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall


performance_calculators: Dict[PerformanceMetric, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred))),
    "mse": lambda y_true, y_pred: float(np.mean((y_true - y_pred) ** 2)),
    "precision": _calculate_precision,
    "recall": _calculate_recall,
}

std_calculators: Dict[PerformanceMetric, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": lambda y_true, y_pred: float(np.std(np.abs(y_true - y_pred))),
    "mse": lambda y_true, y_pred: float(np.std((y_true - y_pred) ** 2)),
}


def create_filtered_error_stats_table(
    data: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    y_scaler: StandardScaler,
    model: torch.nn.Module,
    filter_column: str,
    performance_metric: Optional[List[PerformanceMetric]] = None,
    validation_method: Optional[ValidationMethod] = None,
) -> Console:
    """
    Get a table of the error statistics for the model predictions on the validation set, filtered by the unique values
    of the filter_column.

    Args:
        data (pd.DataFrame): DataFrame containing the data (Ideally scaled and normalized)
        x_columns (List[str]): List of column names to be used as input features
        y_columns (List[str]): List of column names to be used as output features
        y_scaler (StandardScaler): Scaler object used to scale the output features
        model (torch.nn.Module): PyTorch model object
        filter_column (Optional[str]): Column name to be used for filtering the data
        performance_metric (List[PerformanceMetric]): Performance metrics to be used for error calculation.
                                                      Default: ["mae", "mae_std"]
        validation_method (Optional[ValidationMethod]): Validation method to be used for error calculation

    Returns:
        Console: Rich Console object containing the table. Useful for exporting the table
    """
    ## Sanity Check
    assert filter_column in data.columns, f"Filter column {filter_column} not found in data"

    if performance_metric is None:
        performance_metric = ["mae", "mae_std"]  # Default Performance Metrics

    unique_values = data[filter_column].unique()
    unique_values.sort()

    if validation_method is None:
        data_splits = [data]
    else:
        data_splits = validation_method.split(data)

    predictions = []  # Unscaled Predictions
    for data_split in data_splits:
        predictions.append(_generate_predictions(model, data_split, x_columns, y_columns, y_scaler))
        data_split[y_columns] = y_scaler.inverse_transform(data_split[y_columns])

    table = Table(title="Error Statistics")
    table.add_column(filter_column, style="green")
    for metric in performance_metric:
        table.add_column(f"Train {metric}", style="cyan", justify="right")
    if validation_method is not None:
        for metric in performance_metric:
            table.add_column(f"Validation {metric}", style="magenta", justify="right")

    ## Creating the rows
    for value in unique_values:
        row = [str(value)]
        for data_split, prediction in zip(data_splits, predictions):
            filtered_data = data_split[data_split[filter_column] == value]
            filtered_prediction = prediction[(data_split[filter_column] == value).to_numpy()]
            if len(filtered_data) == 0:
                # Append N/A values corresponding to the performance metrics (Padding when no data is present)
                row += ["N/A"] * len(performance_metric)
                continue
            y_true = filtered_data[y_columns].to_numpy()
            y_pred = filtered_prediction[y_columns].to_numpy()
            for metric in performance_metric:
                calculated_metric = performance_calculators[metric](y_true, y_pred)
                row.append(f"{calculated_metric:.4f}")
        table.add_row(*row)

    console = Console(record=True)
    console.print(table)
    return console


def create_per_column_error_stats_table(
    data: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    y_scaler: StandardScaler,
    model: torch.nn.Module,
    performance_metric: PerformanceMetric = "mae",
    validation_method: Optional[ValidationMethod] = None,
) -> Console:
    if validation_method is None:
        data_splits = [data]
    else:
        data_splits = validation_method.split(data)

    predictions = []  # Unscaled Predictions
    for data_split in data_splits:
        predictions.append(_generate_predictions(model, data_split, x_columns, y_columns, y_scaler))
        data_split[y_columns] = y_scaler.inverse_transform(data_split[y_columns])

    # Create Table Headers
    table = Table(title="Error Statistics")
    table.add_column("Label", style="green")
    table.add_column("Train Mean", justify="right", style="cyan")
    table.add_column("Train Std", justify="right", style="cyan")
    if validation_method is not None:
        table.add_column("Validation Mean", justify="right", style="magenta")
        table.add_column("Validation Std", justify="right", style="magenta")

    # Each row corresponds to a different column
    ## For the Training and Validation Set, create a row for each column
    for y_column in y_columns:
        row = [y_column]
        for data_split, prediction in zip(data_splits, predictions):
            y_true = data_split[y_column].to_numpy()
            y_pred = prediction[y_column].to_numpy()
            error = performance_calculators[performance_metric](y_true, y_pred)
            std = std_calculators[performance_metric](y_true, y_pred)
            row.append(f"{error:.4f}")
            row.append(f"{std:.4f}")
        table.add_row(*row)

    console = Console(record=True)
    console.print(table)
    return console


def create_error_stats_table(train_error: pd.DataFrame, val_error: pd.DataFrame) -> Console:
    """
    **Deprecated** - TODO: Remove this function from all the ipynb files!
    Plot a table containing the mean and standard deviation of the errors for both the training and validation sets.
    Args:
        train_error (pd.DataFrame): DataFrame containing per datapoint training errors in individual columns
        val_error (pd.DataFrame): DataFrame containing per datapoint validation errors in individual columns

    Returns:
        Console: Rich Console object containing the table
    """
    table = Table(title="Error Statistics")
    table.add_column("Label", style="green")
    table.add_column("Train Mean", justify="right", style="cyan")
    table.add_column("Train Std", justify="right", style="cyan")
    table.add_column("Validation Mean", justify="right", style="magenta")
    table.add_column("Validation Std", justify="right", style="magenta")
    for error_column in train_error.columns:
        train_mean = train_error[error_column].mean()
        train_std = train_error[error_column].std()
        val_mean = val_error[error_column].mean()
        val_std = val_error[error_column].std()
        table.add_row(error_column, f"{train_mean:.4f}", f"{train_std:.4f}", f"{val_mean:.4f}", f"{val_std:.4f}")

    console = Console(record=True)
    console.print(table)
    return console
