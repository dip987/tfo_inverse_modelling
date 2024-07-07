"""
A bunch of functions dedicated to the creation of tables for the visualization of the results.
"""

from typing import Callable, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rich.table import Table
from rich.console import Console
import torch
import numpy as np
from model_trainer import ValidationMethod
from inverse_modelling_tfo.visualization.visualize import _generate_predictions, PerformanceMetric


performance_calculators: Dict[PerformanceMetric, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred))),
    "mse": lambda y_true, y_pred: float(np.mean((y_true - y_pred) ** 2)),
}


def create_filtered_error_stats_table(
    data: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    y_scaler: StandardScaler,
    model: torch.nn.Module,
    filter_column: str,
    performance_metric: PerformanceMetric = "mae",
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
        performance_metric (PerformanceMetric): Performance metric to be used for error calculation
        validation_method (Optional[ValidationMethod]): Validation method to be used for error calculation

    Returns:
        Console: Rich Console object containing the table. Useful for exporting the table
    """
    ## Sanity Check
    assert filter_column in data.columns, f"Filter column {filter_column} not found in data"

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
    table.add_column("Train Mean", justify="right", style="cyan")
    table.add_column("Train Std", justify="right", style="cyan")
    if validation_method is not None:
        table.add_column("Validation Mean", justify="right", style="magenta")
        table.add_column("Validation Std", justify="right", style="magenta")

    ## Creating the rows
    for value in unique_values:
        row = [str(value)]
        for data_split, prediction in zip(data_splits, predictions):
            filtered_data = data_split[data_split[filter_column] == value]
            filtered_prediction = prediction[data_split[filter_column] == value]
            if len(filtered_data) == 0:
                # Append 2 N/A values corresponding to the mean and std
                row.append("N/A")
                row.append("N/A")
                continue
            y_true = filtered_data[y_columns].to_numpy()
            y_pred = filtered_prediction[y_columns].to_numpy()
            error = performance_calculators[performance_metric](y_true, y_pred)
            row.append(f"{error:.4f}")
        table.add_row(*row)

    console = Console(record=True)
    console.print(table)
    return console


def create_error_stats_table(train_error: pd.DataFrame, val_error: pd.DataFrame) -> Console:
    """
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


# def create_filtered_error_stats_table(train_error: pd.DataFrame, val_error: pd.DataFrame, filter_column: pd.Series):
#     """
#     Plot a table containing the mean and standard deviation of the errors for both the training and validation sets
#     where each row is filtered by the filter_column.
#     """
#     ## Sanity Check
#     assert len(train_error) == len(filter_column), "filter column and error must have the same length"

#     filter_column_name = filter_column.name
#     if filter_column_name is None:
#         filter_column_name = "Filter Value"
#     else:
#         filter_column_name = str(filter_column_name)
#     unique_values = filter_column.unique()
#     unique_values.sort()
#     table = Table(title="Error Statistics")
#     table.add_column("Label", style="green")
#     table.add_column(filter_column_name, style="green")
#     table.add_column("Train Mean", justify="right", style="cyan")
#     table.add_column("Train Mean", justify="right", style="cyan")
#     table.add_column("Train Std", justify="right", style="cyan")
#     table.add_column("Validation Mean", justify="right", style="magenta")
#     table.add_column("Validation Std", justify="right", style="magenta")
#     for value in unique_values:
