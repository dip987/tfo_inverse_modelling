"""
Functions Related to visualization of the results
"""

from typing import List, Literal, Optional, Tuple
from enum import IntEnum, auto
from model_trainer import ValidationMethod
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class PlotTypes(IntEnum):
    """
    Enum class to define the different types of plots that can be created.
    """

    ERROR_DISTRIBUTION = auto()
    PREDICTION_DISTRIBUTION = auto()
    TRUTH_DISTRIBUTION = auto()

    def __str__(self):
        return self.name.replace("_", " ").title()


PerformanceMetric = Literal["mae", "mse"]


def plot_error_pred_truth_dist(
    dataset: Dataset,
    train_pred: pd.DataFrame,
    val_pred: pd.DataFrame,
    train_error: pd.DataFrame,
    val_error: pd.DataFrame,
    y_columns: List[str],
    y_scaler: StandardScaler,
    columns_to_plot: Optional[List[str]] = None,
    bin_count: int = 50,
    figsize: Tuple[float, float] = (17, 8),
) -> Figure:
    """
    Plots the distribution of the errors, predictions and ground truth values. This creates a new figure with 3 rows
    of plots.

    Args:
        dataset: Torch Dataset used to get the ground truth values
        train_pred (pd.DataFrame): DataFrame containing the predictions for the training data
        val_pred (pd.DataFrame): DataFrame containing the predictions for the validation data
        train_error (pd.DataFrame): DataFrame containing the errors for the training data
        val_error (pd.DataFrame): DataFrame containing the errors for the validation data
        y_columns (List[str]): List of columns to plot
        y_scaler (StandardScaler): Scaler used to scale the labels
        columns_to_plot (Optional[List[str]]): List of columns to plot, defaults to None
        bin_count (int): Number of bins to use for the histograms, defaults to 50
        figsize (Tuple[float, float]): Size of the figure, defaults to (17, 8)

    Returns:
        Figure: Matplotlib Figure containing 3 rows of plots. The first row contains the error distributions, the second
                row contains the prediction distributions and the third row contains the ground truth distributions.

    """
    if columns_to_plot is None:
        columns_to_plot = y_columns
    fig_dist, axes = plt.subplots(3, len(columns_to_plot), squeeze=False, figsize=figsize, sharey=True)

    train_data_truth = y_scaler.inverse_transform(dataset[:][1].cpu())
    train_data_truth = np.array(train_data_truth)  # Might be a Sparse Matrix, force cast to np.array
    val_data_truth = y_scaler.inverse_transform(dataset[:][1].cpu())
    val_data_truth = np.array(val_data_truth)  # Might be a Sparse Matrix, force cast to np.array

    for i, original_column_name in enumerate(columns_to_plot):
        # Plot Errors
        ax = axes[0, i]
        plt.sca(ax)
        column_name = train_error.columns[i]
        plt.hist(train_error[column_name], bins=bin_count, color="blue", alpha=0.5, label="Train")
        plt.hist(val_error[column_name], bins=bin_count, color="orange", alpha=0.5, label="Validation")

        # Plot Predictions
        ax = axes[1, i]
        plt.sca(ax)
        column_name = train_pred.columns[i]
        plt.hist(train_pred[column_name], bins=bin_count, color="blue", alpha=0.5, label="Train")
        plt.hist(val_pred[column_name], bins=bin_count, color="orange", alpha=0.5, label="Validation")

        # Plot Ground Truth
        ax = axes[2, i]
        plt.sca(ax)
        plt.hist(train_data_truth[:, i], bins=bin_count, color="blue", alpha=0.5, label="Train")
        plt.hist(val_data_truth[:, i], bins=bin_count, color="orange", alpha=0.5, label="Validation")

        # X Label for the bottommost row
        plt.xlabel(original_column_name)

    # Add text to the left of each row of plots
    for i, label in enumerate(["MAE Error", "Prediction", "Ground Truth"]):
        axes[i, 0].set_ylabel(f"{label} Count")

    # Add labels to top-left subplot
    axes[0, 0].legend()

    plt.tight_layout()

    # Return the figure in case we want further modify it
    return fig_dist


def _generate_predictions(
    model: torch.nn.Module, data: pd.DataFrame, x_columns: List[str], y_columns: List[str], y_scaler: StandardScaler
) -> pd.DataFrame:
    """
    Generate predictions from the model for a given dataset.

    Note: consumes the entire data at once, might run into MEM issues for large datasets.
    Args:
        model (torch.nn.Module): Model to use for predictions
        data (pd.DataFrame): DataFrame containing the data to predict
        x_columns (List[str]): List of columns to use as input
        y_columns (List[str]): List of columns to use as output
        y_scaler (StandardScaler): Scaler used to scale the labels

    Returns:
        pd.DataFrame: DataFrame containing the predictions
    """
    model = model.eval()
    x = data[x_columns].values
    x = torch.tensor(x, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(x.device)
    predictions = y_scaler.inverse_transform(model(x).detach().cpu().numpy())
    return pd.DataFrame(data=predictions, columns=y_columns)


def plot_error_distribution(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    y_column: str,
    performance_metric: PerformanceMetric,
    legend: str,
    filter_column: Optional[str] = None,
) -> None:
    """
    Plot the error distribution for a given column.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        predictions (pd.DataFrame): DataFrame containing the predictions
        y_column (str): Column to plot
        performance_metric (PERFORMANCE_METRIC): Performance metric to use
        legend (str): Legend to use for the plot
        filter_column (Optional[str]): Column to use for filtering the data, defaults to None
    """
    ## Convert to same dtype
    data[y_column] = data[y_column].astype(predictions[y_column].dtype)
    error = data[y_column].to_numpy() - predictions[y_column].to_numpy()
    if performance_metric == "mae":
        error = np.abs(error)
    elif performance_metric == "mse":
        error = error**2
    else:
        pass

    plt.hist(error, alpha=0.5, label=legend)


def plot_prediction_distribution(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    y_column: str,
    performance_metric: PerformanceMetric,
    legend: str,
    filter_column: Optional[str] = None,
) -> None:
    """
    Plot the prediction distribution for a given column.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        predictions (pd.DataFrame): DataFrame containing the predictions
        y_column (str): Column to plot
        performance_metric (PERFORMANCE_METRIC): Performance metric to use
        legend (str): Legend to use for the plot
        filter_column (Optional[str]): Column to use for filtering the data, defaults to None
    """
    plt.hist(predictions[y_column], alpha=0.5, label=legend)


def plot_ground_truth_distribution(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    y_column: str,
    performance_metric: PerformanceMetric,
    legend: str,
    filter_column: Optional[str] = None,
) -> None:
    """
    Plot the ground truth distribution for a given column.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        predictions (pd.DataFrame): DataFrame containing the predictions
        y_column (str): Column to plot
        performance_metric (PERFORMANCE_METRIC): Performance metric to use
        legend (str): Legend to use for the plot
        fitler_column (Optional[str]): Column to use for filtering the data, defaults to None
    """
    plt.hist(data[y_column], alpha=0.5, label=legend)


plot_types_implementation = {
    PlotTypes.ERROR_DISTRIBUTION: plot_error_distribution,
    PlotTypes.PREDICTION_DISTRIBUTION: plot_prediction_distribution,
    PlotTypes.TRUTH_DISTRIBUTION: plot_ground_truth_distribution,
}


def plot_performance_distributions(
    data: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    y_scaler: StandardScaler,
    model: torch.nn.Module,
    filter_column: Optional[str] = None,
    performance_metric: PerformanceMetric = "mae",
    validation_method: Optional[ValidationMethod] = None,
    plot_types: Optional[List[PlotTypes]] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """
    Plot diffent types of distributions to visualize the performance of the model.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot (Should include both x and y columns)
        The data should be scaled where the [y_scaler] can used to undo the scaling
        x_columns (List[str]): List of columns to use as input
        y_columns (List[str]): List of columns to use as output
        y_scaler (StandardScaler): Scaler used to scale the labels
        model (torch.nn.Module): Model to use for predictions
        filter_column (Optional[str]): Column to use for filtering the data. Defaults to None, i.e. no filtering
        performance_metric (PERFORMANCE_METRIC): Performance metric to use, defaults to "mae" (Can also be set to mse)
        validation_method (Optional[ValidationMethod]): Validation method to use, defaults to None/No split
        plot_types (List[PlotTypes]): List of PlotTypes to plot. Check out the PlotTypes Enum for all options.
        defaults to [PlotTypes.ERROR_DISTRIBUTION, PlotTypes.PREDICTION_DISTRIBUTION, PlotTypes.TRUTH_DISTRIBUTION]

    Returns:
        Figure: Matplotlib Figure containing the plots
    """
    ## Setup
    if validation_method is None:
        data_splits = [data]
        split_legend = ["All Data"]
    else:
        data_splits = list(validation_method.split(data))
        split_legend = ["Train", "Validation"]

    predictions = []  # Unscaled predictions
    for data_split in data_splits:
        predictions += [_generate_predictions(model, data_split, x_columns, y_columns, y_scaler)]
        data_split[y_columns] = y_scaler.inverse_transform(data_split[y_columns])
        # Undo the scaling before passing to plotter
    # All DataFrames have their y_columns unscaled by this point

    ## Plot
    if plot_types is None:
        plot_types = [PlotTypes.ERROR_DISTRIBUTION, PlotTypes.PREDICTION_DISTRIBUTION, PlotTypes.TRUTH_DISTRIBUTION]

    fig, axes = plt.subplots(len(plot_types), len(y_columns), figsize=figsize, sharey=True, sharex=True, squeeze=False)

    for split_index, data_split in enumerate(data_splits):
        for row_index, plot_type in enumerate(plot_types):
            for col_index, y_column in enumerate(y_columns):
                ax = axes[row_index, col_index]
                plt.sca(ax)
                plot_function = plot_types_implementation[plot_type]
                plot_function(
                    data_split,
                    predictions[split_index],
                    y_column,
                    performance_metric,
                    split_legend[split_index],
                    filter_column,
                )

    ## Legend
    for ax in axes[0]:
        ax.legend()

    ## Labels
    for i, plot_type in enumerate(plot_types):
        axes[i, 0].set_ylabel(f"{plot_type}")
    for i, y_column in enumerate(y_columns):
        axes[-1, i].set_xlabel(y_column)

    return fig
