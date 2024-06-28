"""
Functions Related to visualization of the results
"""

from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


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
    fig_dist, axes = plt.subplots(3, len(columns_to_plot), squeeze=True, figsize=figsize, sharey=True)

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
