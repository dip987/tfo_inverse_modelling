"""
Functions Related to visualization of the results
"""

from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def plot_error_pred_truth_dist(
    dataset,
    train_pred,
    val_pred,
    train_error,
    val_error,
    y_columns: List[str],
    y_scaler: StandardScaler,
    columns_to_plot: Optional[List[str]] = None,
    bin_count: int = 50,
):
    # TODO: complete this function
    """
    Plots the distribution of the errors, predictions and ground truth values. This creates a new figure with 3 rows
    of plots.
    """
    if columns_to_plot is None:
        columns_to_plot = y_columns
    fig_dist, axes = plt.subplots(3, len(columns_to_plot), squeeze=True, figsize=(17, 8), sharey=True)

    train_data_truth = y_scaler.inverse_transform(dataset[:][1].cpu())
    val_data_truth = y_scaler.inverse_transform(dataset[:][1].cpu())

    for i in range(len(columns_to_plot)):
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
        plt.xlabel(y_columns[i])

    # Add text to the left of each row of plots
    for i, label in enumerate(["MAE Error", "Prediction", "Ground Truth"]):
        fig_dist.text(0, (2.5 - i) / 3, label, ha="center", va="center", rotation="vertical")

    # Y Labels
    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel("Count")

    # Add labels to top-left subplot
    axes[0, 0].legend()
