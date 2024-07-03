"""
A bunch of functions dedicated to the creation of tables for the visualization of the results.
"""
import pandas as pd
from rich.table import Table
from rich.console import Console  

def create_error_stats_table(train_error: pd.DataFrame, val_error: pd.DataFrame) -> None:
    """
    Plot a table containing the mean and standard deviation of the errors for both the training and validation sets.
    Args:
        train_error (pd.DataFrame): DataFrame containing per datapoint training errors in individual columns
        val_error (pd.DataFrame): DataFrame containing per datapoint validation errors in individual columns
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
    
    console = Console()
    console.print(table)
