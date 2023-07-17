"""
Contains methods to split the data into train/validation based on some validation strategy.

Use these methods to generate two non-overlapping tables before passing them into DataLoaders
"""

import pandas as pd
import numpy as np


def random_split(table: pd.DataFrame, train_split: float = 0.8):
    """Radomly split the table into two parts - Train, Validation

    Args:
        table (pd.DataFrame): Data Table
        train_split (float, optional): Defaults to 0.6.

    Returns:
        _type_: Train, Validation Table
    """
    row_ids = np.arange(0, len(table), 1)
    np.random.shuffle(row_ids)
    train_ids = row_ids[:int(len(row_ids) * train_split)]
    validation_ids = row_ids[int(len(row_ids) * train_split):]
    train_table = table.iloc[train_ids, :].copy().reset_index(drop=True)
    validation_table = table.iloc[validation_ids, :].copy().reset_index(drop=True)
    return train_table, validation_table
