"""
Contains methods to split the data into train/validation based on some validation strategy.

Use these methods to generate two non-overlapping tables before passing them into DataLoaders
"""
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np

class ValidationMethod(ABC):
    """
    Base class for validation methods 
    """
    @abstractmethod
    def split(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the Train and Validation split for the input table
        """

class RandomSplit(ValidationMethod):
    """
    Randomly shuffle the data into two parts. The length of each part depends on the [train_split] parameter
    """
    def __init__(self, train_split:float = 0.8) -> None:
        self.train_split = train_split
    
    def split(self, table: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
        row_ids = np.arange(0, len(table), 1)
        np.random.shuffle(row_ids)
        train_ids = row_ids[:int(len(row_ids) * self.train_split)]
        validation_ids = row_ids[int(len(row_ids) * self.train_split):]
        train_table = table.iloc[train_ids, :].copy().reset_index(drop=True)
        validation_table = table.iloc[validation_ids, :].copy().reset_index(drop=True)
        return train_table, validation_table

    def __str__(self) -> str:
        return f'Split the data randomly using np.random.shuffle with a split of {self.train_split}'

class CVSplit(ValidationMethod):
    """
    Special split for doing Cross-Validation. Crops out a certain window from the data as validation. Splits the data 
    into [cv_count] windows and returns the [window_number]'th cut from the data as validation. *Does not randomize*.
    The window count must lie between [0, cv_count - 1]. If a perfect split is not possible, the last split has reduced
    member count to compensate
    """
    def __init__(self, cv_count: int=5, window_number: int=0) -> None:
        assert window_number < cv_count, "window_number must be between [0, cv_count - 1]"
        assert (window_number > -1) & (cv_count > 0), "Unexpected values for cv_count or window_count"
        self.cv_count = cv_count
        self.window_number = window_number
    
    def split(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        table_splits = np.array_split(table, self.cv_count)
        validation_table = table_splits[self.window_number]
        train_table = table_splits[:self.window_number] + table_splits[:self.window_number + 1:]
        train_table = pd.concat(train_table, axis=0)
        return train_table, validation_table
        
    def __str__(self) -> str:
        return f"Splits the data into {self.cv_count} parts and returns the {self.window_number}-th window as \
            validation and the rest as training. Does not randomize!"

class HoldOneOut(ValidationMethod):
    """
    Hold one sample of a specific column out for the training set and the held out one for validation
    """
    def __init__(self, holdout_col_name: str, holdout_value):
        self.holdout_col_name = holdout_col_name
        self.holdout_value = holdout_value
    
    def split(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        validation_table = table[table[self.holdout_col_name] == self.holdout_value].copy()
        train_table = table[table[self.holdout_col_name] != self.holdout_value].copy()
        return train_table, validation_table
    
    def __str__(self) -> str:
        return f'Holds out f{self.holdout_col_name} columns {self.holdout_value} for validation. The rest are used \
            for training'

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
