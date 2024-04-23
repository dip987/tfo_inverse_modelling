"""
Dataclasses for training/testing models using our simulation data
"""

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Setting convention for the data loader. When a dataloader returns a tuple, the following indices are used
# To determine which part of the tuple is the input, label and extra data
DATA_LOADER_INPUT_INDEX = 0
DATA_LOADER_LABEL_INDEX = 1
DATA_LOADER_EXTRA_INDEX = 2


class CustomDataset(Dataset):
    """
    Custom dataset generated from a table with the x_columns as the predictors and the y_columns
    as the lables

    PS: This does not shuffle the dataset. Set shuffle to True during training for best results
    """

    def __init__(
        self,
        table: pd.DataFrame,
        columns_directory: List[List[str]],
        device: torch.device = torch.device("cuda"),
    ):
        """
        Custom torch dataset object. Meant to be used with a dataloader on top of it.
        Args:
            table (pd.DataFrame): Data in the form of a Pandas Dataframe
            columns_directory (List[List[str]]): the columns used from the table to create each iterable component of
            the dataset. Each set of columns gets treated as single element in the __getitem__ method
            device (torch.device): Device to move the data to (Should usually be cuda for GPU training)

        Note: The data is always saved as a torch float32 tensor!

        Example:
            columns_directory = [['x1', 'x2'], ['y1', 'y2']]
            table = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'y1': [7, 8, 9], 'y2': [10, 11, 12]})
            dataset = CustomDataset(table, columns_directory)

            # The created dataset will return the following when iterated over
            print(dataset[0])
            -> ([1, 4], [7, 10])
            # Two separate lists, one composed of x1 and x2, the other composed of y1 and y2, returned as a tuple
        """
        super().__init__()
        # Sanity Check
        assert len(columns_directory) > 0, "No columns to use in the dataset"
        assert all([len(x) > 0 for x in columns_directory]), "Empty columns in the columns_directory"
        assert all([x in table.columns for y in columns_directory for x in y]), "Column not found in the table"

        self.table = torch.Tensor(table.values.astype(float)).to(device)
        self.row_ids = np.arange(0, len(table), 1)
        # Convert the column names to indices - since we are using torch tensors instead of pandas dataframes
        self.columns_directory = []
        for columns in columns_directory:
            self.columns_directory.append([table.columns.get_loc(x) for x in columns])

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, item):
        return tuple([self.table[item, x] for x in self.columns_directory])


class DifferentialCombinationDataset(Dataset):
    """Create a dataset by combining the columns of 2 random rows with a specific key.

    Example:
        Combine the observations from two different simulation setups with all but one different
        tissue model parameter
    """

    def __init__(
        self,
        table: pd.DataFrame,
        fixed_columns: List[str],
        x_columns: List[str],
        differential_column: str,
        total_length: int,
        allow_zero_diff: bool = True,
    ):
        # TODO: Unlike other datasets, this one does not move the data to a torch Tensor. Making this very slow!
        super().__init__()
        self.allow_zero_diff = allow_zero_diff
        temp_table = table.groupby(fixed_columns)
        self.all_data_splits = []
        # Use the length of each table as weight - bigger tables = more data = more weight
        self.split_weights = []
        self.split_fixed_columns = []

        for index, split_df in temp_table:
            # tables with only a single row would do us no good -> ignore
            if len(split_df) > 1:
                self.all_data_splits.append(torch.Tensor(split_df.values.astype(float)))
                self.split_weights.append(len(split_df))
                self.split_fixed_columns.append(index)
        # Normalize weights
        self.split_weights = np.array(self.split_weights, dtype=float)
        self.split_weights /= np.sum(self.split_weights)

        self.x_columns = [table.columns.get_loc(x) for x in x_columns]
        self.differential_column = table.columns.get_loc(differential_column)
        self.fixed_columns = [table.columns.get_loc(x) for x in fixed_columns]
        self.total_length = total_length
        # If the loader is empty, skip initialization
        if self.total_length > 0:
            self.randomized_indices_list = np.random.choice(
                range(len(self.all_data_splits)), total_length, p=self.split_weights
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, item):
        relevant_split = self.all_data_splits[self.randomized_indices_list[item]]
        # Randomly(Uniform) pick 2 rows within the table
        pick_index = np.random.choice(range(len(relevant_split)), 2, replace=self.allow_zero_diff)
        relevant_row1 = relevant_split[pick_index[0]]
        relevant_row2 = relevant_split[pick_index[1]]
        combined_x = torch.concat([relevant_row1[self.x_columns], relevant_row2[self.x_columns]])
        differential_y = relevant_row1[self.differential_column] - relevant_row2[self.differential_column]
        return combined_x, differential_y.view(
            1,
        )
