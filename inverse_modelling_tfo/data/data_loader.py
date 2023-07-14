"""
Dataclasses for training/testing models using our simulation data
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """Custom dataset generated from a table with the x_columns as the predictors and the y_columns
    as the lables
    """

    def __init__(self, table: pd.DataFrame, row_ids: List, x_columns: List[str],
                 y_columns: List[str]):
        super().__init__()
        self.table = torch.Tensor(table.values.astype(float))
        self.row_ids = row_ids
        self.x_columns = [table.columns.get_loc(x) for x in x_columns]
        self.y_columns = [table.columns.get_loc(x) for x in y_columns]

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, item):
        x = self.table[item, self.x_columns]
        y = self.table[item, self.y_columns]
        # x = Tensor(self.table.iloc[item, self.x_columns])
        # y = Tensor(self.table.iloc[item, self.y_columns])
        return x, y


class DifferentialCombinationDataset(Dataset):
    """Create a dataset by combining the columns of 2 random rows with a specific key. 

    Example:
        Combine the observations from two different simulation setups with all but one different 
        tissue model parameter
    """

    def __init__(self, table: pd.DataFrame, fixed_columns: List[str], x_columns: List[str], differential_column: str, total_length: int):
        super().__init__()
        temp_table = table.groupby(fixed_columns)
        self.all_splits = []
        self.split_index = []
        for index, split_df in temp_table:
            self.all_splits.append(torch.Tensor(split_df.values.astype(float)))
            self.split_index.append(index)
        self.x_columns = [table.columns.get_loc(x) for x in x_columns]
        self.differential_column = table.columns.get_loc(differential_column)
        self.fixed_columns = [table.columns.get_loc(x) for x in fixed_columns]
        # self.variable_columns = list(range(len(table.columns)))
        # for fixed_column_index in self.fixed_columns:
        #     self.variable_columns.remove(fixed_column_index)
        self.total_length = total_length
        self.randomized_table_index = np.random.randint(
            0, len(self.all_splits), total_length)
        self.randomized_row_indices = np.random.randint(
            0, len(self.all_splits[0]), (total_length, 2))

    def __len__(self):
        return self.total_length

    def __getitem__(self, item):
        relevant_split = self.all_splits[self.randomized_table_index[item]]
        relevant_row1 = relevant_split[self.randomized_row_indices[item, 0]]
        relevant_row2 = relevant_split[self.randomized_row_indices[item, 1]]
        combined_x = torch.concat([relevant_row1[self.x_columns],
                            relevant_row2[self.x_columns]])
        differential_y = relevant_row1[self.differential_column] - \
            relevant_row2[self.differential_column]
        differential_y = differential_y.view(-1, 1)
        # return x, y
        return combined_x, differential_y


def generate_data_loaders(intensity_data_table: pd.DataFrame, params: Dict, x_columns: List[str],
                          y_columns: List[str], train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Convenience function. Creates a shuffled training and validation data loader with the given 
    params using a given Dataframe. Pass in which column names should be included as features and
    which columns are labels.
    Note: Both x and y column lists need to be Lists. Even if there is only a single column.

    params example:
    params = {
        'batch_size': 2,
        'shuffle': False,   # Set to True to shuffle data on each turn. 
                            Otherwise its shuffled initially
        'num_workers': 2
        }
    :return: training dataloader, validation dataloader

    Args:
        intensity_data_table (pd.DataFrame): Data in the form of a Pandas Dataframe
        params (Dict): Params which get directly passed onto pytorch's DataLoader base class. This 
        does not interact with anything else within this function
        x_columns (List[str]): Which columns will be treated as the predictors
        y_columns (List[str]): Which columns will be treated as the labels
        train_split (float, optional): What fraction of the data to use for training.
        Defaults to 0.8. The rest 0.20 goes to validation

    Returns:
        Tuple[DataLoader, DataLoader]: Training DataLoader, Validation DataLoader
    """
    # Shuffle and create training + validation row IDs
    randomized_array = np.random.choice(
        len(intensity_data_table), size=len(intensity_data_table))
    training_indices = randomized_array[:int(
        len(randomized_array) * train_split)]
    validation_indices = randomized_array[int(
        len(randomized_array) * train_split):]

    # Create the datasets
    training_dataset = CustomDataset(
        intensity_data_table, training_indices, x_columns, y_columns)
    validation_dataset = CustomDataset(
        intensity_data_table, validation_indices, x_columns, y_columns)

    # Create the data loaders
    train_loader = DataLoader(training_dataset, **params)
    validation_loader = DataLoader(validation_dataset, **params)

    return train_loader, validation_loader


def generate_differential_data_loaders(intensity_data_table: pd.DataFrame, params: Dict,
                                       fixed_columns: List[str], x_columns: List[str],
                                       differential_column: List[str], data_length: int,
                                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    training_dataset = DifferentialCombinationDataset(intensity_data_table, fixed_columns, x_columns,
                                                      differential_column, int(data_length * train_split))
    validation_dataset = DifferentialCombinationDataset(intensity_data_table, fixed_columns, x_columns,
                                                        differential_column, int(data_length * (1 - train_split)))
    # Create the data loaders
    train_loader = DataLoader(training_dataset, **params)
    validation_loader = DataLoader(validation_dataset, **params)

    return train_loader, validation_loader


if __name__ == '__main__':
    params1 = {
        'batch_size': 10,
        # data is already shuffled. Set a seed before calling this function for consistency
        'shuffle': False,
        'num_workers': 2
    }
    features = ['SDD', 'Uterus Thickness', 'Maternal Wall Thickness', 'Maternal Mu_a',
                'Fetal Mu_a', 'Wave Int']
    outputs = ['Intensity']
    data_path = r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl'
    data = pd.read_pickle(data_path)
    train, val = generate_data_loaders(data, params1, features, outputs)
    for x, y in train:
        print(x, y)
        break

    print("HALT")
