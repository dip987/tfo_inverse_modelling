"""
Dataclasses for training/testing models using our simulation data
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from inverse_modelling_tfo.models import RandomSplit, ValidationMethod


class CustomDataset(Dataset):
    """Custom dataset generated from a table with the x_columns as the predictors and the y_columns
    as the lables

    PS: This does not shuffle the dataset. Set shuffle to True during training for best results
    """

    def __init__(self, table: pd.DataFrame, x_columns: List[str], y_columns: List[str]):
        super().__init__()
        self.table = torch.Tensor(table.values.astype(float))
        self.row_ids = np.arange(0, len(table), 1)
        self.x_columns = [table.columns.get_loc(x) for x in x_columns]
        self.y_columns = [table.columns.get_loc(x) for x in y_columns]

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, item):
        predictors = self.table[item, self.x_columns]
        target = self.table[item, self.y_columns]
        return predictors, target


class DifferentialCombinationDataset(Dataset):
    """Create a dataset by combining the columns of 2 random rows with a specific key. 

    Example:
        Combine the observations from two different simulation setups with all but one different 
        tissue model parameter
    """

    def __init__(self, table: pd.DataFrame, fixed_columns: List[str], x_columns: List[str],
                 differential_column: str, total_length: int, allow_zero_diff: bool = True):
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
            self.randomized_indices_list = np.random.choice(range(len(self.all_data_splits)),
                                                           total_length, p=self.split_weights)

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
        return combined_x, differential_y.view(1,)


def generate_data_loaders(table: pd.DataFrame, data_loader_params: Dict, x_columns: List[str], y_columns: List[str], 
                          validation_method: ValidationMethod = RandomSplit(0.8)) -> Tuple[DataLoader, DataLoader]:
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
        table (pd.DataFrame): Data in the form of a Pandas Dataframe
        data_loader_params (Dict): Params which get directly passed onto pytorch's DataLoader base class. This 
        does not interact with anything else within this function
        x_columns (List[str]): Which columns will be treated as the predictors
        y_columns (List[str]): Which columns will be treated as the labels
        validation_method (ValidationMethod): How to create the validation dataset? Defaults to a random split of 80%
        training to 20% validation

    Returns:
        Tuple[DataLoader, DataLoader]: Training DataLoader, Validation DataLoader
    """
    # Shuffle and create training + validation row IDs
    train_table, validation_table = validation_method.split(table)

    # Create the datasets
    training_dataset = CustomDataset(train_table, x_columns, y_columns)
    validation_dataset = CustomDataset(validation_table, x_columns, y_columns)

    # Create the data loaders
    train_loader = DataLoader(training_dataset, **data_loader_params)
    validation_loader = DataLoader(validation_dataset, **data_loader_params)

    return train_loader, validation_loader


def generate_differential_data_loaders(table: pd.DataFrame, data_loader_params: Dict, fixed_columns: List[str],
                                       x_columns: List[str], differential_column: List[str], data_length: int,
                                       allow_zero_diff: bool, validation_method: ValidationMethod = RandomSplit(0.8),
                                        train_split: float = 0.8 )-> Tuple[DataLoader, DataLoader]:
    """
    Generate a training and testing dataloader set for Differential data. i.e, it takes two rows from the dataset 
    and creates a new pair with the target(y) being the difference in the [differential_column]. The fixed columns 
    stay unchanged. 
    The table is initially split using the validation_method. Combinations are generated in each split independently.
    

    Args:
        table (pd.DataFrame): Original data table
        data_loader_params (Dict): Params which are passed directly onto the dataloader
        fixed_columns (List[str]): Which columns remain unchanged 
        x_columns (List[str]): Features
        differential_column (List[str]): Which columns change their values between the 2 chosen rows
        data_length (int): length of all combinations generated
        allow_zero_diff (bool): Is zero difference between 2 rows allowed?
        validation_method (ValidationMethod, optional): How to initially split the table into two non-overlapping parts.
        Defaults to RandomSplit(0.8). We calculate combinations into each part separately 
        train_split (float, optional): How much of the total data_length is used during training?. Defaults to 0.8.

    Returns:
        Tuple[DataLoader, DataLoader]: _description_
    """
    train_table, validation_table = validation_method.split(table)
    training_dataset = DifferentialCombinationDataset(train_table, fixed_columns,
                                                      x_columns,
                                                      differential_column,
                                                      int(data_length * train_split),
                                                      allow_zero_diff)
    validation_dataset = DifferentialCombinationDataset(validation_table, fixed_columns,
                                                        x_columns,
                                                        differential_column,
                                                        int(data_length * (1 - train_split)),
                                                        allow_zero_diff)
    # Create the data loaders
    train_loader = DataLoader(training_dataset, **data_loader_params)
    validation_loader = DataLoader(validation_dataset, **data_loader_params)

    return train_loader, validation_loader


if __name__ == '__main__':
    params1 = {
        'batch_size': 10,
        'shuffle': True,
        'num_workers': 2
    }
    features = ['SDD', 'Uterus Thickness', 'Maternal Wall Thickness', 'Maternal Mu_a',
                'Fetal Mu_a', 'Wave Int']
    outputs = ['Intensity']
    PATH = r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl'
    data = pd.read_pickle(PATH)
    train, val = generate_data_loaders(data, params1, features, outputs)
    for x, y in train:
        print(x, y)
        break

    print("HALT")
