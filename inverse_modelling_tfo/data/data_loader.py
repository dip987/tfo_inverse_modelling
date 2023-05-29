from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """Custom dataset generated from a table with the x_columns as the predictors and the y_columns
    as the lables
    """
    def __init__(self, table: pd.DataFrame, row_ids: List, x_columns: List[str],
                 y_columns: List[str]):
        super().__init__()
        self.table = table
        self.row_ids = row_ids
        self.x_columns = [table.columns.get_loc(x) for x in x_columns]  # integer column #
        self.y_columns = [table.columns.get_loc(x) for x in y_columns]  # integer column #

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, item):
        x = Tensor(self.table.iloc[item, self.x_columns])
        y = Tensor(self.table.iloc[item, self.y_columns])
        return x, y


def generate_data_loaders(intensity_data_table: pd.DataFrame, params: Dict, x_columns: List[str],
                          y_columns: List[str], train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Convenience function. Creates a shuffled training and validation data loader with the given 
    params using a given Dataframe. Pass in which column names should be included as features and
    which columns are labels.
    Note: Both x and y column lists need to be Lists. Even if there is only a single column.

    params example:
    params = {
        'batch_size': 2,
        'shuffle': False,   # Set to True to shuffle data on each turn. Otherwise its shuffled initially
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


if __name__ == '__main__':
    params1 = {
        'batch_size': 10,
        # data is already shuffled. Set a seed before calling this function for consistency
        'shuffle': False,
        'num_workers': 2
    }
    data = pd.read_pickle(
        r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl')
    train, val = generate_data_loaders(data, params1, [
                                       'SDD', 'Uterus Thickness', 'Maternal Wall Thickness', 'Maternal Mu_a', 'Fetal Mu_a', 'Wave Int'], ['Intensity'])
    for x, y in train:
        print(x, y)
        break

    print("HALT")
