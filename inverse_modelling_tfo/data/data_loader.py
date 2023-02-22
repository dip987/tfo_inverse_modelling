import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Tuple
from torch import Tensor


class IntensityDataset(Dataset):
    def __init__(self, table: pd.DataFrame, row_IDs: List, x_columns: List[str], y_columns: List[str]):
        super().__init__()
        self.table = table
        self.row_IDs = row_IDs
        self.x_columns = [table.columns.get_loc(x) for x in x_columns]
        self.y_columns = [table.columns.get_loc(x) for x in y_columns]

    def __len__(self):
        return len(self.row_IDs)

    def __getitem__(self, item):
        x = Tensor(self.table.iloc[item, self.x_columns])
        y = Tensor(self.table.iloc[item, self.y_columns])
        return x, y


def generate_data_loaders(intensity_data_table: pd.DataFrame, params: Dict, x_columns: List[str], y_columns: List[str],
                          train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function. Creates a shuffled training and validation data loader with the given params using a given Dataframe.
    Pass in which column names should be included as features and which ones are labels

    params example:
    params = {
        'batch_size': 2,
        'shuffle': False,   # data is already shuffled. Set a numpy seed before calling this function for consistency
        'num_workers': 2
        }
    :return: training dataloader, validation dataloader
    """

    # Shuffle and create training + validation row IDs
    randomized_array = np.random.choice(
        len(intensity_data_table), size=len(intensity_data_table))
    training_indices = randomized_array[:int(
        len(randomized_array) * train_split)]
    validation_indices = randomized_array[int(
        len(randomized_array) * train_split):]

    # Create the datasets
    training_dataset = IntensityDataset(
        intensity_data_table, training_indices, x_columns, y_columns)
    validation_dataset = IntensityDataset(
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
