import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
from torch import Tensor
from inverse_modelling_tfo.data.normalize import normalize_zero_one


class IntensityDataset(Dataset):
    def __init__(self, table: pd.DataFrame, row_IDs: List):
        super().__init__()
        self.table = table
        self.row_IDs = row_IDs
        self.intensity_index = table.columns.get_loc('Intensity')
        self.non_intensity_indices = list(range(len(table.columns)))
        self.non_intensity_indices.remove(self.intensity_index)

    def __len__(self):
        return len(self.row_IDs)

    def __getitem__(self, item):
        x = Tensor(self.table.iloc[item, self.non_intensity_indices])
        y = Tensor([self.table.iloc[item, self.intensity_index]])
        return x, y


def generate_data_loaders(params, train_split: float = 0.8, normalize_function=normalize_zero_one) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function. Creates a training and a validation data loader with the given params. Uses a given
    normalize_function for data normalization. This function should take the DataFrame as the input. Set to None to
    avoid normalization
    params example:
    params = {
        'batch_size': 2,
        'shuffle': False,   # data is already shuffled. Set a seed before calling this function for consistency
        'num_workers': 2
        }
    :return: training dataloader, validation dataloader
    """
    intensity_data_table = pd.read_pickle(r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl')

    # Normalize
    if normalize_function is not None:
        intensity_data_table = normalize_function(intensity_data_table)


    # Shuffle and create training + validation row IDs
    randomized_array = np.random.choice(len(intensity_data_table), size=len(intensity_data_table))
    training_indices = randomized_array[:int(len(randomized_array) * train_split)]
    validation_indices = randomized_array[int(len(randomized_array) * train_split):]

    # Create the datasets
    training_dataset = IntensityDataset(intensity_data_table, training_indices)
    validation_dataset = IntensityDataset(intensity_data_table, validation_indices)

    # Create the data loaders
    train_loader = DataLoader(training_dataset, **params)
    validation_loader = DataLoader(validation_dataset, **params)

    return train_loader, validation_loader





if __name__ == '__main__':
    params1 = {
        'batch_size': 10,
        'shuffle': False,  # data is already shuffled. Set a seed before calling this function for consistency
        'num_workers': 2
    }
    train, val = generate_data_loaders(params1, 0.7)

    print("HALT")
