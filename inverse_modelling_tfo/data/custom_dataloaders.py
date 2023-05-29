"""
A set of custom Dataloaders used for training NN models
"""
from itertools import product
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import pandas as pd
import numpy as np

class DualWaveDataset(Dataset):
    """
    Internal private class.
    Args:
        table: The entire dataset
        row_ids: Which rows of the given table to actually use during generation
    """

    def __init__(self, table: pd.DataFrame, row_ids: List, x_columns: List[str],
                 y_columns: List[str], data_length: int, seed: int) -> None:
        super().__init__()
        self.table = table
        self.row_ids = row_ids
        self.x_columns = [table.columns.get_loc(
            x) for x in x_columns]  # integer column #
        self.y_columns = [table.columns.get_loc(
            x) for x in y_columns]  # integer column #
        self.data_length = data_length

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        x = Tensor(self.table.iloc[item, self.x_columns])
        y = Tensor(self.table.iloc[item, self.y_columns])
        return x, y


def generate_dualwave_data_loaders(intensity_data_table: pd.DataFrame, params: Dict,
                                   x_columns: List[str], y_columns: List[str],
                                   train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Simulation data loaders which mixes and matches two differnt absorption co-efficients and 
    generates 2 sets of intensity data. Essentially, this mimics a tissue model with two different 
    sets of absorption co-effs at 2 diffent wavelengths.

    Pass it the data directly loaded from the TFO_dataset library to get

    started(With any normalization we need doing -> For example, log10(intensity), normalized TMP, 
    etc.)
    The mixing and matching is random. 

    params example:
    params = {
        'batch_size': 2,
        'num_workers': 2
        }
    :return: training dataloader, validation dataloader

    Args:
        intensity_data_table (pd.DataFrame): Intensity Data in the form of a Pandas Dataframe
        params (Dict): Params which get directly passed onto pytorch's DataLoader base class. This 
        does not interact with anything else within this function
        x_columns (List[str]): Which columns will be treated as the predictors
        y_columns (List[str]): Which columns will be treated as the labels
        train_split (float, optional): What fraction of the data to use for training.
        Defaults to 0.8. The rest 0.20 goes to validation

    Returns:
        Tuple[DataLoader, DataLoader]: Training DataLoader, Validation DataLoader
    """
    maternal_mu_a = intensity_data_table['Maternal Mu_a'].unique()
    fetal_mu_a = intensity_data_table['Maternal Mu_a'].unique()
    possible_combos = list(product(maternal_mu_a, fetal_mu_a))

    # Shuffle and create training + validation row IDs
    randomized_array = np.random.choice(len(intensity_data_table), size=len(intensity_data_table))
    training_indices = randomized_array[:int(len(randomized_array) * train_split)]
    validation_indices = randomized_array[int(len(randomized_array) * train_split):]
    


    



