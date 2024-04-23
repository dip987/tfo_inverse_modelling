"""
Generate data loaders for training and validation datasets
"""

from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.data import DataLoader
import torch
from inverse_modelling_tfo.data.data_loader import CustomDataset, TripleOutputDataset
from inverse_modelling_tfo.model_training.validation_methods import RandomSplit, ValidationMethod


def generate_data_loader_triple_output(
    table: pd.DataFrame,
    data_loader_params: Dict,
    x_columns: List[str],
    y_columns: List[str],
    extra_columns: List[str],
    validation_method: ValidationMethod = RandomSplit(0.8),
    device: torch.device = torch.device("cuda"),
) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        table (pd.DataFrame): Data in the form of a Pandas Dataframe
        data_loader_params (Dict): Params which get directly passed onto pytorch's DataLoader base class. This
        does not interact with anything else within this function
        x_columns (List[str]): Which columns will be treated as the predictors
        y_columns (List[str]): Which columns will be treated as the labels
        extra_columns (List[str]): Which columns will be treated as the extra outputs
        validation_method (ValidationMethod): How to create the validation dataset? Defaults to a random split of 80%
        training to 20% validation

    Returns:
        Tuple[DataLoader, DataLoader]: Training DataLoader, Validation DataLoader
    """
    # Shuffle and create training + validation row IDs
    train_table, validation_table = validation_method.split(table)

    # Create the datasets
    training_dataset = TripleOutputDataset(train_table, x_columns, y_columns, extra_columns, device)
    validation_dataset = TripleOutputDataset(validation_table, x_columns, y_columns, extra_columns, device)

    # Create the data loaders
    train_loader = DataLoader(training_dataset, **data_loader_params)
    validation_loader = DataLoader(validation_dataset, **data_loader_params)

    return train_loader, validation_loader


def generate_data_loaders(
    table: pd.DataFrame,
    data_loader_params: Dict,
    x_columns: List[str],
    y_columns: List[str],
    validation_method: ValidationMethod = RandomSplit(0.8),
    device: torch.device = torch.device("cuda"),
) -> Tuple[DataLoader, DataLoader]:
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
    training_dataset = CustomDataset(train_table, x_columns, y_columns, device)
    validation_dataset = CustomDataset(validation_table, x_columns, y_columns, device)

    # Create the data loaders
    train_loader = DataLoader(training_dataset, **data_loader_params)
    validation_loader = DataLoader(validation_dataset, **data_loader_params)

    return train_loader, validation_loader
