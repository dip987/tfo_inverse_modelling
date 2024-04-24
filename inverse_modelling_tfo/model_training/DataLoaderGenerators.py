"""
ModelTrainer Components that create the Dataloaders used for training/validation
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader

from inverse_modelling_tfo.data.datasets import CustomDataset
from inverse_modelling_tfo.model_training.validation_methods import ValidationMethod


class DataLoaderGenerator:
    """
    A class that generates dataloaders for training and validation datasets
    """

    def __init__(
        self,
        table: DataFrame,
        x_columns: List[str],
        y_columns: List[str],
        batch_size: int,
        data_loader_params: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        A class that generates dataloaders for training and validation datasets

        Args:
            table (DataFrame): Data in the form of a Pandas Dataframe
            x_columns (List[str]): Which columns will be treated as the predictors
            y_columns (List[str]): Which columns will be treated as the labels
            batch_size (int): Batch size for the dataloaders
            data_loader_params (Optional[Dict]): Params which get directly passed onto pytorch's DataLoader base class.
            This does not interact with anything else within this function. Defaults to None
            device (torch.device): Device to move the data to (Should usually be cuda for GPU training). Defaults to
            torch.device("cuda")

        Example:
            table = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'y1': [7, 8, 9], 'y2': [10, 11, 12]})
            x_columns = ['x1', 'x2']
            y_columns = ['y1', 'y2']
            data_loader_params = {'shuffle': True, 'num_workers': 0}
            dataloader_gen = DataLoaderGenerator(table, x_columns, y_columns, 2, data_loader_params)
            train_loader, validation_loader = dataloader_gen.generate()
        """
        self.table = table
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.batch_size = batch_size
        self.device = device
        # Process the data_loader_param
        if data_loader_params is None:
            self._data_loader_params = {}
        else:
            self._data_loader_params = deepcopy(data_loader_params)  # Necessary since the params might be arrays
            if "batch_size" in data_loader_params.keys():
                print("Ignoring batch_size in data_loader_params for DataLoaderGenerator.")
        self._data_loader_params["batch_size"] = batch_size

    def change_batch_size(self, new_batch_size: int) -> None:
        """
        Changes the batch size of the dataloaders

        Args:
            new_batch_size (int): New batch size
        """
        self._data_loader_params["batch_size"] = new_batch_size
        self.batch_size = new_batch_size

    def generate(self, validation_method: ValidationMethod) -> Tuple[DataLoader, DataLoader]:
        """
        Generates the training and validation dataloaders

        Args:
            validation_method (ValidationMethod): How to create the validation dataset? Check the
            model_training.validation_methods module for more info

        Returns:
            Tuple[DataLoader]: Training DataLoader, Validation DataLoader
        """
        # Shuffle and create training + validation row IDs
        train_table, validation_table = validation_method.split(self.table)

        # Create the datasets
        training_dataset = CustomDataset(train_table, [self.x_columns, self.y_columns], self.device)
        validation_dataset = CustomDataset(validation_table, [self.x_columns, self.y_columns], self.device)

        # Create the data loaders
        train_loader = DataLoader(training_dataset, **self._data_loader_params)
        validation_loader = DataLoader(validation_dataset, **self._data_loader_params)

        return train_loader, validation_loader


class DataLoaderGenerator3(DataLoaderGenerator):
    """
    A modified DataLoaderGenerator that generates TripleOutput Datasets in each dataloader. The third output can be
    useful additional information being fed into the loss function
    """

    def __init__(
        self,
        table: DataFrame,
        x_columns: List[str],
        y_columns: List[str],
        extra_columns: List[str],
        batch_size: int,
        data_loader_params: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        A modified DataLoaderGenerator that generates TripleOutput Datasets in each dataloader. The third output can be
        useful additional information being fed into the loss function
        Args:
            table (DataFrame): Data in the form of a Pandas Dataframe
            x_columns (List[str]): Which columns will be treated as the predictors
            y_columns (List[str]): Which columns will be treated as the labels
            extra_columns (List[str]): Which columns will be treated as the extra outputs
            batch_size (int): Batch size for the dataloaders
            data_loader_params (Optional[Dict]): Params which get directly passed onto pytorch's DataLoader base class.
            This does not interact with anything else within this function. Defaults to None
            device (torch.device): Device to move the data to (Should usually be cuda for GPU training). Defaults to
            torch.device("cuda")
        """
        super().__init__(table, x_columns, y_columns, batch_size, data_loader_params, device)
        self.extra_columns = extra_columns

    def generate(self, validation_method: ValidationMethod) -> Tuple[DataLoader, DataLoader]:
        """
        Generates the training and validation dataloaders

        Args:
            validation_method (ValidationMethod): How to create the validation dataset? Check the
            model_training.validation_methods module for more info

        Returns:
            Tuple[DataLoader]: Training DataLoader, Validation DataLoader
        """
        # Shuffle and create training + validation row IDs
        train_table, val_table = validation_method.split(self.table)

        # Create the datasets
        training_dataset = CustomDataset(train_table, [self.x_columns, self.y_columns, self.extra_columns], self.device)
        validation_dataset = CustomDataset(val_table, [self.x_columns, self.y_columns, self.extra_columns], self.device)

        # Create the data loaders
        train_loader = DataLoader(training_dataset, **self._data_loader_params)
        validation_loader = DataLoader(validation_dataset, **self._data_loader_params)

        return train_loader, validation_loader
