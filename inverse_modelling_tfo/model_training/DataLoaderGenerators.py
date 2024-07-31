"""
Custom DataLoader Generators for the inverse modelling task
"""

from typing import List, Optional, Dict, Tuple
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from model_trainer import DataLoaderGenerator, ValidationMethod
from inverse_modelling_tfo.data.datasets import SignDetectionDataset


def _groupby_to_tensor(
    table: DataFrame, group_columns: List[str], data_columns: List[str], device: torch.device
) -> List[torch.Tensor]:
    """
    Helper function to convert a groupby object into a list of tensors
    """
    table = table.reset_index(drop=True)
    col_indices = [table.columns.tolist().index(x) for x in data_columns]
    x_groups = [
        table.iloc[row_indices, col_indices] for row_indices in list(table.groupby(group_columns).groups.values())
    ]
    x_groups = [torch.tensor(x.values, device=device) for x in x_groups]
    return x_groups


class ChangeDetectionDataLoaderGenerator(DataLoaderGenerator):
    """
    Generates two dataloaders for change detection. Generates a combination of two rows as each datapoint. The values in
    contrast_fixed_columns are the same for both rows. While the values in y_column are different. The 2
    values for the y_column is chosen randomly. The y_column is used to generate the target value. If
    the y_column for the first row is greater than the second row, the target is 1, else 0.
    """

    def __init__(
        self,
        table: DataFrame,
        x_columns: List[str],
        y_column: str,
        contrast_fixed_columns: List[str],
        batch_size: int,
        data_loader_params: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ):
        ## Sanity checks
        assert [x in table.columns for x in contrast_fixed_columns], f"Columns {contrast_fixed_columns} not in table"

        self.contrast_fixed_columns = contrast_fixed_columns
        # Initialize the parent class
        super().__init__(table, x_columns, [y_column], batch_size, data_loader_params, device)

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
        
        # Group the data by the contrast_fixed_columns and convert the groups into a List of Tensors
        train_x_groups = _groupby_to_tensor(train_table, self.contrast_fixed_columns, self.x_columns, self.device)
        train_y_groups = _groupby_to_tensor(train_table, self.contrast_fixed_columns, self.y_columns, self.device)
        val_x_groups = _groupby_to_tensor(validation_table, self.contrast_fixed_columns, self.x_columns, self.device)
        val_y_groups = _groupby_to_tensor(validation_table, self.contrast_fixed_columns, self.y_columns, self.device)

        # Create Datasets using the Tensor List
        training_dataset = SignDetectionDataset(train_x_groups, train_y_groups)
        validation_dataset = SignDetectionDataset(val_x_groups, val_y_groups)

        # Create the data loaders
        train_loader = DataLoader(training_dataset, **self._data_loader_params)
        validation_loader = DataLoader(validation_dataset, **self._data_loader_params)

        return train_loader, validation_loader
