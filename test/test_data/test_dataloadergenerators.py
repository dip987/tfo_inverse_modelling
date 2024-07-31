"""
Test custom dataloader generators
"""

import unittest
import torch
import pandas as pd
from model_trainer import HoldOneOut
from inverse_modelling_tfo.model_training.DataLoaderGenerators import ChangeDetectionDataLoaderGenerator


class TestChangeDetectionDataLoaderGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame({"key": [1, 1, 2, 2, 3, 3], "value": [5, 4, 2, 1, 8, 9], "x": [1, 2, 3, 4, 5, 6]})
        self.device = torch.device("cuda")
        self.x_columns = ["x"]
        self.y_column = "value"
        self.dataloadergen = ChangeDetectionDataLoaderGenerator(
            self.data, self.x_columns, self.y_column, ["key"], 1, device=self.device
        )
        self.validation_method = HoldOneOut("key", 2)
        self.train_loader, self.val_loader = self.dataloadergen.generate(self.validation_method)

    def test_train_loader_loads_data(self):
        for x, y in self.train_loader:
            print("x:", x)
            print("y:", y)
            self.assertEqual(x.shape, torch.Size([1, 2 * len(self.x_columns)]))
            self.assertEqual(y.shape, torch.Size([1, 1]))

    def test_val_loader_loads_data(self):
        for x, y in self.val_loader:
            print("x:", x)
            print("y:", y)
            self.assertEqual(x.shape, torch.Size([1, 2 * len(self.x_columns)]))
            self.assertEqual(y.shape, torch.Size([1, 1]))

    def test_loader_lengths(self):
        self.assertEqual(len(self.train_loader), 2)
        self.assertEqual(len(self.val_loader), 1)


if __name__ == "__main__":
    unittest.main()
