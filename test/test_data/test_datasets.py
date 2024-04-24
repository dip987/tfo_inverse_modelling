import unittest
import pandas as pd
import torch
from inverse_modelling_tfo.data.datasets import CustomDataset


class TestCustomDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]})
        self.device = torch.device("cuda")

    def test_custom_dataset(self):
        """
        This test checks if the CustomDataset class is working as expected
        """
        dataset = CustomDataset(self.data, [["a", "b"], ["c", "d"]], self.device)
        self.assertEqual(len(dataset), 3)
        torch.testing.assert_close(dataset[0][0], (torch.tensor([1.0, 4.0], device=self.device)))

    def test_wrong_columns(self):
        """
        This test checks if the CustomDataset class raises an error when the columns are not found in the table
        """
        with self.assertRaises(AssertionError):
            CustomDataset(self.data, [["a", "b"], ["c", "e"]], self.device)

    def test_empty_columns(self):
        """
        This test checks if the CustomDataset class raises an error when the columns are empty
        """
        with self.assertRaises(AssertionError):
            CustomDataset(self.data, [["a", "b"], []], self.device)

    def test_correct_device(self):
        """
        This test checks if the CustomDataset class is using the correct device
        """
        dataset1 = CustomDataset(self.data, [["a", "b"], ["c", "d"]], self.device)
        self.assertEqual(dataset1[0][0].device, torch.device("cuda", index=0))  # First GPU
        dataset2 = CustomDataset(self.data, [["a", "b"], ["c", "d"]], torch.device("cpu"))
        self.assertEqual(dataset2[0][0].device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
