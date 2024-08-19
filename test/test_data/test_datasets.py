"""
Test Datasets
"""

import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from inverse_modelling_tfo.data.datasets import CustomDataset, SignDetectionDataset, RowCombinationDataset


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


class TestSignDetectionDataset(unittest.TestCase):
    def setUp(self):
        """
        Sets up group_x -> 10 random tensors of size 10x10 (group size x feature size) and
        group_y -> 10 tensors of size 10 (group size x 1) sorted from 0 to 9 (unique values)
        """
        # Create a dataset with 10 groups of 10 data points each
        self.data_groups_x = [torch.randn(10, 10) for _ in range(10)]
        self.data_groups_y = [torch.arange(0, 10) for _ in range(10)]
        self.dataset = SignDetectionDataset(self.data_groups_x, self.data_groups_y, 10)

    def test_dataset_initialized_properly(self):
        # Test the length
        self.assertEqual(len(self.dataset), 10)

        # Test the data generation
        for i in range(100):
            x_data, label = self.dataset[i % 10]
            self.assertEqual(x_data.shape, torch.Size([20]))
            self.assertIn(label, [0, 1])

    def test_batches_work_in_a_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=5, shuffle=True)
        for _, (x_data, labels) in enumerate(dataloader):
            self.assertEqual(x_data.shape, torch.Size([5, 20]))
            self.assertEqual(labels.shape, torch.Size([5, 1]))

    def test_works_with_gpu_tensors(self):
        gpu_x_columns = [x.cuda() for x in self.data_groups_x]
        gpu_y_columns = [y.cuda() for y in self.data_groups_y]
        dataset = SignDetectionDataset(gpu_x_columns, gpu_y_columns, 10)
        self.assertEqual(len(dataset), 10)
        for i in range(100):
            x_data, label = dataset[i % 10]
            self.assertEqual(x_data.shape, torch.Size([20]))
            self.assertIn(label, [0, 1])

    def test_y_has_the_same_device_as_x(self):
        for i in range(100):
            x_data, label = self.dataset[i % 10]
            self.assertEqual(x_data.device, label.device)

    def test_x_and_y_are_float32(self):
        for i in range(10):
            x_data, label = self.dataset[i]
            self.assertEqual(x_data.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.float32)

    def test_not_all_y_have_same_value(self):
        """
        Test for a large number of samples drawn that not all labels are the same
        """
        all_labels = torch.cat([self.dataset[i % 10][1] for i in range(1000)])
        one_count = torch.sum(all_labels == 1).item()
        zero_count = torch.sum(all_labels == 0).item()
        self.assertTrue(one_count != 10)
        self.assertTrue(zero_count != 10)


class TestRowCombinationDataset(unittest.TestCase):
    def setUp(self):
        """
        Sets up group_x -> 10 random tensors of size 10x10 (group size x feature size) and
        group_y -> 10 tensors of size 10 (group size x 1) sorted from 0 to 9 (unique values)
        """
        # Create a dataset with 10 groups of 10 data points each
        self.data_groups_x = [torch.randn(10, 10) for _ in range(10)]
        self.dataset = RowCombinationDataset(self.data_groups_x)

    def test_dataset_initialized_properly(self):
        # Test the length
        self.assertEqual(len(self.dataset), 10)

        # Test the data generation
        for i in range(100):
            (x_data,) = self.dataset[i % 10]
            self.assertEqual(x_data.shape, torch.Size([20]))

    def test_row_combination_chooses_different_rows(self):
        for i in range(100):
            (x_data,) = self.dataset[i % 10]
            x_data1, x_data2 = torch.chunk(x_data, 2)
            self.assertFalse(torch.allclose(x_data1, x_data2))


if __name__ == "__main__":
    unittest.main()
