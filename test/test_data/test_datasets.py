"""
Test Datasets
"""

import unittest
import pandas as pd
import torch
from inverse_modelling_tfo.data.datasets import CustomDataset, SignDetectionDataset


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
        # Create a dataset with 10 groups of 10 data points each
        self.data_groups_x = [torch.randn(10, 10) for _ in range(10)]
        self.data_groups_y = [torch.randint(0, 2, (10,)) for _ in range(10)]
        self.dataset = SignDetectionDataset(self.data_groups_x, self.data_groups_y)

    def test_dataset_initialized_properly(self):
        # Test the length
        self.assertEqual(len(self.dataset), 10)

        # Test the data generation
        for i in range(100):
            x_data, label = self.dataset[i % 10]
            self.assertEqual(x_data.shape, torch.Size([20]))
            self.assertIn(label, [0, 1])

    def test_batches_work_in_a_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=5, shuffle=True)
        for _, (x_data, labels) in enumerate(dataloader):
            self.assertEqual(x_data.shape, torch.Size([5, 20]))
            self.assertEqual(labels.shape, torch.Size([5, 1]))

    def test_works_with_gpu_tensors(self):
        gpu_x_columns = [x.cuda() for x in self.data_groups_x]
        gpu_y_columns = [y.cuda() for y in self.data_groups_y]
        dataset = SignDetectionDataset(gpu_x_columns, gpu_y_columns)
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


if __name__ == "__main__":
    unittest.main()
