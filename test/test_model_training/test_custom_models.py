"""
Test the custom models defined in the model_training module
"""
import unittest
import torch
from inverse_modelling_tfo.model_training.custom_models import FeatureResidualNetwork, PMFEstimatorNet


class TestFeatureResidualNetowrk(unittest.TestCase):
    """
    Test the FeatureResidualNetwork class
    """

    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.batch_size = 1
        self.lookup_table = torch.rand(100, self.in_features + self.out_features)
        self.lookup_key_indices = torch.arange(self.in_features, self.out_features + self.in_features)
        self.feature_indices = torch.arange(0, self.in_features)
        self.model = FeatureResidualNetwork(
            [self.in_features, 4, self.lookup_key_indices.shape[0]],
            [0.5, 0.5],
            [self.in_features, 8, self.out_features],
            [0.5, 0.5],
            self.lookup_table,
            self.lookup_key_indices,
            self.feature_indices,
        )

        # Training/Testing with only one row of data messes up BatchNorm's moving-mean calcualtion.
        # Set to eval mode to avoid this
        self.model = self.model.eval()

    def test_network_produces_output(self):
        x = torch.rand(self.batch_size, self.in_features)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_features))


class TestPMFEstimatorNet(unittest.TestCase):
    """
    Test the PMFEstimatorNet class
    """

    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.model1 = PMFEstimatorNet([self.in_features, self.out_features], [0.5])
        self.model2 = PMFEstimatorNet([self.in_features, 8, self.out_features])
        self.model3 = PMFEstimatorNet([self.in_features, 8, 6, self.out_features])
        # Create input and output tensors
        self.x = torch.rand(32, self.in_features)
        self.y = torch.abs(torch.rand(32, self.out_features))
        self.y = self.y / self.y.sum(dim=1, keepdim=True)  # Normalize to sum to 1

    def test_network_produces_output(self):
        y = self.model1(self.x)
        self.assertEqual(y.shape, (32, self.out_features))

    def test_network_produces_output_without_dropout(self):
        y = self.model2(self.x)
        self.assertEqual(y.shape, (32, self.out_features))

    def test_network_produces_output_with_hidden_layer(self):
        y = self.model3(self.x)
        self.assertEqual(y.shape, (32, self.out_features))

    def test_outputs_sum_to_one(self):
        y = self.model1(self.x)
        self.assertTrue(torch.allclose(y.sum(dim=1), torch.ones(32)))

        y = self.model2(self.x)
        self.assertTrue(torch.allclose(y.sum(dim=1), torch.ones(32)))

        y = self.model3(self.x)
        self.assertTrue(torch.allclose(y.sum(dim=1), torch.ones(32)))

    def test_outputs_all_positive(self):
        y = self.model1(self.x)
        self.assertTrue(torch.all(y >= 0))

        y = self.model2(self.x)
        self.assertTrue(torch.all(y >= 0))

        y = self.model3(self.x)
        self.assertTrue(torch.all(y >= 0))

if __name__ == "__main__":
    unittest.main()
